#!/usr/bin/env python3
"""
Streaming Multi-Teacher Precomputation System
Processes all 10GB+ data with Qwen2.5-7B text + Higgs Audio V2 multimodal teachers
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEXT_TEACHER = "Qwen/Qwen2.5-7B"
AUDIO_TEACHER = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"
OUTPUT_DIR = Path("training/precomputed_teachers_streaming")
BATCH_SIZE = 4  # Per GPU
MAX_LENGTH = 512
CHECKPOINT_EVERY = 100  # Save progress every N batches


class StreamingDataLoader:
    """Load and stream all available training data"""
    
    def __init__(self):
        self.data_dir = Path("training")
        self.total_samples = 0
        
    def load_all_data(self) -> Iterator[Dict]:
        """Stream all available training data"""
        logger.info("=" * 80)
        logger.info("LOADING ALL TRAINING DATA SOURCES")
        logger.info("=" * 80)
        
        # 1. Local JSONL files
        jsonl_files = [
            "comprehensive_data.jsonl",
            "merged_clean.jsonl", 
            "merged_samples.jsonl",
            "UNIFIED_DATASET.jsonl",
            "repo_samples.jsonl",
        ]
        
        # Add all precomputed teacher files
        jsonl_files.extend([
            f for f in os.listdir(self.data_dir) 
            if f.startswith("precomputed_") and f.endswith(".jsonl")
        ])
        
        sample_count = 0
        
        for filename in jsonl_files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                continue
                
            logger.info(f"Loading: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            # Extract text from various formats
                            text = self._extract_text(data)
                            if text:
                                sample_count += 1
                                yield {
                                    'text': text,
                                    'source': filename,
                                    'original': data
                                }
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading {filename}: {e}")
                continue
        
        # 2. HuggingFace streaming datasets
        hf_datasets = [
            "openai/gsm8k",
            "meta-math/MetaMathQA",
            "garage-bAInd/Open-Platypus",
            "teknium/OpenHermes-2.5",
        ]
        
        for dataset_name in hf_datasets:
            logger.info(f"Streaming HuggingFace: {dataset_name}")
            try:
                dataset = load_dataset(dataset_name, split="train", streaming=True)
                for item in dataset.take(50000):  # Limit per dataset
                    text = self._extract_text(item)
                    if text:
                        sample_count += 1
                        yield {
                            'text': text,
                            'source': dataset_name,
                            'original': item
                        }
            except Exception as e:
                logger.warning(f"Error loading {dataset_name}: {e}")
                continue
        
        self.total_samples = sample_count
        logger.info(f"Total samples loaded: {sample_count:,}")
    
    def _extract_text(self, data: Dict) -> str:
        """Extract text from various data formats"""
        # Try common keys
        for key in ['text', 'content', 'prompt', 'instruction', 'question', 'input']:
            if key in data and data[key]:
                return str(data[key])
        
        # Try conversation formats
        if 'messages' in data:
            return " ".join([m.get('content', '') for m in data['messages']])
        
        if 'conversations' in data:
            return " ".join([m.get('value', '') for m in data['conversations']])
        
        # Try precomputed teacher format
        if 'teacher_outputs' in data and 'input_text' in data['teacher_outputs']:
            return data['teacher_outputs']['input_text']
        
        return ""


class MultiTeacherPrecomputer:
    """Precompute outputs from both text and audio teachers"""
    
    def __init__(self):
        self.text_model = None
        self.text_tokenizer = None
        self.audio_teacher = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
    
    def initialize(self):
        """Initialize both teacher models"""
        if self.initialized:
            return
        
        logger.info("=" * 80)
        logger.info("INITIALIZING MULTI-TEACHER SYSTEM")
        logger.info("=" * 80)
        
        # 1. Load text teacher (Qwen2.5-7B) on GPU 1
        logger.info(f"[1/2] Loading text teacher: {TEXT_TEACHER} on GPU 1")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            TEXT_TEACHER,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "1GB", 1: "10GB", 2: "1GB"}  # Prefer GPU 1
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_TEACHER)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_model.eval()
        logger.info("[OK] Text teacher loaded on GPU 1")
        
        # 2. Load audio teacher (Higgs Audio V2) on GPU 2
        logger.info(f"[2/2] Loading audio teacher: {AUDIO_TEACHER} on GPU 2")
        try:
            from training.higgs_audio_teacher import HiggsAudioTeacher
            self.audio_teacher = HiggsAudioTeacher(
                model_id=AUDIO_TEACHER,
                tokenizer_id=AUDIO_TOKENIZER,
                device="cuda:2",  # Force GPU 2
            )
            # Pre-initialize it once here
            self.audio_teacher._ensure_initialized()
            logger.info("[OK] Audio teacher loaded on GPU 2")
        except Exception as e:
            logger.warning(f"[SKIP] Audio teacher failed (OOM on GPU 2): {e}")
            self.audio_teacher = None
            logger.info("[OK] Continuing with text-only mode (saves ~3GB memory)")
        
        self.initialized = True
        logger.info("=" * 80)
        logger.info("[COMPLETE] Multi-teacher system ready")
        logger.info("=" * 80)
        print()
    
    def precompute_batch(self, texts: List[str]) -> Dict:
        """Precompute outputs from both teachers"""
        # Text teacher outputs on GPU 1
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to("cuda:1")
        
        with torch.no_grad():
            outputs = self.text_model(**inputs, output_hidden_states=True)
        
        results = {
            'text_logits': outputs.logits.cpu(),
            'text_hidden_states': [h.cpu() for h in outputs.hidden_states[-4:]],
        }
        
        # Audio teacher outputs (optional)
        if self.audio_teacher:
            try:
                audio_outputs = []
                for text in texts:
                    audio_result = self.audio_teacher.generate_audio(
                        text[:200],  # Limit text for audio generation
                        max_new_tokens=64,
                        temperature=0.7
                    )
                    audio_outputs.append(audio_result)
                results['audio_features'] = audio_outputs
            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")
        
        return results


def main():
    print("=" * 80)
    print("STREAMING MULTI-TEACHER PRECOMPUTATION")
    print("=" * 80)
    print()
    
    # Setup
    data_loader = StreamingDataLoader()
    precomputer = MultiTeacherPrecomputer()
    precomputer.initialize()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint_file = OUTPUT_DIR / "checkpoint.json"
    start_batch = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            start_batch = checkpoint.get('last_batch', 0)
            logger.info(f"Resuming from batch {start_batch}")
    
    # Stream and process data
    batch = []
    batch_idx = 0
    processed_samples = 0
    
    logger.info("Starting streaming precomputation...")
    print()
    
    for sample in data_loader.load_all_data():
        if batch_idx < start_batch:
            batch_idx += 1
            continue
        
        batch.append(sample['text'])
        
        if len(batch) >= BATCH_SIZE:
            logger.info(f"Processing batch {batch_idx + 1} ({processed_samples} samples)...")
            
            try:
                results = precomputer.precompute_batch(batch)
                
                # Save batch
                output_file = OUTPUT_DIR / f"batch_{batch_idx:08d}.pt"
                torch.save(results, output_file)
                
                processed_samples += len(batch)
                logger.info(f"[OK] Saved batch {batch_idx + 1}")
                
                # Save checkpoint
                if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_batch': batch_idx + 1,
                            'total_samples': processed_samples
                        }, f)
                    logger.info(f"[CHECKPOINT] Saved at batch {batch_idx + 1}")
                
            except Exception as e:
                logger.error(f"[ERROR] Batch {batch_idx} failed: {e}")
            
            batch = []
            batch_idx += 1
    
    # Process final partial batch
    if batch:
        logger.info(f"Processing final batch {batch_idx + 1}...")
        try:
            results = precomputer.precompute_batch(batch)
            output_file = OUTPUT_DIR / f"batch_{batch_idx:08d}.pt"
            torch.save(results, output_file)
            processed_samples += len(batch)
            logger.info(f"[OK] Saved final batch")
        except Exception as e:
            logger.error(f"[ERROR] Final batch failed: {e}")
    
    print()
    print("=" * 80)
    print(f"[COMPLETE] Precomputation finished!")
    print(f"Total samples processed: {processed_samples:,}")
    print(f"Total batches: {batch_idx + 1}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
