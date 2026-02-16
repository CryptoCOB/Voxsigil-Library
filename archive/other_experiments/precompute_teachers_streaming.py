#!/usr/bin/env python3
"""
Streaming Teacher Precomputation System
Integrates with HuggingFace datasets, streaming pipeline, and multi-GPU processing
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for teacher precomputation"""
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "training/precomputed_teachers"
    batch_size: int = 8
    max_length: int = 512
    num_hidden_states: int = 4  # Last N hidden states to save
    
    # Dataset options
    dataset_source: str = "local"  # local, huggingface, streaming
    dataset_path: Optional[str] = "training/data/samples.json"
    hf_dataset_name: Optional[str] = None  # e.g., "wikitext"
    hf_dataset_config: Optional[str] = None  # e.g., "wikitext-2-raw-v1"
    hf_split: str = "train"
    streaming: bool = False
    max_samples: Optional[int] = None
    
    # Text field detection
    text_field: Optional[str] = None  # Auto-detect if None
    

class StreamingTeacherPrecomputer:
    """
    Precompute teacher model outputs for distillation with streaming support
    """
    
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device(s)
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            self.device = torch.device("cuda:0")
            logger.info(f"Found {self.device_count} GPU(s)")
        else:
            self.device_count = 1
            self.device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Load teacher model
        logger.info(f"Loading teacher model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device_count > 1 else None
        )
        
        if self.device_count == 1:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"✅ Teacher model loaded on {self.device_count} GPU(s)")
    
    def detect_text_field(self, sample: Dict) -> str:
        """Auto-detect text field in dataset sample"""
        if self.config.text_field:
            return self.config.text_field
        
        # Common field names in order of preference
        candidates = [
            'text', 'content', 'prompt', 'instruction', 
            'question', 'input', 'document', 'passage',
            'code', 'body', 'message', 'raw'
        ]
        
        for field in candidates:
            if field in sample and isinstance(sample[field], str):
                logger.info(f"Auto-detected text field: '{field}'")
                return field
        
        # Fallback: first string field
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                logger.info(f"Using first string field: '{key}'")
                return key
        
        raise ValueError(f"Could not detect text field in sample: {list(sample.keys())}")
    
    def load_dataset(self):
        """Load dataset from configured source"""
        logger.info(f"Loading dataset from: {self.config.dataset_source}")
        
        if self.config.dataset_source == "local":
            # Load from local file
            data_path = Path(self.config.dataset_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            logger.info(f"Loaded {len(data)} samples from {data_path}")
            return Dataset.from_list(data)
        
        elif self.config.dataset_source == "huggingface":
            # Load from HuggingFace Hub
            if not self.config.hf_dataset_name:
                raise ValueError("hf_dataset_name required for huggingface source")
            
            logger.info(f"Loading HF dataset: {self.config.hf_dataset_name}")
            ds = load_dataset(
                self.config.hf_dataset_name,
                self.config.hf_dataset_config,
                split=self.config.hf_split,
                streaming=self.config.streaming
            )
            
            if self.config.max_samples and not self.config.streaming:
                ds = ds.select(range(min(len(ds), self.config.max_samples)))
            
            return ds
        
        elif self.config.dataset_source == "streaming":
            # Load from streaming pipeline
            from scripts.training.streaming_training_pipeline import StreamingTrainingPipeline
            
            workspace = Path("training/streaming_workspace")
            pipeline = StreamingTrainingPipeline(workspace_dir=workspace)
            
            # Process existing batches in streaming workspace
            logger.info("Loading from streaming pipeline workspace")
            return self._load_from_streaming_workspace(workspace)
        
        else:
            raise ValueError(f"Unknown dataset source: {self.config.dataset_source}")
    
    def _load_from_streaming_workspace(self, workspace: Path):
        """Load data from streaming workspace batches"""
        incoming_dir = workspace / "incoming"
        all_samples = []
        
        for batch_file in sorted(incoming_dir.glob("*.json")):
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch = json.load(f)
                if isinstance(batch, list):
                    all_samples.extend(batch)
                else:
                    all_samples.append(batch)
        
        logger.info(f"Loaded {len(all_samples)} samples from streaming workspace")
        return Dataset.from_list(all_samples)
    
    @torch.no_grad()
    def precompute_batch(self, texts: List[str], batch_idx: int):
        """Precompute teacher outputs for a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract logits and last N hidden states
        logits = outputs.logits.cpu()
        hidden_states = outputs.hidden_states[-self.config.num_hidden_states:]
        hidden_states = [h.cpu() for h in hidden_states]
        
        # Save batch
        batch_data = {
            'logits': logits,
            'hidden_states': hidden_states,
            'input_ids': inputs['input_ids'].cpu(),
            'attention_mask': inputs['attention_mask'].cpu(),
            'texts': texts
        }
        
        output_path = self.output_dir / f"batch_{batch_idx:06d}.pt"
        torch.save(batch_data, output_path)
        
        return len(texts)
    
    def run(self):
        """Run precomputation on configured dataset"""
        logger.info("="*80)
        logger.info("🧠 STREAMING TEACHER PRECOMPUTATION")
        logger.info("="*80)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Source: {self.config.dataset_source}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Detect text field
        if isinstance(dataset, (Dataset, list)):
            sample = dataset[0] if isinstance(dataset, Dataset) else dataset[0]
        else:  # IterableDataset
            sample = next(iter(dataset))
        
        text_field = self.detect_text_field(sample)
        
        # Process in batches
        batch_texts = []
        batch_idx = 0
        total_processed = 0
        
        # Handle both regular and streaming datasets
        if isinstance(dataset, IterableDataset):
            dataset_iter = iter(dataset)
            total_samples = self.config.max_samples or "unknown"
        else:
            dataset_iter = iter(dataset)
            total_samples = min(len(dataset), self.config.max_samples) if self.config.max_samples else len(dataset)
        
        logger.info(f"Processing {total_samples} samples...")
        
        pbar = tqdm(total=total_samples if isinstance(total_samples, int) else None, 
                   desc="Precomputing")
        
        try:
            for sample in dataset_iter:
                # Extract text
                text = sample.get(text_field, "")
                if not text:
                    continue
                
                batch_texts.append(text)
                
                # Process batch when full
                if len(batch_texts) >= self.config.batch_size:
                    processed = self.precompute_batch(batch_texts, batch_idx)
                    total_processed += processed
                    batch_idx += 1
                    batch_texts = []
                    pbar.update(processed)
                
                # Check max samples
                if self.config.max_samples and total_processed >= self.config.max_samples:
                    break
            
            # Process remaining texts
            if batch_texts:
                processed = self.precompute_batch(batch_texts, batch_idx)
                total_processed += processed
                batch_idx += 1
                pbar.update(processed)
        
        finally:
            pbar.close()
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ Precomputation complete!")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Total batches: {batch_idx}")
        logger.info(f"📊 Total samples: {total_processed}")
        logger.info("="*80)
        
        return {
            'total_batches': batch_idx,
            'total_samples': total_processed,
            'output_dir': str(self.output_dir)
        }


def main():
    """Main entry point with CLI argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute teacher model outputs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Teacher model name")
    parser.add_argument("--source", choices=["local", "huggingface", "streaming"], 
                       default="local", help="Dataset source")
    parser.add_argument("--dataset-path", help="Path to local dataset file")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset name")
    parser.add_argument("--hf-config", help="HuggingFace dataset config")
    parser.add_argument("--hf-split", default="train", help="HuggingFace dataset split")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    parser.add_argument("--output-dir", default="training/precomputed_teachers",
                       help="Output directory")
    parser.add_argument("--text-field", help="Field containing text (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Create config
    config = TeacherConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dataset_source=args.source,
        dataset_path=args.dataset_path,
        hf_dataset_name=args.hf_dataset,
        hf_dataset_config=args.hf_config,
        hf_split=args.hf_split,
        streaming=args.streaming,
        max_samples=args.max_samples,
        text_field=args.text_field
    )
    
    # Run precomputation
    precomputer = StreamingTeacherPrecomputer(config)
    results = precomputer.run()
    
    logger.info(f"\n✅ Complete! Processed {results['total_samples']} samples in {results['total_batches']} batches")


if __name__ == "__main__":
    main()
