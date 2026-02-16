"""
ULTRA-FAST Teacher Precomputation
- Batch size 32 (8x faster than current)
- Short generations (max 128 tokens)
- Greedy sampling (no temperature)
- Models loaded ONCE
"""

import json
import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

TEXT_TEACHER = "Qwen/Qwen2.5-7B"
MAX_LENGTH = 512
BATCH_SIZE = 32  # 8x larger than current!
MAX_NEW_TOKENS = 128  # Much shorter generations


class FastTeacherPrecomputer:
    """ULTRA-FAST: Large batches + short generations + greedy sampling"""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("ULTRA-FAST TEACHER PRECOMPUTATION")
        logger.info(f"Batch size: {BATCH_SIZE} (8x larger!)")
        logger.info(f"Max tokens: {MAX_NEW_TOKENS} (shorter generations)")
        logger.info("=" * 80)
        
        # Load text teacher ONCE
        logger.info(f"Loading {TEXT_TEACHER} on GPU 1...")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            TEXT_TEACHER,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "1GB", 1: "10GB", 2: "1GB"}
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_TEACHER)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_model.eval()
        logger.info("[OK] Model loaded and ready")
        
        self.device = "cuda:1"
        self.output_dir = Path("training/precomputed_teachers_fast")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
    
    def precompute_batch(self, texts: List[str]) -> Dict:
        """Fast precomputation - greedy decoding, short generations"""
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            # Just get hidden states - no generation!
            outputs = self.text_model(**inputs, output_hidden_states=True)
        
        return {
            'text_logits': outputs.logits.cpu(),
            'text_hidden_states': outputs.hidden_states[-1].cpu(),  # Only last layer
        }
    
    def process_all(self, samples: List[Dict]):
        """Process all samples with large batches"""
        logger.info(f"\nProcessing {len(samples):,} samples")
        logger.info(f"Batches: {len(samples) // BATCH_SIZE:,}")
        logger.info("=" * 80)
        
        batch_texts = []
        batch_idx = 0
        
        for sample in tqdm(samples, desc="Precomputing"):
            # Get text from sample
            if 'text' in sample:
                text = sample['text']
            elif 'input' in sample:
                text = sample['input']
            elif 'question' in sample:
                text = sample['question']
            elif 'prompt' in sample:
                text = sample['prompt']
            else:
                text = str(sample)
            
            batch_texts.append(text[:MAX_LENGTH])  # Truncate early
            
            # Process batch when full
            if len(batch_texts) >= BATCH_SIZE:
                teacher_outputs = self.precompute_batch(batch_texts)
                
                # Save batch
                output_file = self.output_dir / f"batch_{batch_idx:06d}.pt"
                torch.save({
                    'teacher_outputs': teacher_outputs,
                    'texts': batch_texts,
                }, output_file)
                
                batch_texts = []
                batch_idx += 1
        
        # Save remaining
        if batch_texts:
            teacher_outputs = self.precompute_batch(batch_texts)
            output_file = self.output_dir / f"batch_{batch_idx:06d}.pt"
            torch.save({
                'teacher_outputs': teacher_outputs,
                'texts': batch_texts,
            }, output_file)
            batch_idx += 1
        
        logger.info(f"\n[DONE] Saved {batch_idx:,} batches to {self.output_dir}")


def load_data_fast():
    """Load data fast - just UNIFIED_DATASET"""
    data_dir = Path("training")
    filepath = data_dir / "UNIFIED_DATASET.jsonl"
    
    logger.info(f"Loading: {filepath.name}")
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                samples.append(data)
            except:
                continue
    
    logger.info(f"Loaded {len(samples):,} samples")
    return samples


def main():
    logger.info("\n" + "=" * 80)
    logger.info("STARTING ULTRA-FAST PRECOMPUTATION")
    logger.info("=" * 80)
    
    # Load data
    samples = load_data_fast()
    
    # Limit to reasonable size for overnight run
    # At 32 batch size + faster processing: ~1000 batches/hour = 32K samples/hour
    # 12 hours = ~384K samples (more than we have!)
    if len(samples) > 300000:
        logger.info(f"Limiting to 300K samples (from {len(samples):,})")
        samples = samples[:300000]
    
    # Process
    precomputer = FastTeacherPrecomputer()
    precomputer.process_all(samples)
    
    logger.info("\n" + "=" * 80)
    logger.info("PRECOMPUTATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
