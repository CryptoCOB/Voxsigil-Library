#!/usr/bin/env python3
"""
Standalone Multi-GPU Teacher Precomputation

Runs teacher precomputation directly using all available GPUs without API dependency.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DataParallel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEACHER_MODEL = "Qwen/Qwen2.5-7B"
OUTPUT_DIR = Path("training/precomputed_teachers")
DATA_FILE = Path("training/data/samples.json")
BATCH_SIZE = 8  # Per GPU
MAX_LENGTH = 512


def setup_multi_gpu_teacher():
    """Load teacher model across all GPUs"""
    logger.info(f"Loading teacher model: {TEACHER_MODEL}")
    
    # Check available GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPUs")
    
    if n_gpus == 0:
        logger.warning("No GPUs found! Using CPU (very slow)")
        device = "cpu"
    else:
        device = "cuda"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if n_gpus > 1 else device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    logger.info(f"✅ Teacher model loaded on {n_gpus} GPU(s)")
    return model, tokenizer


def load_samples() -> List[Dict]:
    """Load training samples"""
    if not DATA_FILE.exists():
        logger.error(f"Data file not found: {DATA_FILE}")
        logger.info("Run: python distillation_pipeline/prepare_data.py")
        return []
    
    with open(DATA_FILE) as f:
        samples = json.load(f)
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def precompute_batch(model, tokenizer, texts: List[str], device: str):
    """Precompute teacher outputs for a batch"""
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device if device != "auto" else "cuda")
    
    # Generate teacher outputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    # Extract what we need for distillation
    results = {
        'logits': outputs.logits.cpu(),
        'hidden_states': [h.cpu() for h in outputs.hidden_states[-4:]],  # Last 4 layers
    }
    
    return results


def main():
    print("=" * 80)
    print("🧠 MULTI-GPU TEACHER PRECOMPUTATION")
    print("=" * 80)
    print()
    
    # Setup
    model, tokenizer = setup_multi_gpu_teacher()
    samples = load_samples()
    
    if not samples:
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Processing {len(samples)} samples in {total_batches} batches")
    logger.info(f"Batch size: {BATCH_SIZE} (per GPU)")
    print()
    
    for i in range(0, len(samples), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE
        batch = samples[i:i + BATCH_SIZE]
        texts = [s['text'] for s in batch if 'text' in s]
        
        if not texts:
            continue
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}...")
        
        try:
            results = precompute_batch(model, tokenizer, texts, device)
            
            # Save batch results
            output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
            torch.save(results, output_file)
            
            logger.info(f"✅ Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error processing batch {batch_idx}: {e}")
            continue
    
    print()
    print("=" * 80)
    print(f"✅ Precomputation complete!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total batches: {total_batches}")
    print("=" * 80)


if __name__ == "__main__":
    main()
