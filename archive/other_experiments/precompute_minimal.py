#!/usr/bin/env python3
"""
MINIMAL Teacher Precomputation - Absolutely bare bones for stability
- Fixed batch size 16
- Processes 1000 samples at a time
- Explicit GC and CUDA sync
- No tqdm or complex state
- Simple logging every 20 batches
"""

import os
import json
import torch
import gc
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("training/precomputed_teachers_efficient")
MODEL_ID = "Qwen/Qwen2.5-3B"
DEVICE = "cuda:0"
BATCH_SIZE = 16
MAX_LENGTH = 192
CHUNK_SIZE = 1000  # Process 1000 samples then GC


def load_model():
    print(f"[INIT] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"[OK] Model loaded on {DEVICE}")
    return model, tokenizer


def process_batch(model, tokenizer, texts):
    """Single forward pass"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)
    
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(**inputs, output_hidden_states=True)
    
    # Only save last hidden state
    result = {
        'text_hidden_states': [outputs.hidden_states[-1].cpu()]
    }
    
    del inputs, outputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return result


def load_chunk(filepath, start_idx, chunk_size):
    """Load chunk of samples from file"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if len(samples) >= chunk_size:
                break
            try:
                data = json.loads(line.strip())
                for key in ['text', 'content', 'prompt']:
                    if key in data and data[key]:
                        samples.append(str(data[key])[:800])
                        break
            except:
                continue
    return samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Count existing batches
    existing = list(OUTPUT_DIR.glob("batch_*.pt"))
    start_batch = len(existing)
    print(f"[RESUME] Starting from batch {start_batch}")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Process main file
    data_file = Path("training/UNIFIED_DATASET.jsonl")
    if not data_file.exists():
        print(f"[ERROR] {data_file} not found")
        return
    
    print(f"[START] Processing {data_file.name}")
    print(f"[CONFIG] Batch size: {BATCH_SIZE}, Chunk size: {CHUNK_SIZE}")
    
    batch_idx = start_batch
    total_processed = start_batch * BATCH_SIZE
    chunk_offset = 0
    start_time = time.time()
    
    while True:
        # Load chunk
        print(f"\n[CHUNK] Loading samples {chunk_offset} to {chunk_offset + CHUNK_SIZE}...")
        samples = load_chunk(data_file, chunk_offset, CHUNK_SIZE)
        
        if not samples:
            print("[DONE] No more samples")
            break
        
        print(f"[CHUNK] Loaded {len(samples)} samples")
        
        # Process chunk in batches
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            
            try:
                result = process_batch(model, tokenizer, batch)
                output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
                torch.save(result, output_file)
                
                total_processed += len(batch)
                batch_idx += 1
                
                if batch_idx % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    print(f"[PROGRESS] Batch {batch_idx} | {total_processed} samples | {rate:.1f} samp/s")
            
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx}: {e}")
                continue
        
        # Clean up chunk
        del samples
        gc.collect()
        torch.cuda.empty_cache()
        
        chunk_offset += CHUNK_SIZE
        
        # Safety: stop if taking too long per chunk
        chunk_time = time.time() - start_time
        if chunk_offset > CHUNK_SIZE and chunk_time / (chunk_offset / CHUNK_SIZE) > 600:
            print("[WARNING] Processing too slow, stopping")
            break
    
    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Processed {total_processed} samples in {batch_idx} batches")
    print(f"[TIME] {elapsed/60:.1f} minutes | {total_processed/elapsed:.1f} samples/sec")


if __name__ == "__main__":
    main()
