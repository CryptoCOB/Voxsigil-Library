#!/usr/bin/env python3
"""
SIMPLE Teacher Precomputation - Sequential processing, no seeking
- Reads files sequentially from start
- Skips saving batches that already exist
- Fixed batch size 16 for stability
- Explicit memory management
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


def load_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}\n")
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
    
    result = {
        'text_hidden_states': [outputs.hidden_states[-1].cpu()]
    }
    
    del inputs, outputs
    torch.cuda.empty_cache()
    return result


def process_file(model, tokenizer, filepath):
    """Process a single data file"""
    if not filepath.exists():
        print(f"SKIP: {filepath} not found")
        return 0, 0
    
    print(f"Processing {filepath.name}...")
    
    batch_buffer = []
    batch_idx = 0
    saved_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # Extract text
            try:
                data = json.loads(line.strip())
                text = None
                for key in ['text', 'content', 'prompt']:
                    if key in data and data[key]:
                        text = str(data[key])[:800]
                        break
                if not text:
                    continue
                    
                batch_buffer.append(text)
                
                # Process when batch is full
                if len(batch_buffer) >= BATCH_SIZE:
                    output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
                    
                    if output_file.exists():
                        # Skip existing
                        skipped_count += 1
                    else:
                        # Process and save
                        result = process_batch(model, tokenizer, batch_buffer)
                        torch.save(result, output_file)
                        saved_count += 1
                        
                        # Log every 50 new batches
                        if saved_count % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = (saved_count * BATCH_SIZE) / elapsed
                            print(f"  Batch {batch_idx:06d} | Saved: {saved_count} | Skipped: {skipped_count} | {rate:.1f} samp/s")
                    
                    batch_idx += 1
                    batch_buffer.clear()
                    
                    # Periodic cleanup
                    if batch_idx % 100 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                if line_num % 10000 == 0:
                    print(f"  Line {line_num}: Parse error")
                continue
    
    elapsed = time.time() - start_time
    total_samples = saved_count * BATCH_SIZE
    print(f"  DONE: {saved_count} batches saved, {skipped_count} skipped")
    print(f"  Time: {elapsed/60:.1f} min | Rate: {total_samples/elapsed:.1f} samp/s\n")
    
    return saved_count, skipped_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SIMPLE TEACHER PRECOMPUTATION")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max length: {MAX_LENGTH}")
    print("="*60 + "\n")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Process data files
    data_files = [
        Path("training/UNIFIED_DATASET.jsonl"),
        Path("training/merged_samples.jsonl"),
        Path("training/merged_clean.jsonl"),
        Path("training/comprehensive_data.jsonl"),
        Path("training/repo_samples.jsonl"),
    ]
    
    total_saved = 0
    total_skipped = 0
    overall_start = time.time()
    
    for filepath in data_files:
        saved, skipped = process_file(model, tokenizer, filepath)
        total_saved += saved
        total_skipped += skipped
    
    overall_elapsed = time.time() - overall_start
    total_samples = total_saved * BATCH_SIZE
    
    print("="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Total batches saved: {total_saved}")
    print(f"Total batches skipped: {total_skipped}")
    print(f"Total samples processed: {total_samples}")
    print(f"Total time: {overall_elapsed/3600:.2f} hours")
    print(f"Overall rate: {total_samples/overall_elapsed:.1f} samples/sec")
    print("="*60)


if __name__ == "__main__":
    main()
