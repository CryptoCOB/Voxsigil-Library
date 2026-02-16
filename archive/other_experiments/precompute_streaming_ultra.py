#!/usr/bin/env python3
"""
ULTRA-FAST Streaming Teacher Precomputation
- Zero RAM bloat: streams data incrementally
- Adaptive batching: automatically reduces batch size when slow
- Memory-safe: checks GPU free memory before forward pass
- Stall detection: if batch takes >target_sec, reduces batch
- Resume support: continues from last saved batch
"""

import os
import json
import torch
import logging
import argparse
import time
from pathlib import Path
from typing import Iterator, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-3B")
OUTPUT_DIR = Path("training/precomputed_teachers_efficient")


def build_args():
    p = argparse.ArgumentParser(description="Ultra-fast streaming precomputation")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--device", default="cuda:0", help="CUDA device")
    p.add_argument("--batch-size", type=int, default=32, help="Initial batch size")
    p.add_argument("--min-batch", type=int, default=4, help="Minimum adaptive batch size")
    p.add_argument("--max-length", type=int, default=192, help="Max input tokens")
    p.add_argument("--target-sec", type=float, default=5.0, help="Target seconds per batch")
    p.add_argument("--hidden-last-n", type=int, default=1, help="Hidden layers to save")
    p.add_argument("--no-logits", action="store_true", help="Skip logits")
    p.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    p.add_argument("--resume", action="store_true", help="Resume from last batch")
    p.add_argument("--perf-log-every", type=int, default=100, help="Perf log interval")
    return p.parse_args()


def stream_data(limit: int = 0) -> Iterator[Dict]:
    """Stream data incrementally without loading all into RAM"""
    data_dir = Path("training")
    files = [
        "UNIFIED_DATASET.jsonl",
        "merged_samples.jsonl",
        "merged_clean.jsonl",
        "comprehensive_data.jsonl",
        "repo_samples.jsonl",
    ]
    
    count = 0
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        logger.info(f"Streaming from: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    return
                try:
                    data = json.loads(line.strip())
                    text = None
                    for key in ['text', 'content', 'prompt', 'instruction']:
                        if key in data and data[key]:
                            text = str(data[key])[:1000]
                            break
                    if text:
                        yield {'text': text, 'source': filename}
                        count += 1
                except:
                    continue


class StreamingPrecomputer:
    def __init__(self, model_id: str, device: str, max_length: int):
        logger.info("=" * 80)
        logger.info("STREAMING ULTRA-FAST PRECOMPUTATION")
        logger.info("=" * 80)
        
        # Normalize device
        if torch.cuda.is_available():
            if device.startswith("cuda:"):
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    logger.warning(f"Device {device} unavailable, using cuda:0")
                    device = "cuda:0"
        else:
            device = "cpu"
        
        self.device = device
        logger.info(f"Loading {model_id} on {device}...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model - try full load first
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except:
                pass
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=None,
            ).to(device)
            logger.info("[OK] Model fully loaded on device")
        except RuntimeError:
            logger.warning("OOM on full load, using device_map='auto'")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        
        self.model.eval()
        self.max_length = max_length
        logger.info("=" * 80)
    
    def get_free_gpu_mem_gb(self) -> float:
        """Get free GPU memory in GB"""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.device)
            return free / 1e9
        return 999.0
    
    def forward_batch(self, texts: List[str], hidden_last_n: int, save_logits: bool) -> Dict:
        """Forward pass with optional outputs"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        use_hidden = hidden_last_n > 0
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            outputs = self.model(**inputs, output_hidden_states=use_hidden, use_cache=True)
        
        result = {}
        if save_logits:
            result['text_logits'] = outputs.logits.cpu()
        if use_hidden:
            layers = outputs.hidden_states
            take = min(hidden_last_n, len(layers))
            result['text_hidden_states'] = [layers[-i].cpu() for i in range(1, take + 1)]
        
        # Free GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return result


def main():
    args = build_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    precomputer = StreamingPrecomputer(args.model, args.device, args.max_length)
    
    # Count existing batches for resume
    start_batch = 0
    if args.resume:
        existing = list(OUTPUT_DIR.glob("batch_*.pt"))
        if existing:
            start_batch = len(existing)
            logger.info(f"[RESUME] Starting from batch {start_batch}")
    
    # Adaptive batching state
    current_batch_size = args.batch_size
    min_batch = max(1, args.min_batch)
    target_sec = args.target_sec
    
    logger.info(f"Initial batch size: {current_batch_size}")
    logger.info(f"Min batch size: {min_batch}")
    logger.info(f"Target time/batch: {target_sec}s")
    logger.info(f"Hidden layers: {args.hidden_last_n} | Save logits: {not args.no_logits}")
    logger.info("=" * 80)
    
    # Stream and process
    batch_buffer = []
    batch_idx = 0
    processed = 0
    skipped = 0
    start_time = time.time()
    last_perf_time = start_time
    last_processed = 0
    
    pbar = tqdm(desc="Precomputing", unit="batch")
    
    for sample in stream_data(limit=args.limit):
        batch_buffer.append(sample['text'])
        
        # Process when buffer reaches current batch size
        if len(batch_buffer) >= current_batch_size:
            output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
            
            # Skip if resuming and file exists
            if batch_idx < start_batch and output_file.exists():
                batch_buffer.clear()
                batch_idx += 1
                skipped += current_batch_size
                pbar.update(1)
                continue
            
            # Check GPU memory
            free_gb = precomputer.get_free_gpu_mem_gb()
            if free_gb < 1.0 and current_batch_size > min_batch:
                current_batch_size = max(min_batch, current_batch_size // 2)
                logger.warning(f"Low GPU mem ({free_gb:.1f}GB), reducing batch to {current_batch_size}")
                continue  # Don't process yet, wait for buffer to shrink
            
            # Forward pass
            batch_start = time.time()
            try:
                results = precomputer.forward_batch(
                    batch_buffer,
                    hidden_last_n=args.hidden_last_n,
                    save_logits=not args.no_logits
                )
                torch.save(results, output_file)
                batch_time = time.time() - batch_start
                
                # Adaptive batch size adjustment
                if batch_time > target_sec * 1.5 and current_batch_size > min_batch:
                    current_batch_size = max(min_batch, int(current_batch_size * 0.8))
                    logger.info(f"Slow batch ({batch_time:.1f}s), reducing to {current_batch_size}")
                elif batch_time < target_sec * 0.5 and current_batch_size < args.batch_size:
                    current_batch_size = min(args.batch_size, int(current_batch_size * 1.2))
                    logger.info(f"Fast batch ({batch_time:.1f}s), increasing to {current_batch_size}")
                
                processed += len(batch_buffer)
                batch_buffer.clear()
                batch_idx += 1
                pbar.update(1)
                
                # Performance logging
                if batch_idx % args.perf_log_every == 0:
                    elapsed = time.time() - start_time
                    interval = time.time() - last_perf_time
                    delta = processed - last_processed
                    samp_per_sec = processed / max(elapsed, 1e-6)
                    interval_sps = delta / max(interval, 1e-6)
                    logger.info(f"PERF | {samp_per_sec:.1f} samp/s total | {interval_sps:.1f} samp/s interval | batch_size={current_batch_size}")
                    last_perf_time = time.time()
                    last_processed = processed
            
            except Exception as e:
                logger.error(f"Error batch {batch_idx}: {e}")
                if current_batch_size > min_batch:
                    current_batch_size = max(min_batch, current_batch_size // 2)
                    logger.warning(f"Reducing batch to {current_batch_size} after error")
                batch_buffer.clear()
                continue
    
    # Process remaining buffer
    if batch_buffer:
        output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
        try:
            results = precomputer.forward_batch(
                batch_buffer,
                hidden_last_n=args.hidden_last_n,
                save_logits=not args.no_logits
            )
            torch.save(results, output_file)
            processed += len(batch_buffer)
            batch_idx += 1
        except Exception as e:
            logger.error(f"Error final batch: {e}")
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"[COMPLETE] Processed {processed:,} samples in {batch_idx} batches")
    logger.info(f"Time: {elapsed/60:.1f} minutes | Avg: {processed/max(elapsed,1e-6):.1f} samples/sec")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
