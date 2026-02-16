#!/usr/bin/env python3
"""
EFFICIENT/FAST Teacher Precomputation - Models Loaded ONCE
Processes large datasets WITHOUT reloading models every batch.

Optimizations added:
- Configurable model id (default smaller/faster model is recommended)
- Tunable batch size and max input length
- Greedy forward-only pass (no sampling)
- torch.inference_mode + autocast for peak throughput
- Resume support (skip already saved batches)
- Optional subset limit for quick tests
- Selectable hidden state depth / skip logits for smaller disk writes
- Periodic performance logging (samples/sec, batches remaining)
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sensible fast defaults (can be overridden via CLI)
DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-3B")
OUTPUT_DIR = Path("training/precomputed_teachers_efficient")
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
CHECKPOINT_EVERY = 200


def build_args():
    p = argparse.ArgumentParser(description="Efficient teacher precomputation")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id for text teacher")
    p.add_argument("--device", default="cuda:1", help="CUDA device to place the model")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max input tokens")
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip batches already saved")
    p.add_argument("--hidden-last-n", type=int, default=1, help="Number of last hidden layers to save (0 disables hidden states)")
    p.add_argument("--no-logits", action="store_true", help="Do not persist logits (reduces size)")
    p.add_argument("--perf-log-every", type=int, default=500, help="Performance summary interval (batches)")
    return p.parse_args()


class EfficientTeacherPrecomputer:
    """
    EFFICIENT: Models loaded ONCE and kept in memory throughout processing.
    No wasteful reloading like the previous version.
    """
    
    def __init__(self, model_id: str, device: str, max_length: int):
        logger.info("=" * 80)
        logger.info("EFFICIENT TEACHER PRECOMPUTATION")
        logger.info("Models loaded ONCE - No reloading per batch!")
        logger.info("=" * 80)
        
        # Load text teacher ONCE
        logger.info(f"Loading text teacher: {model_id} on {device}...")
        # IMPORTANT: Avoid accelerate offload for speed. Prefer smaller model that fits fully.
        # Qwen 7B typically needs 14GB+ in fp16; use 3B on 12GB GPUs for speed.
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Enable TF32 where available for speed
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Try to fully place on the chosen device; fall back to auto map if OOM
        try:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=None,
            ).to(device)
            offloaded = False
        except RuntimeError as e:
            logger.warning(f"Direct load to {device} failed ({e}); falling back to device_map='auto' (may be slower)")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            offloaded = True

        self.max_length = max_length
        self.text_model.eval()
        if offloaded:
            logger.info("[OK] Text teacher loaded (auto device map; may offload some weights)")
        else:
            logger.info("[OK] Text teacher loaded and fully on device")
        
        # Skip audio for now (causes OOM)
        logger.info("Skipping audio teacher (text-only mode for efficiency)")
        self.device = device
        logger.info(f"Using device: {self.device}")
        logger.info("=" * 80)
    
    def precompute_batch(self, texts: List[str], hidden_last_n: int, save_logits: bool) -> Dict:
        """Precompute text teacher outputs - models already in memory"""
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        use_hidden = hidden_last_n > 0
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = self.text_model(**inputs, output_hidden_states=use_hidden, use_cache=True)

        result: Dict[str, torch.Tensor | list] = {}
        if save_logits:
            result['text_logits'] = outputs.logits.cpu()
        if use_hidden:
            layers = outputs.hidden_states
            take = min(hidden_last_n, len(layers))
            result['text_hidden_states'] = [layers[-i].cpu() for i in range(1, take + 1)]
        return result


def load_all_data(limit: int = 0):
    """Load all training data efficiently"""
    data_dir = Path("training")
    all_samples = []
    
    # Priority files (largest first)
    files = [
        "UNIFIED_DATASET.jsonl",        # 8.6GB
        "merged_samples.jsonl",         # 11.4MB
        "merged_clean.jsonl",           # 9.7MB
        "comprehensive_data.jsonl",     # 6.1MB
        "repo_samples.jsonl",           # 5.3MB
    ]
    
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        logger.info(f"Loading: {filename}")
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Extract text
                        text = None
                        for key in ['text', 'content', 'prompt', 'instruction']:
                            if key in data and data[key]:
                                text = str(data[key])[:1000]  # Limit length
                                break
                        
                        if text:
                            all_samples.append({'text': text, 'source': filename})
                            line_count += 1
                            if limit and len(all_samples) >= limit:
                                break
                    except:
                        continue
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}")
        
        logger.info(f"  Loaded {line_count:,} samples from {filename}")
        if limit and len(all_samples) >= limit:
            break
    
    logger.info(f"Total samples: {len(all_samples):,}")
    return all_samples


def main():
    args = build_args()
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize precomputer (loads models ONCE)
    # Validate/normalize device
    device = args.device
    if torch.cuda.is_available():
        try:
            if device.startswith("cuda:"):
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    logger.warning(f"Requested {device} but only {torch.cuda.device_count()} CUDA device(s) available; falling back to cuda:0")
                    device = "cuda:0"
        except Exception:
            device = "cuda:0"
    else:
        logger.warning("CUDA not available; using CPU (will be slow)")
        device = "cpu"

    # Initialize precomputer (loads models ONCE)
    precomputer = EfficientTeacherPrecomputer(model_id=args.model, device=device, max_length=args.max_length)
    
    # Load data
    logger.info("\nLoading training data...")
    samples = load_all_data(limit=args.limit)
    
    if not samples:
        logger.error("No data found!")
        return
    
    # Process in batches
    batch_size = max(1, args.batch_size)
    total_batches = (len(samples) + batch_size - 1) // batch_size
    logger.info(f"\nProcessing {len(samples):,} samples in {total_batches:,} batches")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 80)
    
    processed = 0
    hidden_last_n = max(0, args.hidden_last_n)
    save_logits = not args.no_logits
    logger.info(f"Hidden layers saved: {hidden_last_n} | Save logits: {save_logits}")
    logger.info(f"Performance log interval: {args.perf_log_every} batches")
    import time
    start_time = time.time()
    last_perf_time = start_time
    last_processed = 0
    for i in tqdm(range(0, len(samples), batch_size), desc="Precomputing"):
        batch = samples[i:i + batch_size]
        texts = [s['text'] for s in batch]
        
        batch_idx = i // batch_size
        output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
        if args.resume and output_file.exists():
            processed += len(texts)
            continue

        try:
            # Precompute (models already loaded!)
            results = precomputer.precompute_batch(texts, hidden_last_n=hidden_last_n, save_logits=save_logits)
            
            # Save batch
            torch.save(results, output_file)
            
            processed += len(texts)
            
            # Checkpoint
            if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
                logger.info(f"Checkpoint: {processed:,}/{len(samples):,} samples")
            if (batch_idx + 1) % args.perf_log_every == 0:
                elapsed = time.time() - start_time
                interval = time.time() - last_perf_time
                delta = processed - last_processed
                samples_per_sec_total = processed / max(elapsed, 1e-6)
                samples_per_sec_interval = delta / max(interval, 1e-6)
                batches_left = total_batches - (batch_idx + 1)
                est_remaining = batches_left * (elapsed / (batch_idx + 1))
                logger.info(f"PERF | total {samples_per_sec_total:.1f} samp/s | interval {samples_per_sec_interval:.1f} samp/s | est remaining {est_remaining/3600:.2f} h")
                last_perf_time = time.time()
                last_processed = processed
        
        except Exception as e:
            logger.error(f"Error batch {i // batch_size}: {e}")
            continue
    
    logger.info("=" * 80)
    logger.info(f"[COMPLETE] Processed {processed:,} samples")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Total batches: {processed // batch_size}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
