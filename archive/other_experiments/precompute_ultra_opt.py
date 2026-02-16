#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Teacher Precomputation - GPU Memory Management
Aggressive cleanup + smaller hidden states + batch-wise tensor creation.
"""

import os
import json
import torch
import logging
import argparse
import gc
from pathlib import Path
from typing import Generator, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-3B")
OUTPUT_DIR = Path("training/precomputed_teachers_efficient")
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))


def build_args():
    p = argparse.ArgumentParser(description="Ultra-optimized teacher precomputation")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id for text teacher")
    p.add_argument("--device", default="cuda:0", help="CUDA device to place the model")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max input tokens")
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip batches already saved")
    p.add_argument("--hidden-last-n", type=int, default=1, help="Number of last hidden layers to save")
    p.add_argument("--no-logits", action="store_true", help="Do not persist logits")
    p.add_argument("--chunk-size", type=int, default=2000, help="Load data chunk size")
    p.add_argument("--perf-log-every", type=int, default=50, help="Performance log interval")
    return p.parse_args()


class UltraOptimizedPrecomputer:
    """
    ULTRA-OPTIMIZED: Aggressive GPU memory management + minimal retention
    """
    
    def __init__(self, model_id: str, device: str, max_length: int):
        logger.info("=" * 80)
        logger.info("ULTRA-OPTIMIZED TEACHER PRECOMPUTATION")
        logger.info("Aggressive GPU memory management active")
        logger.info("=" * 80)
        
        logger.info(f"Loading text teacher: {model_id} on {device}...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        try:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=None,
            ).to(device)
            logger.info("[OK] Model fully on device")
        except RuntimeError as e:
            logger.warning(f"Direct load failed; using device_map='auto'")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            logger.info("[OK] Model loaded with device_map='auto'")

        self.max_length = max_length
        self.text_model.eval()
        self.device = device
        
        # Disable gradient computation completely
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        logger.info("=" * 80)
    
    def precompute_batch(self, texts: List[str], hidden_last_n: int, save_logits: bool) -> Dict:
        """Precompute with aggressive cleanup"""
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        use_hidden = hidden_last_n > 0
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = self.text_model(**inputs, output_hidden_states=use_hidden)
        
        result: Dict[str, torch.Tensor] = {}
        
        # Move outputs immediately to CPU to free GPU memory
        if save_logits:
            result['text_logits'] = outputs.logits.cpu().float()
            del outputs.logits
        
        if use_hidden:
            # Only save the last N hidden states (not all)
            layers = outputs.hidden_states
            take = min(hidden_last_n, len(layers))
            result['text_hidden_states'] = [
                layers[-i].cpu().float() for i in range(1, take + 1)
            ]
        
        # Cleanup
        del outputs, inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return result


def stream_data(chunk_size: int, limit: int = 0) -> Generator[List[Dict], None, None]:
    """Stream data in small chunks"""
    data_dir = Path("training")
    files = [
        "UNIFIED_DATASET.jsonl",
        "merged_samples.jsonl",
        "merged_clean.jsonl",
        "comprehensive_data.jsonl",
        "repo_samples.jsonl",
    ]
    
    total_yielded = 0
    chunk_buffer = []
    
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        logger.info(f"Streaming: {filename}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = None
                        for key in ['text', 'content', 'prompt', 'instruction']:
                            if key in data and data[key]:
                                text = str(data[key])[:500]  # Smaller text
                                break
                        
                        if text:
                            chunk_buffer.append({'text': text})
                            
                            if len(chunk_buffer) >= chunk_size:
                                yield chunk_buffer
                                total_yielded += len(chunk_buffer)
                                chunk_buffer = []
                                
                                if limit and total_yielded >= limit:
                                    return
                    except:
                        continue
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}")
    
    if chunk_buffer:
        yield chunk_buffer
        total_yielded += len(chunk_buffer)
    
    logger.info(f"Total samples streamed: {total_yielded:,}")


def main():
    args = build_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if torch.cuda.is_available():
        try:
            if device.startswith("cuda:"):
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    device = "cuda:0"
        except Exception:
            device = "cuda:0"

    precomputer = UltraOptimizedPrecomputer(
        model_id=args.model,
        device=device,
        max_length=args.max_length
    )
    
    batch_size = max(1, args.batch_size)
    hidden_last_n = max(0, args.hidden_last_n)
    save_logits = not args.no_logits
    
    logger.info(f"Batch size: {batch_size} | Chunk size: {args.chunk_size}")
    logger.info(f"Hidden layers: {hidden_last_n} | Save logits: {save_logits}")
    logger.info("=" * 80)
    
    import time
    start_time = time.time()
    last_perf_time = start_time
    last_processed = 0
    batch_idx = 0
    processed = 0
    total_estimate = 9939
    
    with tqdm(desc="Precomputing", total=total_estimate) as pbar:
        for chunk in stream_data(chunk_size=args.chunk_size, limit=args.limit):
            for i in range(0, len(chunk), batch_size):
                batch = chunk[i:i + batch_size]
                texts = [s['text'] for s in batch]
                
                output_file = OUTPUT_DIR / f"batch_{batch_idx:06d}.pt"
                
                if args.resume and output_file.exists():
                    batch_idx += 1
                    processed += len(texts)
                    pbar.update(1)
                    continue
                
                try:
                    results = precomputer.precompute_batch(
                        texts,
                        hidden_last_n=hidden_last_n,
                        save_logits=save_logits
                    )
                    torch.save(results, output_file)
                    
                    processed += len(texts)
                    batch_idx += 1
                    pbar.update(1)
                    
                    if batch_idx % args.perf_log_every == 0:
                        elapsed = time.time() - start_time
                        interval = time.time() - last_perf_time
                        delta = processed - last_processed
                        sps_total = processed / max(elapsed, 1e-6)
                        sps_interval = delta / max(interval, 1e-6)
                        logger.info(
                            f"PERF | Batch {batch_idx:5d} | "
                            f"Total: {sps_total:6.1f} samp/s | Interval: {sps_interval:6.1f} samp/s"
                        )
                        last_perf_time = time.time()
                        last_processed = processed
                
                except Exception as e:
                    logger.error(f"Error batch {batch_idx}: {e}")
                    continue
    
    logger.info("=" * 80)
    logger.info(f"[COMPLETE] Processed {processed:,} samples in {batch_idx} batches")
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.2f}h")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
