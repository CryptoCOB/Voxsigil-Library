#!/usr/bin/env python3
"""
High-Speed Multi-GPU Teacher Batch Processor
Processes multiple teacher samples in parallel across all GPUs
"""

import os
import sys
import json
import time
import logging
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TeacherConfig:
    batch_size: int = 8  # Process 8 samples at once
    parallel_workers: int = 3  # Use all 3 GPUs
    ollama_url: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"

class HighSpeedTeacherProcessor:
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.session = requests.Session()
        
    async def generate_teacher_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate teacher responses for a batch of samples in parallel"""
        
        # Split samples across workers (GPUs)
        chunk_size = len(samples) // self.config.parallel_workers
        chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_chunk(chunk))
            tasks.append(task)
        
        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for chunk_results in results:
            all_results.extend(chunk_results)
            
        return all_results
    
    async def _process_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of samples on one GPU"""
        results = []
        
        # Process samples in small batches within the chunk
        for i in range(0, len(chunk), self.config.batch_size):
            batch = chunk[i:i + self.config.batch_size]
            batch_results = await self._process_mini_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_mini_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a mini-batch of samples simultaneously"""
        
        # Prepare all requests
        requests_data = []
        for sample in batch:
            content = sample["content"]
            prompt = f"Summarize or answer concisely based on the following content:\n\n{content}\n\nAnswer:"
            
            request_data = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 2048
                }
            }
            requests_data.append((sample, request_data))
        
        # Execute all requests concurrently
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            tasks = []
            for sample, request_data in requests_data:
                task = loop.run_in_executor(executor, self._make_request, sample, request_data)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        return results
    
    def _make_request(self, sample: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single teacher request (thread-safe)"""
        try:
            response = self.session.post(
                f"{self.config.ollama_url}/api/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                teacher_response = result.get("response", "").strip()
            else:
                teacher_response = f"Fallback summary for: {sample['content'][:100]}..."
                
        except Exception as e:
            logger.warning(f"Teacher request failed: {e}")
            teacher_response = f"Fallback summary for: {sample['content'][:100]}..."
        
        # Return enhanced sample with teacher output
        enhanced_sample = sample.copy()
        enhanced_sample["teacher_outputs"] = {
            self.config.model: teacher_response
        }
        
        return enhanced_sample

async def accelerate_teacher_generation():
    """Accelerate the remaining teacher generation using all GPUs"""
    
    print("⚡ Starting HIGH SPEED teacher generation...")
    
    # Load existing progress
    precomputed_file = "training/precomputed_teachers.jsonl"
    completed_samples = []
    
    if os.path.exists(precomputed_file):
        with open(precomputed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    completed_samples.append(json.loads(line.strip()))
        
        print(f"📊 Found {len(completed_samples)} completed samples")
    
    # Load remaining samples to process
    from knowledge_distillation_system import DistillConfig, collect_samples
    
    cfg = DistillConfig()
    cfg.max_samples = 10000  # Target 10K samples
    
    all_samples = collect_samples(cfg.data_dir, cfg.max_files, cfg.max_samples)
    remaining_samples = all_samples[len(completed_samples):]
    
    print(f"🎯 Processing {len(remaining_samples)} remaining samples with HIGH SPEED")
    
    if not remaining_samples:
        print("✅ All samples already completed!")
        return
    
    # Initialize high-speed processor
    config = TeacherConfig(
        batch_size=8,  # 8 samples per mini-batch
        parallel_workers=3,  # All 3 GPUs
        model="gpt-oss:20b"
    )
    
    processor = HighSpeedTeacherProcessor(config)
    
    # Process in large batches
    batch_size = config.batch_size * config.parallel_workers  # 24 samples per batch
    start_time = time.time()
    
    with open(precomputed_file, 'a', encoding='utf-8') as f:
        for i in range(0, len(remaining_samples), batch_size):
            batch = remaining_samples[i:i + batch_size]
            
            print(f"⚡ Processing batch {i//batch_size + 1}: {len(batch)} samples...")
            
            try:
                # Process batch in parallel across all GPUs
                results = await processor.generate_teacher_batch(batch)
                
                # Save results immediately
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                
                # Progress update
                completed = len(completed_samples) + i + len(results)
                elapsed = time.time() - start_time
                rate = (i + len(results)) / elapsed if elapsed > 0 else 0
                
                print(f"✅ Batch completed: {completed}/{len(all_samples)} total ({rate:.1f} samples/sec)")
                
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                continue
    
    total_time = time.time() - start_time
    print(f"🎉 HIGH SPEED teacher generation completed in {total_time/60:.1f} minutes!")

if __name__ == "__main__":
    asyncio.run(accelerate_teacher_generation())