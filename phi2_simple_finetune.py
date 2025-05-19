#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Phi-2 Fine-tuning Script

This is a simplified version of the Phi-2 fine-tuning script
that uses a manual training loop instead of HF Trainer to avoid
potential issues with the Trainer API.
"""

import os
import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phi2_simple_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./models/phi2_arc_finetuned_simple"
DATASET_PATH = "./data/phi2_arc_training.jsonl"

class SimpleARCDataset(Dataset):
    """Simple dataset for ARC fine-tuning with prompt-completion pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        completion = example["completion"]
        
        # Combine prompt and completion for causal language modeling
        full_text = prompt + completion
        
        # Tokenize
        tokens = self.tokenizer.encode(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens.squeeze(),
            "labels": tokens.squeeze(),
        }

def train_simple_model():
    """Train the model using a simplified approach."""
    print("Starting simplified training...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 32-bit mode for CPU
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )
    
    # Create dataset and dataloader
    dataset = SimpleARCDataset(DATASET_PATH, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Very small batch size for CPU
        shuffle=True,
    )
    
    # Prepare optimizer
    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    
    # Train the model for one epoch
    print("Training for 1 epoch...")
    model.train()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")

if __name__ == "__main__":
    try:
        print("Starting simplified Phi-2 fine-tuning")
        train_simple_model()
        print("Training completed successfully")
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
