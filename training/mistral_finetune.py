#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mistral-7B Fine-tuning Script for ARC Dataset

This standalone script fine-tunes the Mistral-7B model on the ARC dataset
using LoRA (Low-Rank Adaptation) to reduce memory requirements.
"""

import os
import argparse
import json
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mistral_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DEFAULT_OUTPUT_DIR = "./models/mistral_arc_finetuned"
DEFAULT_DATASET_PATH = "./data/mistral_arc_training.jsonl"
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-5

class ARCDataset(Dataset):
    """Dataset for ARC fine-tuning with prompt-completion pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
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
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create sample with input_ids and attention_mask
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Mask out the prompt part in labels to avoid training on it
        prompt_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])
        labels[:prompt_len] = -100  # -100 is ignored in loss calculation
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def create_model_and_tokenizer(model_name):
    """Create and prepare the model and tokenizer for training."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with 4-bit quantization to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model, tokenizer

def train(args):
    """Train the model using the provided arguments."""
    # Validate inputs
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset file not found: {args.dataset}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.model_name)
    
    # Create dataset
    dataset = ARCDataset(args.dataset, tokenizer)
    
    # Calculate training steps
    num_examples = len(dataset)
    total_steps = (num_examples // args.batch_size) * args.epochs
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Higher value for larger model
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B on ARC dataset")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Print training configuration
    logger.info("Starting Mistral-7B fine-tuning with the following configuration:")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Start training
    train(args)
