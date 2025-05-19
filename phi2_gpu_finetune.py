#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phi-2 GPU Fine-tuning Script for ARC Dataset

This standalone script fine-tunes the Microsoft Phi-2 model on the ARC dataset
using LoRA (Low-Rank Adaptation) with optimized GPU settings.

This version doesn't use bitsandbytes but instead uses native PyTorch FP16
to ensure compatibility with most GPUs.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm

# Add necessary paths to system path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "VoxSigilRag"))
sys.path.append(str(root_dir / "voxsigil_supervisor"))

# Import VoxSigil components
try:
    from VoxSigilRag import voxsigil_middleware, voxsigil_rag, voxsigil_rag_compression
    from VoxSigilRag import voxsigil_blt, hybrid_blt
    from voxsigil_supervisor import supervisor_engine
    VOXSIGIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VoxSigil components could not be imported: {e}")
    VOXSIGIL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phi2_gpu_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "microsoft/phi-2"
DEFAULT_OUTPUT_DIR = "./models/phi2_arc_finetuned"
DEFAULT_DATASET_PATH = "./data/phi2_arc_training.jsonl"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5

class ARCDataset(Dataset):
    """Dataset for ARC fine-tuning with prompt-completion pairs and optional VoxSigil enhancement."""
    
    def __init__(self, data_path, tokenizer, max_length=2048, use_voxsigil=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.use_voxsigil = use_voxsigil and VOXSIGIL_AVAILABLE
        
        # Setup VoxSigil components if available
        if self.use_voxsigil:
            logger.info("Initializing VoxSigil components for dataset processing")
            try:
                self.rag_processor = voxsigil_rag.RAGProcessor()
                self.middleware = voxsigil_middleware.VoxSigilMiddleware()
                self.blt_processor = voxsigil_blt.BLTProcessor()
                logger.info("VoxSigil components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize VoxSigil components: {e}")
                self.use_voxsigil = False
        
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples")
        if self.use_voxsigil:
            logger.info("Dataset will use VoxSigil enhancements during training")
        else:
            logger.info("Dataset will use standard processing (VoxSigil not available or disabled)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        completion = example["completion"]
        
        # Apply VoxSigil processing if available and enabled
        if self.use_voxsigil:
            try:
                # Use middleware to enhance the prompt
                enhanced_prompt = self.middleware.process_text(prompt)
                
                # Apply RAG processing if applicable (for context-aware responses)
                if "question" in prompt.lower() or "problem" in prompt.lower():
                    rag_results = self.rag_processor.process(prompt)
                    if rag_results and isinstance(rag_results, str):
                        enhanced_prompt += f"\nAdditional Context: {rag_results}\n"
                
                # Apply BLT processing for logical reasoning tasks
                if "logic" in prompt.lower() or "reason" in prompt.lower():
                    blt_results = self.blt_processor.process(prompt)
                    if blt_results and isinstance(blt_results, str):
                        enhanced_prompt += f"\nReasoning Path: {blt_results}\n"
                
                # Use the enhanced prompt if processing was successful
                prompt = enhanced_prompt
            except Exception as e:
                logger.warning(f"Error in VoxSigil processing: {e}. Using original prompt.")
        
        # Combine prompt and completion for causal language modeling
        full_text = prompt + completion
        
        # Ensure the tokenizer has a pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Tokenize with additional safety measures
        try:
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
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            # Return empty tensors with proper shapes as fallback
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            labels = torch.ones(self.max_length, dtype=torch.long) * -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

def create_model_and_tokenizer(model_name):
    """Create and prepare the model and tokenizer for GPU training."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Check GPU availability
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected. Training will be extremely slow.")
        logger.warning("Consider using a machine with a GPU for faster training.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token (Phi-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token since it was not set")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with FP16 for GPU efficiency (without quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if has_gpu else torch.float32,
        device_map="auto",
    )
    
    # Configure LoRA adapter for memory-efficient fine-tuning
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters %: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def train(args):
    """Train the model using the provided arguments with VoxSigil integration if available."""
    # Validate inputs
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset file not found: {args.dataset}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if VoxSigil components are available
    use_voxsigil = VOXSIGIL_AVAILABLE and not args.disable_voxsigil
    if use_voxsigil:
        logger.info("VoxSigil components available - will be used during training")
    else:
        logger.warning("VoxSigil components not available or disabled - training without enhanced capabilities")
    
    # Get GPU device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.model_name)
      # Create dataset
    dataset = ARCDataset(args.dataset, tokenizer, use_voxsigil=use_voxsigil)
    
    # Configure training settings for GPU
    logger.info("Setting up GPU-optimized training configuration")
    
    # Optimize batch size based on available GPU memory
    if device.type == "cuda":
        if gpu_memory >= 24:  # High-end GPU
            effective_batch = args.batch_size
        elif gpu_memory >= 16:  # Mid-range GPU
            effective_batch = min(args.batch_size, 4) 
        elif gpu_memory >= 8:  # Entry GPU
            effective_batch = min(args.batch_size, 2)
        else:  # Low memory GPU
            effective_batch = 1
        
        logger.info(f"Using batch size {effective_batch} based on available GPU memory")
    else:
        effective_batch = 1  # CPU fallback
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=8 // effective_batch,  # Adjust based on batch size
        learning_rate=args.learning_rate,
        fp16=device.type == "cuda",  # Only use fp16 on GPU
        logging_steps=1,
        save_steps=20,
        save_total_limit=3,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
    
    # Show final message with usage instructions
    print("\n" + "="*80)
    print(f"Model successfully trained and saved to {args.output_dir}")
    print("To use this model, you can load it with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
    print("="*80 + "\n")

def create_sample_data(output_path, num_examples=5):
    """Create a small sample dataset if none exists."""
    print(f"Creating simplified training data...")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create sample ARC-like examples
    examples = [
        {
            "prompt": "<|system|>\nYou are a helpful, precise assistant specialized in solving abstract reasoning tasks.</s>\n<|user|>\nGiven the following input, provide the corresponding output:\nInput: [[1, 2], [3, 4]]\n</s>\n<|assistant|>\n",
            "completion": "Output: [[2, 3], [4, 5]]</s>",
            "task_id": "sample_task_1"
        },
        {
            "prompt": "<|system|>\nYou are a helpful, precise assistant specialized in solving abstract reasoning tasks.</s>\n<|user|>\nGiven the following input, provide the corresponding output:\nInput: [[5, 0], [0, 5]]\n</s>\n<|assistant|>\n",
            "completion": "Output: [[0, 5], [5, 0]]</s>",
            "task_id": "sample_task_2"
        },
        {
            "prompt": "<|system|>\nYou are a helpful, precise assistant specialized in solving abstract reasoning tasks.</s>\n<|user|>\nGiven the following input, provide the corresponding output:\nInput: [[1, 1, 1], [2, 2, 2]]\n</s>\n<|assistant|>\n",
            "completion": "Output: [[1, 1, 1], [1, 1, 1], [2, 2, 2]]</s>",
            "task_id": "sample_task_3"
        },
        {
            "prompt": "<|system|>\nYou are a helpful, precise assistant specialized in solving abstract reasoning tasks.</s>\n<|user|>\nGiven the following input, provide the corresponding output:\nInput: [[7, 8], [9, 10]]\n</s>\n<|assistant|>\n",
            "completion": "Output: [[8, 9], [10, 11]]</s>",
            "task_id": "sample_task_4"
        },
        {
            "prompt": "<|system|>\nYou are a helpful, precise assistant specialized in solving abstract reasoning tasks.</s>\n<|user|>\nGiven the following input, provide the corresponding output:\nInput: [[3, 3], [3, 3]]\n</s>\n<|assistant|>\n",
            "completion": "Output: [[3, 3, 3], [3, 3, 3], [3, 3, 3]]</s>",
            "task_id": "sample_task_5"
        }
    ]
    
    # Write examples to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created sample training data at {output_path}")
    return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 on ARC dataset with GPU")
    
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
    parser.add_argument(
        "--disable_voxsigil",
        action="store_true",
        help="Disable VoxSigil integration even if available",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Print welcome message
    print("\n" + "="*80)
    print("PHI-2 GPU FINE-TUNING FOR ARC DATASET")
    print("="*80)
    
    try:
        # Parse arguments
        args = parse_args()
          # Print training configuration
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Output directory: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Disable VoxSigil: {args.disable_voxsigil}")
        
        # Check VoxSigil availability
        if VOXSIGIL_AVAILABLE and not args.disable_voxsigil:
            print("VoxSigil components available and will be used for enhanced training")
        elif VOXSIGIL_AVAILABLE and args.disable_voxsigil:
            print("VoxSigil components available but disabled by user")
        else:
            print("VoxSigil components not available - training without enhanced capabilities")
        
        # Check GPU availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU available: {device_name}")
        else:
            print("WARNING: No GPU detected. Training will be extremely slow.")
            
        # Check if dataset exists
        if not os.path.exists(args.dataset):
            print(f"Training data not found at {args.dataset}")
            # Create sample data instead
            args.dataset = create_sample_data("./data/phi2_arc_training.jsonl")
        else:
            print(f"Found training data at {args.dataset}")
            
        # Start training
        print("\nStarting Phi-2 fine-tuning with GPU optimization...")
        train(args)
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)