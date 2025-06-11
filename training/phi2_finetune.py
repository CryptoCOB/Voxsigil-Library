#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phi-2 Fine-tuning Script for ARC Dataset

This standalone script fine-tunes the Microsoft Phi-2 model on the ARC dataset
using LoRA (Low-Rank Adaptation) to reduce memory requirements.

The script automatically detects if a GPU is available and adjusts its configuration:
- With GPU: Uses 4-bit quantization and optimized training settings
- Without GPU: Uses standard precision and CPU-friendly training settings
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
    prepare_model_for_kbit_training,
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
        logging.FileHandler("phi2_finetune.log"),
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

def create_model_and_tokenizer(model_name, force_cpu=False):
    """Create and prepare the model and tokenizer for training with optional VoxSigil integration."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token (Phi-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token since it was not set")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if GPU is available
    has_gpu = torch.cuda.is_available() and not force_cpu
    if force_cpu and torch.cuda.is_available():
        logger.info("GPU is available but CPU mode is forced")
    else:
        logger.info(f"GPU available: {has_gpu}")
    
    # Initialize VoxSigil supervisor if available
    voxsigil_supervisor = None
    if VOXSIGIL_AVAILABLE:
        try:
            logger.info("Initializing VoxSigil supervisor for model integration")
            voxsigil_supervisor = supervisor_engine.SupervisorEngine()
            logger.info("VoxSigil supervisor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize VoxSigil supervisor: {e}")
    
    try:
        if has_gpu:
            # Try to load model with standard GPU settings first (no quantization)
            logger.info("Attempting to load model on GPU with standard settings")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("Successfully loaded model on GPU with standard settings")
        else:
            # Load model without quantization for CPU
            logger.info("Loading model without quantization (CPU mode)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",  # Explicitly set to CPU
            )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    )
    
    # Prepare model for training - adjust based on whether we're using quantization
    logger.info("Preparing model for training")
    if has_gpu:
        # For quantized model (GPU)
        model = prepare_model_for_kbit_training(model)
    else:
        # For non-quantized model (CPU) - skip prepare_model_for_kbit_training
        logger.info("Skipping quantized training preparation for CPU mode")
        
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model, tokenizer

def train(args):
    """Train the model using the provided arguments with VoxSigil integration if available."""
    # Validate inputs
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset file not found: {args.dataset}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if VoxSigil components are available
    use_voxsigil = VOXSIGIL_AVAILABLE
    if use_voxsigil:
        logger.info("VoxSigil components available - will be used during training")
    else:
        logger.warning("VoxSigil components not available - training without enhanced capabilities")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.model_name, force_cpu=args.force_cpu)
    
    # Create dataset with VoxSigil integration if available
    dataset = ARCDataset(args.dataset, tokenizer, use_voxsigil=use_voxsigil)
    
    # Check if GPU is available for training configuration
    has_gpu = torch.cuda.is_available() and not args.force_cpu
    if args.force_cpu and torch.cuda.is_available():
        logger.info("Force CPU mode enabled - using CPU despite GPU availability")
    else:
        logger.info(f"Configuring training for {'GPU' if has_gpu else 'CPU'} mode")
    
    # Set up training arguments based on available hardware
    if has_gpu:
        # GPU configuration
        logger.info("Using GPU-optimized training settings")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            report_to="none",
            remove_unused_columns=False,
        )
    else:
        # CPU configuration - use smaller batch size and simpler optimizer
        logger.info("Using CPU-optimized training settings")
        # Adjust batch size if needed
        cpu_batch_size = min(2, args.batch_size)  # Use smaller batch size for CPU
        logger.info(f"Using reduced batch size for CPU: {cpu_batch_size} (requested: {args.batch_size})")
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=cpu_batch_size,
            gradient_accumulation_steps=8,  # Increase for CPU
            learning_rate=args.learning_rate,
            logging_steps=1,  # Log more frequently for debugging
            save_steps=5,     # Save more frequently
            fp16=False,       # Disable mixed precision for CPU
            optim="adamw_torch",  # Use standard optimizer for CPU
            lr_scheduler_type="linear",  # Simpler scheduler
            warmup_ratio=0.03,
            weight_decay=0.01,
            report_to="none",
            remove_unused_columns=False,
            no_cuda=True,     # Force CPU usage
            dataloader_drop_last=False,  # Keep all samples
            dataloader_num_workers=0,    # No multiprocessing for small datasets
        )
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
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
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 on ARC dataset")
    
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
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--disable_voxsigil",
        action="store_true",
        help="Disable VoxSigil integration even if available",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Add more explicit print statements for debugging
    print("SCRIPT STARTED: Phi-2 fine-tuning")
    
    try:        # Parse arguments
        args = parse_args()
        
        # Print training configuration
        print(f"CONFIGURATION: Model={args.model_name}, Dataset={args.dataset}")
        print(f"CONFIGURATION: Output={args.output_dir}, Epochs={args.epochs}, Batch={args.batch_size}, LR={args.learning_rate}")
        print(f"CONFIGURATION: Force CPU={args.force_cpu}")
        logger.info("Starting Phi-2 fine-tuning with the following configuration:")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Force CPU: {args.force_cpu}")
        
        # Check VoxSigil availability
        if VOXSIGIL_AVAILABLE:
            logger.info("VoxSigil components successfully imported")
            print("SUCCESS: VoxSigil components are available and will be used")
        else:
            logger.warning("VoxSigil components not available - training without enhanced capabilities")
            print("WARNING: VoxSigil components are not available - training will proceed without enhanced capabilities")
        
        # Check if dataset exists
        if not os.path.exists(args.dataset):
            print(f"ERROR: Dataset file not found at {args.dataset}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'data directory not found'}")
            exit(1)
        else:
            print(f"SUCCESS: Dataset file found at {args.dataset}")
        
        # Verify peft module
        print("Verifying peft module installation...")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            print("SUCCESS: peft module is correctly installed")
        except ImportError as e:
            print(f"ERROR: Failed to import peft module: {e}")
            exit(1)
            
        # Start training
        print("Starting training process...")
        train(args)
        print("Training completed successfully")
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
