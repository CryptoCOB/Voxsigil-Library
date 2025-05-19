#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phi-2 Multi-GPU Fine-tuning Script

This script is optimized for running Phi-2 fine-tuning on multiple consumer-grade GPUs
with limited VRAM (12GB RTX 3060s). It uses:
1. 4-bit quantization for reduced memory usage
2. Small batch sizes with gradient accumulation
3. Multi-GPU support with proper parallel processing
4. VoxSigil integration for enhanced capabilities
"""

import os
import sys
import argparse
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
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm
import json

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
        logging.FileHandler("phi2_multi_gpu_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "microsoft/phi-2"
DEFAULT_OUTPUT_DIR = "./models/phi2_voxsigil"
DEFAULT_DATASET_PATH = "./data/phi2_arc_training.jsonl"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
MAX_LENGTH = 512  # Reduced from 2048 to save memory


class ARCDataset(Dataset):
    """Dataset for ARC fine-tuning with prompt-completion pairs."""
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH, use_voxsigil=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.use_voxsigil = use_voxsigil and VOXSIGIL_AVAILABLE
        
        # Setup VoxSigil components if available
        if self.use_voxsigil:
            logger.info("Initializing VoxSigil components for dataset processing")
            try:
                # Create configuration for middleware
                from VoxSigilRag.hybrid_blt import HybridMiddlewareConfig
                config = HybridMiddlewareConfig(
                    entropy_threshold=0.25,
                    blt_hybrid_weight=0.7,
                    entropy_router_fallback="token_based",
                    cache_ttl_seconds=300,
                    log_level="INFO"
                )
                
                # Initialize VoxSigil middleware
                self.middleware = voxsigil_middleware.VoxSigilMiddleware()
                
                # Initialize hybrid middleware with config
                self.hybrid_middleware = hybrid_blt.HybridMiddleware(config=config)
                
                # Initialize RAG processor
                self.rag_processor = voxsigil_rag.RAGProcessor()
                
                # Initialize BLT processor
                self.blt_processor = voxsigil_blt.BLTProcessor()
                
                # Initialize VoxSigil supervisor for comprehensive processing
                from voxsigil_supervisor.interfaces.rag_interface import SimpleRagInterface
                from voxsigil_supervisor.interfaces.llm_interface import LocalLlmInterface
                from voxsigil_supervisor.strategies.scaffold_router import BasicScaffoldRouter
                from voxsigil_supervisor.strategies.evaluation_heuristics import SimpleEvaluator
                from voxsigil_supervisor.strategies.retry_policy import BasicRetryPolicy
                
                # Create the necessary components for supervisor
                rag_interface = SimpleRagInterface(self.rag_processor)
                llm_interface = LocalLlmInterface()  # Will use the model we're fine-tuning
                scaffold_router = BasicScaffoldRouter()
                evaluator = SimpleEvaluator()
                retry_policy = BasicRetryPolicy()
                
                # Initialize supervisor
                self.supervisor = supervisor_engine.VoxSigilSupervisor(
                    rag_interface=rag_interface,
                    llm_interface=llm_interface,
                    scaffold_router=scaffold_router,
                    evaluation_heuristics=evaluator,
                    retry_policy=retry_policy,
                    max_iterations=2  # Limit iterations to save compute during training
                )
                
                logger.info("VoxSigil components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize VoxSigil components: {e}")
                import traceback
                logger.warning(traceback.format_exc())
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
                # Use middleware for basic enhancement
                enhanced_prompt = self.middleware.process_text(prompt)
                
                # Use hybrid middleware for advanced processing 
                hybrid_enhanced = self.hybrid_middleware.process(enhanced_prompt)
                if hybrid_enhanced and isinstance(hybrid_enhanced, str):
                    enhanced_prompt = hybrid_enhanced
                
                # Apply BLT processing for logical and structural reasoning
                if "logic" in prompt.lower() or "reason" in prompt.lower():
                    blt_output = self.blt_processor.process(enhanced_prompt)
                    if blt_output and isinstance(blt_output, str):
                        # Add reasoning structure from BLT
                        enhanced_prompt += f"\nReasoning Structure: {blt_output}\n"
                
                # Use RAG to add contextual information if available
                if "question" in prompt.lower() or "problem" in prompt.lower():
                    rag_output = self.rag_processor.process(enhanced_prompt)
                    if rag_output and isinstance(rag_output, str):
                        enhanced_prompt += f"\nAdditional Context: {rag_output}\n"
                
                # Finally, run through supervisor for comprehensive processing
                # This will orchestrate all VoxSigil components
                try:
                    supervisor_output = self.supervisor.process_query(
                        query=enhanced_prompt,
                        system_prompt="You are an assistant that helps with answering queries using all available information."
                    )
                    if supervisor_output and isinstance(supervisor_output, str):
                        enhanced_prompt = supervisor_output
                except Exception as e:
                    logger.warning(f"Supervisor processing failed, using basic enhancement: {e}")
                
                # Use the enhanced prompt
                prompt = enhanced_prompt
                logger.info(f"Successfully enhanced prompt {idx} using VoxSigil components")
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
            # Without device specification to let the trainer handle device placement
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


def setup_model_and_tokenizer(args):
    """Set up the model and tokenizer for multi-GPU training."""
    logger.info(f"Loading model: {args.model_name}")
    
    # Count available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s)")
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
      # Quantization config for low-memory training (4-bit)
    if torch.cuda.is_available():
        logger.info("Setting up 4-bit quantization for GPU training")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # For multiple GPUs, we need to be careful with device mapping
        # Instead of "auto", we'll set explicit device map with model parallelism
        if gpu_count > 1:
            # Use the first GPU as the primary device
            torch.cuda.set_device(0)
            device_map = {
                "model.embed_tokens": 0,
                "model.norm": gpu_count - 1,  # Last GPU
                "lm_head": gpu_count - 1,     # Last GPU
            }
            
            # Distribute the transformer layers across GPUs
            num_layers = 24  # Phi-2 has approximately 24 layers (adjust if different)
            layers_per_gpu = num_layers // gpu_count
            
            # Distribute layers across GPUs
            for i in range(num_layers):
                gpu_id = min(i // layers_per_gpu, gpu_count - 1)
                device_map[f"model.layers.{i}"] = gpu_id
                
            logger.info(f"Using custom device map for {gpu_count} GPUs")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            # Single GPU - use device 0
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
        model = prepare_model_for_kbit_training(model)
        logger.info("Using 4-bit quantization for training")
    else:
        logger.warning("No GPU available, using CPU with standard precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    logger.info("Applying LoRA adapter")
    lora_config = LoraConfig(
        r=8,  # Reduced rank to save memory
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train_model(args):
    """Train the model using the specified arguments with multi-GPU support."""
    logger.info("Starting model training with multi-GPU optimization")
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
      # Configure model to disable cache when using gradient checkpointing
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        logger.info("Disabling model cache for gradient checkpointing compatibility")
        model.config.use_cache = False
    
    # Prepare dataset
    dataset = ARCDataset(args.dataset_path, tokenizer, 
                        max_length=args.max_length,
                        use_voxsigil=not args.disable_voxsigil)
    
    # Training arguments optimized for multiple consumer GPUs
    gpu_count = torch.cuda.device_count()
    per_device_batch_size = 1  # Very small batch size per device to avoid OOM
    
    # Calculate gradient accumulation steps based on GPU count and desired effective batch size
    target_batch_size = 8  # This is the effective batch size we want to simulate
    gradient_accumulation_steps = max(1, target_batch_size // (per_device_batch_size * gpu_count))
    
    logger.info(f"Training configuration: {gpu_count} GPUs, {per_device_batch_size} batch size per device")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {per_device_batch_size * gpu_count * gradient_accumulation_steps}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",  # Disable reporting to save memory
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,  # Save CPU memory
        optim="adamw_torch_fused",  # Use fused optimizer for better performance
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        no_cuda=False,                # Ensure GPU usage
        auto_find_batch_size=True,    # Enable auto find batch size if OOM errors occur
        tf32=True,                    # Use TensorFloat-32 where available
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Custom Trainer with device mapping fix
    class MultiGPUTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # For multi-GPU training, set primary device
            if torch.cuda.device_count() > 1:
                self.primary_device = torch.device("cuda:0")
            else:
                self.primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        def _prepare_inputs(self, inputs):
            """
            Ensure all inputs tensors are on the same device to avoid
            "Expected all tensors to be on the same device" errors.
            """
            prepared_inputs = super()._prepare_inputs(inputs)
            
            # Ensure all tensors are on the same device
            for k, v in prepared_inputs.items():
                if isinstance(v, torch.Tensor) and v.device != self.primary_device:
                    prepared_inputs[k] = v.to(self.primary_device)
            
            return prepared_inputs
    
    # Use custom trainer instead of standard Trainer
    trainer = MultiGPUTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Beginning training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Phi-2 Fine-tuning with VoxSigil integration"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset_path",
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
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--disable_voxsigil",
        action="store_true",
        help="Disable VoxSigil integration even if available",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log VoxSigil availability
    if VOXSIGIL_AVAILABLE:
        logger.info("VoxSigil components successfully imported")
    else:
        logger.warning("VoxSigil components not available - training without enhanced capabilities")
    
    # Print training overview
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info.append(f"GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    print("\n" + "="*80)
    print("MULTI-GPU PHI-2 TRAINING WITH VOXSIGIL")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"VoxSigil integration: {'Disabled' if args.disable_voxsigil else 'Enabled'}")
    print("\nGPU Information:")
    for info in gpu_info:
        print(f"- {info}")
    print("="*80 + "\n")
    
    # Start training
    success = train_model(args)
    
    # Print final message
    print("\n" + "="*80)
    if success:
        print("TRAINING COMPLETED SUCCESSFULLY")
    else:
        print("TRAINING FAILED - CHECK LOGS FOR DETAILS")
    print("="*80 + "\n")
