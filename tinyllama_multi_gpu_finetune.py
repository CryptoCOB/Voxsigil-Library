#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyLlama Low-Memory Multi-GPU Training Script

This script is optimized for running TinyLlama fine-tuning on multiple consumer-grade GPUs
with limited VRAM (12GB RTX 3060). It uses:
1. 4-bit quantization for reduced memory usage
2. Small batch sizes with gradient accumulation
3. Multi-GPU support with proper parallel processing
4. VoxSigil integration for enhanced capabilities (if components are available)
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
    AutoConfig, # Added for dynamic layer count
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
import traceback # Added for more detailed error logging

# Add necessary paths to system path
current_dir = Path(__file__).parent.resolve() # Use resolve for absolute path
root_dir = current_dir.parent
sys.path.append(str(root_dir))
# Assuming VoxSigilRag and voxsigil_supervisor are siblings to the script's parent directory
# or are installed in the Python path. If they are subdirectories of root_dir:
# sys.path.append(str(root_dir / "VoxSigilRag"))
# sys.path.append(str(root_dir / "voxsigil_supervisor"))
# If they are at the same level as the script's parent, the current sys.path.append(str(root_dir)) might be sufficient
# depending on Python's import resolution.
# For robustness, ensure these paths correctly point to VoxSigil modules.
# Example: if VoxSigilRag is inside a directory named 'libs' within root_dir:
# sys.path.append(str(root_dir / "libs" / "VoxSigilRag"))


# Import VoxSigil components
try:
    # These imports assume VoxSigil modules are structured and accessible.
    from VoxSigilRag import voxsigil_middleware, voxsigil_rag, voxsigil_rag_compression
    from VoxSigilRag import voxsigil_blt, hybrid_blt
    from voxsigil_supervisor import supervisor_engine
    # Attempt to import specific processors, fall back if necessary
    try:
        from VoxSigilRag.voxsigil_processors import RAGProcessor as SpecificRAGProcessor
        from VoxSigilRag.voxsigil_processors import BLTProcessor as SpecificBLTProcessor
    except ImportError:
        SpecificRAGProcessor = None # Fallback handled later
        SpecificBLTProcessor = None # Fallback handled later
    
    from voxsigil_supervisor.interfaces.rag_interface import SimpleRagInterface
    from voxsigil_supervisor.interfaces.llm_interface import LocalLlmInterface
    from voxsigil_supervisor.strategies.scaffold_router import BasicScaffoldRouter
    from voxsigil_supervisor.strategies.evaluation_heuristics import SimpleEvaluator
    from voxsigil_supervisor.strategies.retry_policy import BasicRetryPolicy
    from VoxSigilRag.hybrid_blt import HybridMiddlewareConfig

    VOXSIGIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"One or more VoxSigil components could not be imported: {e}. VoxSigil features will be disabled.")
    VOXSIGIL_AVAILABLE = False
    # Define dummy classes if not available to prevent NameErrors later if use_voxsigil is True due to logic error
    class DummySupervisorComponent: pass
    voxsigil_middleware = DummySupervisorComponent()
    hybrid_blt = DummySupervisorComponent()
    voxsigil_rag = DummySupervisorComponent()
    voxsigil_blt = DummySupervisorComponent()
    supervisor_engine = DummySupervisorComponent()
    SimpleRagInterface = DummySupervisorComponent
    LocalLlmInterface = DummySupervisorComponent
    BasicScaffoldRouter = DummySupervisorComponent
    SimpleEvaluator = DummySupervisorComponent
    BasicRetryPolicy = DummySupervisorComponent
    HybridMiddlewareConfig = DummySupervisorComponent
    SpecificRAGProcessor = None
    SpecificBLTProcessor = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # Ensure logs go to stdout
        logging.FileHandler("tinyllama_multi_gpu_finetune.log", mode='w'), # Overwrite log file each run
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "./models/tinyllama_voxsigil"
DEFAULT_DATASET_PATH = "./data/voxsigil_train.jsonl" # Ensure this file exists
DEFAULT_EPOCHS = 1 # Reduced for quick testing
DEFAULT_LEARNING_RATE = 5e-5 # Adjusted common learning rate
MAX_LENGTH = 512


class VoxSigilDataset(Dataset):
    """Dataset for VoxSigil fine-tuning with prompt-completion pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH, use_voxsigil_processing=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.use_voxsigil = use_voxsigil_processing and VOXSIGIL_AVAILABLE
        
        self.middleware = None
        self.hybrid_middleware = None
        self.rag_processor = None
        self.blt_processor = None
        self.supervisor = None

        if self.use_voxsigil:
            logger.info("Attempting to initialize VoxSigil components for dataset processing...")
            try:
                middleware_config = HybridMiddlewareConfig(
                    entropy_threshold=0.25,
                    blt_hybrid_weight=0.7,
                    entropy_router_fallback="token_based",
                    cache_ttl_seconds=300,
                    log_level="INFO"
                )
                self.middleware = voxsigil_middleware.VoxSigilMiddleware()
                self.hybrid_middleware = hybrid_blt.HybridMiddleware(config=middleware_config)
                
                if SpecificRAGProcessor:
                    self.rag_processor = SpecificRAGProcessor()
                    logger.info("Using SpecificRAGProcessor from VoxSigilRag.voxsigil_processors")
                else:
                    self.rag_processor = voxsigil_rag.RAGProcessor() # Fallback
                    logger.info("Using fallback RAGProcessor from VoxSigilRag.voxsigil_rag")

                if SpecificBLTProcessor:
                    self.blt_processor = SpecificBLTProcessor()
                    logger.info("Using SpecificBLTProcessor from VoxSigilRag.voxsigil_processors")
                else:
                    self.blt_processor = voxsigil_blt.BLTProcessor() # Fallback
                    logger.info("Using fallback BLTProcessor from VoxSigilRag.voxsigil_blt")
                
                rag_interface = SimpleRagInterface(self.rag_processor)
                # LocalLlmInterface might not need the actual model instance during dataset prep
                # It might act as a placeholder or configuration for the supervisor.
                llm_interface = LocalLlmInterface() 
                scaffold_router = BasicScaffoldRouter()
                evaluator = SimpleEvaluator()
                retry_policy = BasicRetryPolicy()
                
                self.supervisor = supervisor_engine.VoxSigilSupervisor(
                    rag_interface=rag_interface,
                    llm_interface=llm_interface,
                    scaffold_router=scaffold_router,
                    evaluation_heuristics=evaluator,
                    retry_policy=retry_policy,
                    max_iterations=2
                )
                logger.info("VoxSigil components initialized successfully for dataset.")
            except Exception as e:
                logger.error(f"Failed to initialize one or more VoxSigil components: {e}", exc_info=True)
                logger.error(traceback.format_exc())
                self.use_voxsigil = False
                logger.warning("VoxSigil processing disabled due to initialization error.")
        
        logger.info(f"Loading dataset from {data_path}")
        if not Path(data_path).exists():
            logger.error(f"Dataset file not found: {data_path}. Please create it or provide a valid path.")
            # Create a dummy example to prevent crash if file is missing, and allow script to proceed to model loading for testing
            self.examples.append({"prompt": "This is a dummy prompt as dataset was not found.", "completion": " This is a dummy completion."})
            # raise FileNotFoundError(f"Dataset file not found: {data_path}") # Or raise error
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(tqdm(f, desc="Loading dataset")):
                    try:
                        example = json.loads(line)
                        if "prompt" not in example or "completion" not in example:
                            logger.warning(f"Skipping malformed example at line {line_idx+1}: {line.strip()}")
                            continue
                        self.examples.append(example)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON at line {line_idx+1}: {line.strip()}")
        
        if not self.examples: # If file was empty or all lines malformed
            logger.warning("No valid examples found in the dataset. Using a single dummy example for training.")
            self.examples.append({"prompt": "This is a dummy prompt as dataset was empty.", "completion": " This is a dummy completion."})

        logger.info(f"Loaded {len(self.examples)} examples.")
        if self.use_voxsigil:
            logger.info("Dataset will attempt to use VoxSigil enhancements during training.")
        else:
            logger.info("Dataset will use standard processing (VoxSigil not available, disabled, or failed to init).")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        completion = example["completion"]
        
        if self.use_voxsigil:
            original_prompt = prompt # For logging/comparison if needed
            try:
                if self.middleware:
                    prompt = self.middleware.process_text(prompt)
                if self.hybrid_middleware:
                    hybrid_enhanced = self.hybrid_middleware.process(prompt)
                    if hybrid_enhanced and isinstance(hybrid_enhanced, str):
                        prompt = hybrid_enhanced
                
                processed_by_blt = False
                if self.blt_processor:
                    blt_output = self.blt_processor.process(prompt)
                    if blt_output and isinstance(blt_output, str):
                        prompt += f"\nReasoning Structure: {blt_output}\n" # Append BLT output
                        processed_by_blt = True
                
                processed_by_rag = False
                if self.rag_processor:
                    rag_output = self.rag_processor.process(prompt)
                    if rag_output and isinstance(rag_output, str):
                        prompt += f"\nAdditional Context: {rag_output}\n" # Append RAG output
                        processed_by_rag = True
                
                if self.supervisor:
                    try:
                        # The supervisor's process_query might use RAG/BLT but its LLM interaction should be minimal/handled
                        supervisor_output = self.supervisor.process_query(
                            query=prompt,
                            system_prompt="You are an assistant. Enhance the query." # Generic prompt
                        )
                        if supervisor_output and isinstance(supervisor_output, str):
                            prompt = supervisor_output
                        # logger.debug(f"Prompt enhanced by VoxSigil Supervisor for example {idx}")
                    except Exception as e_sup:
                        logger.warning(f"Supervisor processing failed for example {idx}: {e_sup}. Using prior enhancements.", exc_info=False) # Set exc_info=True for debugging
            except Exception as e_vs_item:
                logger.warning(f"Error during VoxSigil processing for example {idx}: {e_vs_item}. Using original/partially_enhanced prompt.", exc_info=False)
                prompt = original_prompt # Fallback to original if major error

        full_text = prompt + completion
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        labels = input_ids.clone() # For Causal LM, labels are usually the same as input_ids
        
        # To prevent loss calculation on padding tokens, set their labels to -100
        # This is standard practice for Hugging Face Trainer with Causal LM
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def setup_model_and_tokenizer(model_name_or_path, trust_remote_code=True):
    """Set up the model and tokenizer with appropriate quantization and adapters."""
    logger.info(f"Loading model and tokenizer for: {model_name_or_path}")
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s).")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    num_hidden_layers = getattr(model_config, 'num_hidden_layers', None) # Dynamically get layer count
    if num_hidden_layers is None:
        # Fallback for models not having num_hidden_layers directly (e.g. older configs)
        # This is a guess; specific model architecture knowledge would be better.
        # For TinyLlama-1.1B, it's 22.
        if "tinyllama" in model_name_or_path.lower() and "1.1b" in model_name_or_path.lower():
            num_hidden_layers = 22
            logger.warning(f"Could not dynamically get num_hidden_layers. Assuming {num_hidden_layers} for {model_name_or_path}.")
        else:
            # A more generic fallback or error if critical for device_map
            logger.error("Could not determine num_hidden_layers for model.Device mapping might be suboptimal or fail.")
            # Set a plausible default or raise an error
            num_hidden_layers = 24 # A common number, but might be wrong
            # raise ValueError("num_hidden_layers could not be determined from model config.")


    if torch.cuda.is_available():
        logger.info("Setting up 4-bit quantization for GPU training (BitsAndBytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if supported
        )
        
        device_map_config = None
        if gpu_count > 1:
            if num_hidden_layers is None:
                logger.warning("Cannot create detailed device_map due to unknown num_hidden_layers. Using 'auto' device_map.")
                device_map_config = "auto"
            else:
                logger.info(f"Creating custom device map for {gpu_count} GPUs and {num_hidden_layers} layers.")
                # This is a basic device map strategy. More sophisticated ones (e.g., FSDP) are more robust.
                device_map = {"model.embed_tokens": 0} 
                
                # Calculate layers per GPU, ensuring last GPU gets any remainder
                layers_per_gpu = [num_hidden_layers // gpu_count] * gpu_count
                remainder = num_hidden_layers % gpu_count
                for i in range(remainder):
                    layers_per_gpu[i] += 1
                
                current_layer = 0
                for gpu_id in range(gpu_count):
                    for i in range(layers_per_gpu[gpu_id]):
                        if current_layer < num_hidden_layers:
                             device_map[f"model.layers.{current_layer}"] = gpu_id
                        current_layer += 1
                
                # Place norm and lm_head on the last GPU or spread them if needed
                device_map["model.norm"] = gpu_count - 1
                device_map["lm_head"] = gpu_count - 1
                device_map_config = device_map
                logger.info(f"Custom device map: {device_map_config}")

            # Set primary device explicitly for multi-GPU when using custom device_map
            # This helps Hugging Face place non-mapped parts or handle internal ops.
            torch.cuda.set_device(0)

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map=device_map_config, # Use the determined device_map_config
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=trust_remote_code,
            )
        else: # Single GPU or no GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto", # Let Hugging Face decide for single GPU
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=trust_remote_code,
            )
            
        logger.info("Preparing model for k-bit training (PEFT).")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Enable GC here
        logger.info("Model uses 4-bit quantization.")
    else:
        logger.warning("No GPU available. Loading model on CPU with standard precision. Training will be very slow.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            # No quantization for CPU, device_map="cpu" implicitly
        )
    
    logger.info("Applying LoRA adapter.")
    lora_config = LoraConfig(
        r=16, # Increased r slightly, common value
        lora_alpha=32, # Increased alpha
        lora_dropout=0.05, # Standard dropout
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Check these for your model
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train_model_on_gpus(args):
    """Train the model using the specified arguments."""
    logger.info("Starting model training with multi-GPU optimization.")
    
    model, tokenizer = setup_model_and_tokenizer(args.model_name, trust_remote_code=True) # Assuming trust_remote_code
    
    # Ensure model config reflects use_cache=False if gradient checkpointing is used by Trainer
    # prepare_model_for_kbit_training might handle this, but explicit is safer.
    if hasattr(model, "config") and args.gradient_checkpointing:
        model.config.use_cache = False
        logger.info("Model cache explicitly disabled for gradient checkpointing compatibility.")
    
    dataset = VoxSigilDataset(
        args.dataset_path, 
        tokenizer, 
        max_length=args.max_length,
        use_voxsigil_processing=not args.disable_voxsigil
    )
    
    if not dataset.examples:
        logger.error("Dataset is empty. Cannot start training.")
        return None # Indicate failure

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    per_device_batch_size = args.per_device_batch_size
    
    if gpu_count == 0:
        logger.warning("Training on CPU. per_device_batch_size will be used as total batch size.")
        actual_gpu_count_for_grad_accum = 1 # Avoid division by zero
    else:
        actual_gpu_count_for_grad_accum = gpu_count

    # Calculate gradient accumulation steps based on GPU count and desired effective batch size
    # target_effective_batch_size = 16 # Example: aim for an effective batch size
    # gradient_accumulation_steps = max(1, target_effective_batch_size // (per_device_batch_size * actual_gpu_count_for_grad_accum))
    # Or use a fixed gradient_accumulation_steps from args
    gradient_accumulation_steps = args.gradient_accumulation_steps

    logger.info(f"Training configuration: {gpu_count} GPUs, {per_device_batch_size} batch size per device.")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}.")
    effective_batch_size = per_device_batch_size * actual_gpu_count_for_grad_accum * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}.")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(), # Use fp16 if bf16 not available on GPU
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # Use bf16 if available
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False, # Important if dataset returns extra columns not used by model
        report_to="none", # ["tensorboard", "wandb"] or "all"
        # ddp_find_unused_parameters=False, # Only relevant for DDP. Not explicitly using DDP here.
        dataloader_pin_memory=True, # Can speed up data loading if CPU RAM allows
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch", # Fused optimizer for GPU
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=args.gradient_checkpointing, # Controlled by arg
        # no_cuda=not torch.cuda.is_available(), # Handled by fp16/bf16 and device_map
        # auto_find_batch_size=True, # Can be useful but sometimes slow; relying on manual config
        tf32=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), # Enable TF32 on Ampere+
        dataloader_num_workers=args.dataloader_num_workers, # Added arg for dataloader workers
        seed=42, # For reproducibility
        # Required if gradient_checkpointing=True and model has PEFT adapters
        # and not using DDP. If using model parallel via device_map this might also be needed.
        ddp_find_unused_parameters=False if gpu_count > 1 and args.gradient_checkpointing else None,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Using standard Trainer. If specific multi-GPU issues arise with device_map,
    # the custom MultiGPUTrainer might be needed, but Hugging Face Trainer
    # has improved significantly with device_map.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer # Useful for some Trainer features like eval with generation
    )
    
    logger.info("Beginning training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed.")
        
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir) # Saves adapter, not full model unless merged
        tokenizer.save_pretrained(args.output_dir)
        
        # Log some training metrics
        metrics = train_result.metrics
        logger.info(f"TrainOutput metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return train_result
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        # Attempt to save model state even if training fails partway
        if trainer and hasattr(trainer, 'model') and trainer.model:
            try:
                logger.warning("Attempting to save model state due to training error...")
                trainer.save_model(os.path.join(args.output_dir, "checkpoint_on_error"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint_on_error"))
                logger.info("Model state saved to checkpoint_on_error directory.")
            except Exception as save_e:
                logger.error(f"Could not save model state on error: {save_e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU TinyLlama Fine-tuning with VoxSigil integration")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--disable_voxsigil", action="store_true", help="Disable VoxSigil integration")
    
    # Added more detailed training arguments
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of workers for dataloader.")

    return parser.parse_args()


if __name__ == "__main__":
    # Forcing expandable_segments can sometimes help with fragmentation on NVIDIA GPUs
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
    # However, recent PyTorch versions manage memory well. Test if this is needed.
    # Setting this might also depend on the specific GPU architecture and driver.
    
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup a file handler for the root logger to catch messages from transformers/peft etc.
    # This supplements the script-specific logger.
    file_handler = logging.FileHandler(Path(args.output_dir) / "full_training_run.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler) # Add to root logger
    logging.getLogger().setLevel(logging.INFO) # Ensure root logger captures INFO

    if VOXSIGIL_AVAILABLE and not args.disable_voxsigil:
        logger.info("VoxSigil components appear available and are enabled.")
    elif args.disable_voxsigil:
        logger.info("VoxSigil integration explicitly disabled via command line argument.")
    else: # Not available
        logger.warning("VoxSigil components not available. Training without enhanced capabilities.")
    
    gpu_info_str = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info_str.append(f"GPU {i}: {props.name} (Total Memory: {props.total_memory / 1024**3:.2f} GB, Arch: {props.major}.{props.minor})")
    else:
        gpu_info_str.append("No CUDA GPUs available. Using CPU.")

    # Create a dummy dataset file if it doesn't exist, for testing purposes
    # In a real scenario, the dataset should be prepared beforehand.
    if not Path(args.dataset_path).exists():
        logger.warning(f"Dataset file {args.dataset_path} not found. Creating a dummy dataset for testing.")
        with open(args.dataset_path, 'w', encoding='utf-8') as f:
            for i in range(10): # Create 10 dummy examples
                f.write(json.dumps({"prompt": f"Dummy prompt {i+1}: What is the capital of France?", "completion": f" Paris is the capital of France. This is example {i+1}."}) + "\n")
    
    print("\n" + "="*80)
    print(" VANTA- Enhanced Multi-GPU TinyLlama Training Script")
    print("="*80)
    print(f"Run Parameters:")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Max Sequence Length: {args.max_length}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Per Device Batch Size: {args.per_device_batch_size}")
    print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    print(f"  VoxSigil Integration: {'Enabled' if VOXSIGIL_AVAILABLE and not args.disable_voxsigil else 'Disabled'}")
    print("\nGPU Information:")
    for info in gpu_info_str:
        print(f"  - {info}")
    if torch.cuda.is_available(): print(f"  BF16 Supported: {torch.cuda.is_bf16_supported()}")
    print("="*80 + "\n")
    
    logger.info("Initializing training...")
    training_result = train_model_on_gpus(args)
    
    print("\n" + "="*80)
    if training_result:
        logger.info("TRAINING COMPLETED SUCCESSFULLY.")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Final metrics: {training_result.metrics}")
    else:
        logger.error("TRAINING FAILED OR WAS INTERRUPTED. Check logs for details.")
        print("TRAINING FAILED - CHECK LOGS FOR DETAILS")
    print(f"Logs saved to: {args.output_dir}/full_training_run.log and tinyllama_multi_gpu_finetune.log")
    print("="*80 + "\n")