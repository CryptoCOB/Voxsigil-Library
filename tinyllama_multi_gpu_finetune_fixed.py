#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyLlama Low-Memory Multi-GPU Training Script (Fixed)

This script is optimized for running TinyLlama fine-tuning on multiple consumer-grade GPUs
with limited VRAM (12GB RTX 3060). It supports different distributed training strategies:
1. DeepSpeed (Zero Stage 3) - Best for multi-GPU with limited VRAM
2. DDP (Distributed Data Parallel) - Standard PyTorch distributed training
3. FSDP (Fully Sharded Data Parallel) - Newer PyTorch distributed training
4. Single GPU / Basic Multi-GPU - Simpler approach for testing
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
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
import traceback

# Clear CUDA cache
torch.cuda.empty_cache()

# Add necessary paths to system path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# Import VoxSigil components
try:
    from VoxSigilRag import voxsigil_middleware, voxsigil_rag, hybrid_blt
    from voxsigil_supervisor import supervisor_engine
    try:
        from VoxSigilRag.voxsigil_processors import RAGProcessor as SpecificRAGProcessor
        from VoxSigilRag.voxsigil_processors import BLTProcessor as SpecificBLTProcessor
    except ImportError:
        SpecificRAGProcessor = None
        SpecificBLTProcessor = None
    
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
    # Define dummy classes if not available
    class DummySupervisorComponent: pass
    voxsigil_middleware = DummySupervisorComponent()
    hybrid_blt = DummySupervisorComponent()
    voxsigil_rag = DummySupervisorComponent()
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
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tinyllama_multi_gpu_finetune.log", mode='w'),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "./models/tinyllama_voxsigil"
DEFAULT_DATASET_PATH = "./data/voxsigil_train.jsonl"
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 5e-5
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
                self.middleware = hybrid_blt.HybridMiddleware(config=middleware_config)
                self.hybrid_middleware = hybrid_blt.HybridMiddleware(config=middleware_config)
                if SpecificRAGProcessor:
                    self.rag_processor = SpecificRAGProcessor()
                    logger.info("Using SpecificRAGProcessor from VoxSigilRag.voxsigil_processors")
                else:
                    self.rag_processor = voxsigil_rag.RAGProcessor()
                    logger.info("Using fallback RAGProcessor from VoxSigilRag.voxsigil_rag")
                
                if SpecificBLTProcessor:
                    self.blt_processor = SpecificBLTProcessor()
                    logger.info("Using SpecificBLTProcessor from VoxSigilRag.voxsigil_processors")
                else:
                    self.blt_processor = hybrid_blt.BLTProcessor()
                    logger.info("Using fallback BLTProcessor from VoxSigilRag.hybrid_blt")
                
                rag_interface = SimpleRagInterface(self.rag_processor)
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
            # Create a dummy example
            self.examples.append({"prompt": "This is a dummy prompt as dataset was not found.", "completion": " This is a dummy completion."})
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
            
            if not self.examples:
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
            original_prompt = prompt
            try:
                if self.middleware:
                    # Use HybridMiddleware as a callable
                    result = self.middleware({"messages": [{"role": "user", "content": prompt}]})
                    if result and "messages" in result and isinstance(result["messages"], list):
                        prompt = result["messages"][-1].get("content", prompt)
                if self.hybrid_middleware:
                    result = self.hybrid_middleware({"messages": [{"role": "user", "content": prompt}]})
                    if result and "messages" in result and isinstance(result["messages"], list):
                        prompt = result["messages"][-1].get("content", prompt)
                
                if self.blt_processor:
                    blt_output = self.blt_processor.process(prompt)
                    if blt_output and isinstance(blt_output, str):
                        prompt += f"\nReasoning Structure: {blt_output}\n"
                
                if self.rag_processor:
                    rag_output = self.rag_processor.process(prompt)
                    if rag_output and isinstance(rag_output, str):
                        prompt += f"\nAdditional Context: {rag_output}\n"
                
                if self.supervisor:
                    try:
                        # process_query returns a tuple (response_string, metadata_dict)
                        supervisor_output = self.supervisor.process_query(
                            prompt
                        )
                        # Extract the response string from the tuple
                        if supervisor_output and isinstance(supervisor_output, tuple) and len(supervisor_output) > 0:
                            response_string = supervisor_output[0]
                            if isinstance(response_string, str):
                                prompt = response_string
                    except Exception as e_sup:
                        logger.error(f"Supervisor processing failed for example {idx}. Error: {e_sup}. Using prior enhancements. Full traceback:", exc_info=True)
            except Exception as e_vs_item:
                logger.warning(f"Error during VoxSigil processing for example {idx}: {e_vs_item}. Using original/partially_enhanced prompt.", exc_info=False)
                prompt = original_prompt

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
        labels = input_ids.clone()
        
        # Set padding tokens labels to -100
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

    if torch.cuda.is_available():
        logger.info("Setting up 4-bit quantization for GPU training (BitsAndBytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        
        # In distributed training, we'll load the model without device_map
        # Each process will load its own shard of the model
        logger.info("Loading model for distributed training")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=None,  # No device_map for distributed training
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=trust_remote_code,
        )
            
        logger.info("Preparing model for kbit training (PEFT).")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("Model uses 4-bit quantization.")
    else:
        logger.warning("No GPU available. Loading model on CPU with standard precision. Training will be very slow.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    
    logger.info("Applying LoRA adapter.")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # Ensure LoRA is in training mode!
    )
    model = get_peft_model(model, lora_config)

    # Force LoRA parameters to require grad (in case of checkpoint or config override)
    trainable_param_count = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            param.requires_grad = True
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} | shape: {param.shape}")
            trainable_param_count += param.numel()
    logger.info(f"Total trainable parameters: {trainable_param_count}")

    # Ensure model is in training mode
    model.train() 

    # Check if any parameters require gradients
    has_trainable_params = any(p.requires_grad for p in model.parameters())
    logger.info(f"Model has trainable parameters: {has_trainable_params}")
    if not has_trainable_params:
        logger.warning("No parameters require gradients. Training will not occur.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable layer: {name}")
    
    return model, tokenizer


from transformers import Trainer as HFTrainer

class DebugTrainer(HFTrainer):
    def __init__(self, *args, **kwargs):
        print("!!! DebugTrainer.__init__ CALLED !!!", flush=True)
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        print("!!! DebugTrainer.training_step CALLED !!!", flush=True)
        logger.info("[DEBUG] DebugTrainer.training_step called!")
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        logger.info(f"[DEBUG] Loss type: {type(loss)}, requires_grad: {getattr(loss, 'requires_grad', None)}, grad_fn: {getattr(loss, 'grad_fn', None)}")
        return loss

    def _training_step(self, model, inputs):
        print("!!! DebugTrainer._training_step CALLED !!!", flush=True)
        return super()._training_step(model, inputs)


def train_model_on_gpus(args):
    """Train the model using the specified arguments."""
    logger.info("Starting model training with multi-GPU optimization.")
    
    # Set the local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        # Initialize the distributed environment
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Initialized distributed training with rank {local_rank}/{world_size-1}")
        # Set GPU for this process
        torch.cuda.set_device(local_rank)
    
    model, tokenizer = setup_model_and_tokenizer(args.model_name, trust_remote_code=True)
    
    # Ensure model is in training mode before passing to Trainer
    model.train()

    # Disable cache for gradient checkpointing
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
        return None

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    per_device_batch_size = args.per_device_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    logger.info(f"Training configuration: {gpu_count} GPUs, {per_device_batch_size} batch size per device.")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}.")
    effective_batch_size = per_device_batch_size * max(1, gpu_count) * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size}.")
    
    # Configure training arguments based on the strategy
    training_args_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_pin_memory": True,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "gradient_checkpointing": args.gradient_checkpointing,
        "tf32": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": 42,
        "local_rank": local_rank,
        "ddp_find_unused_parameters": False,
    }
    
    # DeepSpeed configuration
    if args.use_deepspeed and args.deepspeed:
        training_args_kwargs["deepspeed"] = args.deepspeed
        logger.info(f"Using DeepSpeed with config from {args.deepspeed}")
    
    # FSDP specific configuration
    if args.use_fsdp and is_distributed:
        training_args_kwargs["fsdp"] = "full_shard"
        training_args_kwargs["fsdp_transformer_layer_cls_to_wrap"] = "LlamaDecoderLayer" if "llama" in args.model_name.lower() else None
        logger.info("Using FSDP (Fully Sharded Data Parallel)")
    
    # Configure training arguments
    training_args = TrainingArguments(**training_args_kwargs)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logger.info("Beginning training...")
    try:
        # Ensure model is in training mode right before trainer.train()
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model.train()
        
        train_result = trainer.train()
        logger.info("Training completed.")
        
        # Only the main process saves the model
        if local_rank <= 0 or not is_distributed:
            logger.info(f"Saving model to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # Log metrics
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
        if local_rank <= 0 or not is_distributed:
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
    
    # Training arguments
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of workers for dataloader.")
    
    # Distributed training strategy flags
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for distributed training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file.")
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP for distributed training.")
    parser.add_argument("--use_ddp", action="store_true", help="Use DDP for distributed training.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
      # Setup a file handler for the root logger
    file_handler = logging.FileHandler(Path(args.output_dir) / "full_training_run.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Check for distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        logger.info(f"Running in distributed mode with local_rank={local_rank}, world_size={world_size}")
    else:
        logger.info("Running in non-distributed mode")

    if VOXSIGIL_AVAILABLE and not args.disable_voxsigil:
        logger.info("VoxSigil components appear available and are enabled.")
    elif args.disable_voxsigil:
        logger.info("VoxSigil integration explicitly disabled via command line argument.")
    else:
        logger.warning("VoxSigil components not available. Training without enhanced capabilities.")
    
    # Only print banner on the main process
    if local_rank <= 0 or not is_distributed:
        gpu_info_str = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info_str.append(f"GPU {i}: {props.name} (Total Memory: {props.total_memory / 1024**3:.2f} GB, Arch: {props.major}.{props.minor})")
        else:
            gpu_info_str.append("No CUDA GPUs available. Using CPU.")

        # Create a dummy dataset file if it doesn't exist for testing
        if not Path(args.dataset_path).exists():
            logger.warning(f"Dataset file {args.dataset_path} not found. Creating a dummy dataset for testing.")
            with open(args.dataset_path, 'w', encoding='utf-8') as f:
                for i in range(10):
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
        print(f"  Distributed Training: {'Yes' if is_distributed else 'No'}")
        if is_distributed:
            strategy = "DeepSpeed" if args.use_deepspeed else "FSDP" if args.use_fsdp else "DDP" if args.use_ddp else "None"
            print(f"  Distribution Strategy: {strategy}")
        
        print("\nGPU Information:")
        for info in gpu_info_str:
            print(f"  - {info}")
        if torch.cuda.is_available(): print(f"  BF16 Supported: {torch.cuda.is_bf16_supported()}")
        print("="*80 + "\n")
    
    logger.info("Initializing training...")
    training_result = train_model_on_gpus(args)
    
    # Only print final output on the main process
    if local_rank <= 0 or not is_distributed:
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
    
    # Cleanup distributed environment
    if is_distributed:
        torch.distributed.destroy_process_group()
