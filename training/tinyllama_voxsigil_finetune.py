import os
import sys
import logging
import json
import argparse
from pathlib import Path
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
        logging.FileHandler("tinyllama_voxsigil_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "./models/tinyllama_voxsigil"
DEFAULT_DATASET_PATH = "./data/voxsigil_train.jsonl"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 5e-4
MAX_LENGTH = 1024
import sys
import logging
import json
import argparse
from pathlib import Path
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
        logging.FileHandler("tinyllama_voxsigil_finetune.log"),
    ],
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "./models/tinyllama_voxsigil"
DEFAULT_DATASET_PATH = "./data/voxsigil_train.jsonl"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 5e-4
MAX_LENGTH = 1024


def save_model_card(output_dir, model_name):
    """Save a model card with usage instructions to the output directory."""
    model_card = f"""
# TinyLlama-Personal-Assistant

This is a fine-tuned version of [{model_name}](https://huggingface.co/{model_name}) customized as a personal assistant with VoxSigil enhancements.

## Model description

This model was fine-tuned to serve as a helpful personal assistant that can:
- Provide practical advice for everyday questions
- Generate creative content like stories and poems
- Assist with task planning and organization
- Help with email drafting and communication
- Offer solutions to common problems
- Perform reasoned calculations and step-by-step problem solving

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Chat example
messages = [
    {{"role": "user", "content": "Can you suggest a quick dinner recipe?"}}
]

# Generate response
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## VoxSigil Integration

This model can optionally use VoxSigil's symbolic reasoning library for enhanced capabilities if available in your environment.
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)


class VoxSigilDataset(Dataset):
    """Dataset for VoxSigil fine-tuning with prompt-completion pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "labels": tokenized.input_ids[0].clone(),
        }


def setup_model_and_tokenizer(model_name, use_lora=True, use_4bit=False):
    """Set up the model and tokenizer with appropriate quantization and adapters."""
    logger.info(f"Loading model: {model_name}")
    
    # Quantization config for low-memory training
    if use_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            logger.info("Using 4-bit quantization for training")
        except ImportError:
            logger.warning("BitsAndBytes not available, falling back to standard precision")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    if use_lora:
        logger.info("Applying LoRA adapter")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


class VoxSigilDataset(Dataset):
    """Dataset for VoxSigil fine-tuning with prompt-completion pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "labels": tokenized.input_ids[0].clone(),
        }


def setup_model_and_tokenizer(model_name, use_lora=True, use_4bit=False):
    """Set up the model and tokenizer with appropriate quantization and adapters."""
    logger.info(f"Loading model: {model_name}")
    
    # Quantization config for low-memory training
    if use_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            logger.info("Using 4-bit quantization for training")
        except ImportError:
            logger.warning("BitsAndBytes not available, falling back to standard precision")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    if use_lora:
        logger.info("Applying LoRA adapter")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def train_model(args):
    """Train the model using the specified arguments."""
    logger.info("Starting model training")
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, 
        use_lora=args.use_lora, 
        use_4bit=args.use_4bit
    )
    
    # Prepare dataset
    dataset = VoxSigilDataset(args.dataset_path, tokenizer, max_length=args.max_length)
    
    # Training arguments
    gradient_accumulation_steps = 4
    if torch.cuda.is_available():
        gradient_accumulation_steps = 2 if torch.cuda.device_count() > 1 else 4
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
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
    
    # Create model card
    save_model_card(args.output_dir, args.model_name)
    
    logger.info("Training complete!")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with VoxSigil integration")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, 
                        help="Name of the pre-trained model to fine-tune")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, 
                        help="Path to the training dataset in JSONL format")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, 
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, 
                        help="Maximum sequence length for training")
    parser.add_argument("--use_lora", action="store_true", default=True, 
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--use_4bit", action="store_true", default=False, 
                        help="Use 4-bit quantization for training (requires bitsandbytes)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log VoxSigil availability
    if VOXSIGIL_AVAILABLE:
        logger.info("VoxSigil components successfully imported")
    else:
        logger.warning("VoxSigil components not available - training without enhanced capabilities")
    
    # Start training
    train_model(args)


if __name__ == "__main__":
    main()