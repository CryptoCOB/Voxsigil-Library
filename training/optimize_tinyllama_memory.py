"""
Script to optimize TinyLlama fine-tuning for lower memory usage
"""

import os
from pathlib import Path

def modify_finetuning_file():
    # Define the paths
    models_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    tinyllama_file = models_dir / "tinyllama_voxsigil_finetune.py"
    
    if not tinyllama_file.exists():
        print(f"Error: {tinyllama_file} not found!")
        return False
    
    # Read the original file
    with open(tinyllama_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Make a backup
    backup_file = tinyllama_file.with_suffix('.py.bak')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup at {backup_file}")
    
    # Modify the content to enable 4-bit quantization and reduce batch size
    # Find the default arguments
    if "DEFAULT_BATCH_SIZE = " not in content:
        content = content.replace(
            "MAX_LENGTH = 1024",
            "MAX_LENGTH = 1024\nDEFAULT_BATCH_SIZE = 2"
        )
    else:
        content = content.replace(
            "DEFAULT_BATCH_SIZE = 4",
            "DEFAULT_BATCH_SIZE = 2"
        )
    
    # Ensure the parser includes batch_size
    if "--batch_size" not in content:
        parser_section = "parser.add_argument(\n        \"--learning_rate\","
        batch_size_arg = """parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size per device",
    )
    """
        content = content.replace(parser_section, parser_section + "\n    " + batch_size_arg)
    
    # Update setup_model_and_tokenizer to use 4-bit by default
    if "def setup_model_and_tokenizer" in content:
        content = content.replace(
            "def setup_model_and_tokenizer(model_name, use_lora=True, use_4bit=False):",
            "def setup_model_and_tokenizer(model_name, use_lora=True, use_4bit=True):"
        )
    
    # Add batch size to the argument parser if not already there
    if "--use_4bit" not in content:
        disable_vox_arg = "parser.add_argument(\n        \"--disable_voxsigil\","
        use_4bit_arg = """parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for training (requires bitsandbytes)",
    )
    """
        content = content.replace(disable_vox_arg, use_4bit_arg + "\n    " + disable_vox_arg)
    
    # Modify the training arguments to use smaller gradient accumulation
    if "TrainingArguments(" in content:
        # Find the TrainingArguments section and adjust gradient_accumulation_steps
        old_line = "gradient_accumulation_steps=gradient_accumulation_steps,"
        new_line = "gradient_accumulation_steps=1,  # Reduced for lower memory usage"
        content = content.replace(old_line, new_line)
        
        # Add memory optimization flags
        old_line = "remove_unused_columns=False,"
        additional_opts = """remove_unused_columns=False,
        optim="adamw_torch",
        torch_compile=False,  # Disable torch.compile to save memory
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory"""
        content = content.replace(old_line, additional_opts)
    
    # Write the modified content back
    with open(tinyllama_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Modified {tinyllama_file} to use less memory:")
    print("  - Reduced batch size to 2")
    print("  - Enabled 4-bit quantization by default")
    print("  - Reduced gradient accumulation steps")
    print("  - Added memory optimization flags")
    
    return True

def run_optimized_finetuning():
    """Run the fine-tuning with extra memory optimization flags"""
    models_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    tinyllama_file = models_dir / "tinyllama_voxsigil_finetune.py"
    
    if not tinyllama_file.exists():
        print(f"Error: {tinyllama_file} not found!")
        return False
    
    command = f"""
    & $env:CONDA_PREFIX\\python.exe {tinyllama_file} --use_4bit --batch_size 1 --learning_rate 2e-4 --max_length 512
    """
    
    print("Running optimized fine-tuning command:")
    print(command)
    
    # Return the command for the user to run
    return command

if __name__ == "__main__":
    if modify_finetuning_file():
        command = run_optimized_finetuning()
        print("\nTo run the optimized fine-tuning, execute this command:")
        print(command)
        print("\nOr you can run the script directly with these optimization flags:")
        print("python models/tinyllama_voxsigil_finetune.py --use_4bit --batch_size 1 --max_length 512")
    else:
        print("Failed to modify fine-tuning script.")
