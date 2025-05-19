"""
TinyLlama Personal Assistant - All-in-One Script

This consolidated script handles all aspects of TinyLlama Personal Assistant:
1. GPU detection and optimization
2. Fine-tuning with optimal parameters
3. Chat interface with the trained model
4. VoxSigil integration for enhanced capabilities

Run with:
python tinyllama_assistant.py --mode [train|chat|info]
"""

# Import all required modules
import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import torch
import time
import json
import platform
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tinyllama_assistant.log"),
    ],
)
logger = logging.getLogger(__name__)

# Add necessary paths to system path for VoxSigil imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "VoxSigilRag"))
sys.path.append(str(root_dir / "voxsigil_supervisor"))

# Import VoxSigil components (with error handling)
try:
    from VoxSigilRag import voxsigil_middleware, voxsigil_rag, voxsigil_rag_compression
    from VoxSigilRag import voxsigil_blt, hybrid_blt, voxsigil_blt_rag
    from voxsigil_supervisor import supervisor_engine
    from voxsigil_supervisor.blt_supervisor_integration import BLTSupervisorRagInterface, TinyLlamaIntegration
    VOXSIGIL_AVAILABLE = True
    BLT_AVAILABLE = True
    logger.info("VoxSigil components successfully imported with BLT enhancements")
except ImportError as e:
    try:
        # Try without BLT components
        from VoxSigilRag import voxsigil_middleware, voxsigil_rag, voxsigil_rag_compression
        from voxsigil_supervisor import supervisor_engine
        VOXSIGIL_AVAILABLE = True
        BLT_AVAILABLE = False
        logger.warning(f"VoxSigil components imported without BLT enhancements: {e}")
    except ImportError as e2:
        logger.warning(f"VoxSigil components could not be imported: {e2}")
        VOXSIGIL_AVAILABLE = False
        BLT_AVAILABLE = False

# Constants
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "./models/tinyllama_voxsigil"
DEFAULT_DATASET_PATH = "./data/voxsigil_train.jsonl"
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 5e-4
MAX_LENGTH = 1024

# Import additional required libraries for model handling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HF_AVAILABLE = True
except ImportError:
    logger.warning("Hugging Face Transformers or PEFT not available")
    HF_AVAILABLE = False


def initialize_voxsigil_components(library_path=None, use_blt=True):
    """Initialize VoxSigil components for the assistant."""
    if not VOXSIGIL_AVAILABLE:
        logger.warning("VoxSigil components not available for initialization")
        return None, None
    
    try:
        library_path = library_path or os.path.join(root_dir, "voxsigil-Library")
        library_path = Path(library_path)
        
        if use_blt and BLT_AVAILABLE:
            logger.info("Initializing BLT-enhanced VoxSigil components...")
            # Initialize using the new BLT integration
            integration = TinyLlamaIntegration(
                voxsigil_library_path=library_path,
                blt_config={
                    "entropy_threshold": 0.3,
                    "blt_hybrid_weight": 0.7
                }
            )
            
            # Create supervisor
            supervisor = integration.create_supervisor()
            
            logger.info("BLT-enhanced VoxSigil supervisor initialized")
            return supervisor, integration.rag_interface
        else:
            # Fall back to standard VoxSigil components
            logger.info("Initializing standard VoxSigil components...")
            from VoxSigilRag.voxsigil_rag import VoxSigilRAG
            from voxsigil_supervisor.interfaces.rag_interface import SupervisorRagInterface
            
            rag = VoxSigilRAG(voxsigil_library_path=library_path)
            rag_interface = SupervisorRagInterface(voxsigil_library_path=library_path)
            
            # Other supervisor components would be initialized here
            
            logger.info("Standard VoxSigil components initialized")
            return None, rag  # Return rag instead of full supervisor for backward compatibility
    except Exception as e:
        logger.error(f"Error initializing VoxSigil components: {e}")
        return None, None
        
        # Check RAG components
        try:
            rag_processor = voxsigil_rag.RAGProcessor()
            logger.info("VoxSigil RAG is functional.")
        except Exception as e:
            logger.warning(f"VoxSigil RAG initialization failed: {e}")
        
        # Check Middleware components
        try:
            middleware = voxsigil_middleware.VoxSigilMiddleware()
            logger.info("VoxSigil Middleware is functional.")
        except Exception as e:
            logger.warning(f"VoxSigil Middleware initialization failed: {e}")
        
        # Check BLT components
        try:
            blt_processor = voxsigil_blt.BLTProcessor()
            logger.info("VoxSigil BLT is functional.")
        except Exception as e:
            logger.warning(f"VoxSigil BLT initialization failed: {e}")
        
        # Check Supervisor components
        try:
            supervisor = supervisor_engine.SupervisorEngine()
            logger.info("VoxSigil Supervisor is functional.")
        except Exception as e:
            logger.warning(f"VoxSigil Supervisor initialization failed: {e}")
    else:
        logger.warning("VoxSigil components are not available.")
    
    return True


def initialize_tinyllama_model(model_name=DEFAULT_MODEL_NAME, load_in_8bit=False, load_in_4bit=True):
    """
    Initialize a TinyLlama model for inference.
    
    Args:
        model_name: Name or path of the TinyLlama model to load (default: TinyLlama-1.1B-Chat-v1.0)
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        
    Returns:
        Loaded TinyLlama model ready for inference
    """
    if not HF_AVAILABLE:
        logger.error("Hugging Face Transformers not available. Cannot initialize TinyLlama model.")
        return None
    
    try:
        logger.info(f"Loading TinyLlama model: {model_name}")
        
        # Set quantization parameters
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info("Using 4-bit quantization for TinyLlama")
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            logger.info("Using 8-bit quantization for TinyLlama")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set tokenizer padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_args = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if quantization_config:
            model_args["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_args
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        logger.info(f"TinyLlama model {model_name} initialized successfully")
        
        # Return both model and tokenizer as a tuple
        return (model, tokenizer)
    
    except Exception as e:
        logger.error(f"Error initializing TinyLlama model: {e}")
        return None


def train_model(args):
    """Train the TinyLlama model using VoxSigil enhancements."""
    logger.info("Starting TinyLlama training with VoxSigil enhancements...")
    
    if not HF_AVAILABLE:
        logger.error("Hugging Face Transformers not available. Cannot train model.")
        return False
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Call the specialized training script if it exists
    train_script = os.path.join(current_dir, "tinyllama_voxsigil_finetune.py")
    if os.path.exists(train_script):
        cmd = [
            sys.executable,
            train_script,
            "--model_name", args.model_name,
            "--output_dir", args.output_dir,
            "--dataset_path", args.dataset,
            "--epochs", str(args.epochs),
            "--learning_rate", str(args.learning_rate),
        ]
        
        if not args.use_voxsigil:
            cmd.append("--disable_voxsigil")
        
        logger.info(f"Running training script: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream the output
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        if process.returncode != 0:
            logger.error("Training failed.")
            return False
        
        logger.info("Training completed successfully.")
        return True
    else:
        logger.error(f"Training script not found: {train_script}")
        logger.info("Attempting direct model training...")
        
        # If we reach here, we need to implement the training logic directly
        # This is a simplified implementation - the specialized script is preferred
        import importlib.util
        spec = importlib.util.find_spec("tinyllama_voxsigil_finetune")
        if spec is not None:
            tinyllama_finetune = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tinyllama_finetune)
            
            # Call the train function from the imported module
            train_args = argparse.Namespace(
                model_name=args.model_name,
                output_dir=args.output_dir,
                dataset_path=args.dataset,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                use_lora=True,
                use_4bit=False,
                batch_size=4,
                max_length=MAX_LENGTH,
                disable_voxsigil=not args.use_voxsigil
            )
            
            try:
                tinyllama_finetune.train_model(train_args)
                logger.info("Training completed successfully.")
                return True
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return False
        else:
            logger.error("Unable to find or import tinyllama_voxsigil_finetune module.")
            return False


def chat_with_model(args):
    """Chat with the trained TinyLlama model with VoxSigil enhancements."""
    logger.info("Starting chat interface...")
    
    if not HF_AVAILABLE:
        logger.error("Hugging Face Transformers not available. Cannot use chat interface.")
        return False
    
    # Check if model exists
    model_path = args.model_path if args.model_path else args.output_dir
    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        return False
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        
        # Initialize VoxSigil components if available and enabled
        voxsigil_processor = None
        if VOXSIGIL_AVAILABLE and args.use_voxsigil:
            try:
                logger.info("Initializing VoxSigil components for enhanced chat...")
                
                # Initialize middleware for text processing
                middleware = voxsigil_middleware.VoxSigilMiddleware()
                
                # Initialize RAG for context retrieval
                rag_processor = voxsigil_rag.RAGProcessor()
                
                # Initialize Supervisor for managing interaction
                supervisor = supervisor_engine.SupervisorEngine()
                
                logger.info("VoxSigil components initialized successfully")
                voxsigil_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize VoxSigil components: {e}")
                voxsigil_enabled = False
        else:
            voxsigil_enabled = False
        
        # Print welcome message
        print("\n" + "="*50)
        print("TinyLlama Personal Assistant")
        print("Type 'exit' or 'quit' to end the conversation")
        if voxsigil_enabled:
            print("VoxSigil enhancements: ENABLED")
        else:
            print("VoxSigil enhancements: DISABLED")
        print("="*50 + "\n")
        
        # Chat loop
        chat_history = []
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            # Apply VoxSigil processing if available
            if voxsigil_enabled:
                try:
                    # Process input through middleware
                    enhanced_input = middleware.process_text(user_input)
                    
                    # Apply RAG for context-aware responses
                    rag_results = rag_processor.process(user_input)
                    if rag_results and isinstance(rag_results, str):
                        context = f"Additional Context: {rag_results}\n"
                    else:
                        context = ""
                    
                    # Use supervisor to manage interaction
                    supervised_input = supervisor.process_input(user_input)
                    
                    # Combine all enhancements
                    processed_input = f"{enhanced_input}\n{context}{supervised_input}"
                    logger.info(f"Enhanced input with VoxSigil: {processed_input}")
                except Exception as e:
                    logger.warning(f"Error in VoxSigil processing: {e}")
                    processed_input = user_input
            else:
                processed_input = user_input
            
            # Prepare input for the model
            chat_history.append({"role": "user", "content": processed_input})
            input_ids = tokenizer.apply_chat_template(
                chat_history, 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to(model.device)
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            
            # Stream tokens for better UX
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                streamer=None,  # We'll manually stream tokens
            )
            
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(response)
            
            end_time = time.time()
            logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
            
            # Add response to chat history
            chat_history.append({"role": "assistant", "content": response})
        
        return True
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        return False


def check_system_info():
    """Check and report system information."""
    logger.info("Checking system information...")
    
    # OS Information
    os_info = platform.platform()
    logger.info(f"Operating System: {os_info}")
    
    # Python Information
    python_version = platform.python_version()
    logger.info(f"Python Version: {python_version}")
    
    # CUDA Information
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        logger.info(f"CUDA Version: {cuda_version}")
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        logger.warning("CUDA not available. Using CPU only.")
    
    # VoxSigil Information
    if VOXSIGIL_AVAILABLE:
        logger.info("VoxSigil components are available.")
        if BLT_AVAILABLE:
            logger.info("BLT-enhanced components are also available.")
        else:
            logger.info("BLT-enhanced components are not available.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TinyLlama Personal Assistant with VoxSigil enhancements"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "chat", "info"],
        default="info",
        help="Mode of operation: train, chat, or system info",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or path for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the training dataset",
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
        "--model_path",
        type=str,
        default="",
        help="Path to a trained model for chat mode",
    )
    parser.add_argument(
        "--use_voxsigil",
        action="store_true",
        default=True,
        help="Use VoxSigil enhancements if available",
    )
    parser.add_argument(
        "--disable_voxsigil",
        action="store_true",
        help="Disable VoxSigil integration even if available",
    )
    
    args = parser.parse_args()
    
    # Handle conflicting arguments
    if args.disable_voxsigil:
        args.use_voxsigil = False
    
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("TinyLlama Personal Assistant with VoxSigil Integration")
    print("="*80)
    
    # Check VoxSigil availability
    if VOXSIGIL_AVAILABLE and args.use_voxsigil:
        print("VoxSigil components: AVAILABLE and ENABLED")
    elif VOXSIGIL_AVAILABLE and not args.use_voxsigil:
        print("VoxSigil components: AVAILABLE but DISABLED by user")
    else:
        print("VoxSigil components: NOT AVAILABLE")
    
    # Execute requested mode
    success = False
    if args.mode == "info":
        success = check_system_info()
    elif args.mode == "train":
        success = train_model(args)
    elif args.mode == "chat":
        success = chat_with_model(args)
    
    # Print footer
    print("\n" + "="*80)
    if success:
        print(f"TinyLlama Personal Assistant - {args.mode.upper()} completed successfully")
    else:
        print(f"TinyLlama Personal Assistant - {args.mode.upper()} completed with errors")
    print("="*80 + "\n")