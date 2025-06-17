#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VoxSigil BLT Middleware Loader

This script loads the BLT middleware for use with the fine-tuned TinyLlama model,
creating a bridge between the chat interface and the BLT components.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("blt_middleware.log"), logging.StreamHandler()],
)
logger = logging.getLogger("blt_middleware_loader")

# Ensure all necessary paths are in the system path
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
if str(current_dir / "VoxSigilRag") not in sys.path:
    sys.path.append(str(current_dir / "VoxSigilRag"))
if str(current_dir / "voxsigil_supervisor") not in sys.path:
    sys.path.append(str(current_dir / "voxsigil_supervisor"))

# Import necessary components with error handling
try:
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    from .voxsigil_blt_rag import (
        BLTEnhancedRAG,  # Import entropy_router conditionally for routing decisions
    )

    try:
        from VoxSigilRag.hybrid_blt import (
            entropy_router_util,  # noqa: F401 - Import used for availability check
        )

        ENTROPY_ROUTER_AVAILABLE = True

        # Create a wrapper function that matches the expected interface
        def entropy_router(entropy_score, threshold=0.3):
            """Wrapper for entropy_router_util to match expected interface"""
            return entropy_score > threshold
    except ImportError:
        ENTROPY_ROUTER_AVAILABLE = False

        # Define a simple fallback entropy router
        def entropy_router(entropy_score, threshold=0.3):
            return entropy_score > threshold

    BLT_AVAILABLE = True
    logger.info("Successfully imported VoxSigil BLT components")

except ImportError as e:
    BLT_AVAILABLE = False
    ENTROPY_ROUTER_AVAILABLE = False

    # Fallback entropy router
    def entropy_router(entropy_score, threshold=0.3):
        return entropy_score > threshold

    logger.error(f"Failed to import VoxSigil BLT components: {e}")
    logger.warning("BLT middleware will not be available")


class VoxSigilBLTMiddleware:
    """
    A class to handle BLT middleware integration with fine-tuned TinyLlama models.
    Provides methods to process text through BLT enhancement before model inference.
    """

    def __init__(self, model_path, voxsigil_library_path, blt_config=None):
        """
        Initialize the BLT middleware with a fine-tuned model and VoxSigil library.

        Args:
            model_path: Path to the fine-tuned TinyLlama model
            voxsigil_library_path: Path to the VoxSigil library
            blt_config: Configuration settings for BLT middleware
        """
        self.model_path = Path(model_path)
        self.voxsigil_library_path = Path(voxsigil_library_path)
        self.blt_config = blt_config or {}

        self.tokenizer = None
        self.model = None
        self.rag = None
        self.supervisor = None
        self.blt_encoder = None
        self.rag_processor = None

        if not BLT_AVAILABLE:
            logger.warning("BLT components are not available. Middleware will be disabled.")
            return

        self._initialize_middleware()

    def _initialize_middleware(self):
        """Initialize all BLT middleware components"""
        try:
            # Load configuration
            entropy_threshold = self.blt_config.get("entropy_threshold", 0.3)
            blt_hybrid_weight = self.blt_config.get("blt_hybrid_weight", 0.7)

            logger.info(
                f"Initializing BLT middleware with entropy_threshold={entropy_threshold}, "
                f"blt_hybrid_weight={blt_hybrid_weight}"
            )

            # Initialize tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer initialized successfully")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                return

            # Load the fine-tuned model
            try:
                # Choose appropriate precision
                dtype = None
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        if torch.cuda.get_device_capability(i)[0] >= 8:  # Ampere or newer (BF16)
                            dtype = torch.bfloat16
                            logger.info(f"Using BF16 precision (GPU {i})")
                            break

                    if dtype is None:
                        dtype = torch.float16
                        logger.info("Using FP16 precision")
                else:
                    dtype = torch.float32
                    logger.info("Using FP32 precision")

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                logger.info(
                    f"Model loaded successfully with {self.model.num_parameters():,} parameters"
                )
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return
                # Initialize BLTEnhancedRAG if VoxSigil library exists
            if self.voxsigil_library_path.exists():
                try:
                    self.rag = BLTEnhancedRAG(
                        voxsigil_library_path=self.voxsigil_library_path,
                        cache_enabled=True,
                        blt_hybrid_weight=blt_hybrid_weight,
                        entropy_threshold=entropy_threshold,
                        enable_patch_validation=True,
                        enable_patch_compression=True,
                    )
                    logger.info("BLT Enhanced RAG initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing BLT RAG: {e}")
            else:
                # Create the directory if it doesn't exist
                os.makedirs(self.voxsigil_library_path, exist_ok=True)
                logger.warning(f"VoxSigil library not found at {self.voxsigil_library_path}")
                logger.info("BLT middleware will operate in standalone mode without RAG features")

        except Exception as e:
            logger.error(f"Error initializing BLT middleware: {e}")

    def is_available(self):
        """Check if BLT middleware is available and properly initialized"""
        # Even without RAG, we can still use BLT middleware features
        # as long as the model and tokenizer are available
        return BLT_AVAILABLE and self.model is not None and self.tokenizer is not None

    def generate_with_blt(
        self,
        prompt,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    ):
        """
        Generate text with BLT enhancement

        Args:
            prompt: Input prompt to process
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty parameter

        Returns:
            Tuple of (generated_text, method_used)"""
        if not self.is_available():
            return "BLT middleware is not available", "Error"

        try:  # First, attempt to use RAG if available
            if self.rag and self.rag.patch_encoder:
                try:
                    # Use entropy router to determine if BLT enhancement should be used
                    entropy_score = self.rag.patch_encoder.compute_average_entropy(prompt)
                    should_use_blt = entropy_router(
                        entropy_score,
                        threshold=self.blt_config.get("entropy_threshold", 0.3),
                    )

                    if should_use_blt:
                        # Use BLT-enhanced RAG
                        results = self.rag.query(prompt, top_k=3)

                        if results and self.tokenizer and self.model:
                            # Extract context from results
                            context = ""
                            for idx, result in enumerate(results):
                                if idx > 0:
                                    context += "\n\n"
                                context += f"{result.get('content', '')}"

                            # Use context to enhance prompt
                            enhanced_prompt = (
                                f"Context information:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
                            )

                            # Generate with enhanced prompt
                            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt")
                            device = next(self.model.parameters()).device
                            inputs = {k: v.to(device) for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=max_length,
                                    do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repetition_penalty=repetition_penalty,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                )

                            generated_text = self.tokenizer.decode(
                                outputs[0], skip_special_tokens=True
                            )
                            # Remove the context and question part
                            response = generated_text[len(enhanced_prompt) :].strip()
                            return (
                                response,
                                "BLT-Enhanced RAG",
                            )  # Fall through to standard generation if BLT is not needed or failed
                    logger.info(f"Falling back to standard generation (entropy={entropy_score})")

                except Exception as e:
                    logger.error(f"Error using BLT-Enhanced RAG: {e}")
                    # Fall through to standard generation

            # Standard generation
            if not self.tokenizer or not self.model:
                return "Model or tokenizer not initialized", "Error"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the response (without the prompt)
            response = generated_text[len(prompt) :].strip()
            return response, "Standard Model"

        except Exception as e:
            logger.error(f"Error generating with BLT: {e}")
            return f"Error generating response: {e}", "Error"

    def create_rag_context(self, query: str, num_results: int = 5) -> str:
        """
        Create a RAG context for the query.
        """
        if not self.is_available():
            return ""

        try:  # Get RAG context
            logger.info(f"Creating RAG context for query: {query[:50]}...")
            if self.rag_processor:
                result = self.rag_processor.create_rag_context(
                    query=query, num_sigils=num_results
                )  # Handle different possible return types
                if result is None:
                    return ""
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Safely unpack the tuple
                    context = result[0]
                    sigils = result[1]
                    if hasattr(sigils, "__len__"):  # Check if it has length (list, tuple, etc.)
                        logger.info(f"Created RAG context with {len(sigils)} sigils")
                    return str(context) if context else ""
                elif isinstance(result, str):
                    return result
                else:
                    logger.warning("Unexpected return type from rag_processor.create_rag_context")
                    return ""
            return ""
        except Exception as e:
            logger.error(f"Error creating RAG context: {e}")
            return ""
