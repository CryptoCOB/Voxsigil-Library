#!/usr/bin/env python
"""
BLT Supervisor Integration

This module integrates the BLT-enhanced RAG system with TinyLlama and the VoxSigil Supervisor.
It provides a BLTSupervisorRagInterface that uses the BLTEnhancedRAG implementation
for enhanced retrieval and processing capabilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch  # Added for TinyLlama interface

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BLTSupervisorIntegration")

# Use standard path helper for imports
try:
    from utils.path_helper import add_project_root_to_path, get_project_root

    add_project_root_to_path()

    # Add additional paths if needed
    root_dir = get_project_root()
    if str(root_dir / "VoxSigilRag") not in sys.path:
        sys.path.append(str(root_dir / "VoxSigilRag"))
    if str(root_dir / "voxsigil_supervisor") not in sys.path:
        sys.path.append(str(root_dir / "voxsigil_supervisor"))
except ImportError:
    # Fallback to manual path setup if standard helper is not available
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    if str(root_dir / "VoxSigilRag") not in sys.path:
        sys.path.append(str(root_dir / "VoxSigilRag"))
    if str(root_dir / "voxsigil_supervisor") not in sys.path:
        sys.path.append(str(root_dir / "voxsigil_supervisor"))

# Add proper import for Vanta Core
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
except ImportError:
    logger.warning(
        "Could not import UnifiedVantaCore. Some functionality may be limited."
    )
    VantaCore = None

# Import VoxSigil components with comprehensive error handling
try:
    from BLT.voxsigil_blt_rag import BLTEnhancedRAG
    from BLT.hybrid_blt import (
        HybridMiddlewareConfig,
        EntropyRouter,
        ByteLatentTransformerEncoder,
    )

    # Define flag for component availability
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some BLT components couldn't be imported: {e}")
    COMPONENTS_AVAILABLE = False

    # Create stub classes for graceful degradation
    class BLTEnhancedRAG:
        """Stub HybridMiddlewareConfig class when actual implementation is not available."""

        def __init__(self, *args, **kwargs):
            self.entropy_threshold = kwargs.get("entropy_threshold", 0.25)
            self.blt_hybrid_weight = kwargs.get("blt_hybrid_weight", 0.7)
            self.entropy_router_fallback = kwargs.get(
                "entropy_router_fallback", "token_based"
            )
            self.cache_ttl_seconds = kwargs.get("cache_ttl_seconds", 300)
            self.log_level = kwargs.get("log_level", "INFO")


try:
    from BLT.hybrid_blt import EntropyRouter
except ImportError:

    class EntropyRouter:
        """Stub EntropyRouter class when actual implementation is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def route(self, text):
            return "token_based", None, [0.5]


try:
    from BLT.hybrid_blt import ByteLatentTransformerEncoder
except ImportError:

    class ByteLatentTransformerEncoder:
        """Stub ByteLatentTransformerEncoder class when actual implementation is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def create_patches(self, *args, **kwargs):
            return []


try:
    from Vanta.interfaces.rag_interface import BaseRagInterface
except ImportError:

    class BaseRagInterface:
        """Stub BaseRagInterface class when supervisor module is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def retrieve_sigils(self, query, top_k=5, filter_conditions=None):
            return []

        def create_rag_context(self, query, num_sigils=5):
            return "", []


# Function stubs with error handling
def entropy_router(entropy_score, threshold=0.3):
    """Wrapper function for entropy routing."""
    return entropy_score > threshold


def hybrid_embedding(*args, **kwargs):
    """Placeholder for hybrid embedding function."""
    return None


# Determine components availability
COMPONENTS_AVAILABLE = True
try:
    # Test basic imports to determine actual availability
    from BLT.voxsigil_blt_rag import BLTEnhancedRAG as _test_import

    logger.info("BLTEnhancedRAG is available")
except ImportError:
    COMPONENTS_AVAILABLE = False
    logger.warning("BLTEnhancedRAG not available, using stub implementation")


class BLTSupervisorRagInterface(BaseRagInterface):
    """
    Implementation of the Supervisor RAG interface using BLTEnhancedRAG.
    This enhances the standard VoxSigilRAG with Byte Latent Transformer capabilities
    for improved embedding, searching, and context optimization.
    """

    def __init__(
        self,
        voxsigil_library_path: Optional[Path] = None,
        embedding_model_name: Optional[str] = None,
        blt_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the BLT-enhanced RAG interface.

        Args:
            voxsigil_library_path: Path to the VoxSigil library.
            embedding_model_name: Name of the embedding model to use.
            blt_config: Configuration for BLT components.
        """
        super().__init__()

        if not COMPONENTS_AVAILABLE:
            logger.error(
                "Required components not available. BLTSupervisorRagInterface will not function."
            )
            self.rag_instance = None
            self._sigil_index_by_id = {}
            return

        blt_config = blt_config or {}

        # Load hybrid middleware configuration
        try:
            self.middleware_config = HybridMiddlewareConfig(
                entropy_threshold=blt_config.get("entropy_threshold", 0.25),
                blt_hybrid_weight=blt_config.get("blt_hybrid_weight", 0.7),
                entropy_router_fallback=blt_config.get(
                    "entropy_router_fallback", "token_based"
                ),
                cache_ttl_seconds=blt_config.get("cache_ttl_seconds", 300),
                log_level=blt_config.get("log_level", "INFO"),
            )
            logger.info(
                f"Initialized BLT middleware config with entropy threshold: {self.middleware_config.entropy_threshold}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize middleware config with custom settings: {e}. Using defaults."
            )
            self.middleware_config = HybridMiddlewareConfig()

        # Initialize the BLTEnhancedRAG with detailed error handling
        try:
            # Check if voxsigil_library_path exists
            if voxsigil_library_path and not voxsigil_library_path.exists():
                logger.warning(
                    f"VoxSigil library path {voxsigil_library_path} does not exist."
                )
                if not os.environ.get("VOXSIGIL_LIBRARY_PATH"):
                    logger.warning(
                        "VOXSIGIL_LIBRARY_PATH environment variable not set. Using default paths."
                    )

            # Validate embedding model name - fallback to default if needed
            embedding_model = embedding_model_name or "all-MiniLM-L6-v2"
            if not embedding_model_name:
                logger.info(
                    f"No embedding model specified, using default: {embedding_model}"
                )

            # Initialize the RAG instance with validated parameters
            self.rag_instance = BLTEnhancedRAG(
                voxsigil_library_path=voxsigil_library_path,
                cache_enabled=True,
                embedding_model=embedding_model,
                # BLT-specific parameters
                blt_hybrid_weight=self.middleware_config.blt_hybrid_weight,
                entropy_threshold=self.middleware_config.entropy_threshold,
                enable_patch_validation=True,
                enable_patch_compression=True,
            )
            logger.info("Initialized BLTSupervisorRagInterface with BLTEnhancedRAG")

            # Pre-load sigils to avoid first-query latency
            self._sigil_index_by_id = {}

            if hasattr(self.rag_instance, "load_all_sigils"):
                # Initialize the components separately - useful for testing
                self.initialize_components()

        except Exception as e:
            logger.error(f"Error initializing BLTEnhancedRAG: {e}")
            self.rag_instance = None
            self._sigil_index_by_id = {}

    def initialize_components(self):
        """
        Initialize and validate all BLT-related components.

        This method ensures all necessary components are properly initialized
        and ready for use in the supervisor.

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        if not self.rag_instance:
            logger.error("Cannot initialize components: RAG instance not available")
            return False

        try:
            # Test loading sigils
            if hasattr(self.rag_instance, "load_all_sigils"):
                loaded_sigils = self.rag_instance.load_all_sigils()
                if loaded_sigils:
                    logger.info(f"Successfully loaded {len(loaded_sigils)} sigils")
                    # Build index for faster lookups
                    for sigil in loaded_sigils:
                        sigil_id = sigil.get("sigil", "")
                        if sigil_id:
                            self._sigil_index_by_id[sigil_id] = sigil

            # Verify the BLT encoder is available
            if (
                hasattr(self.rag_instance, "blt_encoder")
                and self.rag_instance.blt_encoder
            ):
                # Test the encoder with a simple input
                test_text = "Testing BLT encoder initialization"
                try:
                    # Test create_patches method
                    if hasattr(self.rag_instance.blt_encoder, "create_patches"):
                        patches = self.rag_instance.blt_encoder.create_patches(
                            test_text
                        )
                        if patches:
                            logger.info(
                                f"BLT encoder successfully created {len(patches)} patches"
                            )
                except Exception as e:
                    logger.warning(f"BLT encoder patch creation test failed: {e}")
            else:
                logger.warning("BLT encoder not available in RAG instance")

            # Verify patch validator
            if (
                hasattr(self.rag_instance, "patch_validator")
                and self.rag_instance.patch_validator
            ):
                # Test the validator
                try:
                    if hasattr(self.rag_instance.patch_validator, "validate_structure"):
                        valid, _ = self.rag_instance.patch_validator.validate_structure(
                            "Test structure"
                        )
                        logger.debug(f"Patch validator test: {valid}")
                except Exception as e:
                    logger.warning(f"Patch validator test failed: {e}")

            logger.info("BLT components initialization complete")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize BLT components: {e}")
            return False

    def retrieve_sigils(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant sigils using BLTEnhancedRAG.

        Args:
            query: The query string.
            top_k: Maximum number of sigils to return.
            filter_conditions: Optional filtering conditions.

        Returns:
            List of sigil dictionaries with similarity scores.
        """
        if not self.rag_instance:
            logger.error("BLTEnhancedRAG instance not available")
            return []

        filter_conditions = filter_conditions or {}

        # Determine if we should use the entropy-aware routing
        query_entropy = self._calculate_query_entropy(query)
        use_enhanced_mode = query_entropy < self.middleware_config.entropy_threshold

        logger.info(
            f"Query entropy: {query_entropy:.4f}, using {'enhanced BLT' if use_enhanced_mode else 'standard'} mode"
        )

        # Convert filter_conditions to BLTEnhancedRAG parameters
        rag_params = {
            "num_sigils": top_k,
            "detail_level": filter_conditions.get("detail_level", "standard"),
            "use_blt_encoding": use_enhanced_mode,  # Use BLT for low-entropy queries
        }

        # Add tag filters if specified
        if "tags" in filter_conditions:
            rag_params["filter_tags"] = filter_conditions["tags"]
            rag_params["tag_operator"] = filter_conditions.get("tag_operator", "OR")

        # Add minimum similarity threshold if specified
        if "min_similarity" in filter_conditions:
            rag_params["min_score_threshold"] = filter_conditions["min_similarity"]

        try:
            # Instead of using the standard context creation, use the enhanced RAG process for BLT benefits
            if use_enhanced_mode and hasattr(self.rag_instance, "enhanced_rag_process"):
                _, retrieved_sigils = self.rag_instance.enhanced_rag_process(
                    query=query,
                    num_sigils=top_k,
                    augment_query=True,
                    apply_recency_boost=True,
                    enable_context_optimization=True,
                    auto_fuse_related=filter_conditions.get("auto_fuse_related", True),
                    max_fusion_sigils=filter_conditions.get("max_fusion_sigils", 3),
                )
                return retrieved_sigils
            else:
                # Fall back to standard method
                if hasattr(self.rag_instance, "create_rag_context"):
                    _, retrieved_sigils = self.rag_instance.create_rag_context(
                        query=query, **rag_params
                    )
                    return retrieved_sigils
                else:
                    logger.warning(
                        "Neither enhanced_rag_process nor create_rag_context available"
                    )
                    return []
        except Exception as e:
            logger.error(f"Error retrieving sigils: {e}")
            return []

    def retrieve_context(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Retrieves formatted context using BLTEnhancedRAG with entropy-aware processing.

        Args:
            query: The query string.
            params: Optional parameters.

        Returns:
            Formatted context string.
        """
        if not self.rag_instance:
            logger.error("BLTEnhancedRAG instance not available")
            return "ERROR: BLTEnhancedRAG not available for context retrieval."

        params = params or {}

        # Extract parameters relevant to BLTEnhancedRAG
        scaffold = params.get("scaffold")

        # If we have a scaffold, adjust the context retrieval
        if scaffold:
            # Add scaffold-specific tags for better context
            scaffold_tags = self._extract_scaffold_tags(scaffold)
            if scaffold_tags:
                params.setdefault("filter_tags", []).extend(scaffold_tags)

        # Determine whether to use enhanced mode based on entropy
        query_entropy = self._calculate_query_entropy(query)
        use_enhanced_mode = query_entropy < self.middleware_config.entropy_threshold

        logger.info(
            f"Context retrieval - Query entropy: {query_entropy:.4f}, using {'enhanced BLT' if use_enhanced_mode else 'standard'} mode"
        )

        # Convert params to BLTEnhancedRAG parameters
        rag_params = {
            "num_sigils": params.get("top_k", 5),
            "detail_level": params.get("detail_level", "standard"),
            "use_blt_encoding": use_enhanced_mode,
        }

        # Add tag filters if specified
        if "filter_tags" in params:
            rag_params["filter_tags"] = params["filter_tags"]
            rag_params["tag_operator"] = params.get("tag_operator", "OR")

        try:
            # Use the enhanced process if available and appropriate
            if use_enhanced_mode and hasattr(self.rag_instance, "enhanced_rag_process"):
                context_text, _ = self.rag_instance.enhanced_rag_process(
                    query=query,
                    num_sigils=rag_params["num_sigils"],
                    augment_query=True,
                    apply_recency_boost=True,
                    enable_context_optimization=True,
                    auto_fuse_related=params.get("auto_fuse_related", True),
                    max_fusion_sigils=params.get("max_fusion_sigils", 3),
                )
                return context_text
            else:
                # Fall back to standard method
                if hasattr(self.rag_instance, "create_rag_context"):
                    context_text, _ = self.rag_instance.create_rag_context(
                        query=query, **rag_params
                    )
                    return context_text
                else:
                    logger.warning(
                        "Neither enhanced_rag_process nor create_rag_context available"
                    )
                    return "ERROR: No context retrieval method available"
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"ERROR: Failed to retrieve context: {e}"

    def _extract_scaffold_tags(self, scaffold_name: str) -> List[str]:
        """
        Extract relevant tags for a given scaffold to enhance context retrieval.

        Args:
            scaffold_name: The name of the scaffold.

        Returns:
            List of relevant tags.
        """
        # Map scaffold names to useful tags for context retrieval
        scaffold_tag_map = {
            "TRIALOGOS": ["problem_solving", "reasoning", "dialogue", "debate"],
            "C_STRUCTURE": ["algorithm", "code", "programming", "structure"],
            "SMART_MRAP": ["goal_oriented", "planning", "strategies"],
            "BLT_HYBRID": ["entropy", "byte_level", "hybrid", "adaptive"],
            "ENTROPY_ROUTER": ["adaptive", "routing", "optimization"],
        }
        return scaffold_tag_map.get(scaffold_name, [])

    def _calculate_query_entropy(self, query: str) -> float:
        """
        Calculate the entropy of a query string.
        Used to determine whether to use BLT-enhanced processing.

        Args:
            query: The query string.

        Returns:
            Entropy value.
        """
        if not query:
            return 0.0

        try:
            # Calculate Shannon entropy
            text_bytes = query.encode("utf-8")
            frequencies = {}
            for byte in text_bytes:
                if byte not in frequencies:
                    frequencies[byte] = 0
                frequencies[byte] += 1

            entropy = 0
            for count in frequencies.values():
                probability = count / len(text_bytes)
                entropy -= probability * np.log2(probability)

            return entropy
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return 1.0  # Default to high entropy on error

    def create_supervisor(self):
        """
        Create a VoxSigil Supervisor with BLT-enhanced RAG and TinyLlama.

        Returns:
            An instance of VoxSigilSupervisor if successful, None otherwise.
        """
        if not COMPONENTS_AVAILABLE:
            logger.error("Cannot create supervisor: Required components not available.")
            return None

        try:
            # Import supervisor components with fallback handling
            try:
                from voxsigil_supervisor.supervisor_engine import SupervisorEngine
            except ImportError:
                logger.error("SupervisorEngine not available for import")
                return None

            # Create a new supervisor engine
            supervisor = SupervisorEngine(
                rag_interface=self,
                config={
                    "blt_enhanced": True,
                    "entropy_threshold": self.middleware_config.entropy_threshold
                    if hasattr(self, "middleware_config")
                    else 0.25,
                    "hybrid_weight": self.middleware_config.blt_hybrid_weight
                    if hasattr(self, "middleware_config")
                    else 0.7,
                },
            )

            logger.info("Successfully created BLT-enhanced supervisor")
            return supervisor

        except Exception as e:
            logger.error(f"Failed to create supervisor: {e}")
            return None


class TinyLlamaIntegration:
    """
    Integrates TinyLlama with the BLT-enhanced Supervisor.
    This class initializes all necessary components and provides methods
    for creating integrated applications.
    """

    def __init__(
        self,
        voxsigil_library_path: Optional[Path] = None,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        blt_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TinyLlama integration.

        Args:
            voxsigil_library_path: Path to the VoxSigil library.
            model_name: Name of the TinyLlama model.
            blt_config: Configuration for BLT components.
        """
        if not COMPONENTS_AVAILABLE:
            logger.error(
                "Required components not available. TinyLlamaIntegration will not function."
            )
            return

        self.model_name = model_name
        self.blt_config = blt_config or {}
        self.middleware_config = HybridMiddlewareConfig(**self.blt_config)

        # Initialize BLT-enhanced RAG interface
        logger.info("Initializing BLT-enhanced RAG interface...")
        self.rag_interface = BLTSupervisorRagInterface(
            voxsigil_library_path=voxsigil_library_path,
            embedding_model_name="all-MiniLM-L6-v2",  # Default model
            blt_config=self.blt_config,
        )
        # Initialize TinyLlama (needs to be implemented based on how you use TinyLlama)
        logger.info(f"Initializing TinyLlama model: {model_name}")
        self.tinyllama_initialized = False
        try:
            # Import TinyLlama-specific components with fallback
            try:
                from tinyllama_assistant import initialize_tinyllama_model

                self.llm = initialize_tinyllama_model(model_name)
                self.tinyllama_initialized = True
                logger.info("TinyLlama model initialized successfully")
            except ImportError:
                logger.warning("TinyLlama assistant module not available")
                self.llm = None
        except Exception as e:
            logger.error(f"Error initializing TinyLlama model: {e}")
            self.llm = None

        logger.info("TinyLlama integration setup complete")

    def create_supervisor(self):
        """
        Create a VoxSigil Supervisor with BLT-enhanced RAG and TinyLlama.

        Returns:
            An instance of VoxSigilSupervisor if successful, None otherwise.
        """
        if not COMPONENTS_AVAILABLE or not self.tinyllama_initialized:
            logger.error("Cannot create supervisor: Required components not available.")
            return None

        try:
            # Import supervisor components with comprehensive fallback handling
            try:
                from voxsigil_supervisor.supervisor_engine import VoxSigilSupervisor
                from Scaffolds.scaffold_router import ScaffoldRouter
                from voxsigil_supervisor.evaluation_heuristics import ResponseEvaluator
                from voxsigil_supervisor.retry_policy import RetryPolicy
            except ImportError as e:
                logger.error(f"Failed to import supervisor components: {e}")
                return None

            # Create a TinyLlama LLM interface (needs to be implemented)
            llm_interface = self._create_tinyllama_interface()

            # Create scaffold router with BLT awareness
            scaffold_router = ScaffoldRouter()

            # Create other required components
            evaluation_heuristics = ResponseEvaluator()
            retry_policy = RetryPolicy()

            # Create the supervisor
            supervisor = VoxSigilSupervisor(
                rag_interface=self.rag_interface,
                llm_interface=llm_interface,
                scaffold_router=scaffold_router,
                evaluation_heuristics=evaluation_heuristics,
                retry_policy=retry_policy,
                default_system_prompt="You are an AI assistant using TinyLlama with BLT-enhanced VoxSigil.",
                max_iterations=3,
            )

            logger.info("VoxSigil Supervisor created with BLT-enhanced components")
            return supervisor

        except Exception as e:
            logger.error(f"Error creating supervisor: {e}")
            return None    def _create_tinyllama_interface(self):
        """
        Create a TinyLlama LLM interface for the supervisor.

        Returns:
            An implementation of BaseLlmInterface for TinyLlama.
        """
        try:
            from ARC.llm.llm_interface import BaseLlmInterface
        except ImportError:
            # Create a stub if not available
            class BaseLlmInterface:
                def generate_response(
                    self, messages, system_prompt_override=None, task_requirements=None
                ):
                    return "Stub response", {}, {}

        class TinyLlamaInterface(BaseLlmInterface):
            def __init__(self, model_tuple):
                # Unpack model and tokenizer from the tuple
                if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
                    self.model, self.tokenizer = model_tuple
                    self.model_name = getattr(self.model, "name", "TinyLlama")
                else:
                    logger.warning("Invalid model tuple provided to TinyLlamaInterface")
                    self.model, self.tokenizer = None, None
                    self.model_name = "TinyLlama"
                
                logger.info(f"TinyLlamaInterface initialized with model: {self.model_name}")

            def generate_response(
                self, messages, system_prompt_override=None, task_requirements=None
            ):
                """
                Generate a response using TinyLlama model based on input messages.
                
                Args:
                    messages: List of message objects with 'role' and 'content'
                    system_prompt_override: Optional system prompt to override default
                    task_requirements: Optional requirements for the task
                
                Returns:
                    Tuple of (response_text, metadata, debug_info)
                """
                try:
                    if not self.model or not self.tokenizer:
                        return "TinyLlama model or tokenizer not available", {}, {}
                    
                    # Format messages into prompt string
                    prompt = self._format_messages(messages, system_prompt_override)
                    
                    # Set generation parameters
                    max_new_tokens = task_requirements.get("max_tokens", 512) if task_requirements else 512
                    temperature = task_requirements.get("temperature", 0.7) if task_requirements else 0.7
                    top_p = task_requirements.get("top_p", 0.9) if task_requirements else 0.9
                    
                    # Tokenize the prompt
                    input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        output = self.model.generate(
                            **input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    # Decode the response
                    response_ids = output[0][input_ids["input_ids"].shape[1]:]
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    # Return the response with metadata
                    metadata = {
                        "model": self.model_name,
                        "tokens_generated": len(response_ids),
                        "generation_parameters": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens
                        }
                    }
                    
                    return response_text, metadata, {}
                except Exception as e:
                    logger.error(f"Error generating response with TinyLlama: {e}")
                    return f"Error generating response: {str(e)}", {}, {"error": str(e)}

            def _format_messages(self, messages, system_prompt=None):
                """
                Format messages for TinyLlama in the expected chat format.
                
                Args:
                    messages: List of message objects with 'role' and 'content'
                    system_prompt: Optional system prompt to override
                
                Returns:
                    Formatted prompt string for TinyLlama
                """
                if not messages:
                    return ""
                
                formatted_messages = []
                
                # Add system prompt if provided, otherwise use the first system message if available
                if system_prompt:
                    formatted_messages.append(f"<|system|>\n{system_prompt}</s>")
                else:
                    # Look for system message in the provided messages
                    system_messages = [msg for msg in messages if msg.get('role') == 'system']
                    if system_messages:
                        formatted_messages.append(f"<|system|>\n{system_messages[0]['content']}</s>")
                
                # Process all messages (except system if we already handled it)
                for message in messages:
                    role = message.get('role', '').lower()
                    content = message.get('content', '')
                    
                    # Skip system messages if we already added a system prompt
                    if role == 'system' and formatted_messages and formatted_messages[0].startswith('<|system|>'):
                        continue
                    
                    # Map roles to TinyLlama format
                    if role == 'user':
                        formatted_messages.append(f"<|user|>\n{content}</s>")
                    elif role == 'assistant':
                        formatted_messages.append(f"<|assistant|>\n{content}</s>")
                    elif role == 'system':
                        formatted_messages.append(f"<|system|>\n{content}</s>")
                
                # Add final assistant prompt
                formatted_messages.append("<|assistant|>")
                
                # Join all formatted messages
                return "\n".join(formatted_messages)

        return TinyLlamaInterface(self.llm)

    def validate_integration(self) -> Dict[str, bool]:
        """
        Validate the BLT-TinyLlama integration by checking all critical components.

        Returns:
            Dictionary with validation results for each component
        """
        validation_results = {
            "tinyllama_model": False,
            "blt_encoder": False,
            "patch_encoder": False,
            "entropy_router": False,
            "rag_instance": False,
            "create_patches": False,
            "calculate_similarity": False,
        }

        # Check TinyLlama
        if hasattr(self, "llm") and self.llm is not None:
            validation_results["tinyllama_model"] = True

        # Verify RAG components
        if hasattr(self, "rag_interface") and self.rag_interface is not None:
            validation_results["rag_instance"] = True

            # Check if rag_interface has blt components
            rag = self.rag_interface.rag_instance
            if rag is not None:
                # Check blt_encoder
                if hasattr(rag, "blt_encoder") and rag.blt_encoder is not None:
                    validation_results["blt_encoder"] = True

                    # Test create_patches method
                    try:
                        test_text = "This is a test text for BLT encoder validation."
                        if hasattr(rag.blt_encoder, "create_patches"):
                            patches = rag.blt_encoder.create_patches(test_text)
                            if patches is not None:
                                validation_results["create_patches"] = True
                                logger.info(
                                    f"create_patches test successful: {len(patches)} patches created"
                                )
                    except Exception as e:
                        logger.error(f"create_patches test failed: {e}")

                    # Test calculate_similarity method
                    try:
                        if hasattr(rag.blt_encoder, "calculate_similarity"):
                            # Create dummy embeddings
                            emb1 = np.random.rand(128)
                            emb2 = np.random.rand(128)
                            sim = rag.blt_encoder.calculate_similarity(emb1, emb2)
                            if 0 <= sim <= 1:
                                validation_results["calculate_similarity"] = True
                                logger.info(
                                    f"calculate_similarity test successful: {sim}"
                                )
                    except Exception as e:
                        logger.error(f"calculate_similarity test failed: {e}")

                # Check patch_encoder
                if hasattr(rag, "patch_encoder") and rag.patch_encoder is not None:
                    validation_results["patch_encoder"] = True

        # Check if entropy_router is available
        if hasattr(self, "middleware_config") and self.middleware_config is not None:
            validation_results["entropy_router"] = True
            if hasattr(self.middleware_config, "entropy_router_fallback"):
                logger.info(
                    f"Entropy router fallback: {self.middleware_config.entropy_router_fallback}"
                )

        return validation_results


# Example usage function
def create_blt_tinyllama_integration(
    voxsigil_library_path: Optional[Path] = None,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    blt_config: Optional[Dict[str, Any]] = None,
) -> TinyLlamaIntegration:
    """
    Convenience function to create a complete BLT-TinyLlama integration.

    Args:
        voxsigil_library_path: Path to the VoxSigil library.
        model_name: Name of the TinyLlama model.
        blt_config: Configuration for BLT components.

    Returns:
        Configured TinyLlamaIntegration instance.
    """
    default_blt_config = {
        "entropy_threshold": 0.25,
        "blt_hybrid_weight": 0.7,
        "entropy_router_fallback": "token_based",
        "cache_ttl_seconds": 300,
        "log_level": "INFO",
    }

    if blt_config:
        default_blt_config.update(blt_config)

    return TinyLlamaIntegration(
        voxsigil_library_path=voxsigil_library_path,
        model_name=model_name,
        blt_config=default_blt_config,
    )
