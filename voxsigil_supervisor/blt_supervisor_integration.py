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

from voxsigil_supervisor.interfaces.llm_interface import BaseLlmInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BLTSupervisorIntegration")

# Ensure all necessary paths are in the system path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(root_dir / "VoxSigilRag") not in sys.path:
    sys.path.append(str(root_dir / "VoxSigilRag"))
if str(root_dir / "voxsigil_supervisor") not in sys.path:
    sys.path.append(str(root_dir / "voxsigil_supervisor"))

# Import VoxSigil components with error handling
try:
    from VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG
    from VoxSigilRag.hybrid_blt import (
        HybridMiddlewareConfig, 
        entropy_router, 
        hybrid_embedding,
        EntropyRouter,
        ByteLatentTransformerEncoder
    )
    from voxsigil_supervisor.interfaces.rag_interface import BaseRagInterface
    
    COMPONENTS_AVAILABLE = True
    logger.info("Successfully imported VoxSigil components for BLT Supervisor integration")
    
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logger.error(f"Failed to import required components: {e}")
    
    # Create stub classes to avoid breaking imports
    class BLTEnhancedRAG:
        def __init__(self, *args, **kwargs): pass    # Import unified interface from Vanta
    from Vanta.interfaces.base_interfaces import BaseRagInterface
    
    # All stub classes removed - use proper imports or handle ImportError
    HybridMiddlewareConfig = None
    ByteLatentTransformerEncoder = None
    EntropyRouter = None
    
    # Stub functions removed - use real implementations
    def entropy_router(text, config=None):
        return "token_based", 0.5
        
    def hybrid_embedding(text, token_embedding, patch_embedding, config=None):
        return token_embedding


class BLTSupervisorRagInterface(BaseRagInterface):
    """
    Implementation of the Supervisor RAG interface using BLTEnhancedRAG.
    This enhances the standard VoxSigilRAG with Byte Latent Transformer capabilities
    for improved embedding, searching, and context optimization.
    """
    
    def __init__(self, 
                 voxsigil_library_path: Optional[Path] = None, 
                 embedding_model_name: Optional[str] = None,
                 blt_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BLT-enhanced RAG interface.
        
        Args:
            voxsigil_library_path: Path to the VoxSigil library.
            embedding_model_name: Name of the embedding model to use.
            blt_config: Configuration for BLT components.
        """
        if not COMPONENTS_AVAILABLE:
            logger.error("Required components not available. BLTSupervisorRagInterface will not function.")
            self.rag_instance = None
            self._sigil_index_by_id = {}
            return
        
        blt_config = blt_config or {}
        
        # Load hybrid middleware configuration
        try:
            self.middleware_config = HybridMiddlewareConfig(
                entropy_threshold=blt_config.get('entropy_threshold', 0.25),
                blt_hybrid_weight=blt_config.get('blt_hybrid_weight', 0.7),
                entropy_router_fallback=blt_config.get('entropy_router_fallback', 'token_based'),
                cache_ttl_seconds=blt_config.get('cache_ttl_seconds', 300),
                log_level=blt_config.get('log_level', 'INFO')
            )
            logger.info(f"Initialized BLT middleware config with entropy threshold: {self.middleware_config.entropy_threshold}")
        except Exception as e:
            logger.warning(f"Failed to initialize middleware config with custom settings: {e}. Using defaults.")
            self.middleware_config = HybridMiddlewareConfig()
        
        # Initialize the BLTEnhancedRAG with detailed error handling
        try:
            # Check if voxsigil_library_path exists
            if voxsigil_library_path and not voxsigil_library_path.exists():
                logger.warning(f"VoxSigil library path {voxsigil_library_path} does not exist.")
                if not os.environ.get("VOXSIGIL_LIBRARY_PATH"):
                    logger.warning("VOXSIGIL_LIBRARY_PATH environment variable not set. Using default paths.")
            
            # Validate embedding model name - fallback to default if needed
            embedding_model = embedding_model_name or "all-MiniLM-L6-v2"
            if not embedding_model_name:
                logger.info(f"No embedding model specified, using default: {embedding_model}")
            
            # Initialize the RAG instance with validated parameters
            self.rag_instance = BLTEnhancedRAG(
                voxsigil_library_path=voxsigil_library_path,
                cache_enabled=True,
                embedding_model=embedding_model,
                # BLT-specific parameters
                blt_hybrid_weight=self.middleware_config.blt_hybrid_weight,
                entropy_threshold=self.middleware_config.entropy_threshold,
                enable_patch_validation=True,
                enable_patch_compression=True
            )
            logger.info(f"Initialized BLTSupervisorRagInterface with BLTEnhancedRAG")
            
            # Pre-load sigils to avoid first-query latency
            self._sigil_index_by_id = {}
            loaded_sigils = None
            
            if hasattr(self.rag_instance, 'load_all_sigils'):
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
            loaded_sigils = self.rag_instance.load_all_sigils()
            if loaded_sigils:
                logger.info(f"Successfully loaded {len(loaded_sigils)} sigils")
                # Build index for faster lookups
                for sigil in loaded_sigils:
                    sigil_id = sigil.get('sigil', '')
                    if sigil_id:
                        self._sigil_index_by_id[sigil_id] = sigil
            
            # Verify the BLT encoder is available
            if hasattr(self.rag_instance, 'blt_encoder'):
                # Test the encoder with a simple input
                test_text = "Testing BLT encoder initialization"
                try:
                    # Test create_patches method
                    patches = self.rag_instance.blt_encoder.create_patches(test_text)
                    if patches:
                        logger.info(f"BLT encoder successfully created {len(patches)} patches")
                except Exception as e:
                    logger.warning(f"BLT encoder patch creation test failed: {e}")
            else:
                logger.warning("BLT encoder not available in RAG instance")
            
            # Verify patch validator
            if hasattr(self.rag_instance, 'patch_validator'):
                # Test the validator
                try:
                    valid, _ = self.rag_instance.patch_validator.validate_structure("Test structure")
                    logger.debug(f"Patch validator test: {valid}")
                except Exception as e:
                    logger.warning(f"Patch validator test failed: {e}")
            
            logger.info("BLT components initialization complete")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize BLT components: {e}")
            return False
    
    def retrieve_sigils(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
        
        logger.info(f"Query entropy: {query_entropy:.4f}, using {'enhanced BLT' if use_enhanced_mode else 'standard'} mode")
        
        # Convert filter_conditions to BLTEnhancedRAG parameters
        rag_params = {
            'num_sigils': top_k,
            'detail_level': filter_conditions.get('detail_level', 'standard'),
            'use_blt_encoding': use_enhanced_mode  # Use BLT for low-entropy queries
        }
        
        # Add tag filters if specified
        if 'tags' in filter_conditions:
            rag_params['filter_tags'] = filter_conditions['tags']
            rag_params['tag_operator'] = filter_conditions.get('tag_operator', 'OR')
            
        # Add minimum similarity threshold if specified
        if 'min_similarity' in filter_conditions:
            rag_params['min_score_threshold'] = filter_conditions['min_similarity']
        
        try:
            # Instead of using the standard context creation, use the enhanced RAG process for BLT benefits
            if use_enhanced_mode and hasattr(self.rag_instance, 'enhanced_rag_process'):
                _, retrieved_sigils = self.rag_instance.enhanced_rag_process(
                    query=query,
                    num_sigils=top_k,
                    augment_query=True,
                    apply_recency_boost=True,
                    enable_context_optimization=True,
                    auto_fuse_related=filter_conditions.get('auto_fuse_related', True),
                    max_fusion_sigils=filter_conditions.get('max_fusion_sigils', 3)
                )
                return retrieved_sigils
            else:
                # Fall back to standard method
                _, retrieved_sigils = self.rag_instance.create_rag_context(query=query, **rag_params)
                return retrieved_sigils
        except Exception as e:
            logger.error(f"Error retrieving sigils: {e}")
            return []
    
    def retrieve_context(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
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
        scaffold = params.get('scaffold')
        
        # If we have a scaffold, adjust the context retrieval
        if scaffold:
            # Add scaffold-specific tags for better context
            scaffold_tags = self._extract_scaffold_tags(scaffold)
            if scaffold_tags:
                params.setdefault('filter_tags', []).extend(scaffold_tags)
        
        # Determine whether to use enhanced mode based on entropy
        query_entropy = self._calculate_query_entropy(query)
        use_enhanced_mode = query_entropy < self.middleware_config.entropy_threshold
        
        logger.info(f"Context retrieval - Query entropy: {query_entropy:.4f}, using {'enhanced BLT' if use_enhanced_mode else 'standard'} mode")
        
        # Convert params to BLTEnhancedRAG parameters
        rag_params = {
            'num_sigils': params.get('top_k', 5),
            'detail_level': params.get('detail_level', 'standard'),
            'use_blt_encoding': use_enhanced_mode
        }
        
        # Add tag filters if specified
        if 'filter_tags' in params:
            rag_params['filter_tags'] = params['filter_tags']
            rag_params['tag_operator'] = params.get('tag_operator', 'OR')
            
        try:
            # Use the enhanced process if available and appropriate
            if use_enhanced_mode and hasattr(self.rag_instance, 'enhanced_rag_process'):
                context_text, _ = self.rag_instance.enhanced_rag_process(
                    query=query,
                    num_sigils=rag_params['num_sigils'],
                    augment_query=True,
                    apply_recency_boost=True,
                    enable_context_optimization=True,
                    auto_fuse_related=params.get('auto_fuse_related', True),
                    max_fusion_sigils=params.get('max_fusion_sigils', 3)
                )
                return context_text
            else:
                # Fall back to standard method
                context_text, _ = self.rag_instance.create_rag_context(query=query, **rag_params)
                return context_text
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"ERROR: Failed to retrieve context: {e}"
    
    def retrieve_scaffolds(self, query: str, filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves reasoning scaffolds relevant to the query using BLT-enhanced retrieval.
        
        Args:
            query: The query string.
            filter_tags: Optional list of tags to filter scaffolds.
            
        Returns:
            List of scaffold dictionaries.
        """
        if not self.rag_instance:
            logger.error("BLTEnhancedRAG instance not available")
            return []
            
        # Scaffolds are stored as sigils with the 'scaffold' tag
        scaffold_tags = ['scaffold']
        if filter_tags:
            scaffold_tags.extend(filter_tags)
            
        try:
            # Use the existing sigil retrieval with scaffold tags
            return self.retrieve_sigils(
                query=query,
                top_k=5,
                filter_conditions={
                    'tags': scaffold_tags,
                    'detail_level': 'full',  # We want the full scaffold definition
                    'auto_fuse_related': True  # Include related scaffolds
                }
            )
        except Exception as e:
            logger.error(f"Error retrieving scaffolds: {e}")
            return []
    
    def get_scaffold_definition(self, scaffold_name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full definition of a specific reasoning scaffold.
        
        Args:
            scaffold_name_or_id: The unique name or sigil glyph of the scaffold.
            
        Returns:
            A dictionary containing the scaffold's definition, or None if not found.
        """
        if not self.rag_instance or not hasattr(self, '_sigil_index_by_id'):
            logger.error("BLTEnhancedRAG instance or sigil index not available")
            return None
            
        # Check if we have this scaffold in our index
        scaffold_data = self._sigil_index_by_id.get(scaffold_name_or_id)
        
        if scaffold_data:
            # Verify if it's indeed a scaffold by checking its tags
            tags = scaffold_data.get("tags", [])
            if isinstance(tags, str): 
                tags = [tags]  # Ensure list
            
            # Check for scaffold-related tags
            scaffold_related_tags = ["scaffold", "reasoning_scaffold", "orchestration_pattern"]
            if any(tag in tags for tag in scaffold_related_tags):
                logger.info(f"Found scaffold definition for '{scaffold_name_or_id}'")
                return scaffold_data
            else:
                logger.warning(f"Construct '{scaffold_name_or_id}' found but not tagged as a scaffold")
                # Return it anyway, but with a warning
                return scaffold_data
        
        logger.warning(f"Scaffold '{scaffold_name_or_id}' not found in sigil index")
        return None
    
    def get_sigil_by_id(self, sigil_id_glyph: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific sigil by its unique ID or glyph.
        
        Args:
            sigil_id_glyph: The unique ID or glyph of the sigil.
            
        Returns:
            The sigil dictionary, or None if not found.
        """
        if not self.rag_instance or not hasattr(self, '_sigil_index_by_id'):
            logger.error("BLTEnhancedRAG instance or sigil index not available")
            return None
            
        return self._sigil_index_by_id.get(sigil_id_glyph)
    
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
            "ENTROPY_ROUTER": ["adaptive", "routing", "optimization"]
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
            text_bytes = query.encode('utf-8')
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
            # Import supervisor components
            from voxsigil_supervisor.supervisor_engine import SupervisorEngine
            
            # Create a new supervisor engine
            supervisor = SupervisorEngine(
                rag_interface=self,
                config={
                    "blt_enhanced": True,
                    "entropy_threshold": self.middleware_config.entropy_threshold if hasattr(self, 'middleware_config') else 0.25,
                    "hybrid_weight": self.middleware_config.blt_hybrid_weight if hasattr(self, 'middleware_config') else 0.7
                }
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
    
    def __init__(self, 
                 voxsigil_library_path: Optional[Path] = None,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 blt_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TinyLlama integration.
        
        Args:
            voxsigil_library_path: Path to the VoxSigil library.
            model_name: Name of the TinyLlama model.
            blt_config: Configuration for BLT components.
        """
        if not COMPONENTS_AVAILABLE:
            logger.error("Required components not available. TinyLlamaIntegration will not function.")
            return
            
        self.model_name = model_name
        self.blt_config = blt_config or {}
        
        # Initialize BLT-enhanced RAG interface
        logger.info("Initializing BLT-enhanced RAG interface...")
        self.rag_interface = BLTSupervisorRagInterface(
            voxsigil_library_path=voxsigil_library_path,
            embedding_model_name="all-MiniLM-L6-v2",  # Default model
            blt_config=self.blt_config
        )
        
        # Initialize TinyLlama (needs to be implemented based on how you use TinyLlama)
        logger.info(f"Initializing TinyLlama model: {model_name}")
        self.tinyllama_initialized = False
        try:
            # Import TinyLlama-specific components
            from models.tinyllama_assistant import initialize_tinyllama_model
            
            self.llm = initialize_tinyllama_model(model_name)
            self.tinyllama_initialized = True
            logger.info("TinyLlama model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TinyLlama model: {e}")
        
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
            # Import supervisor components
            from voxsigil_supervisor.supervisor_engine import VoxSigilSupervisor
            from voxsigil_supervisor.strategies.scaffold_router import ScaffoldRouter
            from voxsigil_supervisor.strategies.evaluation_heuristics import ResponseEvaluator
            from voxsigil_supervisor.strategies.retry_policy import RetryPolicy
            
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
                max_iterations=3
            )
            
            logger.info("VoxSigil Supervisor created with BLT-enhanced components")
            return supervisor
            
        except Exception as e:
            logger.error(f"Error creating supervisor: {e}")
            return None
            
    def _create_tinyllama_interface(self) -> 'BaseLlmInterface':
        """
        Create a TinyLlama LLM interface for the supervisor.
        
        Returns:
            An implementation of BaseLlmInterface for TinyLlama.
        """
        # This would need to be implemented based on your TinyLlama interface
        # For demonstration, a stub is provided:
        from voxsigil_supervisor.interfaces.llm_interface import BaseLlmInterface
        
        class TinyLlamaInterface(BaseLlmInterface):
            def __init__(self, model):
                self.model = model
                
            def generate_response(self, messages, system_prompt_override=None, task_requirements=None):
                # Implementation needed to generate responses with TinyLlama
                try:
                    # Example implementation
                    prompt = self._format_messages(messages, system_prompt_override)
                    response = self.model.generate(prompt)
                    return response, {"model": self.model.name}, {}
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    return "Error generating response", {}, {}
                    
            def _format_messages(self, messages, system_prompt):
                # Format messages for TinyLlama
                return "formatted_prompt"  # Placeholder
        
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
            "calculate_similarity": False
        }
        
        # Check TinyLlama
        if hasattr(self, 'llm') and self.llm is not None:
            validation_results["tinyllama_model"] = True
            
        # Verify RAG components
        if hasattr(self, 'rag_interface') and self.rag_interface is not None:
            validation_results["rag_instance"] = True
            
            # Check if rag_interface has blt components
            rag = self.rag_interface.rag_instance
            if rag is not None:
                # Check blt_encoder
                if hasattr(rag, 'blt_encoder') and rag.blt_encoder is not None:
                    validation_results["blt_encoder"] = True
                    
                    # Test create_patches method
                    try:
                        test_text = "This is a test text for BLT encoder validation."
                        patches = rag.blt_encoder.create_patches(test_text)
                        if patches is not None:
                            validation_results["create_patches"] = True
                            logger.info(f"create_patches test successful: {len(patches)} patches created")
                    except Exception as e:
                        logger.error(f"create_patches test failed: {e}")
                        
                    # Test calculate_similarity method
                    try:
                        if hasattr(rag.blt_encoder, 'calculate_similarity'):
                            # Create dummy embeddings
                            emb1 = np.random.rand(128)
                            emb2 = np.random.rand(128)
                            sim = rag.blt_encoder.calculate_similarity(emb1, emb2)
                            if 0 <= sim <= 1:
                                validation_results["calculate_similarity"] = True
                                logger.info(f"calculate_similarity test successful: {sim}")
                    except Exception as e:
                        logger.error(f"calculate_similarity test failed: {e}")
                
                # Check patch_encoder
                if hasattr(rag, 'patch_encoder') and rag.patch_encoder is not None:
                    validation_results["patch_encoder"] = True
        
        # Check if entropy_router is available
        if hasattr(self, 'middleware_config') and self.middleware_config is not None:
            validation_results["entropy_router"] = True
            if hasattr(self.middleware_config, 'entropy_router_fallback'):
                validation_results["entropy_router"] = True
                logger.info(f"Entropy router fallback: {self.middleware_config.entropy_router_fallback}")