#!/usr/bin/env python
"""
BLT-enhanced middleware for the VoxSigil system.

This module extends the standard VoxSigil middleware with Byte Latent Transformer
capabilities for improved performance and robustness.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path

# Import the HybridMiddleware class which is now the preferred middleware
from VoxSigilRag.hybrid_blt import HybridMiddleware as VoxSigilMiddleware
from VoxSigilRag.voxsigil_rag import VoxSigilRAG
from VoxSigilRag.voxsigil_blt import (
    ByteLatentTransformerEncoder, 
    SigilPatchEncoder, 
    PatchAwareValidator,
    PatchAwareCompressor
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoxSigilBLTMiddleware")

class BLTEnhancedMiddleware(VoxSigilMiddleware):
    """
    BLT-enhanced middleware that extends the standard VoxSigil middleware.
    
    This middleware integrates Byte Latent Transformer concepts for improved
    entropy-based byte-level processing, patch-based embeddings, and dynamic
    computation allocation.
    """
    
    def __init__(self, 
                 voxsigil_rag_instance: Optional[VoxSigilRAG] = None,
                 # Pass through standard middleware parameters
                 conversation_history_size: int = 5,
                 num_sigils: int = 5,
                 min_score_threshold: float = 0.4,
                 detail_level: str = "standard",
                 include_explanations: bool = True,
                 rag_off_keywords: Optional[List[str]] = None,
                 min_prompt_len_for_rag: int = 5,
                 history_truncation_strategy: str = "tail",
                 enable_intent_detection: bool = False,
                 # BLT-specific parameters
                 blt_hybrid_weight: float = 0.5,
                 entropy_threshold: float = 0.5,
                 enable_patch_validation: bool = True,
                 enable_patch_compression: bool = True):
        """
        Initialize the BLT-enhanced middleware.
        
        Args:
            voxsigil_rag_instance: VoxSigil RAG instance 
            conversation_history_size: Max turns for history context
            num_sigils: Default max sigils to include
            min_score_threshold: Default min score for sigils
            detail_level: Default detail level for sigil formatting
            include_explanations: Default for including retrieval explanations
            rag_off_keywords: Keywords in user prompt to disable RAG
            min_prompt_len_for_rag: Minimum length to trigger RAG
            history_truncation_strategy: How to truncate history
            enable_intent_detection: Enable intent detection for filtering
            blt_hybrid_weight: Weight of BLT vs. standard embeddings (0-1)
            entropy_threshold: Entropy threshold for patch boundaries
            enable_patch_validation: Enable BLT-based schema validation
            enable_patch_compression: Enable entropy-based compression
        """
        # Initialize the parent middleware first
        super().__init__(
            voxsigil_rag_instance=voxsigil_rag_instance,
            conversation_history_size=conversation_history_size,
            num_sigils=num_sigils,
            min_score_threshold=min_score_threshold,
            detail_level=detail_level,
            include_explanations=include_explanations,
            rag_off_keywords=rag_off_keywords,
            min_prompt_len_for_rag=min_prompt_len_for_rag,
            history_truncation_strategy=history_truncation_strategy,
            enable_intent_detection=enable_intent_detection
        )
        
        # Initialize BLT components
        self.blt_encoder = ByteLatentTransformerEncoder(
            entropy_threshold=entropy_threshold
        )
        
        # Create patch encoder (or retrieve from RAG if available)
        base_model = None
        if self.voxsigil_rag and hasattr(self.voxsigil_rag, 'embedding_model'):
            base_model = self.voxsigil_rag.embedding_model
            
        self.patch_encoder = SigilPatchEncoder(
            base_embedding_model=base_model,
            entropy_threshold=entropy_threshold
        )
        
        # Initialize validator and compressor
        self.patch_validator = PatchAwareValidator(
            entropy_threshold=entropy_threshold + 0.1  # Slightly higher threshold for validation
        )
        
        self.patch_compressor = PatchAwareCompressor(
            entropy_threshold=entropy_threshold
        )
        
        # BLT configuration
        self.blt_hybrid_weight = blt_hybrid_weight
        self.enable_patch_validation = enable_patch_validation
        self.enable_patch_compression = enable_patch_compression
        
        # Track additional analytics
        self.patch_analytics = {
            "total_patches_processed": 0,
            "avg_entropy": 0.0,
            "validation_issues": []
        }
        
        logger.info("BLT-enhanced middleware initialized")

    def _blt_enhance_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply BLT enhancements to the request.
        
        Args:
            request: Original request dictionary
            
        Returns:
            Enhanced request dictionary
        """
        # Make a copy to avoid modifying original
        enhanced_request = request.copy()
        
        # Extract the user message if present
        user_message = ""
        if 'messages' in enhanced_request:
            for msg in enhanced_request['messages']:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
        
        # If we have a user message, analyze it with BLT
        if user_message:
            # Create patches for entropy analysis
            patches = self.blt_encoder.create_patches(user_message)
            
            # Update analytics
            self.patch_analytics["total_patches_processed"] += len(patches)
            if patches:
                avg_entropy = sum(entropy for _, entropy in patches) / len(patches)
                self.patch_analytics["avg_entropy"] = (
                    (self.patch_analytics["avg_entropy"] + avg_entropy) / 2  # Running average
                )
            
            # Log insights
            high_entropy_count = sum(1 for _, e in patches if e > self.blt_encoder.entropy_threshold)
            logger.debug(f"BLT analysis: {len(patches)} patches, {high_entropy_count} high-entropy")
            
            # Add metadata to request for downstream processing
            if 'blt_metadata' not in enhanced_request:
                enhanced_request['blt_metadata'] = {}
                
            enhanced_request['blt_metadata']['patch_count'] = len(patches)
            enhanced_request['blt_metadata']['high_entropy_ratio'] = (
                high_entropy_count / len(patches) if patches else 0
            )
        
        return enhanced_request
    
    def preprocess_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override the parent method to add BLT preprocessing.
        
        Args:
            request: Original API request
            
        Returns:
            Processed request with BLT enhancements
        """
        # Apply BLT enhancements first
        blt_enhanced_request = self._blt_enhance_request(request)
        
        # Then apply the parent middleware processing
        return super().preprocess_request(blt_enhanced_request)
    
    def _extract_message_content(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Extract system and user content from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (system_content, user_content)
        """
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_content = msg.get('content', '')
        
        return system_content, user_content
    
    def _validate_sigil_with_blt(self, sigil_data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a sigil with BLT-based patch validation.
        
        Args:
            sigil_data: Sigil data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if not self.enable_patch_validation:
            return True, []
            
        # Convert to string for validation
        sigil_str = str(sigil_data)
        
        # Validate using patch-aware validator
        is_valid, issues = self.patch_validator.validate_schema(sigil_str)
        
        # Add issues to analytics
        if issues:
            self.patch_analytics["validation_issues"].extend(issues)
            
        return is_valid, issues
    
    def _compress_content_with_blt(self, content: str, max_size: Optional[int] = None) -> str:
        """
        Compress content using BLT patch-aware compression.
        
        Args:
            content: Content to compress
            max_size: Optional maximum size in characters
            
        Returns:
            Compressed content
        """
        if not self.enable_patch_compression or not content:
            return content
            
        # If content is already smaller than max_size, return as is
        if max_size and len(content) <= max_size:
            return content
            
        # Apply patch-aware compression
        compressed, ratio = self.patch_compressor.compress(content)
        
        logger.debug(f"BLT compression: {len(content)} → {len(compressed)} chars ({ratio:.2f} ratio)")
        
        # If we have a max size and compressed is still too big, truncate
        if max_size and len(compressed) > max_size:
            return compressed[:max_size]
            
        return compressed
    
    def wrap_llm_api(self, llm_api_call: Callable[..., Any]) -> Callable[..., Any]:
        """
        Override to use BLT-enhanced wrapping.
        """
        # Get the parent's wrapped function
        parent_wrapped = super().wrap_llm_api(llm_api_call)
        
        # Define our enhanced wrapper
        def blt_enhanced_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Apply BLT-specific processing
            model_config = None
            messages = None
            
            # Try to extract model config and messages
            if args and isinstance(args[0], dict):
                model_config = args[0]
                if len(args) > 1 and isinstance(args[1], list):
                    messages = args[1]
            
            # Log BLT processing
            if model_config and messages:
                logger.debug(f"BLT-enhanced request for model {model_config.get('name', 'unknown')}")
                
                # Apply potential BLT optimizations based on message content
                system_content, user_content = self._extract_message_content(messages)
                
                # Apply compression if needed
                if user_content and self.enable_patch_compression:
                    # Determine if we should compress based on entropy
                    patches = self.blt_encoder.create_patches(user_content)
                    avg_entropy = sum(entropy for _, entropy in patches) / len(patches) if patches else 0
                    
                    # Compress high-entropy content more aggressively
                    if avg_entropy > self.blt_encoder.entropy_threshold:
                        compressed_user = self._compress_content_with_blt(user_content)
                        
                        # Replace user content in messages
                        for msg in messages:
                            if msg.get('role') == 'user':
                                msg['content'] = compressed_user
                                break
            
            # Call the parent's wrapped function
            return parent_wrapped(*args, **kwargs)
        
        return blt_enhanced_wrapper
