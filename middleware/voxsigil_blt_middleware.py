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
import time

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
    
    def _extract_message_content(self, messages: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Extract system and user content from messages for BLT processing.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Tuple of (system_content, user_content)
        """
        system_content = ""
        user_content = ""
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                system_content += content + "\n"
            elif role == 'user':
                user_content += content + "\n"
        
        return system_content.strip(), user_content.strip()
    
    def _adaptive_compress(self, content: str, context_size: int = 0) -> str:
        """
        Apply adaptive compression based on content characteristics and context.
        
        Args:
            content: Content to compress
            context_size: Size of existing context
            
        Returns:
            Compressed content
        """
        if not content:
            return content
        
        # Determine compression strategy based on content type and size
        content_length = len(content)
        
        # For very short content, don't compress
        if content_length < 100:
            return content
        
        # Calculate compression level based on context pressure
        context_pressure = context_size / self.max_context_size if self.max_context_size else 0
        
        if context_pressure > 0.8:
            # High pressure: aggressive compression
            compression_level = 'aggressive'
        elif context_pressure > 0.5:
            # Medium pressure: balanced compression
            compression_level = 'balanced'
        else:
            # Low pressure: light compression
            compression_level = 'light'
        
        # Apply BLT compression with appropriate level
        if compression_level == 'aggressive':
            # Use maximum patch size for aggressive compression
            patches = self.blt_encoder.create_patches(content, max_patch_size=128)
        elif compression_level == 'balanced':
            # Use medium patch size
            patches = self.blt_encoder.create_patches(content, max_patch_size=64)
        else:
            # Use small patch size for light compression
            patches = self.blt_encoder.create_patches(content, max_patch_size=32)
        
        # Encode patches
        encoded = self.blt_encoder.encode_patches(patches)
        
        logger.debug(f"Adaptive compression ({compression_level}): {content_length} → {len(encoded)} chars")
        
        return encoded
    
    def create_context_manager(self, max_tokens: int = None) -> 'BLTContextManager':
        """
        Create a BLT-enhanced context manager for handling conversation context.
        
        Args:
            max_tokens: Maximum token limit for context
            
        Returns:
            BLTContextManager instance
        """
        return BLTContextManager(
            middleware=self,
            max_tokens=max_tokens or self.max_context_size
        )
    
    def optimize_for_model(self, model_name: str) -> None:
        """
        Optimize BLT settings for specific model characteristics.
        
        Args:
            model_name: Name of the target model
        """
        # Model-specific optimizations
        if 'gpt-4' in model_name.lower():
            # GPT-4 has good context handling, use lighter compression
            self.blt_encoder.entropy_threshold = 0.7
            self.enable_adaptive_compression = True
        elif 'gpt-3.5' in model_name.lower():
            # GPT-3.5 benefits from more compression
            self.blt_encoder.entropy_threshold = 0.5
            self.enable_adaptive_compression = True
        elif 'claude' in model_name.lower():
            # Claude has excellent context handling
            self.blt_encoder.entropy_threshold = 0.8
            self.enable_adaptive_compression = False
        else:
            # Default settings for unknown models
            self.blt_encoder.entropy_threshold = 0.6
            self.enable_adaptive_compression = True
        
        logger.info(f"Optimized BLT settings for model: {model_name}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get statistics about BLT compression performance.
        
        Returns:
            Dictionary with compression statistics
        """
        stats = {
            'total_compressions': getattr(self, '_compression_count', 0),
            'total_bytes_saved': getattr(self, '_bytes_saved', 0),
            'average_compression_ratio': getattr(self, '_avg_compression_ratio', 0.0),
            'encoder_stats': self.blt_encoder.get_stats() if hasattr(self.blt_encoder, 'get_stats') else {}
        }
        
        if hasattr(self.patch_compressor, 'get_stats'):
            stats['patch_compressor_stats'] = self.patch_compressor.get_stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self._compression_count = 0
        self._bytes_saved = 0
        self._avg_compression_ratio = 0.0
        
        if hasattr(self.blt_encoder, 'reset_stats'):
            self.blt_encoder.reset_stats()
        
        if hasattr(self.patch_compressor, 'reset_stats'):
            self.patch_compressor.reset_stats()


class BLTContextManager:
    """
    Context manager for handling BLT-enhanced conversation context.
    Provides intelligent context truncation and compression.
    """
    
    def __init__(self, middleware: 'VoxSigilBLTMiddleware', max_tokens: int = 4000):
        """
        Initialize BLT context manager.
        
        Args:
            middleware: VoxSigil BLT middleware instance
            max_tokens: Maximum token limit
        """
        self.middleware = middleware
        self.max_tokens = max_tokens
        self.context_history: List[Dict[str, Any]] = []
        self.compressed_context: Optional[str] = None
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the context with BLT optimization.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
        """
        message = {'role': role, 'content': content, 'timestamp': time.time()}
        self.context_history.append(message)
        
        # Check if we need to compress or truncate
        if self._estimate_token_count() > self.max_tokens:
            self._optimize_context()
    
    def get_context(self) -> List[Dict[str, Any]]:
        """
        Get optimized context for API calls.
        
        Returns:
            List of optimized message dictionaries
        """
        if not self.context_history:
            return []
        
        # If we have compressed context, use it
        if self.compressed_context:
            return [{'role': 'system', 'content': self.compressed_context}] + self.context_history[-2:]
        
        return self.context_history
    
    def _estimate_token_count(self) -> int:
        """
        Estimate token count for current context.
        
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg['content']) for msg in self.context_history)
        # Rough estimation: 1 token ≈ 4 characters
        return total_chars // 4
    
    def _optimize_context(self) -> None:
        """
        Optimize context using BLT compression and intelligent truncation.
        """
        if len(self.context_history) < 3:
            return
        
        # Keep the most recent system message and last user/assistant exchange
        recent_messages = self.context_history[-2:]
        older_messages = self.context_history[:-2]
        
        # Compress older messages
        if older_messages:
            combined_content = "\n".join(f"{msg['role']}: {msg['content']}" for msg in older_messages)
            
            # Apply BLT compression
            self.compressed_context = self.middleware._adaptive_compress(
                combined_content,
                context_size=self._estimate_token_count()
            )
            
            # Clear older messages from history
            self.context_history = recent_messages
            
            logger.debug(f"Compressed {len(older_messages)} messages into BLT context")
    
    def clear(self) -> None:
        """Clear all context."""
        self.context_history.clear()
        self.compressed_context = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context management statistics.
        
        Returns:
            Dictionary with context stats
        """
        return {
            'message_count': len(self.context_history),
            'estimated_tokens': self._estimate_token_count(),
            'has_compressed_context': self.compressed_context is not None,
            'compressed_context_size': len(self.compressed_context) if self.compressed_context else 0
        }


# Export the main classes
__all__ = ['VoxSigilBLTMiddleware', 'BLTContextManager']
