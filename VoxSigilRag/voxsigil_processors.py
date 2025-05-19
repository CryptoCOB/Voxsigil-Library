#!/usr/bin/env python
"""
VoxSigil RAG and BLT Processors for the VoxSigil system.

This module provides RAGProcessor and BLTProcessor classes used in training scripts
to enhance inputs with VoxSigil functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoxSigilProcessors")

class RAGProcessor:
    """
    Retrieval-Augmented Generation processor for VoxSigil.
    
    This class provides functionality to enhance prompts with retrieved sigil context.
    """
    
    def __init__(self, voxsigil_rag_instance=None):
        """
        Initialize the RAG processor.
        
        Args:
            voxsigil_rag_instance: VoxSigilRAG instance (if None, will create or import)
        """
        self.voxsigil_rag = voxsigil_rag_instance
        
        # Import VoxSigilRAG if not provided
        if not self.voxsigil_rag:
            try:
                from VoxSigilRag.voxsigil_rag import VoxSigilRAG
                self.voxsigil_rag = VoxSigilRAG()
                logger.info("Initialized VoxSigilRAG for RAGProcessor")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilRAG: {e}")
                self.voxsigil_rag = None
        
        # Track additional state
        self.last_processed = None
        self.last_sigils = []
    
    def process(self, text: str, **kwargs) -> str:
        """
        Process text with RAG enhancements.
        
        Args:
            text: The input text to enhance
            **kwargs: Additional parameters for RAG processing
            
        Returns:
            Enhanced text with RAG context
        """
        if not self.voxsigil_rag:
            logger.warning("VoxSigilRAG not available. Returning original text.")
            return text
        
        try:
            # Default parameters
            num_sigils = kwargs.get('num_sigils', 5)
            min_score = kwargs.get('min_score_threshold', 0.4)
            detail_level = kwargs.get('detail_level', 'standard')
            include_explanations = kwargs.get('include_explanations', True)
            
            # Get RAG context using underlying VoxSigilRAG
            if hasattr(self.voxsigil_rag, 'inject_voxsigil_context'):
                enhanced_text, retrieved_sigils = self.voxsigil_rag.inject_voxsigil_context(
                    prompt=text,
                    query=text,
                    num_sigils=num_sigils,
                    min_score_threshold=min_score,
                    detail_level=detail_level,
                    include_explanations=include_explanations
                )
            elif hasattr(self.voxsigil_rag, 'create_rag_context'):
                context_str, retrieved_sigils = self.voxsigil_rag.create_rag_context(
                    query=text,
                    num_sigils=num_sigils
                )
                
                # Create enhanced text with RAG context
                enhanced_text = f"--- VoxSigil Context ---\n{context_str}\n\n--- Original Query ---\n{text}"
            else:
                logger.warning("VoxSigilRAG instance is missing required methods. Returning original text.")
                return text
            
            # Store last processed for tracking
            self.last_processed = text
            self.last_sigils = retrieved_sigils
            
            return enhanced_text
        
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}", exc_info=True)
            return text  # Fall back to original text on error


class BLTProcessor:
    """
    Byte-Latent Transformer processor for VoxSigil.
    
    This class provides functionality to enhance prompts with BLT processing.
    """
    
    def __init__(self, blt_instance=None):
        """
        Initialize the BLT processor.
        
        Args:
            blt_instance: VoxSigilBLT instance (if None, will create or import)
        """
        self.blt_instance = blt_instance
        
        # Import VoxSigilBLT if not provided
        if not self.blt_instance:
            try:
                from VoxSigilRag.voxsigil_blt import ByteLatentTransformerEncoder
                self.blt_instance = ByteLatentTransformerEncoder()
                logger.info("Initialized ByteLatentTransformerEncoder for BLTProcessor")
            except ImportError as e:
                logger.warning(f"Could not import ByteLatentTransformerEncoder: {e}")
                self.blt_instance = None
        
        # Track additional state
        self.last_processed = None
        self.last_patches = []
    
    def process(self, text: str, **kwargs) -> str:
        """
        Process text with BLT enhancements.
        
        Args:
            text: The input text to enhance
            **kwargs: Additional parameters for BLT processing
            
        Returns:
            Enhanced text with BLT structures
        """
        if not self.blt_instance:
            logger.warning("BLT instance not available. Returning original text.")
            return text
        
        try:
            # Create patches based on entropy
            if hasattr(self.blt_instance, 'create_patches'):
                patches = self.blt_instance.create_patches(text)
                self.last_patches = patches
                
                # Extract logical structure from the text
                high_entropy_count = sum(1 for patch in patches if getattr(patch, 'entropy', 0) > 0.5)
                low_entropy_count = len(patches) - high_entropy_count
                
                # Simple tag-based formatting for demo
                if high_entropy_count > low_entropy_count:
                    structure_desc = "High-entropy natural language"
                    context_type = "NLU"
                else:
                    structure_desc = "Low-entropy structured information"
                    context_type = "STRUCTURED"
                
                # Add simple BLT analysis as scaffold
                scaffold = f"BLT Analysis: {structure_desc} ({context_type})"
                enhanced_text = f"{text}\n\nScaffold: {scaffold}\nPatches: {len(patches)}\nHigh-Entropy: {high_entropy_count}, Low-Entropy: {low_entropy_count}"
            else:
                logger.warning("BLT instance is missing required methods. Returning original text.")
                return text
            
            # Store last processed for tracking
            self.last_processed = text
            
            return enhanced_text
        
        except Exception as e:
            logger.error(f"Error in BLT processing: {e}", exc_info=True)
            return text  # Fall back to original text on error
