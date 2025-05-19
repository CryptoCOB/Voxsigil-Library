# filepath: c:\Users\16479\Desktop\ARC2025\VoxSigilRag\__init__.py
"""
VoxSigilRag Package

This package provides retrieval-augmented generation (RAG) capabilities
for enhancing model prompts with VoxSigil sigils from the VoxSigil-Library.
"""

# Import main classes for easy access
from .voxsigil_rag import VoxSigilRAG
# Legacy middleware now replaced by HybridMiddleware
# from .voxsigil_middleware import VoxSigilMiddleware
from .voxsigil_evaluator import VoxSigilResponseEvaluator
from .voxsigil_evaluator import VoxSigilConfig, VoxSigilError
from .voxsigil_rag_compression import RAGCompressionEngine, RAGCompressionError
from .voxsigil_blt import ByteLatentTransformerEncoder, BLTEnhancedMiddleware
from .sigil_patch_encoder import SigilPatchEncoder
from .hybrid_blt import HybridMiddleware
from .voxsigil_semantic_cache import SemanticCacheManager

# Alias for backward compatibility - pointing to new hybrid middleware
from .hybrid_blt import HybridMiddleware as VoxSigilMiddleware

try:
    from .sigil_patch_encoder import SigilPatchEncoder
except ImportError:
    pass  # Silently skip if not available

__version__ = "0.1.0"

# Package-level logger setup
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
