# filepath: c:\Users\16479\Desktop\ARC2025\VoxSigilRag\__init__.py
"""
VoxSigilRag Package

This package provides retrieval-augmented generation (RAG) capabilities
for enhancing model prompts with VoxSigil sigils from the VoxSigil-Library.
"""
import logging

# Lazy imports to avoid circular dependencies
def _lazy_import():
    """Import main classes when needed to avoid circular dependencies."""
    globals_ = {}

    try:
        from .hybrid_blt import HybridMiddleware

        globals_["HybridMiddleware"] = HybridMiddleware
        # Alias for backward compatibility
        globals_["VoxSigilMiddleware"] = HybridMiddleware
    except ImportError:
        pass

    try:
        from .sigil_patch_encoder import SigilPatchEncoder

        globals_["SigilPatchEncoder"] = SigilPatchEncoder
    except ImportError:
        pass

    try:
        from .voxsigil_blt import BLTEnhancedMiddleware, ByteLatentTransformerEncoder

        globals_["BLTEnhancedMiddleware"] = BLTEnhancedMiddleware
        globals_["ByteLatentTransformerEncoder"] = ByteLatentTransformerEncoder
    except ImportError:
        pass

    try:
        from .voxsigil_evaluator import VoxSigilConfig, VoxSigilError, VoxSigilResponseEvaluator

        globals_["VoxSigilConfig"] = VoxSigilConfig
        globals_["VoxSigilError"] = VoxSigilError
        globals_["VoxSigilResponseEvaluator"] = VoxSigilResponseEvaluator
    except ImportError:
        pass

    try:
        from .voxsigil_rag_compression import RAGCompressionEngine, RAGCompressionError

        globals_["RAGCompressionEngine"] = RAGCompressionEngine
        globals_["RAGCompressionError"] = RAGCompressionError
    except ImportError:
        pass

    try:
        from .voxsigil_semantic_cache import SemanticCacheManager

        globals_["SemanticCacheManager"] = SemanticCacheManager
    except ImportError:
        pass

    try:
        from .voxsigil_rag import VoxSigilRAG

        globals_["VoxSigilRAG"] = VoxSigilRAG
    except ImportError:
        # Create stub if not available
        class VoxSigilRAG:
            def __init__(self, *args, **kwargs):
                pass

        globals_["VoxSigilRAG"] = VoxSigilRAG
    return globals_


# Module-level getattr to handle lazy imports
def __getattr__(name):
    """Lazy import mechanism."""
    _globals = _lazy_import()
    if name in _globals:
        return _globals[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Legacy middleware now replaced by HybridMiddleware
# from .voxsigil_middleware import VoxSigilMiddleware


__version__ = "0.1.0"

# Make sure RagSystem is available at module level
__all__ = [
    "VoxSigilRAG",
    "RagSystem",
    "VoxSigilResponseEvaluator",
    "VoxSigilConfig",
    "VoxSigilError",
    "RAGCompressionEngine",
    "RAGCompressionError",
    "ByteLatentTransformerEncoder",
    "BLTEnhancedMiddleware",
    "SigilPatchEncoder",
    "HybridMiddleware",
    "VoxSigilMiddleware",
    "SemanticCacheManager",
]

# Package-level logger setup


logging.getLogger(__name__).addHandler(logging.NullHandler())
