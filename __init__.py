"""
BLT (Bi-directional Learning Transfer) System

This package provides the BLT system components:
- BLT Encoder: Core encoding functionality
- Hybrid Middleware: Adaptive processing middleware
- RAG Components: Retrieval Augmented Generation integrations
- ARC Integration: Specialized adapters for ARC tasks
"""

__all__ = [
    # Core BLT components
    "BLTEncoder",
    "ByteLatentTransformerEncoder",
    "HybridMiddleware",
    "EntropyRouter",
    "SigilPatchEncoder",
    # RAG integration
    "VoxSigilRAG",
    "BLTEnhancedRAG",
    # Compression components
    "RAGCompressionEngine",
    "RAGCompressionError",
    "PatchAwareValidator",
    "PatchAwareCompressor",
    # ARC integration
    "ARCGridFormerBLT",
    "GridFormerBLTAdapter",
]

# Ensure all modules are properly importable
try:
    from .blt_encoder import BLTEncoder, SigilPatchEncoder

    # Use BLTEncoder as ByteLatentTransformerEncoder alias to avoid conflicts
    ByteLatentTransformerEncoder = BLTEncoder

    from .arc_gridformer_blt_adapter import GridFormerBLTAdapter
    from .blt_rag_compression import (
        PatchAwareCompressor,
        PatchAwareValidator,
        RAGCompressionEngine,
        RAGCompressionError,
    )
    from .hybrid_blt_fixed import BLTEnhancedRAG, EntropyRouter, HybridMiddleware
    from .voxsigil_rag_clean import VoxSigilRAG

    # Import ARC components separately to avoid circular imports
    try:
        from .arc_gridformer_blt import ByteLatentTransformerEncoder as ARCGridFormerBLT
    except ImportError:
        # Fallback to main BLT encoder if ARC version not available
        ARCGridFormerBLT = BLTEncoder

except ImportError as e:
    # Fallback imports if main components fail
    import warnings

    warnings.warn(f"Some BLT components not available: {e}")

    # Define minimal stubs
    class BLTEncoder:
        def encode(self, text, task_type=None):
            return [0.0] * 128

    ByteLatentTransformerEncoder = BLTEncoder
    SigilPatchEncoder = None
    ARCGridFormerBLT = BLTEncoder
