"""
VoxSigil BLT-Memory Engine (VME)
Single-call memory codec library for LLMs
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["build_context", "ContextPack"]


@dataclass
class ContextPack:
    """Output from build_context: compressed, signed context."""
    query: str
    compressed_content: bytes
    signature: str
    version: str
    budget_tokens: int
    mode: str
    metadata: Dict[str, Any]


def build_context(
    query: str,
    budget_tokens: int = 2048,
    mode: str = "balanced",
    device: Optional[str] = None,
    cache: bool = True,
) -> ContextPack:
    """
    Single-call memory codec: build optimized context from query.
    
    Args:
        query: Input query string (what LLM needs context for)
        budget_tokens: Token budget for context (default 2048)
        mode: Compression mode - 'aggressive', 'balanced', 'quality' (default 'balanced')
        device: 'cuda' or 'cpu' (auto-detect if None)
        cache: Whether to cache the result (default True)
    
    Returns:
        ContextPack: Compressed context + metadata
    
    Raises:
        ValueError: If budget_tokens < 128 or mode invalid
        RuntimeError: If semantic layer or protocol layer fails
    
    Example:
        >>> result = build_context("what is machine learning", budget_tokens=1024)
        >>> print(result.signature)  # Deterministic signature
    """
    # Phase 1: Input validation
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    
    if budget_tokens < 128:
        raise ValueError(f"budget_tokens must be >= 128, got {budget_tokens}")
    
    valid_modes = {"aggressive", "balanced", "quality"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
    
    if device is None:
        device = "cpu"  # Default to cpu for Phase 1
    elif device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device}")
    
    # Phase 1: Prepare metadata
    metadata = {
        "device": device,
        "cache": cache,
        "query_length": len(query),
    }
    
    # Phase 1-2: Call semantic layer (stub for now)
    try:
        from voxsigil_memory.protocol import ProtocolVersioner  # noqa: F401
        # Placeholder: phase 2 will implement compression
        compressed_content = f"[PHASE_1_STUB] query={query[:50]}... mode={mode}".encode()
        metadata["semantic_status"] = "phase_1_stub"
    except Exception as e:
        raise RuntimeError(f"Semantic layer error: {e}") from e
    
    # Phase 1-3: Call protocol layer (stub for now)
    try:
        from voxsigil_memory.protocol import ProtocolVersioner
        # Placeholder: phase 3 will implement deterministic signing
        signature = "phase_1_stub_signature"
        versioner = ProtocolVersioner()
        version = versioner.current_version
        metadata["protocol_status"] = "phase_1_stub"
    except Exception as e:
        raise RuntimeError(f"Protocol layer error: {e}") from e
    
    # Phase 1: Return ContextPack
    return ContextPack(
        query=query,
        compressed_content=compressed_content,
        signature=signature,
        version=version,
        budget_tokens=budget_tokens,
        mode=mode,
        metadata=metadata,
    )


def decompress_context(context_pack: ContextPack) -> str:
    """Decompress and verify a context pack."""
    raise NotImplementedError("Phase 1: Implement context decompression")
