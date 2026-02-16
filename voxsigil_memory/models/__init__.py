"""Data models and type definitions."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextPack:
    """Compressed context package."""
    query: str
    compressed_content: bytes
    signature: str
    version: str
    budget_tokens: int
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class CompressionMetrics:
    """Metrics from compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    time_ms: float
    device: str


__all__ = ["ContextPack", "CompressionMetrics"]
