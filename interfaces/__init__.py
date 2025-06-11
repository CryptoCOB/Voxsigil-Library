# voxsigil_supervisor/interfaces/__init__.py
"""
Interfaces for the VoxSigil Supervisor.

This sub-package defines abstract base classes or protocols for external
systems like RAG engines, LLMs, and memory stores, allowing for
pluggable implementations.
"""
from .rag_interface import BaseRagInterface
from .llm_interface import BaseLlmInterface
from .memory_interface import BaseMemoryInterface

__all__ = [
    "BaseRagInterface",
    "BaseLlmInterface",
    "BaseMemoryInterface",
]