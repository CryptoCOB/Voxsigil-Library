"""VoxSigil Library - LLM components package."""

from .register_llm_module import register_llm, LLMModuleAdapter

__all__ = [
    "register_llm",
    "LLMModuleAdapter",
]
