# interfaces/__init__.py
"""
Legacy Interface Wrappers (DEPRECATED)
======================================

⚠️  DEPRECATION NOTICE: These interfaces are now deprecated.
    Use unified interfaces from Vanta.interfaces instead.

This sub-package contains legacy interface wrappers that are maintained
for backward compatibility. New code should import directly from:

    from Vanta.interfaces import (
        BaseRagInterface,
        BaseLlmInterface, 
        BaseMemoryInterface
    )

These legacy imports will be removed in a future version.
"""

# Legacy imports - redirect to Vanta unified interfaces
try:
    from Vanta.interfaces import (
        BaseRagInterface,
        BaseLlmInterface,
        BaseMemoryInterface
    )
    print("✅ Successfully imported unified interfaces from Vanta")
except ImportError:
    # Fallback to local implementations for backward compatibility
    from .rag_interface import BaseRagInterface
    from .llm_interface import BaseLlmInterface  
    from .memory_interface import BaseMemoryInterface
    print("⚠️  Using legacy interface implementations - consider updating to Vanta.interfaces")

__all__ = [
    "BaseRagInterface",
    "BaseLlmInterface", 
    "BaseMemoryInterface",
]