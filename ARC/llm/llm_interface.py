"""
LLM Interface Bridge Module
========================

This module re-exports the BaseLlmInterface class from Vanta.interfaces.base_interfaces
for backward compatibility.
"""

# Re-export the BaseLlmInterface class from the actual implementation
from Vanta.interfaces.base_interfaces import BaseLlmInterface

# Export all the re-exported symbols
__all__ = [
    "BaseLlmInterface",
]
