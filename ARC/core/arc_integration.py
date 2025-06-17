"""
ARC Integration Bridge Module
===========================

This module re-exports classes and functions from ARC.arc_integration
for backward compatibility.
"""

# Re-export the classes and functions from the actual implementation
from ARC.arc_integration import (
    HybridARCSolver,
    integrate_with_vantacore,
    # Include other exports as needed
)

# Export all the re-exported symbols
__all__ = [
    "HybridARCSolver",
    "integrate_with_vantacore",
    # Include other exports as needed
]
