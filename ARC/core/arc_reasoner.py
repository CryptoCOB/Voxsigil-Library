"""
ARC Reasoner Bridge Module
=======================

This module re-exports classes and functions from ARC.arc_reasoner
for backward compatibility.
"""

# Re-export the classes and functions from the actual implementation
from ARC.arc_reasoner import (
    ARCReasoner,
    # Include other exports as needed
)

# Export all the re-exported symbols
__all__ = [
    "ARCReasoner",
    # Include other exports as needed
]
