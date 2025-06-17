"""
ARC Data Processor Bridge Module
==============================

This module re-exports classes and functions from ARC.arc_data_processor
for backward compatibility.
"""

# Re-export the classes and functions from the actual implementation
from ..arc_data_processor import (
    ARCGridDataProcessor,
    create_arc_dataloaders,
    visualize_grid,
    # Include other exports as needed
)

# Export all the re-exported symbols
__all__ = [
    "ARCGridDataProcessor",
    "visualize_grid",
    "create_arc_dataloaders",
    # Include other exports as needed
]
