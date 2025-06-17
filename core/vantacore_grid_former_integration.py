"""
Proxy module for the vantacore_grid_former_integration.py module in Vanta.integration.
This file exists to maintain backward compatibility after the module was moved.
"""

# Import the GridFormerVantaIntegration from the new location
from Vanta.integration.vantacore_grid_former_integration import (
    GridFormerVantaIntegration,
    HybridARCSolver,
    integrate_with_vantacore,
    main,
    parse_arguments,
)

# Re-export all classes and functions
__all__ = [
    "GridFormerVantaIntegration",
    "parse_arguments",
    "main",
    "HybridARCSolver",
    "integrate_with_vantacore",
]
