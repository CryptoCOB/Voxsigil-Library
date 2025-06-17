"""
Proxy module for the vantacore_grid_former_integration.py module in Vanta.integration.
This file exists to maintain backward compatibility after the module was moved.
"""

# Import only the GridFormerVantaIntegration class and integrate_with_vantacore function
# HybridARCSolver will be lazily imported in the Vanta.integration module to avoid circular imports
from Vanta.integration.vantacore_grid_former_integration import (
    GridFormerVantaIntegration,
    integrate_with_vantacore,
    main,
    parse_arguments,
)

# Re-export only the classes and functions that don't cause circular imports
__all__ = [
    "GridFormerVantaIntegration",
    "parse_arguments",
    "main",
    "integrate_with_vantacore",
]

# HybridARCSolver should be imported directly from ARC.arc_integration when needed
