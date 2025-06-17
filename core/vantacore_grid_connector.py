"""
Proxy module for the vantacore_grid_connector.py module in Vanta.integration.
This file exists to maintain backward compatibility after the module was moved.
"""

# Import the GridFormerConnector from the new location
from Vanta.integration.vantacore_grid_connector import (
    GridFormerConnector,
    # Also re-export any other classes or functions that might be used
)

# Re-export the GridFormerConnector class
__all__ = ["GridFormerConnector"]
