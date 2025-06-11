#!/usr/bin/env python
"""
Run script for the VantaCore Grid Connector.

This script runs the test function in the vantacore_grid_connector module.
"""

# Add the root directory to path for imports
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import the test function
from Voxsigil_Library.Gridformer.core.vantacore_grid_connector import (
    test_grid_former_connector,
)

if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run the test function
    print("Running VantaCore Grid Connector test...")
    test_grid_former_connector()
