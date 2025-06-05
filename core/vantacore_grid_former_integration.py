#!/usr/bin/env python
"""
vantacore_grid_former_integration.py - VantaCore Integration for GRID-Former

This script integrates the GRID-Former direct neural network training approach
with the VantaCore meta-learning system in VoxSigil.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Use standard path helper for imports
try:
    from utils.path_helper import add_project_root_to_path

    add_project_root_to_path()
except ImportError:
    # Fallback if path_helper isn't available
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import required components
try:
    from Gridformer.core.grid_former import GRID_Former
    from Gridformer.core.vantacore_grid_connector import GridFormerConnector
    from Gridformer.core.enhanced_grid_connector import EnhancedGridFormerConnector
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
except ImportError as e:
    logging.error(f"Failed to import required components: {e}")
    # Fallback for relative imports
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    try:
        from Gridformer.core.grid_former import GRID_Former
        from Gridformer.core.vantacore_grid_connector import GridFormerConnector
        from Gridformer.core.enhanced_grid_connector import EnhancedGridFormerConnector
        from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
    except ImportError as e:
        logging.error(f"Still failed to import required components: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vantacore_grid_former_integration.log"),
    ],
)

logger = logging.getLogger("VantaCore.Integration")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Integrate GRID-Former with VantaCore")

    parser.add_argument(
        "--vantacore-module", type=str, required=True, help="Path to VantaCore module"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained GRID-Former model",
    )

    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to VantaCore configuration"
    )

    parser.add_argument(
        "--hybrid-mode",
        action="store_true",
        help="Enable hybrid mode combining neural nets with LLMs",
    )

    return parser.parse_args()


class GridFormerVantaIntegration:
    """
    Integration class for connecting GRID-Former with VantaCore.

    This class provides a high-level interface for working with both
    GRID-Former and VantaCore together in a unified manner.
    """

    def __init__(
        self, vanta_core=None, model_path=None, config=None, hybrid_mode=False
    ):
        """
        Initialize the integration.

        Args:
            vanta_core: VantaCore instance or None to create new one
            model_path: Path to pre-trained GRID-Former model
            config: Configuration dictionary for VantaCore
            hybrid_mode: Whether to enable hybrid mode
        """
        self.logger = logging.getLogger("GridFormer.VantaIntegration")
        self.hybrid_mode = hybrid_mode

        # Initialize VantaCore
        if vanta_core is not None:
            self.vanta_core = vanta_core
        else:
            self.logger.info("Initializing VantaCore...")
            self.vanta_core = VantaCore(config=config or {})

        # Initialize GRID-Former
        self.logger.info("Initializing GRID-Former...")
        self.grid_former = GRID_Former()

        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading pre-trained model from {model_path}")
            self.grid_former = GRID_Former.load_from_file(model_path)

        # Create connector
        self.logger.info("Creating GridFormerConnector...")
        self.connector = EnhancedGridFormerConnector(
            grid_former=self.grid_former,
            vanta_core=self.vanta_core,
            hybrid_mode=self.hybrid_mode,
            model_dir="./grid_former_models",
        )

    def initialize(self):
        """Initialize the integration."""
        self.logger.info("Initializing integration...")
        self.connector.initialize()

    def run_integration(self):
        """Run the integration process."""
        self.logger.info("Running integration...")
        self.connector.run_integration()

    def predict(self, input_grid):
        """
        Make a prediction using the integrated system.

        Args:
            input_grid: Input grid to make prediction for

        Returns:
            Prediction result
        """
        if self.connector is None:
            self.logger.error("Cannot predict: connector not initialized")
            return None

        return self.connector.predict(input_grid)


def main():
    """Main entry point for integration script."""
    # Parse command-line arguments
    args = parse_arguments()

    # Log startup
    logger.info("Starting VantaCore + GRID-Former integration")

    try:
        # Initialize VantaCore
        logger.info("Initializing VantaCore...")
        config_path = args.config_path

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                import json

                config = json.load(f)
        else:
            config = {}

        # Create integration
        integration = GridFormerVantaIntegration(
            vanta_core=None,
            model_path=args.model_path,
            config=config,
            hybrid_mode=args.hybrid_mode,
        )

        # Initialize and run integration
        integration.initialize()
        integration.run_integration()

        logger.info("Integration complete!")
        return 0

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
