#!/usr/bin/env python
"""
enhanced_grid_connector.py - Enhanced Integration between GRID-Former and VantaCore

Provides a connector class that allows VantaCore to use GRID-Former models
for ARC tasks, enabling meta-learning and knowledge transfer.
This enhanced version adds proper integration support and fixes connectivity issues.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Use standard path helper for imports
try:
    from utils.path_helper import add_project_root_to_path

    add_project_root_to_path()
except ImportError:
    # Fallback if path_helper isn't available
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
except ImportError:
    logger = logging.getLogger("VoxSigil.GRID-Former.Connector")
    logger.error("Failed to import VantaCore. Integration will not work properly.")
    VantaCore = None

# Import GRID-Former modules
try:
    # Try relative imports first (when imported as a module)
    from ARC.core.arc_data_processor import (
        ARCGridDataProcessor,
        visualize_grid,
    )
    from Gridformer.training.grid_model_trainer import (
        GridFormerTrainer,
    )

    from .grid_former import GRID_Former
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from ARC.core.arc_data_processor import ARCGridDataProcessor
    from Gridformer.core.grid_former import GRID_Former
    from Gridformer.training.grid_model_trainer import (
        GridFormerTrainer,
    )

logger = logging.getLogger("VoxSigil.GRID-Former.Connector")


@vanta_core_module(
    name="enhanced_grid_connector",
    subsystem="grid_processing",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="Enhanced GRID-Former connector for VantaCore integration with neural-symbolic grid processing",
    capabilities=["grid_neural_synthesis", "vanta_grid_integration", "model_connector", "arc_task_processing", "hybrid_reasoning"],
    cognitive_load=4.0,
    symbolic_depth=3,
    collaboration_patterns=["neural_symbolic_bridge", "model_integration", "grid_synthesis"]
)
class EnhancedGridFormerConnector(BaseCore):
    """
    Enhanced connector class for integrating GRID-Former with VantaCore.

    This class manages the interface between VantaCore's meta-learning system
    and the GRID-Former neural network models, with proper integration support.
    """

    def __init__(
        self,
        vanta_core: Any,
        config: Dict[str, Any],
        grid_former: Optional[GRID_Former] = None,
        model_dir: str = "./grid_former_models",
        default_model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_grid_size: int = 30,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        hybrid_mode: bool = False,
    ):
        """
        Initialize the connector.

        Args:
            vanta_core: VantaCore instance for HOLO-1.5 compliance
            config: Configuration dictionary for BaseCore
            grid_former: Existing GRID-Former instance or None to create new one
            model_dir: Directory for model storage
            default_model_path: Path to default model or None to create new one
            device: Device for model computation
            max_grid_size: Maximum grid size for padding
            hidden_dim: Hidden dimension for model architecture
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hybrid_mode: Whether to enable hybrid neural-symbolic mode
        """
        # Initialize BaseCore
        super().__init__(vanta_core, config)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)

        # Set directories
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Store Vanta Core reference
        self.vanta_core = vanta_core
        self.hybrid_mode = hybrid_mode
        self.initialized = False

        # Set configuration
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Create data processor
        self.processor = ARCGridDataProcessor(
            max_grid_size=max_grid_size, augment_data=False
        )

        # Track active models
        self.models: Dict[str, GRID_Former] = {}
        self.trainers: Dict[str, GridFormerTrainer] = {}
        self.default_model_id = "default"

        # Use provided grid_former or initialize default model
        if grid_former is not None:
            logger.info("Using provided GRID-Former model")
            self.models[self.default_model_id] = grid_former

            # Create trainer for the model
            self.trainers[self.default_model_id] = GridFormerTrainer(
                model=self.models[self.default_model_id],
                output_dir=str(self.model_dir),
                device=self.device,
            )
        else:
            # Load or create default model
            self._initialize_default_model(default_model_path)

    def _initialize_default_model(self, model_path: Optional[str]) -> None:
        """
        Initialize the default model.

        Args:
            model_path: Path to model or None to create new one
        """
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading default model from {model_path}")
            self.models[self.default_model_id] = GRID_Former.load_from_file(
                model_path, str(self.device)
            )
        else:
            logger.info("Creating new default model")
            self.models[self.default_model_id] = GRID_Former(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                max_grid_size=self.max_grid_size,
            ).to(self.device)

        # Create trainer for the default model
        self.trainers[self.default_model_id] = GridFormerTrainer(
            model=self.models[self.default_model_id],
            output_dir=str(self.model_dir),
            device=self.device,
        )    
    
    async def initialize(self) -> bool:
        """
        Initialize the integration with VantaCore.

        This method sets up the necessary connections between
        the GRID-Former model and VantaCore's meta-learning system.

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        if self.vanta_core is None:
            logger.warning("Cannot initialize: VantaCore not provided")
            return False

        try:
            logger.info("Initializing GRID-Former connector with VantaCore")

            # Register GRID-Former models with VantaCore
            if hasattr(self.vanta_core, "register_model"):
                for model_id, model in self.models.items():
                    self.vanta_core.register_model(
                        model_id=f"grid_former_{model_id}",
                        model_type="neural_grid",
                        model_interface=self,
                        description=f"GRID-Former model for ARC tasks ({model_id})",
                    )
                logger.info(
                    f"Registered {len(self.models)} GRID-Former models with VantaCore"
                )
            else:
                logger.warning("VantaCore does not support model registration")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def run_integration(self) -> bool:
        """
        Run the integration process between GRID-Former and VantaCore.

        This method performs the active integration process, enabling
        bi-directional learning transfer between the systems.

        Returns:
            bool: True if integration succeeded, False otherwise
        """
        if not self.initialized:
            logger.warning("Cannot run integration: not initialized")
            if self.vanta_core is not None:
                logger.info("Attempting to initialize first...")
                if not self.initialize():
                    return False
            else:
                return False

        try:
            logger.info("Running GRID-Former integration with VantaCore")

            # Set up event handlers for bi-directional learning
            if hasattr(self.vanta_core, "subscribe_to_learning_events"):
                self.vanta_core.subscribe_to_learning_events(
                    event_type="new_reasoning_pattern",
                    callback=self._handle_reasoning_pattern,
                )
                logger.info("Subscribed to VantaCore reasoning pattern events")

            # Trigger initial knowledge sharing
            self._share_grid_former_knowledge()

            logger.info("Integration completed successfully")
            return True

        except Exception as e:
            logger.error(f"Integration failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _handle_reasoning_pattern(self, pattern_data):
        """Handle new reasoning patterns from VantaCore."""
        logger.info("Received new reasoning pattern from VantaCore")
        # Implement pattern adaptation logic here
        pass

    def _share_grid_former_knowledge(self):
        """Share GRID-Former knowledge with VantaCore."""
        logger.info("Sharing GRID-Former knowledge with VantaCore")
        # Implement knowledge sharing logic here
        pass

    def predict(
        self,
        input_grid: Union[List[List[int]], np.ndarray, torch.Tensor],
        target_shape: Optional[Tuple[int, int]] = None,
        model_id: str = "default",
    ) -> np.ndarray:
        """
        Generate a prediction for an input grid.

        Args:
            input_grid: Input grid data
            target_shape: Target shape for output grid or None to infer
            model_id: ID of model to use

        Returns:
            Predicted output grid as numpy array
        """
        # Ensure model exists
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found, using default model")
            model_id = self.default_model_id

        # Get the model
        model = self.models[model_id]

        # Convert input to tensor if needed
        if isinstance(input_grid, list):
            input_grid = np.array(input_grid)
        if isinstance(input_grid, np.ndarray):
            input_tensor = torch.tensor(input_grid, dtype=torch.long).unsqueeze(
                0
            )  # Add batch dimension
        else:
            input_tensor = input_grid
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Ensure input grid fits in max size
        h, w = input_grid.shape[-2:]  # Get height and width
        if h > self.max_grid_size or w > self.max_grid_size:
            raise ValueError(
                f"Input grid size {h}x{w} exceeds maximum size {self.max_grid_size}"
            )

        # Pad input if needed
        if h < self.max_grid_size or w < self.max_grid_size:
            # Create padded tensor
            padded = torch.full(
                (1, self.max_grid_size, self.max_grid_size),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            padded[0, :h, :w] = input_tensor[0, :h, :w]
            input_tensor = padded

        # Move to device
        input_tensor = input_tensor.to(self.device)

        # Generate prediction
        model.eval()
        with torch.no_grad():
            # Generate prediction
            output_logits = model(input_tensor, target_shape)
            predictions = torch.argmax(output_logits, dim=3).squeeze(
                0
            )  # Remove batch dimension

        # Convert to numpy
        output_grid = predictions.cpu().numpy()

        # Extract the actual output grid of the target shape or the size of the input
        if target_shape:
            out_h, out_w = target_shape
        else:
            out_h, out_w = h, w

        output_grid = output_grid[:out_h, :out_w]

        return output_grid
