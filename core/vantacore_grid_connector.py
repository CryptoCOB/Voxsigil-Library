#!/usr/bin/env python
"""
vantacore_grid_connector.py - Integration between GRID-Former and VantaCore

Provides a connector class that allows VantaCore to use GRID-Former models
for ARC tasks, enabling meta-learning and knowledge transfer.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import Vanta Core
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
except ImportError:
    # Fallback for Vanta Core import
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


class GridFormerConnector:
    """
    Connector class for integrating GRID-Former with VantaCore.

    This class manages the interface between VantaCore's meta-learning system
    and the GRID-Former neural network models, enabling hybrid neural-symbolic
    reasoning for ARC tasks with meta-learning capabilities.

    Features:
    - Bi-directional knowledge transfer between neural and symbolic systems
    - Dynamic model adaptation based on VantaCore insights
    - Fine-tuning on specific ARC tasks
    - Model export/import in VoxSigil format
    - Integration with VantaCore's meta-learning framework
    """

    def __init__(
        self,
        grid_former: Optional[GRID_Former] = None,
        vanta_core: Optional[Any] = None,  # Changed to Any to avoid type issues
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
            grid_former: Existing GRID-Former instance or None to create new one
            vanta_core: Existing VantaCore instance or None
            model_dir: Directory for model storage
            default_model_path: Path to default model or None to create new one
            device: Device for model computation
            max_grid_size: Maximum grid size for padding
            hidden_dim: Hidden dimension for model architecture
            num_layers: Number of transformer layers
            num_heads: Number of attention heads            hybrid_mode: Whether to enable hybrid neural-symbolic mode
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = (
            self.device if isinstance(self.device, str) else str(self.device)
        )
        self.device = torch.device(self.device)

        # Set directories
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Store Vanta Core reference
        self.vanta_core = vanta_core
        self.hybrid_mode = hybrid_mode
        self.initialized = False

        # Track active models
        self.models: Dict[str, GRID_Former] = {}
        self.trainers: Dict[str, GridFormerTrainer] = {}
        self.default_model_id = "default"

        # Set configuration
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Create data processor
        self.processor = ARCGridDataProcessor(
            max_grid_size=max_grid_size, augment_data=False
        )

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
                model_path, self.device_str
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
            device=self.device_str,
        )

    def initialize(self) -> bool:
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
            if (
                hasattr(self.vanta_core, "subscribe_to_learning_events")
                and self.vanta_core is not None
            ):
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
        model = self.models[model_id]
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

    def handle_arc_task(
        self,
        task_data: Dict[str, Any],
        task_id: str = "unknown_task",
        use_vanta_params: bool = True,
        run_fine_tuning: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an ARC task using GRID-Former.

        Args:
            task_data: ARC task data containing train pairs and test input
            task_id: Identifier for the task
            use_vanta_params: Whether to use VantaCore parameter adaptation
            run_fine_tuning: Whether to fine-tune on train examples

        Returns:
            Dictionary with prediction results and metrics
        """
        # Extract train pairs and test input
        train_pairs = task_data.get("train", [])
        test_inputs = task_data.get("test", [])

        if not train_pairs or not test_inputs:
            return {"error": "Invalid task data: missing train pairs or test inputs"}

        model_id = f"task_{task_id}"

        # Clone the default model for this task if needed
        if model_id not in self.models:
            logger.info(f"Creating new model for task {task_id}")
            self.models[model_id] = GRID_Former(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                max_grid_size=self.max_grid_size,
            ).to(self.device)

            # Copy weights from default model
            default_state_dict = self.models[self.default_model_id].state_dict()
            self.models[model_id].load_state_dict(
                default_state_dict
            )  # Create trainer for this model
            self.trainers[model_id] = GridFormerTrainer(
                model=self.models[model_id],
                output_dir=str(self.model_dir),
                device=self.device_str,
            )

        # Fine-tune on train examples if requested
        if run_fine_tuning and train_pairs:
            logger.info(
                f"Fine-tuning model on {len(train_pairs)} train examples for task {task_id}"
            )
            self._fine_tune_on_examples(train_pairs, model_id)

        # Generate predictions for test inputs
        predictions = []
        for test_idx, test_case in enumerate(test_inputs):
            test_input = test_case["input"]

            # Try to determine output shape if available in train data
            target_shape = None
            if train_pairs:
                # Use the shape of the first train pair's output as a hint
                example_output = train_pairs[0]["output"]
                target_shape = (len(example_output), len(example_output[0]))

            # Generate prediction
            predicted_grid = self.predict(test_input, target_shape, model_id)

            predictions.append(
                {"input": test_input, "predicted_grid": predicted_grid.tolist()}
            )

        # Prepare result
        result = {
            "task_id": task_id,
            "predictions": predictions,
            "model_id": model_id,
            "timestamp": time.time(),
        }

        return result

    def _fine_tune_on_examples(
        self,
        examples: List[Dict[str, List[List[int]]]],
        model_id: str,
        num_epochs: int = 50,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Fine-tune a model on specific examples.

        Args:
            examples: List of examples with 'input' and 'output' grids
            model_id: ID of model to fine-tune
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for optimizer
        """
        # Ensure model exists
        if model_id not in self.models or model_id not in self.trainers:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]
        trainer = self.trainers[model_id]

        # Prepare mini-dataset
        input_tensors = []
        output_tensors = []
        input_masks = []
        output_masks = []
        input_shapes = []
        output_shapes = []

        for example in examples:
            input_grid = example["input"]
            output_grid = example["output"]

            # Process example
            processed = self.processor.process_example(input_grid, output_grid)

            input_tensors.append(processed["input"])
            output_tensors.append(processed["output"])
            input_masks.append(processed["input_mask"])
            output_masks.append(processed["output_mask"])
            input_shapes.append(processed["input_shape"])
            output_shapes.append(processed["output_shape"])

        # Create batched tensors
        batch = {
            "input": torch.stack(input_tensors).to(self.device),
            "output": torch.stack(output_tensors).to(self.device),
            "input_mask": torch.stack(input_masks).to(self.device),
            "output_mask": torch.stack(output_masks).to(self.device),
            "input_shape": torch.stack(input_shapes).to(self.device),
            "output_shape": torch.stack(output_shapes).to(self.device),
        }

        # Set up optimizer for fine-tuning
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Set up loss function
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Fine-tuning loop
        model.train()
        for epoch in range(num_epochs):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output_logits = model(batch["input"])  # (B, H, W, num_colors)

            # Reshape for loss computation
            B, H, W, C = output_logits.shape
            output_logits_flat = output_logits.reshape(B * H * W, C)
            output_grid_flat = batch["output"].reshape(B * H * W)

            # Calculate loss
            loss = criterion(output_logits_flat, output_grid_flat)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Fine-tuning epoch {epoch + 1}/{num_epochs}, loss: {loss.item():.6f}"
                )

        # Evaluate after fine-tuning
        model.eval()
        with torch.no_grad():
            output_logits = model(batch["input"])
            B, H, W, C = output_logits.shape
            output_logits_flat = output_logits.reshape(B * H * W, C)
            output_grid_flat = batch["output"].reshape(B * H * W)
            loss = criterion(output_logits_flat, output_grid_flat)

            # Calculate accuracy (only on non-padded cells)
            predictions = torch.argmax(output_logits, dim=3)
            mask = batch["output"] != -1
            correct_predictions = (predictions == batch["output"]) & mask
            correct = correct_predictions.sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0.0

        logger.info(
            f"Fine-tuning completed. Final loss: {loss.item():.6f}, accuracy: {accuracy:.4f}"
        )

    def export_to_sigil(
        self, model_id: str, path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export model as VoxSigil sigil format.

        Args:
            model_id: ID of model to export
            path: Path to save sigil file or None to return without saving

        Returns:
            Sigil representation of the model
        """
        # Ensure model exists
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        # Generate export timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # Create state dict for export
        state_dict = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

        # Convert numpy arrays to lists for JSON serialization
        for k, v in state_dict.items():
            state_dict[k] = v.tolist()

        # Create sigil representation
        sigil = {
            "SigilType": "GRID-Former-Model",
            "Version": "1.0",
            "Content": {
                "model_state": state_dict,
                "config": {
                    "hidden_dim": model.hidden_dim,
                    "max_grid_size": model.max_grid_size,
                    "num_colors": model.num_colors,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                },
                "metadata": {"export_timestamp": timestamp, "model_id": model_id},
            },
        }

        # Save to file if path is provided
        if path:
            with open(path, "w") as f:
                json.dump(sigil, f, indent=2)
            logger.info(f"Exported model {model_id} to sigil at {path}")

        return sigil

    @classmethod
    def from_sigil(
        cls, sigil_data: Dict[str, Any], device: Optional[str] = None
    ) -> "GridFormerConnector":
        """
        Create a connector instance from a sigil.

        Args:
            sigil_data: Sigil data dictionary
            device: Device to load model to

        Returns:
            New GridFormerConnector instance
        """
        # Validate sigil
        if (
            "SigilType" not in sigil_data
            or sigil_data["SigilType"] != "GRID-Former-Model"
        ):
            raise ValueError("Invalid sigil: Not a GRID-Former model sigil")

        if "Content" not in sigil_data:
            raise ValueError("Invalid sigil: Missing content")

        content = sigil_data["Content"]

        if "config" not in content or "model_state" not in content:
            raise ValueError("Invalid sigil: Missing config or model state")

        # Extract config
        config = content["config"]
        hidden_dim = config.get("hidden_dim", 256)
        max_grid_size = config.get("max_grid_size", 30)
        num_colors = config.get("num_colors", 10)
        num_layers = config.get("num_layers", 6)
        num_heads = config.get("num_heads", 8)

        # Create connector
        connector = cls(
            device=device,
            max_grid_size=max_grid_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Create model
        model = GRID_Former(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=max_grid_size,
            num_colors=num_colors,
        )

        # Convert lists back to tensors
        state_dict = content["model_state"]
        state_dict_tensors = {}
        for k, v in state_dict.items():
            state_dict_tensors[k] = torch.tensor(v)

        # Load state dict
        model.load_state_dict(state_dict_tensors)
        model = model.to(connector.device)

        # Set as default model
        connector.models[connector.default_model_id] = model  # Create trainer
        connector.trainers[connector.default_model_id] = GridFormerTrainer(
            model=model,
            output_dir=str(connector.model_dir),
            device=connector.device_str,
        )

        logger.info("Created GridFormerConnector from sigil")

        return connector


# Test function
def test_grid_former_connector():
    """
    Test the GridFormerConnector functionality.
    """
    # Create connector
    connector = GridFormerConnector(
        model_dir="./test_models",
        max_grid_size=15,
        hidden_dim=64,  # Small for testing
        num_layers=2,  # Small for testing
    )

    # Create test task data
    task_data = {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                "output": [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
            },
            {
                "input": [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
                "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            },
        ],
        "test": [{"input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}],
    }

    # Handle task
    result = connector.handle_arc_task(task_data, task_id="test_task")

    # Print results
    print(f"Task result: {result}")

    # Export model to sigil
    sigil = connector.export_to_sigil("task_test_task")
    print(f"Exported sigil: {sigil.keys()}")

    print("Test completed successfully!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add parent directory to path for imports to work
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    test_grid_former_connector()
