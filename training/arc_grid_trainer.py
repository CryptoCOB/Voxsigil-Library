#!/usr/bin/env python
"""
arc_grid_trainer.py - VantaCore + GRID-Former ARC Training Integration

This module integrates VantaCore's meta-learning capabilities with the GRID-Former
neural architecture for training on the ARC dataset. It provides a comprehensive
training pipeline that combines:
- VantaCore's task adaptation profiles and cross-task knowledge indexing
- GRID-Former's transformer-based 2D grid pattern recognition
- MetaConsciousness ART training integration for pattern learning
- Persistent training state management and metrics tracking
"""

import json
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Voxsigil_Library.voxsigil_supervisor.vanta.Vanta_Core import VantaCore

# Import GRID-Former components
try:
    # Import main GRID-Former model
    from Voxsigil_Library.ARC.core.arc_data_processor import (
        ARCGridDataProcessor,
        create_arc_dataloaders,
    )
    from Voxsigil_Library.Gridformer.core.grid_former import GRID_Former
    from Voxsigil_Library.Gridformer.core.vantacore_grid_connector import (
        GridFormerConnector,
    )
    from Voxsigil_Library.Gridformer.training.grid_model_trainer import (
        GridFormerTrainer,
    )
    from Voxsigil_Library.Gridformer.training.grid_sigil_handler import GridSigilHandler

    logging.info("Successfully imported GRID-Former components")
except ImportError as e:
    logging.warning(f"GRID-Former components not available: {e}")
    GRID_Former = None
    GridFormerTrainer = None
    GridFormerConnector = None
    GridSigilHandler = None
    create_arc_dataloaders = None
    ARCGridDataProcessor = None

# Set to False as MetaConsciousness is not available
ARTController = None
ARTTrainer = None
EnhancedMetaConsciousness = None
METACONSCIOUSNESS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaCore.ARCGridTrainer")


class VantaGridFormerBridge(nn.Module):
    """
    Bridge component that integrates VantaCore with GRID-Former.
    Handles pattern encoding, attention mechanisms, and meta-learning feedback.
    Inherits from nn.Module to support parameter optimization.
    """

    def __init__(self, vanta_core: VantaCore, grid_former: Any, config: Dict[str, Any]):
        super(VantaGridFormerBridge, self).__init__()
        self.vanta_core = vanta_core
        self.grid_former = grid_former
        self.config = config

        # Pattern encoding dimensions
        self.grid_feature_dim = config.get("grid_feature_dim", 256)
        self.meta_feature_dim = config.get("meta_feature_dim", 128)

        # Create projection layers
        self.grid_to_meta_projection = nn.Linear(
            self.grid_feature_dim, self.meta_feature_dim
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.grid_to_meta_projection.weight)
        nn.init.zeros_(self.grid_to_meta_projection.bias)

        # Training metrics
        self.training_metrics = {
            "pattern_recognition_accuracy": [],
            "meta_learning_convergence": [],
            "cross_task_transfer": [],
        }

    def encode_grid_pattern(self, grid_data: torch.Tensor) -> torch.Tensor:
        """Extract pattern features from grid using GRID-Former encoder."""
        try:
            with torch.no_grad():
                # Safety check for grid_former
                if self.grid_former is None:
                    logger.warning(
                        "grid_former is None in encode_grid_pattern - returning zeros"
                    )
                    dummy_features = torch.zeros(
                        grid_data.size(0) if grid_data.dim() > 1 else 1,
                        self.grid_feature_dim,
                        device=grid_data.device
                        if hasattr(grid_data, "device")
                        else "cpu",
                    )
                    return self.grid_to_meta_projection(dummy_features)

                # Safety check for tensor
                if isinstance(self.grid_former, torch.Tensor):
                    logger.warning(
                        "grid_former is a tensor in encode_grid_pattern - using fallback"
                    )
                    # If it's somehow a tensor already, just use it directly
                    dummy_features = torch.ones(
                        grid_data.size(0) if grid_data.dim() > 1 else 1,
                        self.grid_feature_dim,
                        device=grid_data.device
                        if hasattr(grid_data, "device")
                        else "cpu",
                    )
                    return self.grid_to_meta_projection(dummy_features)

                # Handle DataParallel wrapper - use helper method if available
                if hasattr(self, "_get_model_module"):
                    try:
                        actual_model = self._get_model_module(self.grid_former)
                    except Exception as e:
                        logger.error(f"Error in _get_model_module: {e}")
                        actual_model = self.grid_former
                else:
                    # Fallback if _get_model_module is not defined
                    actual_model = (
                        self.grid_former.module
                        if hasattr(self.grid_former, "module")
                        else self.grid_former
                    )

                # Use GRID-Former's pattern recognition layers with error handling
                try:
                    if hasattr(actual_model, "pattern_layers"):
                        features = actual_model.pattern_layers(grid_data)
                    elif hasattr(actual_model, "grid_embedding"):
                        # Fallback to basic grid embedding
                        features = actual_model.grid_embedding(grid_data)
                    else:
                        # Final fallback
                        logger.warning(
                            "Neither pattern_layers nor grid_embedding found - using fallback"
                        )
                        features = torch.ones(
                            grid_data.size(0) if grid_data.dim() > 1 else 1,
                            self.grid_feature_dim,
                            device=grid_data.device
                            if hasattr(grid_data, "device")
                            else "cpu",
                        )
                except Exception as e:
                    logger.error(f"Error extracting features: {e}")
                    features = torch.ones(
                        grid_data.size(0) if grid_data.dim() > 1 else 1,
                        self.grid_feature_dim,
                        device=grid_data.device
                        if hasattr(grid_data, "device")
                        else "cpu",
                    )

            # Project to meta-learning space with error handling
            try:
                meta_features = self.grid_to_meta_projection(features)
                return meta_features
            except Exception as e:
                logger.error(f"Error in grid_to_meta_projection: {e}")
                return features  # Return unprojected features as fallback

        except Exception as e:
            logger.error(f"Unhandled error in encode_grid_pattern: {e}")
            # Return a dummy tensor as last resort
            return torch.zeros(
                grid_data.size(0) if grid_data.dim() > 1 else 1,
                self.meta_feature_dim if hasattr(self, "meta_feature_dim") else 128,
                device=grid_data.device if hasattr(grid_data, "device") else "cpu",
            )

    def forward(self, grid_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VantaGridFormerBridge.

        Args:
            grid_input: Grid pattern tensor

        Returns:
            Dictionary containing grid features and meta-learning features
        """
        # Extract grid pattern features using GRID-Former
        grid_features = self.grid_former(grid_input)

        # Extract meta features for VantaCore integration
        meta_features = self.encode_grid_pattern(grid_input)

        # Combine into output dictionary
        output = {
            "grid_features": grid_features,
            "meta_features": meta_features,
        }

        return output

    def _get_model_module(self, model):
        """Helper to extract the actual model from wrapper classes."""
        if hasattr(model, "module"):
            return model.module
        return model


class ARCGridTrainer:
    """
    Main trainer class for ARC grid-based problem-solving using GRID-Former and VantaCore.
    Handles the full training pipeline, data management, and integration with MetaConsciousness.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vanta_core: Optional[VantaCore] = None,
        grid_former: Optional[Any] = None,
    ):
        """
        Initialize the ARC Grid Trainer with VantaCore and GRID-Former.

        Args:
            config: Configuration dictionary
            vanta_core: Optional VantaCore instance
            grid_former: Optional GRID-Former model instance
        """
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and config.get("use_cuda", True)
            else "cpu"
        )

        # Load or create VantaCore
        self.vanta_core = vanta_core or self._initialize_vanta_core()

        # Load or create GRID-Former
        self.grid_former = grid_former or self._initialize_grid_former()

        # Create bridge component
        self.bridge = VantaGridFormerBridge(
            vanta_core=self.vanta_core,
            grid_former=self.grid_former,
            config=config,
        ).to(self.device)

        # Setup optimizers
        self.optimizer = self._setup_optimizer()

        # Initialize data processor
        self.data_processor = ARCGridDataProcessor(
            data_path=config.get("arc_data_path", "./arc_data"),
            grid_size=config.get("grid_size", 30),
        )

        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []

        # Initialize MetaConsciousness if available
        self.art_controller = None
        self.metaconsciousness = None
        if METACONSCIOUSNESS_AVAILABLE and config.get("use_metaconsciousness", False):
            self._initialize_metaconsciousness()

    def _initialize_vanta_core(self) -> VantaCore:
        """Initialize and configure VantaCore instance."""
        vanta_config = self.config.get("vanta_config", {})
        vanta_core = VantaCore(
            config=vanta_config,
            device=self.device,
            initialize_subsystems=True,
        )
        return vanta_core

    def _initialize_grid_former(self) -> Any:
        """Initialize and configure GRID-Former model."""
        if GRID_Former is None:
            logger.warning("GRID-Former is not available")
            return None

        grid_config = self.config.get("grid_former_config", {})
        grid_former = GRID_Former(
            grid_size=grid_config.get("grid_size", 30),
            embedding_dim=grid_config.get("embedding_dim", 256),
            num_layers=grid_config.get("num_layers", 6),
            num_heads=grid_config.get("num_heads", 8),
            dropout=grid_config.get("dropout", 0.1),
        ).to(self.device)

        return grid_former

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for training."""
        # Combine parameters from both models and bridge
        params = list(self.bridge.parameters())

        if self.grid_former is not None:
            params.extend(list(self.grid_former.parameters()))

        # Create optimizer with configured settings
        optimizer = optim.Adam(
            params,
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        return optimizer

    def _initialize_metaconsciousness(self) -> None:
        """Initialize MetaConsciousness components if available."""
        if not METACONSCIOUSNESS_AVAILABLE:
            logger.warning("MetaConsciousness components not available")
            return

        try:
            # Create ART controller
            art_config = self.config.get("art_config", {})
            self.art_controller = ARTController(
                config=art_config,
                device=self.device,
            )

            # Initialize enhanced metaconsciousness
            self.metaconsciousness = EnhancedMetaConsciousness(
                art_controller=self.art_controller,
                vanta_core=self.vanta_core,
                config=self.config.get("metaconsciousness_config", {}),
            )

            # Connect grid former to metaconsciousness
            self.metaconsciousness.register_external_model(
                model=self.grid_former,
                model_type="grid_pattern_processor",
            )

            logger.info("MetaConsciousness initialized and connected")
        except Exception as e:
            logger.error(f"Failed to initialize MetaConsciousness: {e}")
            self.art_controller = None
            self.metaconsciousness = None

    def train(
        self,
        num_epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the integrated model on ARC grid pattern data.

        Args:
            num_epochs: Number of training epochs
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training metrics and history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Training metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "best_accuracy": 0.0,
            "convergence_epoch": None,
        }

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            train_metrics = self._train_epoch(train_dataloader)
            metrics["train_loss"].append(train_metrics["loss"])

            # Validation
            if val_dataloader is not None:
                val_metrics = self._validate(val_dataloader)
                metrics["val_loss"].append(val_metrics["loss"])
                metrics["accuracy"].append(val_metrics["accuracy"])

                # Check for best model
                if val_metrics["accuracy"] > metrics["best_accuracy"]:
                    metrics["best_accuracy"] = val_metrics["accuracy"]

                    # Save checkpoint
                    if checkpoint_dir:
                        self.save_checkpoint(
                            os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
                        )

            # Regular checkpoint
            if (
                checkpoint_dir
                and (epoch + 1) % self.config.get("checkpoint_interval", 10) == 0
            ):
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                )

            # Update training history
            self.training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"] if val_dataloader else None,
                    "accuracy": val_metrics["accuracy"] if val_dataloader else None,
                }
            )

            # Early stopping check
            if self._check_early_stopping(metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                metrics["convergence_epoch"] = epoch
                break

        logger.info("Training complete")
        return metrics

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for a single epoch."""
        self.bridge.train()
        if self.grid_former is not None:
            self.grid_former.train()

        total_loss = 0.0
        samples_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            # Process batch data
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.grid_former(inputs)

            # Calculate loss
            loss = self._compute_loss(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            samples_processed += batch_size

            # Log progress
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                logger.info(
                    f"Train Epoch: {self.current_epoch + 1} [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # Calculate average loss
        avg_loss = total_loss / samples_processed

        return {"loss": avg_loss}

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.bridge.eval()
        if self.grid_former is not None:
            self.grid_former.eval()

        total_loss = 0.0
        correct = 0
        samples_processed = 0

        with torch.no_grad():
            for batch in dataloader:
                # Process batch data
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.grid_former(inputs)

                # Calculate loss
                loss = self._compute_loss(outputs, targets)

                # Calculate accuracy
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()

                # Update metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                samples_processed += batch_size

        # Calculate average metrics
        avg_loss = total_loss / samples_processed
        accuracy = 100.0 * correct / samples_processed

        logger.info(
            f"Validation: Average loss: {avg_loss:.4f}, "
            f"Accuracy: {correct}/{samples_processed} ({accuracy:.2f}%)"
        )

        return {"loss": avg_loss, "accuracy": accuracy}

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss with optional metaconsciousness integration."""
        # Base loss calculation
        criterion = nn.CrossEntropyLoss()
        base_loss = criterion(outputs, targets)

        # Metaconsciousness integration for enhanced loss signal
        if self.metaconsciousness is not None:
            try:
                meta_loss = self.metaconsciousness.compute_enhanced_loss(
                    base_loss=base_loss,
                    outputs=outputs,
                    targets=targets,
                    model=self.grid_former,
                )
                return meta_loss
            except Exception as e:
                logger.error(f"Error in metaconsciousness loss calculation: {e}")
                return base_loss

        return base_loss

    def _check_early_stopping(self, metrics: Dict[str, Any]) -> bool:
        """Check if early stopping criteria are met."""
        if len(metrics["val_loss"]) < self.config.get("patience", 10):
            return False

        # Check for plateau in validation loss
        patience = self.config.get("patience", 10)
        min_delta = self.config.get("min_delta", 1e-4)

        recent_losses = metrics["val_loss"][-patience:]
        best_recent_loss = min(recent_losses)
        first_loss = recent_losses[0]

        # Stop if not enough improvement
        return abs(first_loss - best_recent_loss) < min_delta

    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "grid_former_state_dict": self.grid_former.state_dict()
            if self.grid_former
            else None,
            "bridge_state_dict": self.bridge.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
            "training_history": self.training_history,
            "config": self.config,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint file {filepath} not found")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model states
        if self.grid_former and checkpoint["grid_former_state_dict"]:
            self.grid_former.load_state_dict(checkpoint["grid_former_state_dict"])

        self.bridge.load_state_dict(checkpoint["bridge_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training state
        self.current_epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"Checkpoint loaded from {filepath} (epoch {self.current_epoch})")

    def predict(self, grid_data: torch.Tensor) -> torch.Tensor:
        """Generate predictions for grid data."""
        self.bridge.eval()
        if self.grid_former is not None:
            self.grid_former.eval()

        with torch.no_grad():
            grid_data = grid_data.to(self.device)
            predictions = self.grid_former(grid_data)

        return predictions


def main():
    """Main entry point for standalone execution."""
    import argparse
    import os

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ARC Grid Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="config/arc_training_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_path", type=str, default="./arc_data", help="Path to ARC dataset"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = {}

    # Override with command-line arguments
    config["arc_data_path"] = args.data_path

    # Create trainer
    trainer = ARCGridTrainer(config=config)

    # Create data loaders
    train_loader, val_loader = create_arc_dataloaders(
        data_path=config.get("arc_data_path", "./arc_data"),
        batch_size=config.get("batch_size", 32),
        val_split=config.get("val_split", 0.2),
    )

    # Train model
    trainer.train(
        num_epochs=args.epochs,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
