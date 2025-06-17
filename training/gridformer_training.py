"""
GridFormer Training Components
=============================

Training components for GridFormer models that were missing from Gridformer.training.
These are now part of the main training module under Vanta control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ML Dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None

# Import existing training components
try:
    from .arc_grid_trainer import ARCGridTrainer

    HAVE_ARC_TRAINER = True
except ImportError:
    HAVE_ARC_TRAINER = False
    ARCGridTrainer = None

logger = logging.getLogger(__name__)


@dataclass
class GridFormerTrainingConfig:
    """Configuration for GridFormer training."""

    model_name: str = "gridformer_v1"
    input_size: int = 30
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 100
    device: str = "auto"
    checkpoint_dir: str = "./checkpoints"
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0


class GridFormerModel(nn.Module):
    """
    Basic GridFormer model for ARC tasks.

    This is a simplified version that can be extended with the full
    GridFormer architecture.
    """

    def __init__(self, config: GridFormerTrainingConfig):
        super().__init__()
        self.config = config

        if not HAVE_TORCH:
            raise RuntimeError("PyTorch is required for GridFormer training")

        # Simple grid processing layers
        self.input_embedding = nn.Linear(config.input_size * config.input_size, config.hidden_size)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=0.1,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_projection = nn.Linear(
            config.hidden_size, config.input_size * config.input_size
        )

    def forward(self, grid_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for grid processing.

        Args:
            grid_input: Input grid tensor of shape (batch, grid_size, grid_size)

        Returns:
            Output grid tensor of same shape
        """
        batch_size = grid_input.size(0)

        # Flatten grid for processing
        flattened = grid_input.view(batch_size, -1)

        # Embed input
        embedded = self.input_embedding(flattened)
        embedded = embedded.unsqueeze(1)  # Add sequence dimension

        # Process through transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded)

        # Project back to grid
        output = self.output_projection(embedded.squeeze(1))
        output = output.view(batch_size, self.config.input_size, self.config.input_size)

        return output


class GridFormerDataset(Dataset):
    """
    Dataset for GridFormer training.

    This should be replaced with actual ARC data loading.
    """

    def __init__(self, grid_size: int = 30, num_samples: int = 1000):
        self.grid_size = grid_size
        self.num_samples = num_samples

        if not HAVE_TORCH:
            raise RuntimeError("PyTorch is required for dataset")

        # Generate synthetic grid data for now
        self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic grid data for training."""
        data = []
        for _ in range(self.num_samples):
            # Create input grid with some pattern
            input_grid = torch.randint(0, 10, (self.grid_size, self.grid_size)).float()

            # Create target (for now, just a simple transformation)
            target_grid = torch.roll(input_grid, shifts=1, dims=0)

            data.append((input_grid, target_grid))

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class GridFormerTrainer:
    """
    GridFormer trainer that integrates with Vanta async training.

    This replaces the missing Gridformer.training.GridFormerTrainer.
    """

    def __init__(self, config: GridFormerTrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = self._setup_device()
        self.training_history = []

        logger.info(f"Initialized GridFormerTrainer with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch is required for training")

        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        return device

    def setup_model(self) -> GridFormerModel:
        """Setup the GridFormer model."""
        self.model = GridFormerModel(self.config)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Setup loss function
        self.criterion = nn.MSELoss()

        logger.info(
            f"Setup GridFormer model with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
        return self.model

    def setup_data(self, dataset: Optional[Dataset] = None) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders."""
        if dataset is None:
            dataset = GridFormerDataset(grid_size=self.config.input_size, num_samples=1000)

        # Split into train/val
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        if self.model is None:
            raise RuntimeError("Model not setup. Call setup_model() first.")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_grids, target_grids) in enumerate(train_loader):
            input_grids = input_grids.to(self.device)
            target_grids = target_grids.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_grids)
            loss = self.criterion(outputs, target_grids)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        if self.model is None:
            raise RuntimeError("Model not setup. Call setup_model() first.")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_grids, target_grids in val_loader:
                input_grids = input_grids.to(self.device)
                target_grids = target_grids.to(self.device)

                outputs = self.model(input_grids)
                loss = self.criterion(outputs, target_grids)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"val_loss": avg_loss}

    def train(self, dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and final metrics
        """
        logger.info("Starting GridFormer training")

        # Setup model and data
        self.setup_model()
        train_loader, val_loader = self.setup_data(dataset)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                **train_metrics,
                **val_metrics,
            }

            self.training_history.append(epoch_metrics)

            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )

            # Early stopping
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0

                # Save best model
                self.save_checkpoint(f"best_model_epoch_{epoch + 1}.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Final results
        results = {
            "training_history": self.training_history,
            "best_val_loss": best_val_loss,
            "total_epochs": len(self.training_history),
            "model_parameters": sum(p.numel() for p in self.model.parameters())
            if self.model
            else 0,
        }

        logger.info("GridFormer training completed")
        return results

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.model is None:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / filename

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "config": self.config,
                "training_history": self.training_history,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is None:
            self.setup_model()

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")


# For compatibility with existing code that expects these imports
def create_gridformer_trainer(config_dict: Dict[str, Any]) -> GridFormerTrainer:
    """Create GridFormer trainer from config dictionary."""
    config = GridFormerTrainingConfig(**config_dict)
    return GridFormerTrainer(config)


# Legacy compatibility functions
def train_gridformer_model(config: Dict[str, Any], dataset=None) -> Dict[str, Any]:
    """Train a GridFormer model (legacy interface)."""
    trainer = create_gridformer_trainer(config)
    return trainer.train(dataset)


__all__ = [
    "GridFormerTrainingConfig",
    "GridFormerModel",
    "GridFormerDataset",
    "GridFormerTrainer",
    "create_gridformer_trainer",
    "train_gridformer_model",
]
