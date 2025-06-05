#!/usr/bin/env python
"""
Simple GRID-Former Model Handler
Unified training and inference with consistent model saving/loading
"""

from math import e
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add the path to import GRID_Former
sys.path.append(str(Path(__file__).parent / "Voxsigil_Library" / "ARC" / "grid_former"))
sys.path.append(str(Path(__file__).parent / "notebooks"))

try:
    from grid_former import GRID_Former
except ImportError:
    logging.error("Failed to import GRID_Former directly. Trying alternative paths...")
    try:
        from Gridformer.core.grid_former import GRID_Former

        logging.info("Imported GRID_Former from Gridformer.core")
    except ImportError:
        logging.error("Failed to import GRID_Former. Ensure the path is correct.")
        raise ImportError("GRID_Former module not found. Check your installation.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleGridFormer")


class SimpleGridFormerHandler:
    """
    Simple, no-nonsense GRID-Former handler for training and inference.
    Focuses on clean model saving/loading that works everywhere.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_grid_size: int = 30,
        num_colors: int = 10,
        device: Optional[str] = None,
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store model config
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_grid_size": max_grid_size,
            "num_colors": num_colors,
        }

        # Create model
        self.model = GRID_Former(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=max_grid_size,
            num_colors=num_colors,
        ).to(self.device)

        logger.info(f"Created GRID-Former model on {self.device}")
        logger.info(f"Model config: {self.config}")

    def save_model_simple(self, filepath: str) -> None:
        """
        Save model in the SIMPLEST possible format.
        This will work with any inference script.
        """
        # Get the actual model (unwrap DataParallel if needed)
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model

        # Create the cleanest possible checkpoint
        checkpoint = {
            "model_state_dict": actual_model.state_dict(),  # Clean state dict
            "config": self.config,  # Model architecture config
            "metadata": self.config,  # Backward compatibility
        }

        # Save to file
        torch.save(checkpoint, filepath)
        logger.info(f"✅ Saved clean model to: {filepath}")
        logger.info(f"Model config: {self.config}")

    def load_model_simple(self, filepath: str) -> None:
        """
        Load model from the simple format.
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Get config if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            logger.info(f"Loaded config: {config}")
        elif "metadata" in checkpoint:
            config = checkpoint["metadata"]
            logger.info(f"Loaded metadata as config: {config}")
        else:
            logger.warning("No config found, using current model")
            config = self.config

        # Create new model with loaded config if different
        if config != self.config:
            logger.info("Config mismatch, creating new model with loaded config")
            self.config = config
            self.model = GRID_Former(**config).to(self.device)

        # Load state dict
        state_dict = checkpoint["model_state_dict"]

        # Handle DataParallel prefixes
        if any(k.startswith("module.") for k in state_dict.keys()):
            logger.info("Removing 'module.' prefix from state dict")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load the state dict
        self.model.load_state_dict(state_dict)
        logger.info(f"✅ Loaded model from: {filepath}")

    def train_simple(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        save_path: str = "simple_grid_former.pt",
    ) -> Dict[str, List[float]]:
        """
        Simple training loop with automatic model saving.
        """
        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_dataloader):
                input_grid = batch["input"].to(self.device)
                output_grid = batch["output"].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output_logits = self.model(input_grid)

                # Compute loss
                B, H, W, C = output_logits.shape
                loss = criterion(
                    output_logits.reshape(B * H * W, C), output_grid.reshape(B * H * W)
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.6f}"
                    )

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_grid = batch["input"].to(self.device)
                    output_grid = batch["output"].to(self.device)

                    output_logits = self.model(input_grid)
                    B, H, W, C = output_logits.shape
                    loss = criterion(
                        output_logits.reshape(B * H * W, C),
                        output_grid.reshape(B * H * W),
                    )
                    val_loss += loss.item()

            # Calculate averages
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_simple(save_path)
                logger.info(f"✅ New best model saved! Val Loss: {val_loss:.6f}")

        logger.info(f"🎉 Training complete! Best model saved to: {save_path}")
        return history

    def predict_simple(
        self, input_grid: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Simple prediction method.
        """
        self.model.eval()

        # Handle DataParallel
        model = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        # Add batch dimension if needed
        if len(input_grid.shape) == 2:
            input_grid = input_grid.unsqueeze(0)

        input_grid = input_grid.to(self.device)

        with torch.no_grad():
            if hasattr(model, "predict_grid_transformation"):
                predictions = model.predict_grid_transformation(
                    input_grid, target_shape
                )
            else:
                # Fallback
                output_logits = model(input_grid, target_shape)
                predictions = torch.argmax(output_logits, dim=3)

        return predictions


def create_kaggle_ready_model(
    model_path: str, kaggle_output_path: str = "grid_former_kaggle.pt"
) -> None:
    """
    Create a Kaggle-ready model file from a trained model.
    This ensures maximum compatibility.
    """
    logger.info(f"Creating Kaggle-ready model from: {model_path}")

    # Load the model
    handler = SimpleGridFormerHandler()
    handler.load_model_simple(model_path)

    # Save in the cleanest possible format
    handler.save_model_simple(kaggle_output_path)

    logger.info(f"✅ Kaggle-ready model saved to: {kaggle_output_path}")
    logger.info("This file is ready to upload to Kaggle!")


if __name__ == "__main__":
    # Example usage
    handler = SimpleGridFormerHandler()

    # For training:
    # history = handler.train_simple(train_dataloader, val_dataloader, num_epochs=10)

    # For creating Kaggle-ready model:
    # create_kaggle_ready_model("path/to/your/trained/model.pt", "kaggle_model.pt")

    print("✅ Simple GRID-Former handler ready!")
    print("Usage:")
    print("1. Train: handler.train_simple(train_dl, val_dl)")
    print("2. Save: handler.save_model_simple('model.pt')")
    print("3. Load: handler.load_model_simple('model.pt')")
    print("4. Predict: handler.predict_simple(input_grid)")
