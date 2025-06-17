"""
Grid Sigil Handler for VoxSigil

Handles saving and loading of trained models as VoxSigil format files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GridSigilHandler:
    """
    Handler for converting trained models to VoxSigil format.
    """

    def __init__(self, output_dir: str = "sigils"):
        """
        Initialize the GridSigilHandler.

        Args:
            output_dir: Directory where sigil files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_model_to_sigil(
        self, model: Any, training_info: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model as a VoxSigil file.

        Args:
            model: The trained model to save
            training_info: Information about the training process
            metadata: Additional metadata to include

        Returns:
            Path to the saved sigil file
        """
        if metadata is None:
            metadata = {}

        # Create sigil filename based on model type and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = getattr(model, "__class__", type(model)).__name__
        filename = f"{model_type.lower()}_grid_{timestamp}.voxsigil"
        filepath = self.output_dir / filename

        # Prepare sigil data structure
        sigil_data = {
            "sigil_type": "trained_model",
            "model_type": model_type,
            "timestamp": timestamp,
            "training_info": training_info,
            "metadata": metadata,
            "model_state": self._extract_model_state(model),
            "voxsigil_version": "1.0",
        }

        # Save to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(sigil_data, f, indent=2, default=str)

            logger.info(f"Model saved to sigil: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save model to sigil: {e}")
            raise

    def load_model_from_sigil(self, sigil_path: str) -> Dict[str, Any]:
        """
        Load model data from a VoxSigil file.

        Args:
            sigil_path: Path to the sigil file

        Returns:
            Dictionary containing model data and metadata
        """
        try:
            with open(sigil_path, "r", encoding="utf-8") as f:
                sigil_data = json.load(f)

            logger.info(f"Model loaded from sigil: {sigil_path}")
            return sigil_data

        except Exception as e:
            logger.error(f"Failed to load model from sigil: {e}")
            raise

    def _extract_model_state(self, model: Any) -> Dict[str, Any]:
        """
        Extract relevant state from a model for storage.

        Args:
            model: The model to extract state from

        Returns:
            Dictionary containing model state information
        """
        state = {}

        # Try to get common model attributes
        if hasattr(model, "state_dict"):
            # PyTorch model
            try:
                # Convert tensors to lists for JSON serialization
                state_dict = model.state_dict()
                state["pytorch_state"] = {
                    k: v.tolist() if hasattr(v, "tolist") else str(v) for k, v in state_dict.items()
                }
            except Exception as e:
                logger.warning(f"Could not extract PyTorch state: {e}")
                state["pytorch_state"] = "extraction_failed"

        if hasattr(model, "get_weights"):
            # TensorFlow/Keras model
            try:
                weights = model.get_weights()
                state["keras_weights"] = [
                    w.tolist() if hasattr(w, "tolist") else str(w) for w in weights
                ]
            except Exception as e:
                logger.warning(f"Could not extract Keras weights: {e}")
                state["keras_weights"] = "extraction_failed"

        # Extract model architecture info
        if hasattr(model, "__dict__"):
            try:
                config = {
                    k: v
                    for k, v in model.__dict__.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                }
                state["model_config"] = config
            except Exception as e:
                logger.warning(f"Could not extract model config: {e}")

        # Add basic model info
        state["model_class"] = model.__class__.__name__
        state["model_module"] = getattr(model.__class__, "__module__", "unknown")

        return state

    def list_sigils(self) -> list[str]:
        """
        List all sigil files in the output directory.

        Returns:
            List of sigil file paths
        """
        return [str(p) for p in self.output_dir.glob("*.voxsigil")]

    def get_sigil_info(self, sigil_path: str) -> Dict[str, Any]:
        """
        Get metadata information from a sigil file without loading the full model.

        Args:
            sigil_path: Path to the sigil file

        Returns:
            Dictionary containing sigil metadata
        """
        try:
            with open(sigil_path, "r", encoding="utf-8") as f:
                sigil_data = json.load(f)

            # Return only metadata, not the full model state
            return {
                "sigil_type": sigil_data.get("sigil_type"),
                "model_type": sigil_data.get("model_type"),
                "timestamp": sigil_data.get("timestamp"),
                "training_info": sigil_data.get("training_info"),
                "metadata": sigil_data.get("metadata"),
                "voxsigil_version": sigil_data.get("voxsigil_version"),
            }

        except Exception as e:
            logger.error(f"Failed to get sigil info: {e}")
            raise


def create_grid_sigil_handler(output_dir: str = "sigils") -> GridSigilHandler:
    """
    Factory function to create a GridSigilHandler instance.

    Args:
        output_dir: Directory where sigil files will be saved

    Returns:
        GridSigilHandler instance
    """
    return GridSigilHandler(output_dir)
