#!/usr/bin/env python3
"""
Model Utilities for VoxSigil System
Provides model loading, discovery, and management functionality.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Advanced model loader for VoxSigil system.
    Handles loading, caching, and management of various model types.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(project_root / "models" / "cache")
        self.loaded_models = {}
        self.model_registry = {}
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self, model_path: Union[str, Path], model_type: str = "auto", **kwargs) -> Any:
        """
        Load a model from the specified path.

        Args:
            model_path: Path to model file or directory
            model_type: Type of model ('pytorch', 'tensorflow', 'onnx', 'auto')
            **kwargs: Additional loading parameters

        Returns:
            Loaded model object
        """
        model_path = Path(model_path)
        cache_key = f"{model_path}_{model_type}"

        # Return cached model if available
        if cache_key in self.loaded_models:
            logger.info(f"Returning cached model: {cache_key}")
            return self.loaded_models[cache_key]

        try:
            if model_type == "auto":
                model_type = self._detect_model_type(model_path)

            model = self._load_by_type(model_path, model_type, **kwargs)
            self.loaded_models[cache_key] = model
            logger.info(f"Successfully loaded model: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def _detect_model_type(self, model_path: Path) -> str:
        """Auto-detect model type based on file extension or directory structure."""
        if model_path.is_file():
            suffix = model_path.suffix.lower()
            if suffix in [".pt", ".pth"]:
                return "pytorch"
            elif suffix in [".pb", ".h5"]:
                return "tensorflow"
            elif suffix == ".onnx":
                return "onnx"
            elif suffix == ".json":
                return "config"
        elif model_path.is_dir():
            # Check for common model files in directory
            if (model_path / "pytorch_model.bin").exists() or (
                model_path / "model.safetensors"
            ).exists():
                return "pytorch"
            elif (model_path / "saved_model.pb").exists():
                return "tensorflow"

        return "generic"

    def _load_by_type(self, model_path: Path, model_type: str, **kwargs) -> Any:
        """Load model based on detected type."""
        if model_type == "pytorch":
            return self._load_pytorch_model(model_path, **kwargs)
        elif model_type == "tensorflow":
            return self._load_tensorflow_model(model_path, **kwargs)
        elif model_type == "onnx":
            return self._load_onnx_model(model_path, **kwargs)
        elif model_type == "config":
            return self._load_config(model_path, **kwargs)
        else:
            # Generic loader - try to load as JSON or return path
            if model_path.suffix == ".json":
                return self._load_config(model_path)
            return str(model_path)

    def _load_pytorch_model(self, model_path: Path, **kwargs) -> Any:
        """Load PyTorch model."""
        try:
            import torch

            if model_path.is_file():
                return torch.load(model_path, map_location="cpu")
            else:
                # Try loading from directory (transformers style)
                try:
                    from transformers import AutoModel

                    return AutoModel.from_pretrained(str(model_path))
                except ImportError:
                    logger.warning("Transformers not available, loading as generic model")
                    return str(model_path)
        except ImportError:
            logger.warning("PyTorch not available, returning path")
            return str(model_path)

    def _load_tensorflow_model(self, model_path: Path, **kwargs) -> Any:
        """Load TensorFlow model."""
        try:
            import tensorflow as tf

            return tf.keras.models.load_model(str(model_path))
        except ImportError:
            logger.warning("TensorFlow not available, returning path")
            return str(model_path)

    def _load_onnx_model(self, model_path: Path, **kwargs) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort

            return ort.InferenceSession(str(model_path))
        except ImportError:
            logger.warning("ONNXRuntime not available, returning path")
            return str(model_path)

    def _load_config(self, config_path: Path, **kwargs) -> Dict[str, Any]:
        """Load configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def register_model(self, name: str, model: Any, metadata: Optional[Dict] = None):
        """Register a model in the registry."""
        self.model_registry[name] = {
            "model": model,
            "metadata": metadata or {},
            "loaded_at": time.time(),
        }

    def get_registered_model(self, name: str) -> Optional[Any]:
        """Get a registered model by name."""
        entry = self.model_registry.get(name)
        return entry["model"] if entry else None

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_registry.keys())

    def clear_cache(self):
        """Clear the model cache."""
        self.loaded_models.clear()
        logger.info("Model cache cleared")

    def discover_models(
        self, search_paths: Optional[List[Union[str, Path]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Discover models in specified paths.

        Args:
            search_paths: List of paths to search for models

        Returns:
            Dictionary with model paths as keys and metadata as values
        """
        # Call the module-level function
        models_list = discover_models_standalone(search_paths)
        # Convert list to dictionary format expected by GUI
        models_dict = {}
        for model_info in models_list:
            path = model_info["path"]
            metadata = {k: v for k, v in model_info.items() if k != "path"}
            models_dict[path] = metadata
        return models_dict


def discover_models_standalone(
    search_paths: Optional[List[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """
    Discover models in specified paths.

    Args:
        search_paths: List of paths to search for models

    Returns:
        List of discovered model information
    """
    if search_paths is None:
        search_paths = [
            project_root / "models",
            project_root / "checkpoints",
            Path.home() / ".cache" / "voxsigil" / "models",
        ]

    discovered = []

    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            continue

        for file_path in search_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".pt", ".pth", ".pb", ".h5", ".onnx"]:
                discovered.append(
                    {
                        "path": str(file_path),
                        "name": file_path.stem,
                        "type": _detect_type_from_suffix(file_path.suffix),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

    return discovered


def get_latest_models(
    model_dir: Optional[Union[str, Path]] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get the latest models from a directory.

    Args:
        model_dir: Directory to search for models
        limit: Maximum number of models to return

    Returns:
        List of latest model information
    """
    if model_dir is None:
        model_dir = project_root / "models"

    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []

    models = discover_models_standalone([model_dir])
    models.sort(key=lambda x: x["modified"], reverse=True)
    return models[:limit]


def _detect_type_from_suffix(suffix: str) -> str:
    """Detect model type from file suffix."""
    suffix = suffix.lower()
    if suffix in [".pt", ".pth"]:
        return "pytorch"
    elif suffix in [".pb", ".h5"]:
        return "tensorflow"
    elif suffix == ".onnx":
        return "onnx"
    else:
        return "unknown"


# For backward compatibility - alias the standalone function
discover_models = discover_models_standalone
