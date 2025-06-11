#!/usr/bin/env python3
"""
🧠 Neural Interface - Handles model inference for GridFormer
Provides tensor conversion and fallback strategies
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using fallback inference only.")


class NeuralInterface:
    """Interface for neural network inference with ARC grids"""

    def __init__(self, model=None, config=None):
        """
        Initialize neural interface.        Args:
            model: Optional pre-loaded model
            config: Optional configuration
        """
        self.model = model
        self.config = config or {}
        self.device = self._get_device()
        self.inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "total_inference_time": 0.0,
            "fallback_used": 0,
        }

    def _get_device(self) -> str:
        """Get the appropriate device (CPU/CUDA/MPS) for inference"""
        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"

    def set_model(self, model):
        """Set the model for inference"""
        self.model = model
        return True

    def prepare_input(
        self, grid: Union[List[List[int]], np.ndarray]
    ) -> Union["torch.Tensor", Any]:
        """
        Prepare grid input for model inference.

        Args:
            grid: Input grid as list of lists or numpy array

        Returns:
            Tensor suitable for model input (or original grid if PyTorch not available)
        """
        if not TORCH_AVAILABLE:
            return grid

        # Convert to numpy if it's a list
        if isinstance(grid, list):
            grid = np.array(grid, dtype=np.int64)

        # Convert to tensor
        grid_tensor = torch.tensor(grid, dtype=torch.long)

        # Add batch dimension if needed
        if len(grid_tensor.shape) == 2:
            grid_tensor = grid_tensor.unsqueeze(0)

        # Move to appropriate device
        grid_tensor = grid_tensor.to(self.device)
        return grid_tensor

    def postprocess_output(
        self,
        output: Union["torch.Tensor", np.ndarray, Any],
        original_shape: Optional[Tuple[int, ...]] = None,
    ) -> Union[np.ndarray, Any]:
        """
        Process model output back to grid format.

        Args:
            output: Model output tensor
            original_shape: Optional original grid shape

        Returns:
            Processed grid as numpy array (or original output if not a tensor)
        """
        if not TORCH_AVAILABLE or not hasattr(output, "detach"):
            return output

        # Move to CPU and convert to numpy
        if isinstance(output, torch.Tensor):
            output_np = output.detach().cpu().numpy()
        else:
            output_np = output  # Assume it's already a NumPy array or compatible format

        # If output has a channel or probability dimension, get the most likely class
        if len(output_np.shape) > 2:
            if len(output_np.shape) == 4:  # [batch, height, width, classes]
                output_np = np.argmax(output_np, axis=-1)
            elif (
                len(output_np.shape) == 3 and output_np.shape[0] == 1
            ):  # [batch=1, height, width]
                output_np = output_np[0]  # Remove batch dimension

        # Reshape to original shape if provided
        if original_shape and output_np.shape != original_shape:
            output_np = output_np.reshape(original_shape)

        return output_np

    def run_inference(
        self, grid: Union[List[List[int]], np.ndarray], use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on input grid.

        Args:
            grid: Input grid
            use_fallback: Whether to use fallback if model inference fails

        Returns:
            Dictionary with results and metadata
        """
        import time

        start_time = time.time()

        # Track statistics
        self.inference_stats["total_inferences"] += 1  # Store original shape
        original_shape = tuple(np.array(grid).shape)

        result = {
            "success": False,
            "grid": None,
            "confidence": 0.0,
            "error": None,
            "inference_time": 0.0,
            "fallback_used": False,
            "device": self.device,
        }

        # Check if model is available
        if self.model is None:
            result["error"] = "No model loaded"
            if use_fallback:
                result.update(self._fallback_inference(grid))
                result["fallback_used"] = True
                self.inference_stats["fallback_used"] += 1
            self.inference_stats["failed_inferences"] += 1
            return result

        try:
            # Prepare input
            input_tensor = self.prepare_input(grid)

            # Run model inference
            with torch.no_grad():
                if TORCH_AVAILABLE:
                    # Move model to appropriate device if it's a torch model
                    if hasattr(self.model, "to") and callable(
                        getattr(self.model, "to")
                    ):
                        self.model.to(self.device)

                    if hasattr(self.model, "eval") and callable(
                        getattr(self.model, "eval")
                    ):
                        self.model.eval()

                # Run forward pass
                output = self.model(input_tensor)

                # Process output
                output_grid = self.postprocess_output(output, original_shape)

                result["grid"] = output_grid
                result["success"] = True
                self.inference_stats["successful_inferences"] += 1

        except Exception as e:
            result["error"] = str(e)
            self.inference_stats["failed_inferences"] += 1

            # Use fallback if enabled
            if use_fallback:
                result.update(self._fallback_inference(grid))
                result["fallback_used"] = True
                self.inference_stats["fallback_used"] += 1

        # Calculate inference time
        inference_time = time.time() - start_time
        result["inference_time"] = inference_time
        self.inference_stats["total_inference_time"] += inference_time

        return result

    def _fallback_inference(
        self, grid: Union[List[List[int]], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Fallback inference when model inference fails.
        Implements a simple identity transformation.

        Args:
            grid: Input grid

        Returns:
            Dictionary with fallback results
        """
        return {
            "success": True,
            "grid": np.array(grid),
            "confidence": 0.1,
            "fallback_used": True,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        stats = self.inference_stats.copy()

        # Calculate average inference time
        if stats["total_inferences"] > 0:
            stats["avg_inference_time"] = (
                stats["total_inference_time"] / stats["total_inferences"]
            )
            stats["success_rate"] = (
                stats["successful_inferences"] / stats["total_inferences"]
            )
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_inferences"]

        return stats
