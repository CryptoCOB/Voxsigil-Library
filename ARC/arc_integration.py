#!/usr/bin/env python
"""
arc_integration.py - Integration between GRID-Former and existing VoxSigil ARC system

Provides a bridge between the new direct neural network approach and the existing
LLM-based ARC solver in the VoxSigil system.
"""

import logging

# --- Robust import for GridFormerConnector ---
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Import VoxSigil ARC components
from .arc_reasoner import ARCReasoner

GRIDFORMER_CORE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Gridformer", "core")
)
if GRIDFORMER_CORE_DIR not in sys.path:
    sys.path.insert(0, GRIDFORMER_CORE_DIR)

try:
    from core.vantacore_grid_connector import GridFormerConnector
except ImportError as e:
    logging.error(f"Could not import GridFormerConnector from {GRIDFORMER_CORE_DIR}: {e}")
    raise

logger = logging.getLogger("VoxSigil.ARC.Integration")


class HybridARCSolver:
    """
    Hybrid ARC solver that combines neural network and LLM approaches.

    This class integrates the GRID-Former direct neural network training approach
    with the existing LLM-based solver, allowing for the best of both worlds.
    """

    def __init__(
        self,
        grid_former_model_path: Optional[str] = None,
        grid_former_confidence_threshold: float = 0.7,
        prefer_neural_net: bool = False,
        device: Optional[str] = None,
        enable_adaptive_routing: bool = True,
    ):
        """
        Initialize the hybrid solver.

        Args:
            grid_former_model_path: Path to GRID-Former model or None to create new one
            grid_former_confidence_threshold: Threshold for using GRID-Former predictions
            prefer_neural_net: Whether to prefer neural net over LLM when confidence is equal
            device: Device for neural network computation
            enable_adaptive_routing: Whether to use adaptive routing based on task characteristics
        """
        # Set up GRID-Former components
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_former_connector = GridFormerConnector(
            default_model_path=grid_former_model_path, device=self.device
        )

        # Set up configuration
        self.confidence_threshold = grid_former_confidence_threshold
        self.prefer_neural_net = prefer_neural_net
        self.enable_adaptive_routing = enable_adaptive_routing

        # Initialize ARC reasoner (LLM-based)
        self.arc_reasoner = ARCReasoner()

        # Track usage statistics
        self.stats = {"neural_net_uses": 0, "llm_uses": 0, "hybrid_uses": 0}

        logger.info(
            f"Initialized HybridARCSolver with preference: {'Neural Net' if prefer_neural_net else 'LLM'}"
        )

    def solve_arc_task(
        self,
        task_data: Dict[str, Any],
        task_id: str = "unknown_task",
        force_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Solve an ARC task using the best available method.

        Args:
            task_data: ARC task data (train and test examples)
            task_id: Task identifier
            force_method: Force a specific method ('neural', 'llm', or None for auto)

        Returns:
            Dictionary with solution results
        """
        # Extract train pairs and test input
        train_pairs = task_data.get("train", [])
        test_inputs = task_data.get("test", [])

        if not train_pairs or not test_inputs:
            return {"error": "Invalid task data: missing train pairs or test inputs"}

        # Determine which method to use based on task characteristics
        method_to_use = force_method or self._select_solving_method(task_data, task_id)

        start_time = time.time()

        # Solve with selected method
        if method_to_use == "neural":
            logger.info(f"Solving task {task_id} with GRID-Former neural network")
            result = self.grid_former_connector.handle_arc_task(task_data, task_id)
            self.stats["neural_net_uses"] += 1

        elif method_to_use == "llm":
            logger.info(f"Solving task {task_id} with LLM")
            llm_result, _ = self.arc_reasoner.solve_with_trace(
                {"train": train_pairs, "test": test_inputs, "id": task_id}
            )
            # Adapt to expected output format
            result = {"predicted_grid": llm_result.get("solution_grid", [])}
            self.stats["llm_uses"] += 1

        elif method_to_use == "hybrid":
            logger.info(f"Solving task {task_id} with hybrid approach")
            result = self._solve_with_hybrid_approach(task_data, task_id)
            self.stats["hybrid_uses"] += 1

        else:
            # Fall back to LLM as default
            logger.warning(f"Unknown method {method_to_use}, falling back to LLM")
            llm_result, _ = self.arc_reasoner.solve_with_trace(
                {"train": train_pairs, "test": test_inputs, "id": task_id}
            )
            result = {"predicted_grid": llm_result.get("solution_grid", [])}
            self.stats["llm_uses"] += 1

        # Add metadata to result
        elapsed_time = time.time() - start_time
        result["solution_metadata"] = {
            "method_used": method_to_use,
            "elapsed_time": elapsed_time,
            "timestamp": time.time(),
        }

        logger.info(
            f"Completed task {task_id} with method {method_to_use} in {elapsed_time:.2f} seconds"
        )
        return result

    def _select_solving_method(self, task_data: Dict[str, Any], task_id: str) -> str:
        """
        Select the best method for solving a task based on its characteristics.

        Args:
            task_data: Task data
            task_id: Task identifier

        Returns:
            Method to use: 'neural', 'llm', or 'hybrid'
        """
        if not self.enable_adaptive_routing:
            return "hybrid"  # Use hybrid approach by default

        # Extract train pairs
        train_pairs = task_data.get("train", [])

        # Simple heuristics for routing:

        # Analyze grid sizes
        max_grid_size = 0
        for example in train_pairs:
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])

            if input_grid:
                h, w = len(input_grid), len(input_grid[0])
                max_grid_size = max(max_grid_size, h, w)

            if output_grid:
                h, w = len(output_grid), len(output_grid[0])
                max_grid_size = max(max_grid_size, h, w)

        # Check for large grids (favor neural for larger grids)
        if max_grid_size > 15:
            logger.info(
                f"Task {task_id} has large grids (size {max_grid_size}), favoring neural approach"
            )
            return "neural" if self.prefer_neural_net else "hybrid"

        # Check for consistent transform patterns (favor neural)
        if self._has_consistent_transforms(train_pairs) and self.prefer_neural_net:
            logger.info(
                f"Task {task_id} has consistent transform patterns, favoring neural approach"
            )
            return "neural"

        # Check for complex reasoning (favor LLM)
        if self._needs_complex_reasoning(train_pairs):
            logger.info(f"Task {task_id} likely needs complex reasoning, favoring LLM approach")
            return "llm" if not self.prefer_neural_net else "hybrid"

        # Default to hybrid
        return "hybrid"

    def _has_consistent_transforms(self, train_pairs: List[Dict[str, Any]]) -> bool:
        """
        Check if the training examples have consistent transformation patterns.

        Args:
            train_pairs: List of training examples

        Returns:
            Whether the examples have consistent patterns
        """
        if len(train_pairs) < 2:
            return False

        # Simple check for consistent grid size changes
        input_shapes = []
        output_shapes = []

        for example in train_pairs:
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])

            if input_grid and output_grid:
                input_shapes.append((len(input_grid), len(input_grid[0])))
                output_shapes.append((len(output_grid), len(output_grid[0])))

        # Check if all examples have the same shape transformation
        shape_diffs = []
        for i in range(len(input_shapes)):
            i_h, i_w = input_shapes[i]
            o_h, o_w = output_shapes[i]
            shape_diffs.append((o_h - i_h, o_w - i_w))

        # If all shape differences are the same, likely consistent transform
        return all(diff == shape_diffs[0] for diff in shape_diffs)

    def _needs_complex_reasoning(self, train_pairs: List[Dict[str, Any]]) -> bool:
        """
        Check if the task likely requires complex reasoning.

        Args:
            train_pairs: List of training examples

        Returns:
            Whether the task likely needs complex reasoning
        """
        # Simple heuristic: if outputs vary a lot in size or structure, likely complex
        if len(train_pairs) < 2:
            return False

        # Check for variation in output shapes
        output_shapes = []
        for example in train_pairs:
            output_grid = example.get("output", [])
            if output_grid:
                output_shapes.append((len(output_grid), len(output_grid[0])))

        # If output shapes vary significantly, likely needs complex reasoning
        return len(set(output_shapes)) > len(train_pairs) // 2

    def _solve_with_hybrid_approach(
        self, task_data: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        """
        Solve a task using a hybrid approach that combines neural network and LLM methods.

        Args:
            task_data: Task data
            task_id: Task identifier

        Returns:
            Solution result
        """
        # Get neural network prediction
        neural_result = self.grid_former_connector.handle_arc_task(task_data, task_id)

        # Get LLM prediction
        train_pairs = task_data.get("train", [])
        test_inputs = task_data.get("test", [])
        llm_result, _ = self.arc_reasoner.solve_with_trace(
            {"train": train_pairs, "test": test_inputs, "id": task_id}
        )

        # Compare and select the best prediction
        neural_prediction = neural_result.get("predictions", [{}])[0].get("predicted_grid", [])
        llm_prediction = llm_result.get("solution_grid", [])

        # Simple ensemble: if predictions match, high confidence
        predictions_match = np.array_equal(np.array(neural_prediction), np.array(llm_prediction))

        # Create combined result
        combined_result = {
            "task_id": task_id,
            "predictions": [
                {
                    "input": test_inputs[0]["input"] if test_inputs else [],
                    "predicted_grid": neural_prediction
                    if self.prefer_neural_net
                    else llm_prediction,
                    "neural_prediction": neural_prediction,
                    "llm_prediction": llm_prediction,
                    "predictions_match": predictions_match,
                }
            ],
        }

        logger.info(f"Hybrid solution for task {task_id}, predictions match: {predictions_match}")
        return combined_result


def integrate_with_vantacore(vantacore_instance, model_path: Optional[str] = None) -> None:
    """
    Integrate GRID-Former with VantaCore.

    Args:
        vantacore_instance: VantaCore instance
        model_path: Path to GRID-Former model or None to create new one
    """
    # Check for device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create connector
    connector = GridFormerConnector(default_model_path=model_path, device=device)

    # Register with VantaCore
    # This is a placeholder - actual registration will depend on VantaCore API
    if hasattr(vantacore_instance, "register_model_handler"):
        vantacore_instance.register_model_handler("grid_former", connector)
        logger.info("Registered GRID-Former with VantaCore")
    else:
        logger.warning("Unable to register with VantaCore: missing registration method")

    # Add VantaCore hooks
    if hasattr(vantacore_instance, "add_task_hook"):
        vantacore_instance.add_task_hook("arc_tasks", connector.handle_arc_task)
        logger.info("Added GRID-Former task hook to VantaCore")
    else:
        logger.warning("Unable to add task hook to VantaCore: missing hook method")


# Command-line utility for testing integration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simple test
    solver = HybridARCSolver(prefer_neural_net=True)

    # Create sample task
    task_data = {
        "train": [{"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]}],
        "test": [{"input": [[4, 5], [6, 7]]}],
    }

    # Solve with different methods
    neural_result = solver.solve_arc_task(task_data, "test_task", force_method="neural")
    llm_result = solver.solve_arc_task(task_data, "test_task", force_method="llm")
    hybrid_result = solver.solve_arc_task(task_data, "test_task", force_method="hybrid")

    # Print results
    print(
        f"Neural network result: {neural_result.get('predictions', [{}])[0].get('predicted_grid')}"
    )
    print(f"LLM result: {llm_result.get('predicted_grid')}")
    print(f"Hybrid result: {hybrid_result.get('predictions', [{}])[0].get('predicted_grid')}")
    print(f"Usage statistics: {solver.stats}")
