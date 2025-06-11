#!/usr/bin/env python3
"""
📊 ARC Data Loader - Handles ARC dataset loading and preprocessing
Supports both original ARC data and synthetic training data
"""

import json
import os
import random
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

# Attempt to import the optional unicode_safe_logging package dynamically
try:
    _usl = importlib.import_module("unicode_safe_logging")
    CHECK_MARK = _usl.CHECK_MARK
    CROSS_MARK = _usl.CROSS_MARK
    WARNING = _usl.WARNING
    get_safe_symbol = _usl.get_safe_symbol
    use_ascii = _usl.use_ascii_fallbacks()
except ModuleNotFoundError:
    # Fallbacks if the unicode_safe_logging module isn't available
    CHECK_MARK, CROSS_MARK, WARNING = "✅", "❌", "⚠️"
    ASCII_CHECK_MARK, ASCII_CROSS_MARK, ASCII_WARNING = "[PASS]", "[FAIL]", "[WARN]"
    use_ascii = os.name == "nt"

    def get_safe_symbol(symbol, use_ascii=False):
        if use_ascii:
            if symbol == CHECK_MARK:
                return ASCII_CHECK_MARK
            elif symbol == CROSS_MARK:
                return ASCII_CROSS_MARK
            elif symbol == WARNING:
                return ASCII_WARNING
        return symbol


class ARCDataLoader:
    """Loads and manages ARC dataset with preprocessing capabilities"""

    def __init__(self, workspace_root: str = r"C:\Users\16479\Desktop\Voxsigil"):
        self.workspace_root = Path(workspace_root)
        self.data_paths = {
            "arc_training": self.workspace_root
            / "Voxsigil_Library"
            / "ARC"
            / "datasamples"
            / "arc-agi_training_challenges.json",
            "arc_training_solutions": self.workspace_root
            / "Voxsigil_Library"
            / "ARC"
            / "datasamples"
            / "arc-agi_training_solutions.json",
            "arc_evaluation": self.workspace_root
            / "Voxsigil_Library"
            / "ARC"
            / "datasamples"
            / "arc-agi_evaluation_challenges.json",
            "arc_evaluation_solutions": self.workspace_root
            / "Voxsigil_Library"
            / "ARC"
            / "datasamples"
            / "arc-agi_evaluation_solutions.json",
            "arc_test": self.workspace_root
            / "data"
            / "arc"
            / "evaluation"
            / "evaluation.json",  # Use evaluation as test
            "sample_submission": self.workspace_root
            / "Voxsigil_Library"
            / "ARC"
            / "datasamples"
            / "sample_submission.json",
            "data_training": self.workspace_root
            / "data"
            / "arc"
            / "training"
            / "training.json",
            "data_test": self.workspace_root / "data" / "arc" / "test" / "test.json",
        }
        self.loaded_data = {}

    def load_arc_data(self, data_type: str = "training") -> Dict[str, Any]:
        """Load ARC data by type"""
        if data_type in self.loaded_data:
            return self.loaded_data[data_type]

        # Try multiple possible paths
        possible_paths = []
        if data_type == "training":
            # Potential paths for training data
            possible_paths.extend(
                [
                    self.data_paths["arc_training"],
                    self.data_paths["data_training"],
                ]
            )
        elif data_type == "test":
            possible_paths = [self.data_paths["arc_test"], self.data_paths["data_test"]]
        elif data_type == "evaluation":
            possible_paths = [self.data_paths["arc_evaluation"]]

        data = None
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    success_mark = get_safe_symbol(CHECK_MARK, use_ascii)
                    print(f"{success_mark} Loaded {data_type} data from: {path}")
                    break
                except Exception as e:
                    error_mark = get_safe_symbol(CROSS_MARK, use_ascii)
                    print(f"{error_mark} Error loading {path}: {e}")
                    continue

        if data is None:
            # Create sample data if no files found
            warning_mark = get_safe_symbol(WARNING, use_ascii)
            print(f"{warning_mark} No {data_type} data found, creating sample data")
            data = self._create_sample_data()

        self.loaded_data[data_type] = data
        return data

    def load_solutions(self, data_type: str = "training") -> Optional[Dict[str, Any]]:
        """Load solution data if available"""
        if data_type == "training":
            path = self.data_paths["arc_training_solutions"]
        elif data_type == "evaluation":
            path = self.data_paths["arc_evaluation_solutions"]
        else:
            return None

        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading solutions: {e}")

        return None

    def load_training_data(
        self, max_samples: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Load training data - wrapper for load_arc_data

        Args:
            max_samples: Optional limit on the number of samples to load
        """
        try:
            data = self.load_arc_data("training")
            if data and max_samples is not None and len(data) > max_samples:
                # Limit the number of samples if max_samples is specified
                limited_data = dict(list(data.items())[:max_samples])
                return limited_data
            return data
        except Exception as e:
            error_mark = get_safe_symbol(CROSS_MARK, use_ascii)
            print(f"{error_mark} Error loading training data: {e}")
            return None

    def load_test_data(self) -> Optional[Dict[str, Any]]:
        """Load test data - wrapper for load_arc_data"""
        try:
            return self.load_arc_data("test")
        except Exception as e:
            error_mark = get_safe_symbol(CROSS_MARK, use_ascii)
            print(f"{error_mark} Error loading test data: {e}")
            return None

    def load_custom_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load custom data from specified file path"""
        try:
            if not Path(file_path).exists():
                error_mark = get_safe_symbol(CROSS_MARK, use_ascii)
                print(f"{error_mark} File not found: {file_path}")
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            success_mark = get_safe_symbol(CHECK_MARK, use_ascii)
            print(f"{success_mark} Loaded custom data from: {file_path}")
            return data
        except Exception as e:
            error_mark = get_safe_symbol(CROSS_MARK, use_ascii)
            print(f"{error_mark} Error loading custom data from {file_path}: {e}")
            return None

    def get_task_by_id(
        self, task_id: str, data_type: str = "training"
    ) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID"""
        data = self.load_arc_data(data_type)
        return data.get(task_id)

    def get_random_tasks(
        self, n: int = 5, data_type: str = "training"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get n random tasks for testing"""
        data = self.load_arc_data(data_type)
        task_ids = list(data.keys())

        if len(task_ids) == 0:
            return []

        n = min(n, len(task_ids))
        selected_ids = random.sample(task_ids, n)

        return [(task_id, data[task_id]) for task_id in selected_ids]

    def preprocess_grid(self, grid: List[List[int]]) -> np.ndarray:
        """
        Preprocess a grid for training/inference

        Args:
            grid: 2D list representing the grid

        Returns:
            Preprocessed grid as numpy array
        """
        # Convert to numpy array
        grid_array = np.array(grid, dtype=np.int32)

        # Validate grid values are in ARC range (0-9)
        if grid_array.min() < 0 or grid_array.max() > 9:
            print(
                f"Warning: Grid values outside ARC range (0-9): min={grid_array.min()}, max={grid_array.max()}"
            )

        return grid_array

    def augment_grid(
        self, grid: np.ndarray, augmentation_type: str = "rotate"
    ) -> np.ndarray:
        """
        Apply data augmentation to a grid

        Args:
            grid: Input grid as numpy array
            augmentation_type: Type of augmentation ("rotate", "flip", "none")

        Returns:
            Augmented grid
        """
        if augmentation_type == "rotate":
            # Rotate 90 degrees clockwise
            return np.rot90(grid, k=-1)
        elif augmentation_type == "flip_horizontal":
            return np.fliplr(grid)
        elif augmentation_type == "flip_vertical":
            return np.flipud(grid)
        elif augmentation_type == "none":
            return grid.copy()
        else:
            print(
                f"Warning: Unknown augmentation type '{augmentation_type}', returning original grid"
            )
            return grid.copy()

    def preprocess_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a complete task for model input"""
        processed = {"train": [], "test": []}

        # Process training examples
        for example in task.get("train", []):
            processed_example = {
                "input": self.preprocess_grid(example["input"]),
                "output": self.preprocess_grid(example["output"]),
            }
            processed["train"].append(processed_example)

        # Process test examples (usually just input)
        for example in task.get("test", []):
            processed_example = {"input": self.preprocess_grid(example["input"])}
            if "output" in example:
                processed_example["output"] = self.preprocess_grid(example["output"])
            processed["test"].append(processed_example)

        return processed

    def get_dataset_stats(self, data_type: str = "training") -> Dict[str, Any]:
        """Get statistics about the dataset"""
        data = self.load_arc_data(data_type)

        stats = {
            "total_tasks": len(data),
            "grid_sizes": [],
            "color_counts": [],
            "train_examples_per_task": [],
        }

        for task_id, task in data.items():
            # Analyze training examples
            train_examples = task.get("train", [])
            stats["train_examples_per_task"].append(len(train_examples))

            for example in train_examples:
                input_grid = example["input"]
                output_grid = example["output"]

                # Grid sizes
                stats["grid_sizes"].append((len(input_grid), len(input_grid[0])))
                stats["grid_sizes"].append((len(output_grid), len(output_grid[0])))

                # Color counts
                input_colors = set()
                output_colors = set()
                for row in input_grid:
                    input_colors.update(row)
                for row in output_grid:
                    output_colors.update(row)

                stats["color_counts"].append(len(input_colors))
                stats["color_counts"].append(len(output_colors))

        # Calculate summary statistics
        if stats["grid_sizes"]:
            heights, widths = zip(*stats["grid_sizes"])
            stats["grid_size_summary"] = {
                "min_height": min(heights),
                "max_height": max(heights),
                "avg_height": sum(heights) / len(heights),
                "min_width": min(widths),
                "max_width": max(widths),
                "avg_width": sum(widths) / len(widths),
            }

        if stats["color_counts"]:
            stats["color_summary"] = {
                "min_colors": min(stats["color_counts"]),
                "max_colors": max(stats["color_counts"]),
                "avg_colors": sum(stats["color_counts"]) / len(stats["color_counts"]),
            }

        if stats["train_examples_per_task"]:
            stats["examples_summary"] = {
                "min_examples": min(stats["train_examples_per_task"]),
                "max_examples": max(stats["train_examples_per_task"]),
                "avg_examples": sum(stats["train_examples_per_task"])
                / len(stats["train_examples_per_task"]),
            }

        return stats

    def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample ARC data for testing when no real data available"""
        sample_data = {}

        for i in range(5):
            task_id = f"sample_{i:03d}"

            # Create simple pattern-based task
            task = {"train": [], "test": []}

            # Generate 3 training examples
            for j in range(3):
                # Simple pattern: input has scattered 1s, output groups them
                input_grid = [[0] * 5 for _ in range(5)]
                output_grid = [[0] * 5 for _ in range(5)]

                # Add some 1s randomly in input
                for _ in range(3):
                    r, c = random.randint(0, 4), random.randint(0, 4)
                    input_grid[r][c] = 1

                # In output, group 1s in top-left
                count_ones = sum(sum(row) for row in input_grid)
                for k in range(count_ones):
                    output_grid[k // 5][k % 5] = 1

                task["train"].append({"input": input_grid, "output": output_grid})

            # Generate 1 test example
            input_grid = [[0] * 5 for _ in range(5)]
            for _ in range(2):
                r, c = random.randint(0, 4), random.randint(0, 4)
                input_grid[r][c] = 1

            task["test"].append({"input": input_grid})

            sample_data[task_id] = task

        return sample_data

    def validate_data_format(self, data: Dict[str, Any]) -> bool:
        """Validate that data follows expected ARC format"""
        try:
            for task_id, task in data.items():
                if not isinstance(task, dict):
                    return False

                if "train" not in task or "test" not in task:
                    return False

                for example in task["train"]:
                    if "input" not in example or "output" not in example:
                        return False

                    if not isinstance(example["input"], list) or not isinstance(
                        example["output"], list
                    ):
                        return False
                for example in task["test"]:
                    if "input" not in example:
                        return False

                    if not isinstance(example["input"], list):
                        return False

            return True

        except Exception:
            return False

    def load_scaffold(self, scaffold_name_or_path: str, **kwargs) -> Any:
        """
        Stub implementation for loading scaffolds.

        This is a placeholder implementation that will be expanded when the
        scaffold loading system is fully implemented.

        Args:
            scaffold_name_or_path: Name or path of the scaffold to load
            **kwargs: Additional loading parameters

        Returns:
            A mock scaffold object with basic methods
        """
        import logging

        logger = logging.getLogger("ARC.DataLoader")

        logger.warning(
            f"Loading scaffold '{scaffold_name_or_path}' - using mock implementation"
        )

        class MockScaffold:
            """Mock scaffold implementation for testing"""

            def __init__(self, name: str):
                self.name = name

            def apply(self, task: Any, **kwargs) -> Any:
                """Mock apply method that returns the task unchanged"""
                return task

            def generate_antithesis(self, thesis: Any, **kwargs) -> Any:
                """Mock antithesis generation for Hegelian scaffolds"""
                return f"antithesis_of_{thesis}"

            def synthesize(self, thesis: Any, antithesis: Any, **kwargs) -> Any:
                """Mock synthesis for Hegelian scaffolds"""
                return f"synthesis_of_{thesis}_and_{antithesis}"

        return MockScaffold(scaffold_name_or_path)


# Standalone utility functions for backward compatibility
def preprocess_grid(grid: List[List[int]]) -> np.ndarray:
    """
    Standalone function to preprocess a grid for training/inference

    Args:
        grid: 2D list representing the grid

    Returns:
        Preprocessed grid as numpy array
    """
    # Convert to numpy array
    grid_array = np.array(grid, dtype=np.int32)

    # Validate grid values are in ARC range (0-9)
    if grid_array.min() < 0 or grid_array.max() > 9:
        print(
            f"Warning: Grid values outside ARC range (0-9): min={grid_array.min()}, max={grid_array.max()}"
        )

    return grid_array


def augment_grid(grid: np.ndarray, augmentation_type: str = "rotate") -> np.ndarray:
    """
    Standalone function to apply data augmentation to a grid

    Args:
        grid: Input grid as numpy array
        augmentation_type: Type of augmentation ("rotate", "flip", "none")

    Returns:
        Augmented grid
    """
    if augmentation_type == "rotate":
        # Rotate 90 degrees clockwise
        return np.rot90(grid, k=-1)
    elif augmentation_type == "flip_horizontal":
        return np.fliplr(grid)
    elif augmentation_type == "flip_vertical":
        return np.flipud(grid)
    else:  # "none" or any other value
        return grid.copy()
