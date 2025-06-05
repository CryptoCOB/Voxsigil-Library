#!/usr/bin/env python
"""
arc_data_processor.py - ARC Data Processing Pipeline

Processes ARC grid data for use with the GRID-Former model.
Handles data loading, preprocessing, augmentation, and batching.
"""

import os
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger("ARC.DataProcessor")


class ARCGridDataProcessor:
    """
    Processes ARC grid data for neural network training.
    Handles loading, preprocessing, and augmentation.
    """

    def __init__(
        self,
        max_grid_size: int = 30,
        padding_value: int = -1,
        augment_data: bool = True,
    ):
        self.max_grid_size = max_grid_size
        self.padding_value = padding_value
        self.augment_data = augment_data

    def load_arc_data(
        self, challenges_path: str, solutions_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load ARC data from JSON files.

        Args:
            challenges_path: Path to challenges JSON file
            solutions_path: Path to solutions JSON file (optional)

        Returns:
            Dictionary of ARC tasks with inputs and outputs
        """
        try:
            # Load challenges
            with open(challenges_path, "r") as f:
                challenges = json.load(f)

            # Initialize result dictionary
            tasks = {}

            # Process challenges
            for task_id, challenge in challenges.items():
                tasks[task_id] = {
                    "train": challenge["train"],
                    "test": challenge["test"],
                }

            # Load solutions if provided
            if solutions_path and os.path.exists(solutions_path):
                with open(solutions_path, "r") as f:
                    solutions = json.load(f)

                # Add solutions to tasks
                for task_id, solution in solutions.items():
                    if task_id in tasks:
                        for test_idx, test_solution in enumerate(solution):
                            if test_idx < len(tasks[task_id]["test"]):
                                tasks[task_id]["test"][test_idx]["output"] = (
                                    test_solution
                                )

            logger.info(f"Loaded {len(tasks)} ARC tasks")
            return tasks

        except Exception as e:
            logger.error(f"Error loading ARC data: {e}")
            raise

    def pad_grid(self, grid: List[List[int]]) -> np.ndarray:
        """
        Pad a grid to a fixed size.

        Args:
            grid: Input grid as a list of lists

        Returns:
            Padded grid as numpy array
        """
        grid_array = np.array(grid)
        h, w = grid_array.shape

        # Create padded grid
        padded_grid = np.full(
            (self.max_grid_size, self.max_grid_size), self.padding_value, dtype=np.int32
        )

        # Copy original grid values
        padded_grid[:h, :w] = grid_array

        return padded_grid

    def create_grid_mask(self, grid: List[List[int]]) -> np.ndarray:
        """
        Create a mask for a grid to track valid cells.

        Args:
            grid: Input grid as a list of lists

        Returns:
            Mask as numpy array (1 for valid cells, 0 for padding)
        """
        h, w = len(grid), len(grid[0])

        # Create mask
        mask = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.float32)
        mask[:h, :w] = 1.0

        return mask

    def augment_grid(
        self, grid: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to a grid.

        Args:
            grid: Input grid as numpy array
            mask: Grid mask as numpy array

        Returns:
            Tuple of (augmented_grid, augmented_mask)
        """
        if not self.augment_data:
            return grid, mask

        # Determine which augmentations to apply
        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        rotate = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees

        # Find the region where mask is 1 (actual grid content)
        rows = np.where(np.any(mask > 0, axis=1))[0]
        cols = np.where(np.any(mask > 0, axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            return grid, mask

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the valid grid and mask
        valid_grid = grid[min_row : max_row + 1, min_col : max_col + 1].copy()
        valid_mask = mask[min_row : max_row + 1, min_col : max_col + 1].copy()

        # Apply flips
        if flip_h:
            valid_grid = np.flip(valid_grid, axis=1)
            valid_mask = np.flip(valid_mask, axis=1)
        if flip_v:
            valid_grid = np.flip(valid_grid, axis=0)
            valid_mask = np.flip(valid_mask, axis=0)

        # Apply rotation
        if rotate > 0:
            valid_grid = np.rot90(valid_grid, k=rotate)
            valid_mask = np.rot90(valid_mask, k=rotate)

        # Create new padded grid and mask
        new_grid = np.full_like(grid, self.padding_value)
        new_mask = np.zeros_like(mask)

        # Insert augmented content back
        h, w = valid_grid.shape
        new_grid[:h, :w] = valid_grid
        new_mask[:h, :w] = valid_mask

        return new_grid, new_mask

    def process_example(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single input-output example.

        Args:
            input_grid: Input grid as a list of lists
            output_grid: Output grid as a list of lists

        Returns:
            Dictionary with processed tensors
        """
        # Pad grids
        input_padded = self.pad_grid(input_grid)
        output_padded = self.pad_grid(output_grid)

        # Create masks
        input_mask = self.create_grid_mask(input_grid)
        output_mask = self.create_grid_mask(output_grid)

        # Apply augmentation
        if self.augment_data:
            input_padded, input_mask = self.augment_grid(input_padded, input_mask)
            output_padded, output_mask = self.augment_grid(output_padded, output_mask)

        # Convert to torch tensors
        input_tensor = torch.tensor(input_padded, dtype=torch.long)
        output_tensor = torch.tensor(output_padded, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.float)
        output_mask_tensor = torch.tensor(output_mask, dtype=torch.float)

        # Store original dimensions
        input_shape = torch.tensor(
            [len(input_grid), len(input_grid[0])], dtype=torch.long
        )
        output_shape = torch.tensor(
            [len(output_grid), len(output_grid[0])], dtype=torch.long
        )

        return {
            "input": input_tensor,
            "output": output_tensor,
            "input_mask": input_mask_tensor,
            "output_mask": output_mask_tensor,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC data.
    """

    def __init__(
        self,
        tasks: Dict[str, Dict[str, Any]],
        split: str = "train",
        data_processor: Optional[ARCGridDataProcessor] = None,
        max_grid_size: int = 30,
        augment_data: bool = True,
    ):
        """
        Initialize ARC dataset.

        Args:
            tasks: Dictionary of ARC tasks
            split: 'train' or 'test'
            data_processor: ARCGridDataProcessor instance or None to create new one
            max_grid_size: Maximum grid size for padding
            augment_data: Whether to apply data augmentation
        """
        self.tasks = tasks
        self.split = split
        self.processor = data_processor or ARCGridDataProcessor(
            max_grid_size=max_grid_size, augment_data=augment_data
        )

        # Create list of examples
        self.examples = []
        for task_id, task in tasks.items():
            for example in task[split]:
                if "input" in example and "output" in example:
                    self.examples.append(
                        {
                            "task_id": task_id,
                            "input": example["input"],
                            "output": example["output"],
                        }
                    )

        logger.info(f"Created ARC dataset with {len(self.examples)} {split} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return self.processor.process_example(example["input"], example["output"])


def create_arc_dataloaders(
    challenges_path: str,
    solutions_path: Optional[str] = None,
    batch_size: int = 32,
    max_grid_size: int = 30,
    augment_train: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for ARC training and testing.

    Args:
        challenges_path: Path to challenges JSON file
        solutions_path: Path to solutions JSON file (optional)
        batch_size: Batch size for DataLoader
        max_grid_size: Maximum grid size for padding
        augment_train: Whether to apply data augmentation to training data
        num_workers: Number of worker processes for DataLoader

    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create data processor
    processor = ARCGridDataProcessor(max_grid_size=max_grid_size, augment_data=False)

    # Load data
    tasks = processor.load_arc_data(challenges_path, solutions_path)

    # Create datasets
    train_dataset = ARCDataset(
        tasks=tasks,
        split="train",
        data_processor=processor,
        max_grid_size=max_grid_size,
        augment_data=augment_train,
    )

    test_dataset = ARCDataset(
        tasks=tasks,
        split="test",
        data_processor=processor,
        max_grid_size=max_grid_size,
        augment_data=False,  # No augmentation for test data
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


# Visualization utility
def visualize_grid(
    grid: Union[List[List[int]], np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """
    Create a visualization of a grid.

    Args:
        grid: Input grid
        mask: Optional mask to highlight valid cells

    Returns:
        RGB image as numpy array
    """
    # Convert to numpy array
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    elif isinstance(grid, list):
        grid = np.array(grid)

    # Handle mask
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

    # Define colors - ARC typically uses 0-9 for colors
    colors = [
        [0, 0, 0],  # 0: Black
        [0, 0, 255],  # 1: Blue
        [255, 0, 0],  # 2: Red
        [0, 255, 0],  # 3: Green
        [255, 255, 0],  # 4: Yellow
        [255, 0, 255],  # 5: Magenta
        [255, 165, 0],  # 6: Orange
        [128, 128, 128],  # 7: Gray
        [165, 42, 42],  # 8: Brown
        [255, 255, 255],  # 9: White
        [50, 50, 50],  # -1: Padding (dark gray)
    ]

    # Get grid dimensions
    if mask is not None:
        rows = np.where(np.any(mask > 0, axis=1))[0]
        cols = np.where(np.any(mask > 0, axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            h, w = grid.shape
        else:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            h, w = max_row - min_row + 1, max_col - min_col + 1
            grid = grid[min_row : max_row + 1, min_col : max_col + 1]
    else:
        h, w = grid.shape

    # Create RGB image
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Fill in colors
    for i in range(h):
        for j in range(w):
            color_idx = grid[i, j]
            if color_idx == -1:  # Padding
                color_idx = 10
            elif color_idx >= 0 and color_idx < 10:
                color_idx = int(color_idx)
            else:
                color_idx = 10  # Default to padding color for invalid values
            img[i, j] = colors[color_idx]

    return img


# Test function
def test_arc_data_processor():
    """
    Test the ARC data processor.
    """
    # Create processor
    processor = ARCGridDataProcessor(max_grid_size=30, augment_data=True)

    # Create sample grids
    input_grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    output_grid = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]

    # Process example
    processed = processor.process_example(input_grid, output_grid)

    # Print results
    print(f"Input tensor shape: {processed['input'].shape}")
    print(f"Output tensor shape: {processed['output'].shape}")
    print(f"Input mask shape: {processed['input_mask'].shape}")
    print(f"Original input shape: {processed['input_shape']}")
    print(f"Original output shape: {processed['output_shape']}")

    # Test visualization
    input_vis = visualize_grid(processed["input"], processed["input_mask"])
    output_vis = visualize_grid(processed["output"], processed["output_mask"])

    print(f"Input visualization shape: {input_vis.shape}")
    print(f"Output visualization shape: {output_vis.shape}")

    print("Test successful!")


if __name__ == "__main__":
    test_arc_data_processor()
