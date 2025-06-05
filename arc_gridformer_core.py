#!/usr/bin/env python
"""
🎯 ARC GridFormer BLT Core
A simplified implementation of ARC-specific color validation and grid processing.

This module provides the essential functionality needed for ARC GridFormer
without the complexity of the full hybrid BLT implementation.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("ARCGridFormerBLT")


class ARCGridValidator:
    """
    ARC grid validation and correction utility.

    Handles validation of grid colors against the ARC palette (0-9)
    and provides correction for out-of-bounds values.
    """

    def __init__(
        self,
        color_correction_threshold: float = 0.8,  # Confidence threshold for corrections
    ):
        self.color_correction_threshold = color_correction_threshold
        # ARC Color Palette (0=background, 1-9=colors)
        self.arc_palette = list(range(10))

        # Correction history for pattern learning
        self.correction_history = []  # Store (prediction, correction, confidence) tuples
        self.max_history = 100  # Keep last 100 corrections for pattern learning

    def validate_color(self, color_value: Any) -> int:
        """Validate a color value against the ARC palette (0-9) and record corrections."""
        try:
            color = int(color_value)
            if color < 0 or color > 9:
                logger.warning(
                    f"Out-of-bounds color value {color} detected. Clamping to valid range."
                )
                corrected = max(0, min(9, color))
                # Record correction in history
                if len(self.correction_history) >= self.max_history:
                    self.correction_history.pop(0)
                self.correction_history.append((color, corrected, 1.0))
                return corrected
            return color
        except (ValueError, TypeError):
            logger.warning(f"Invalid color value {color_value}. Defaulting to 0.")
            # Record correction in history
            if len(self.correction_history) >= self.max_history:
                self.correction_history.pop(0)
            self.correction_history.append((color_value, 0, 1.0))
            return 0

    def correct_grid_colors(self, grid: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply color validation and correction to an ARC grid"""
        if not isinstance(grid, np.ndarray):
            logger.warning(f"Invalid grid type {type(grid)}. Expected numpy array.")
            if hasattr(grid, "__array__"):
                grid = np.array(grid)
            else:
                return np.zeros((3, 3), dtype=np.int32), 0.0

        # Create a copy to avoid modifying the original
        corrected = grid.copy()

        # Track how many corrections were made
        correction_count = 0
        total_cells = grid.size

        # Validate each cell
        for idx in np.ndindex(grid.shape):
            original = grid[idx]
            valid = self.validate_color(original)
            if original != valid:
                corrected[idx] = valid
                correction_count += 1

                # Store correction in history for learning
                if len(self.correction_history) >= self.max_history:
                    self.correction_history.pop(0)  # Remove oldest
                self.correction_history.append((original, valid, 1.0))

        # Calculate confidence (higher = more corrections needed)
        confidence = 1.0 - (correction_count / total_cells) if total_cells > 0 else 1.0

        return corrected, confidence

    def batch_correct_grids(
        self, grids: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Apply color validation and correction to a batch of ARC grids"""
        corrected_grids = []
        confidences = []

        for grid in grids:
            corrected, confidence = self.correct_grid_colors(grid)
            corrected_grids.append(corrected)
            confidences.append(confidence)

        return corrected_grids, confidences

    def get_correction_patterns(self) -> Dict[int, int]:
        """Analyze correction history to identify common patterns (all originals included)"""
        if not self.correction_history:
            return {}

        # Count frequency of corrections
        correction_counts = {}
        for original, corrected, _ in self.correction_history:
            key = (original, corrected)
            correction_counts[key] = correction_counts.get(key, 0) + 1

        # For each original value, always include its most frequent correction
        patterns = {}
        originals = set(original for original, _, _ in self.correction_history)
        for original in originals:
            # Find the most frequent correction for this original
            candidates = [
                (corr, count)
                for (orig, corr), count in correction_counts.items()
                if orig == original
            ]
            if candidates:
                # Pick the correction with the highest count
                corrected, _ = max(candidates, key=lambda x: x[1])
                patterns[original] = corrected

        return patterns
