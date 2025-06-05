#!/usr/bin/env python3
"""
🎨 VoxSigil Visualization Utilities
Provides visualization tools for ARC grids and performance analysis
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class GridVisualizer:
    """Visualizer for ARC task grids and predictions"""

    def __init__(self):
        """Initialize with default settings"""
        # ARC color palette (matches visualization standard)
        self.arc_colors = [
            "#000000",  # 0: Black (background)
            "#0074D9",  # 1: Blue
            "#FF4136",  # 2: Red
            "#2ECC40",  # 3: Green
            "#FFDC00",  # 4: Yellow
            "#AAAAAA",  # 5: Grey
            "#F012BE",  # 6: Magenta
            "#FF851B",  # 7: Orange
            "#7FDBFF",  # 8: Sky blue
            "#870C25",  # 9: Brown
        ]

    def _get_arc_colormap(self):
        """Create a matplotlib colormap from ARC colors"""
        import matplotlib.colors as mcolors

        # Convert hex to RGB
        rgb_colors = []
        for hex_color in self.arc_colors:
            # Convert hex to RGB normalized to [0,1]
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            rgb_colors.append((r, g, b))

        # Create colormap
        return mcolors.ListedColormap(rgb_colors)

    def visualize_grid(self, grid: List[List[int]], title: str = "", ax=None):
        """Visualize a single ARC grid with proper coloring"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        # Convert grid to numpy array
        grid_array = np.array(grid)

        # Get ARC colormap
        cmap = self._get_arc_colormap()

        # Display grid
        ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)

        # Set title if provided
        if title:
            ax.set_title(title)

        # Add grid lines
        for i in range(grid_array.shape[0] + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
        for j in range(grid_array.shape[1] + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def visualize_task_example(
        self,
        task: Dict[str, Any],
        prediction: Optional[List[List[int]]] = None,
        title: Optional[str] = None,
    ) -> Optional[Figure]:
        """Visualize a complete ARC task with prediction"""

        if "train" not in task or not task["train"]:
            # Can't visualize without at least one training example
            return None

        # Use first training example
        example = task["train"][0]
        input_grid = example["input"]
        output_grid = example["output"]

        # Create figure with proper layout
        if prediction is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            titles = ["Input", "Expected Output", "Prediction"]
            grids = [input_grid, output_grid, prediction]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            titles = ["Input", "Output"]
            grids = [input_grid, output_grid]

        # Add main title if provided
        if title:
            fig.suptitle(title, fontsize=16)

        # Visualize each grid
        for ax, grid, subtitle in zip(axes, grids, titles):
            self.visualize_grid(grid, title=subtitle, ax=ax)

        plt.tight_layout()
        return fig

    def visualize_batch_results(
        self,
        results: List[Dict[str, Any]],
        max_samples: int = 5,
        save_dir: Optional[str] = None,
    ) -> List[Figure]:
        """Visualize a batch of results"""
        figures = []

        for i, result in enumerate(results[:max_samples]):
            # Skip if missing required data
            if "task_data" not in result or "prediction" not in result:
                continue

            task_data = result["task_data"]
            prediction = result["prediction"]
            task_id = result.get("task_id", f"task_{i}")

            fig = self.visualize_task_example(
                task_data, prediction=prediction, title=f"Task: {task_id}"
            )

            # Save if requested
            if save_dir and fig:
                save_path = Path(save_dir) / f"result_{task_id}.png"
                fig.savefig(save_path)

            figures.append(fig)

        return figures


class PerformanceVisualizer:
    """Visualizer for model performance metrics"""

    def __init__(self):
        """Initialize with default settings"""
        self.colors = {
            "primary": "#0074D9",  # Blue
            "secondary": "#FF4136",  # Red
            "accent": "#2ECC40",  # Green
            "highlight": "#FFDC00",  # Yellow
            "neutral": "#AAAAAA",  # Grey
        }

    def plot_inference_performance(
        self, confidences: List[float], times: List[float], total_samples: int
    ) -> Figure:
        """Plot confidence distribution and inference timing"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot confidence distribution
        ax1.hist(confidences, bins=20, color=self.colors["primary"], alpha=0.7)
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Confidence Distribution")
        ax1.grid(True, alpha=0.3)

        # Plot inference timing
        ax2.plot(times, "o-", color=self.colors["secondary"], linewidth=2, markersize=4)
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Inference Time (s)")
        ax2.set_title("Inference Timing")
        ax2.grid(True, alpha=0.3)

        # Add performance summary as text
        avg_conf = np.mean(confidences) if confidences else 0
        avg_time = np.mean(times) if times else 0
        total_time = sum(times) if times else 0

        summary = f"Total Samples: {total_samples}\n"
        summary += f"Avg Confidence: {avg_conf:.3f}\n"
        summary += f"Avg Time/Sample: {avg_time:.3f}s\n"
        summary += f"Total Time: {total_time:.2f}s\n"
        summary += (
            f"Throughput: {total_samples / total_time:.1f} samples/sec"
            if total_time > 0
            else ""
        )

        fig.text(
            0.5,
            0.01,
            summary,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.5),
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        return fig

    def plot_training_curves(self, training_history: Dict[str, List[float]]) -> Figure:
        """Plot training and validation curves"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for key, values in training_history.items():
            if "loss" in key.lower():
                color = self.colors["secondary"]
            elif "acc" in key.lower() or "accuracy" in key.lower():
                color = self.colors["primary"]
            elif "val" in key.lower():
                color = self.colors["accent"]
            else:
                color = self.colors["neutral"]

            ax.plot(values, label=key, color=color, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training Performance")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_accuracy_comparison(self, model_accuracies: Dict[str, float]) -> Figure:
        """Plot accuracy comparison between models"""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(model_accuracies.keys())
        accuracies = list(model_accuracies.values())

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        models = [models[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]

        # Create horizontal bar chart
        ax.barh(models, accuracies, color=self.colors["primary"], alpha=0.7)

        # Add value labels
        for i, v in enumerate(accuracies):
            ax.text(v + 0.01, i, f"{v:.2%}", va="center")

        ax.set_xlabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlim(0, max(accuracies) * 1.1)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig
