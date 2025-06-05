#!/usr/bin/env python3
# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
"""
VoxSigil Visualization Tab Interface
Modular component for visualization functionality

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Encapsulated visualization interface for Dynamic GridFormer GUI
"""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .gui_styles import VoxSigilStyles


class VoxSigilVisualizationInterface:
    """Visualization interface for VoxSigil Dynamic GridFormer"""

    def __init__(self, parent_gui, notebook):
        """
        Initialize the visualization interface

        Args:
            parent_gui: Reference to the main GUI class
            notebook: ttk.Notebook to add the visualization tab to
        """
        self.parent_gui = parent_gui
        self.notebook = notebook

        # Visualization state
        self.current_sample = None
        self.current_prediction = None
        self.figure = None
        self.canvas = None

        # Create the visualization tab
        self.create_visualization_tab()

    def create_visualization_tab(self):
        """Create the visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📊 Visualization")

        # Main container
        main_container = tk.Frame(viz_frame, **VoxSigilStyles.get_frame_config())
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel (top)
        self._create_control_panel(main_container)

        # Visualization area (bottom)
        self._create_visualization_area(main_container)

    def _create_control_panel(self, parent):
        """Create the visualization control panel"""
        control_panel = tk.LabelFrame(
            parent, **VoxSigilStyles.get_label_frame_config("Visualization Controls")
        )
        control_panel.pack(fill=tk.X, padx=5, pady=5)

        # Control buttons
        button_frame = tk.Frame(control_panel, **VoxSigilStyles.get_frame_config())
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame,
            text="🎨 Visualize Sample",
            command=self.visualize_sample,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="📈 Show Performance",
            command=self.show_performance,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="🔍 Analyze Predictions",
            command=self.analyze_predictions,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="🔄 Clear Display",
            command=self.clear_display,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        # Sample selection
        selection_frame = tk.Frame(control_panel, **VoxSigilStyles.get_frame_config())
        selection_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(selection_frame, text="Sample Index:", style="Info.TLabel").pack(
            side=tk.LEFT, padx=5
        )

        self.sample_index = tk.IntVar(value=0)
        sample_spinbox = tk.Spinbox(
            selection_frame,
            from_=0,
            to=100,
            textvariable=self.sample_index,
            bg=VoxSigilStyles.COLORS["bg_tertiary"],
            fg=VoxSigilStyles.COLORS["text_primary"],
            width=10,
        )
        sample_spinbox.pack(side=tk.LEFT, padx=5)

        # Visualization options
        options_frame = tk.Frame(control_panel, **VoxSigilStyles.get_frame_config())
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.show_grid_lines = tk.BooleanVar(value=True)
        self.show_differences = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=False)

        tk.Checkbutton(
            options_frame,
            text="Grid Lines",
            variable=self.show_grid_lines,
            bg=VoxSigilStyles.COLORS["bg_secondary"],
            fg=VoxSigilStyles.COLORS["text_primary"],
            selectcolor=VoxSigilStyles.COLORS["accent_cyan"],
        ).pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(
            options_frame,
            text="Show Differences",
            variable=self.show_differences,
            bg=VoxSigilStyles.COLORS["bg_secondary"],
            fg=VoxSigilStyles.COLORS["text_primary"],
            selectcolor=VoxSigilStyles.COLORS["accent_cyan"],
        ).pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(
            options_frame,
            text="Confidence Map",
            variable=self.show_confidence,
            bg=VoxSigilStyles.COLORS["bg_secondary"],
            fg=VoxSigilStyles.COLORS["text_primary"],
            selectcolor=VoxSigilStyles.COLORS["accent_cyan"],
        ).pack(side=tk.LEFT, padx=5)

    def _create_visualization_area(self, parent):
        """Create the main visualization area"""
        viz_area = tk.LabelFrame(
            parent, **VoxSigilStyles.get_label_frame_config("Grid Visualization")
        )
        viz_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup matplotlib with VoxSigil styling
        plt.style.use("dark_background")

        self.figure = Figure(
            figsize=(12, 8), facecolor=VoxSigilStyles.COLORS["bg_primary"]
        )
        self.figure.patch.set_facecolor(VoxSigilStyles.COLORS["bg_primary"])

        self.canvas = FigureCanvasTkAgg(self.figure, viz_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize with placeholder
        self._create_placeholder_plot()

    def _create_placeholder_plot(self):
        """Create a placeholder visualization"""
        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "📊 VoxSigil Visualization\n\nLoad data and select a sample to visualize",
            ha="center",
            va="center",
            fontsize=16,
            color=VoxSigilStyles.COLORS["accent_cyan"],
            transform=ax.transAxes,
        )

        ax.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])
        ax.set_xticks([])
        ax.set_yticks([])

        self.figure.patch.set_facecolor(VoxSigilStyles.COLORS["bg_primary"])
        self.canvas.draw()

    def visualize_sample(self):
        """Visualize a specific sample with input, target, and prediction"""
        try:
            # Get test data from testing interface
            if (
                hasattr(self.parent_gui, "testing_interface")
                and self.parent_gui.testing_interface.test_data
            ):
                test_data = self.parent_gui.testing_interface.test_data
            else:
                messagebox.showwarning(
                    "No Data", "Please load test data in the Testing tab first."
                )
                return

            sample_idx = self.sample_index.get()
            if sample_idx >= len(test_data):
                messagebox.showwarning(
                    "Invalid Index", f"Sample index must be less than {len(test_data)}"
                )
                return

            sample = test_data[sample_idx]
            self.current_sample = sample

            # Get prediction if model is loaded
            prediction = None
            if self.parent_gui.current_model:
                try:
                    result = self.parent_gui.inference_engine.run_single_inference(
                        sample
                    )
                    if result and "prediction" in result:
                        prediction = result["prediction"]
                        self.current_prediction = prediction
                except Exception as e:
                    print(f"Error getting prediction: {e}")

            # Create visualization
            self._create_sample_visualization(sample, prediction)

        except Exception as e:
            messagebox.showerror(
                "Visualization Error", f"Failed to visualize sample: {str(e)}"
            )

    def _create_sample_visualization(self, sample, prediction=None):
        """Create the main sample visualization"""
        self.figure.clear()

        # Extract grids from sample
        input_grids = sample.get("input", [])
        output_grid = sample.get("output")

        if not input_grids:
            self._create_error_plot("No input grids found in sample")
            return

        # Setup grid visualization
        num_inputs = len(input_grids)
        cols = min(
            4,
            num_inputs
            + (2 if output_grid is not None else 1)
            + (1 if prediction is not None else 0),
        )
        rows = max(1, (num_inputs + cols - 1) // cols)

        if output_grid is not None or prediction is not None:
            rows = max(rows, 2)

        # Create custom colormap for ARC
        cmap = plt.get_cmap("Set3")

        # Plot input grids
        for i, input_grid in enumerate(input_grids):
            if i >= 8:  # Limit to prevent overcrowding
                break

            ax = self.figure.add_subplot(rows, cols, i + 1)
            self._plot_grid(ax, input_grid, f"Input {i + 1}", cmap)

        # Plot target output if available
        if output_grid is not None:
            ax = self.figure.add_subplot(rows, cols, cols * (rows - 1) + 1)
            self._plot_grid(ax, output_grid, "Expected Output", cmap)

        # Plot prediction if available
        if prediction is not None:
            ax_pos = cols * (rows - 1) + (2 if output_grid is not None else 1)
            if ax_pos <= rows * cols:
                ax = self.figure.add_subplot(rows, cols, ax_pos)

                # Check if prediction matches target
                is_correct = False
                if output_grid is not None:
                    is_correct = np.array_equal(prediction, output_grid)

                title = "Prediction ✓" if is_correct else "Prediction ✗"
                title_color = (
                    VoxSigilStyles.COLORS["accent_mint"]
                    if is_correct
                    else VoxSigilStyles.COLORS["accent_coral"]
                )

                self._plot_grid(ax, prediction, title, cmap, title_color=title_color)

        # Add difference visualization if enabled and both target and prediction exist
        if (
            self.show_differences.get()
            and output_grid is not None
            and prediction is not None
            and rows * cols >= cols * rows
        ):
            try:
                difference = self._compute_difference(output_grid, prediction)
                ax = self.figure.add_subplot(rows, cols, cols * rows)
                self._plot_difference(ax, difference, "Differences")
            except Exception as e:
                print(f"Error computing differences: {e}")

        self.figure.suptitle(
            f"ARC Sample {self.sample_index.get()}",
            color=VoxSigilStyles.COLORS["accent_cyan"],
            fontsize=14,
        )

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_grid(self, ax, grid, title, cmap, title_color=None):
        """Plot a single grid with VoxSigil styling"""
        if title_color is None:
            title_color = VoxSigilStyles.COLORS["accent_cyan"]

        grid_array = np.array(grid)
        im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)

        ax.set_title(title, color=title_color, fontsize=12)
        ax.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])

        # Grid lines if enabled
        if self.show_grid_lines.get():
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(
                which="minor",
                color=VoxSigilStyles.COLORS["border_inactive"],
                linestyle="-",
                linewidth=1,
                alpha=0.5,
            )

        ax.tick_params(colors=VoxSigilStyles.COLORS["accent_mint"], labelsize=8)

        # Add value annotations for small grids
        if grid_array.shape[0] <= 10 and grid_array.shape[1] <= 10:
            for i in range(grid_array.shape[0]):
                for j in range(grid_array.shape[1]):
                    text_color = "white" if grid_array[i, j] < 5 else "black"
                    ax.text(
                        j,
                        i,
                        str(grid_array[i, j]),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        weight="bold",
                    )

    def _plot_difference(self, ax, difference, title):
        """Plot difference map between target and prediction"""
        diff_cmap = plt.get_cmap("RdBu_r")
        im = ax.imshow(difference, cmap=diff_cmap, vmin=-1, vmax=1)

        ax.set_title(title, color=VoxSigilStyles.COLORS["accent_coral"], fontsize=12)
        ax.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])
        ax.tick_params(colors=VoxSigilStyles.COLORS["accent_mint"], labelsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Difference", color=VoxSigilStyles.COLORS["text_primary"])
        cbar.ax.tick_params(colors=VoxSigilStyles.COLORS["text_primary"])

    def _compute_difference(self, target, prediction):
        """Compute difference between target and prediction"""
        target_array = np.array(target)
        pred_array = np.array(prediction)

        # Ensure same shape
        if target_array.shape != pred_array.shape:
            max_h = max(target_array.shape[0], pred_array.shape[0])
            max_w = max(target_array.shape[1], pred_array.shape[1])

            target_padded = np.zeros((max_h, max_w))
            pred_padded = np.zeros((max_h, max_w))

            target_padded[: target_array.shape[0], : target_array.shape[1]] = (
                target_array
            )
            pred_padded[: pred_array.shape[0], : pred_array.shape[1]] = pred_array

            target_array, pred_array = target_padded, pred_padded

        # Compute difference (-1: target higher, 0: same, 1: prediction higher)
        difference = np.sign(pred_array - target_array)

        return difference

    def _create_error_plot(self, error_message):
        """Create an error visualization"""
        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"❌ Error\n\n{error_message}",
            ha="center",
            va="center",
            fontsize=14,
            color=VoxSigilStyles.COLORS["accent_coral"],
            transform=ax.transAxes,
        )

        ax.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])
        ax.set_xticks([])
        ax.set_yticks([])

        self.canvas.draw()

    def show_performance(self):
        """Show performance visualization"""
        try:
            # Get test results from testing interface
            if (
                hasattr(self.parent_gui, "testing_interface")
                and self.parent_gui.testing_interface.test_results
            ):
                results = self.parent_gui.testing_interface.test_results
                self._create_performance_plot(results)
            else:
                messagebox.showwarning(
                    "No Results", "Please run tests first to view performance."
                )

        except Exception as e:
            messagebox.showerror(
                "Performance Error", f"Failed to show performance: {str(e)}"
            )

    def _create_performance_plot(self, results):
        """Create performance visualization"""
        self.figure.clear()

        if not results:
            self._create_error_plot("No test results available")
            return

        # Extract performance metrics
        accuracies = [r.get("accuracy", 0) for r in results if "accuracy" in r]

        if not accuracies:
            self._create_error_plot("No accuracy data in results")
            return

        # Create subplots
        ax1 = self.figure.add_subplot(221)  # Accuracy over time
        ax2 = self.figure.add_subplot(222)  # Accuracy histogram
        ax3 = self.figure.add_subplot(223)  # Success rate
        ax4 = self.figure.add_subplot(224)  # Statistics

        # Accuracy over time
        ax1.plot(accuracies, color=VoxSigilStyles.COLORS["accent_cyan"], linewidth=2)
        ax1.set_title("Accuracy Over Tests", color=VoxSigilStyles.COLORS["accent_cyan"])
        ax1.set_xlabel("Test Number")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])

        # Accuracy histogram
        ax2.hist(
            accuracies, bins=20, color=VoxSigilStyles.COLORS["accent_mint"], alpha=0.7
        )
        ax2.set_title(
            "Accuracy Distribution", color=VoxSigilStyles.COLORS["accent_cyan"]
        )
        ax2.set_xlabel("Accuracy")
        ax2.set_ylabel("Frequency")
        ax2.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])

        # Success rate pie chart
        perfect_scores = sum(1 for acc in accuracies if acc >= 0.99)
        partial_scores = sum(1 for acc in accuracies if 0.01 <= acc < 0.99)
        zero_scores = sum(1 for acc in accuracies if acc < 0.01)

        labels = ["Perfect", "Partial", "Failed"]
        sizes = [perfect_scores, partial_scores, zero_scores]
        colors = [
            VoxSigilStyles.COLORS["accent_mint"],
            VoxSigilStyles.COLORS["accent_gold"],
            VoxSigilStyles.COLORS["accent_coral"],
        ]

        ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax3.set_title("Success Rate", color=VoxSigilStyles.COLORS["accent_cyan"])
        ax3.set_facecolor(VoxSigilStyles.COLORS["bg_secondary"])

        # Statistics text
        ax4.axis("off")
        stats_text = f"""Performance Statistics
        
Total Tests: {len(accuracies)}
Average Accuracy: {np.mean(accuracies):.3f}
Median Accuracy: {np.median(accuracies):.3f}
Best Accuracy: {np.max(accuracies):.3f}
Worst Accuracy: {np.min(accuracies):.3f}
Standard Deviation: {np.std(accuracies):.3f}

Perfect Scores: {perfect_scores} ({perfect_scores / len(accuracies) * 100:.1f}%)
Partial Scores: {partial_scores} ({partial_scores / len(accuracies) * 100:.1f}%)
Failed Tests: {zero_scores} ({zero_scores / len(accuracies) * 100:.1f}%)"""

        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            color=VoxSigilStyles.COLORS["text_primary"],
            bbox=dict(
                boxstyle="round",
                facecolor=VoxSigilStyles.COLORS["bg_secondary"],
                alpha=0.8,
            ),
        )

        self.figure.suptitle(
            "Model Performance Analysis",
            color=VoxSigilStyles.COLORS["accent_cyan"],
            fontsize=14,
        )

        self.figure.tight_layout()
        self.canvas.draw()

    def analyze_predictions(self):
        """Analyze prediction patterns"""
        messagebox.showinfo(
            "Feature Coming Soon",
            "Advanced prediction analysis will be available in a future update.",
        )

    def clear_display(self):
        """Clear the visualization display"""
        self._create_placeholder_plot()
        self.current_sample = None
        self.current_prediction = None

    def update_sample_range(self, max_samples):
        """Update the sample index range based on available data"""
        if hasattr(self, "sample_index"):
            # Update spinbox range
            # Note: This would require accessing the spinbox widget
            pass
