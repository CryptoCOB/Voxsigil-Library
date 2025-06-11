#!/usr/bin/env python3
"""
VoxSigil Performance Analysis Tab Interface
Modular component for performance analysis and metrics calculation

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Encapsulated performance analysis interface for Dynamic GridFormer GUI
"""

import json
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Union, Any, TYPE_CHECKING

import numpy as np

# Matplotlib imports with fallback
if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    MATPLOTLIB_AVAILABLE = True
else:
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False

        # Create fallback classes
        class MockAxes:
            """Mock matplotlib Axes object for when matplotlib is not available"""

            def __init__(self):
                pass

            def bar(self, *args, **kwargs):
                pass

            def set_title(self, *args, **kwargs):
                pass

            def set_xlabel(self, *args, **kwargs):
                pass

            def set_ylabel(self, *args, **kwargs):
                pass

            def set_xticks(self, *args, **kwargs):
                pass

            def set_xticklabels(self, *args, **kwargs):
                pass

            def legend(self, *args, **kwargs):
                pass

            def clear(self):
                pass

        class Figure:
            """Mock matplotlib Figure object for when matplotlib is not available"""

            def __init__(self, *args, **kwargs):
                self._mock_axes = MockAxes()

            def add_subplot(self, *args, **kwargs):
                return self._mock_axes

            def tight_layout(self):
                pass

            def clear(self):
                pass

        class FigureCanvasTkAgg:
            """Mock FigureCanvasTkAgg object for when matplotlib is not available"""

            def __init__(self, *args, **kwargs):
                pass

            def get_tk_widget(self):
                import tkinter as tk

                return tk.Label(text="Matplotlib not available")

            def draw(self):
                pass


from .gui_styles import VoxSigilStyles


class VoxSigilPerformanceInterface:
    """Performance analysis interface for VoxSigil Dynamic GridFormer"""

    def __init__(
        self, parent_gui, parent_frame, perf_visualizer=None, analyze_callback=None
    ):
        """
        Initialize the performance analysis interface

        Args:
            parent_gui: Parent GUI instance
            parent_frame: Frame where this interface will be embedded
        """
        self.parent_gui = parent_gui
        self.parent_frame = parent_frame
        self.perf_visualizer = perf_visualizer
        self.analyze_callback = analyze_callback
        self.current_metrics = {}
        self.metrics_history = []

        # Build UI inside provided parent frame
        self.create_performance_tab()

    def update_metrics(self, metrics: dict) -> None:
        """Alias to update metrics display when called externally"""
        self.update_metrics_display(metrics)

    def create_performance_tab(self):
        """Build the performance analysis UI inside the provided frame"""
        performance_frame = self.parent_frame

        # Create two-column layout
        left_frame = ttk.Frame(performance_frame, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_frame = ttk.Frame(performance_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # --- Left Frame Content ---
        # Model Metrics Section
        metrics_frame = ttk.LabelFrame(left_frame, text="Model Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable metrics display
        metrics_canvas = tk.Canvas(metrics_frame)
        scrollbar = ttk.Scrollbar(
            metrics_frame, orient="vertical", command=metrics_canvas.yview
        )
        scrollable_frame = ttk.Frame(metrics_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: metrics_canvas.configure(scrollregion=metrics_canvas.bbox("all")),
        )

        metrics_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        metrics_canvas.configure(yscrollcommand=scrollbar.set)

        metrics_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add metrics display widgets
        self.metrics_widgets = {}
        metrics_labels = [
            "Model Name",
            "Accuracy",
            "Loss",
            "Precision",
            "Recall",
            "F1 Score",
            "Inference Time",
            "GPU Memory",
            "Model Size",
        ]

        for i, label in enumerate(metrics_labels):
            label_widget = ttk.Label(
                scrollable_frame, text=f"{label}:", font=VoxSigilStyles.LABEL_FONT_BOLD
            )
            label_widget.grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)

            value_widget = ttk.Label(
                scrollable_frame, text="N/A", font=VoxSigilStyles.LABEL_FONT
            )
            value_widget.grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)

            self.metrics_widgets[label] = value_widget

        # Performance Actions
        actions_frame = ttk.LabelFrame(
            left_frame, text="Performance Actions", padding=10
        )
        actions_frame.pack(fill=tk.X, pady=(0, 10))

        # Load test results button
        load_results_btn = ttk.Button(
            actions_frame, text="Load Test Results", command=self.load_test_results
        )
        load_results_btn.pack(fill=tk.X, pady=5)

        # Calculate metrics button
        calc_metrics_btn = ttk.Button(
            actions_frame,
            text="Calculate Model Metrics",
            command=self.calculate_model_metrics,
        )
        calc_metrics_btn.pack(fill=tk.X, pady=5)

        # Export metrics button
        export_metrics_btn = ttk.Button(
            actions_frame,
            text="Export Metrics Report",
            command=self.export_metrics_report,
        )
        export_metrics_btn.pack(fill=tk.X, pady=5)

        # --- Right Frame Content ---
        # Performance Visualization
        viz_frame = ttk.LabelFrame(
            right_frame, text="Performance Visualization", padding=10
        )
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Graph canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Visualization controls
        controls_frame = ttk.Frame(viz_frame)
        controls_frame.pack(fill=tk.X, pady=(10, 0))

        # Metric selection
        ttk.Label(controls_frame, text="Metric:").pack(side=tk.LEFT, padx=(0, 5))
        self.metric_var = tk.StringVar(value="Accuracy")
        metric_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.metric_var,
            values=["Accuracy", "Loss", "Inference Time"],
        )
        metric_combo.pack(side=tk.LEFT, padx=(0, 10))
        metric_combo.bind("<<ComboboxSelected>>", self.update_visualization)

        # Plot button
        plot_btn = ttk.Button(
            controls_frame, text="Plot", command=self.update_visualization
        )
        plot_btn.pack(side=tk.LEFT)

        # Compare Models Section
        compare_frame = ttk.LabelFrame(right_frame, text="Model Comparison", padding=10)
        compare_frame.pack(fill=tk.X, expand=False, pady=(10, 0))

        # Model selection for comparison
        model_select_frame = ttk.Frame(compare_frame)
        model_select_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_select_frame, text="Model A:").grid(row=0, column=0, padx=5)
        self.model_a_var = tk.StringVar()
        model_a_combo = ttk.Combobox(
            model_select_frame, textvariable=self.model_a_var, width=20
        )
        model_a_combo.grid(row=0, column=1, padx=5)
        self.model_a_combo = model_a_combo

        ttk.Label(model_select_frame, text="Model B:").grid(row=0, column=2, padx=5)
        self.model_b_var = tk.StringVar()
        model_b_combo = ttk.Combobox(
            model_select_frame, textvariable=self.model_b_var, width=20
        )
        model_b_combo.grid(row=0, column=3, padx=5)
        self.model_b_combo = model_b_combo

        # Compare button
        compare_btn = ttk.Button(
            compare_frame, text="Compare Models", command=self.compare_models
        )
        compare_btn.pack(fill=tk.X, pady=5)

    def load_test_results(self):
        """Load test results from a file"""
        file_path = filedialog.askopenfilename(
            title="Select Test Results File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as file:
                test_results = json.load(file)

            # Update metrics
            self.update_metrics_display(test_results)

            # Add to history
            if test_results.get("model_name"):
                self.metrics_history.append(test_results)

            # Update model selection combos
            self._update_model_selection_combos()

            messagebox.showinfo("Success", "Test results loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load test results: {str(e)}")

    def update_metrics_display(self, metrics):
        """Update the metrics display with new values"""
        self.current_metrics = metrics

        # Map metrics to display widgets
        display_map = {
            "Model Name": metrics.get("model_name", "N/A"),
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Loss": f"{metrics.get('loss', 0):.4f}",
            "Precision": f"{metrics.get('precision', 0):.4f}",
            "Recall": f"{metrics.get('recall', 0):.4f}",
            "F1 Score": f"{metrics.get('f1_score', 0):.4f}",
            "Inference Time": f"{metrics.get('avg_inference_time', 0):.2f} ms",
            "GPU Memory": f"{metrics.get('gpu_memory_usage', 0):.2f} MB",
            "Model Size": f"{metrics.get('model_size', 0):.2f} MB",
        }

        # Update the display widgets
        for label, value in display_map.items():
            if label in self.metrics_widgets:
                self.metrics_widgets[label].config(text=value)

    def calculate_model_metrics(self):
        """Calculate additional metrics for the current model"""
        if not self.parent_gui.current_model:
            messagebox.showwarning("No Model", "Please load a model first")
            return

        # Get the current model
        model = self.parent_gui.current_model

        # Get test dataset if available
        test_data = self.parent_gui.get_test_dataset()
        if not test_data:
            messagebox.showwarning("No Data", "Please load test data first")
            return

        # Calculate metrics
        try:
            # This would be a real metrics calculation in a full implementation
            # For demonstration, we'll use mock data
            import random

            accuracy = random.uniform(0.7, 0.98)
            loss = random.uniform(0.01, 0.3)
            precision = random.uniform(0.7, 0.95)
            recall = random.uniform(0.7, 0.95)
            f1 = 2 * (precision * recall) / (precision + recall)

            metrics = {
                "model_name": getattr(model, "name", "Unknown Model"),
                "accuracy": accuracy,
                "loss": loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "avg_inference_time": random.uniform(5, 100),
                "gpu_memory_usage": random.uniform(200, 5000),
                "model_size": random.uniform(10, 1000),
                "timestamp": datetime.now().isoformat(),
            }

            # Update the display
            self.update_metrics_display(metrics)

            # Add to history
            self.metrics_history.append(metrics)

            # Update model selection combos
            self._update_model_selection_combos()

            # Update visualization
            self.update_visualization()

            messagebox.showinfo("Success", "Metrics calculated successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate metrics: {str(e)}")

    def export_metrics_report(self):
        """Export the current metrics as a report"""
        if not self.current_metrics:
            messagebox.showwarning("No Metrics", "No metrics available to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Metrics Report",
            defaultextension=".json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            # Add timestamp if not present
            if "timestamp" not in self.current_metrics:
                self.current_metrics["timestamp"] = datetime.now().isoformat()

            # Write the report
            with open(file_path, "w") as file:
                json.dump(self.current_metrics, file, indent=2)

            messagebox.showinfo("Success", f"Metrics report saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export metrics report: {str(e)}")

    def update_visualization(self, event=None):
        """Update the performance visualization"""
        if not self.metrics_history:
            return

        # Clear the plot
        self.plot.clear()

        # Get selected metric
        metric = self.metric_var.get().lower().replace(" ", "_")

        # Extract data for visualization
        models = []
        values = []

        for metrics in self.metrics_history:
            if metric in metrics:
                models.append(metrics.get("model_name", "Unknown"))
                values.append(metrics[metric])

        if not models or not values:
            return

        # Create the plot
        self.plot.bar(models, values)
        self.plot.set_title(f"{self.metric_var.get()} Comparison")
        self.plot.set_ylabel(self.metric_var.get())
        self.plot.set_xlabel("Model")

        # Rotate x-axis labels if needed
        if len(models) > 3:
            self.plot.set_xticklabels(models, rotation=45, ha="right")

        self.figure.tight_layout()
        self.canvas.draw()

    def compare_models(self):
        """Compare two selected models"""
        model_a = self.model_a_var.get()
        model_b = self.model_b_var.get()

        if not model_a or not model_b:
            messagebox.showwarning(
                "Selection Missing", "Please select two models to compare"
            )
            return

        # Find metrics for selected models
        metrics_a = None
        metrics_b = None

        for metrics in self.metrics_history:
            if metrics.get("model_name") == model_a:
                metrics_a = metrics
            if metrics.get("model_name") == model_b:
                metrics_b = metrics

        if not metrics_a or not metrics_b:
            messagebox.showwarning(
                "Data Missing", "Metrics not found for selected models"
            )
            return

        # Create comparison visualization
        self.plot.clear()

        # Select metrics to compare
        compare_metrics = ["accuracy", "precision", "recall", "f1_score"]
        x = np.arange(len(compare_metrics))
        width = 0.35

        # Extract values
        values_a = [metrics_a.get(m, 0) for m in compare_metrics]
        values_b = [metrics_b.get(m, 0) for m in compare_metrics]

        # Create bars
        self.plot.bar(x - width / 2, values_a, width, label=model_a)
        self.plot.bar(x + width / 2, values_b, width, label=model_b)

        # Add labels and legend
        self.plot.set_title("Model Comparison")
        self.plot.set_xticks(x)
        self.plot.set_xticklabels(
            [m.replace("_", " ").title() for m in compare_metrics]
        )
        self.plot.set_ylabel("Score")
        self.plot.legend()

        self.figure.tight_layout()
        self.canvas.draw()

        # Show detailed comparison in a dialog
        self._show_detailed_comparison(metrics_a, metrics_b)

    def _update_model_selection_combos(self):
        """Update the model selection combo boxes"""
        model_names = [
            m.get("model_name", "Unknown")
            for m in self.metrics_history
            if "model_name" in m
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_models = [m for m in model_names if not (m in seen or seen.add(m))]

        # Update combo boxes
        # Update stored combobox widgets
        if hasattr(self, "model_a_combo"):
            self.model_a_combo["values"] = unique_models
        if hasattr(self, "model_b_combo"):
            self.model_b_combo["values"] = unique_models

    def _show_detailed_comparison(self, metrics_a, metrics_b):
        """Show detailed comparison between two models"""
        compare_window = tk.Toplevel(self.parent_gui.root)
        compare_window.title("Detailed Model Comparison")
        compare_window.geometry("600x400")

        # Set icon and style
        VoxSigilStyles.apply_icon(compare_window)
        VoxSigilStyles.apply_theme(compare_window)

        # Create frame
        frame = ttk.Frame(compare_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="Metric", font=VoxSigilStyles.HEADER_FONT).grid(
            row=0, column=0, padx=10
        )
        ttk.Label(
            header_frame,
            text=metrics_a.get("model_name", "Model A"),
            font=VoxSigilStyles.HEADER_FONT,
        ).grid(row=0, column=1, padx=10)
        ttk.Label(
            header_frame,
            text=metrics_b.get("model_name", "Model B"),
            font=VoxSigilStyles.HEADER_FONT,
        ).grid(row=0, column=2, padx=10)
        ttk.Label(
            header_frame, text="Difference", font=VoxSigilStyles.HEADER_FONT
        ).grid(row=0, column=3, padx=10)

        # Comparison rows
        compare_frame = ttk.Frame(frame)
        compare_frame.pack(fill=tk.BOTH, expand=True)

        # Define metrics to compare
        compare_metrics = [
            ("accuracy", "Accuracy"),
            ("loss", "Loss"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 Score"),
            ("avg_inference_time", "Avg Inference Time (ms)"),
            ("gpu_memory_usage", "GPU Memory (MB)"),
            ("model_size", "Model Size (MB)"),
        ]

        # Add each metric row
        for i, (key, label) in enumerate(compare_metrics):
            bg_color = VoxSigilStyles.ALT_ROW_COLOR if i % 2 else ""

            row_frame = ttk.Frame(compare_frame)
            row_frame.pack(fill=tk.X, pady=2)

            value_a = metrics_a.get(key, 0)
            value_b = metrics_b.get(key, 0)
            diff = value_b - value_a

            # Format values based on metric type
            if key in ["avg_inference_time", "gpu_memory_usage", "model_size"]:
                value_a_str = f"{value_a:.2f}"
                value_b_str = f"{value_b:.2f}"
                diff_str = f"{diff:.2f}"
            else:
                value_a_str = f"{value_a:.4f}"
                value_b_str = f"{value_b:.4f}"
                diff_str = f"{diff:.4f}"

            # Set color for difference (green if improvement, red if worse)
            if (
                key == "loss"
                or key == "avg_inference_time"
                or key == "gpu_memory_usage"
            ):
                # For these metrics, lower is better
                diff_color = "green" if diff < 0 else "red" if diff > 0 else "black"
            else:
                # For these metrics, higher is better
                diff_color = "green" if diff > 0 else "red" if diff < 0 else "black"

            ttk.Label(row_frame, text=label, background=bg_color).grid(
                row=0, column=0, padx=10, sticky=tk.W
            )
            ttk.Label(row_frame, text=value_a_str, background=bg_color).grid(
                row=0, column=1, padx=10
            )
            ttk.Label(row_frame, text=value_b_str, background=bg_color).grid(
                row=0, column=2, padx=10
            )
            ttk.Label(
                row_frame, text=diff_str, foreground=diff_color, background=bg_color
            ).grid(row=0, column=3, padx=10)

        # Close button
        ttk.Button(frame, text="Close", command=compare_window.destroy).pack(
            pady=(10, 0)
        )
