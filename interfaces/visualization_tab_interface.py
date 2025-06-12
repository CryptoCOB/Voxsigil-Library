#!/usr/bin/env python3
"""
VoxSigil Visualization Tab Interface - Qt5 Version
Modular component for visualization functionality with enhanced features

Created by: GitHub Copilot
Purpose: Encapsulated visualization interface for Dynamic GridFormer GUI
"""

import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QCheckBox, QGroupBox, QMessageBox, QComboBox, QSlider, QTextEdit,
    QSplitter, QFrame, QGridLayout, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette


class VoxSigilStyles:
    """Qt5 Color scheme and styling for VoxSigil"""
    COLORS = {
        "bg_primary": "#1a1a1a",
        "bg_secondary": "#2d2d2d", 
        "bg_tertiary": "#404040",
        "text_primary": "#ffffff",
        "text_secondary": "#cccccc",
        "accent_cyan": "#00ffff",
        "accent_mint": "#00ff88",
        "accent_coral": "#ff6b6b",
        "accent_gold": "#ffd700",
        "border_inactive": "#555555"
    }

    @classmethod
    def apply_dark_theme(cls, widget):
        """Apply dark theme to widget"""
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {cls.COLORS['bg_primary']};
                color: {cls.COLORS['text_primary']};
            }}
            QGroupBox {{
                border: 2px solid {cls.COLORS['border_inactive']};
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: {cls.COLORS['accent_cyan']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {cls.COLORS['bg_secondary']};
                border: 2px solid {cls.COLORS['accent_cyan']};
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                color: {cls.COLORS['text_primary']};
            }}
            QPushButton:hover {{
                background-color: {cls.COLORS['accent_cyan']};
                color: {cls.COLORS['bg_primary']};
            }}
            QPushButton:pressed {{
                background-color: {cls.COLORS['bg_tertiary']};
            }}
            QCheckBox {{
                color: {cls.COLORS['text_primary']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.COLORS['accent_cyan']};
            }}
            QSpinBox, QComboBox {{
                background-color: {cls.COLORS['bg_secondary']};
                border: 1px solid {cls.COLORS['border_inactive']};
                border-radius: 3px;
                padding: 5px;
                color: {cls.COLORS['text_primary']};
            }}
            QSlider::groove:horizontal {{
                background: {cls.COLORS['bg_secondary']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {cls.COLORS['accent_cyan']};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QTextEdit {{
                background-color: {cls.COLORS['bg_secondary']};
                border: 1px solid {cls.COLORS['border_inactive']};
                border-radius: 3px;
                color: {cls.COLORS['text_primary']};
            }}
        """)


class AnimationController:
    """Feature 1: Animation controller for smooth transitions"""
    
    def __init__(self, parent):
        self.parent = parent
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate_step)
        self.animation_steps = []
        self.current_step = 0
        
    def animate_grid_transition(self, from_grid, to_grid, duration_ms=1000):
        """Animate transition between two grids"""
        steps = 30  # 30 frames
        self.animation_steps = []
        
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated = self._interpolate_grids(from_grid, to_grid, alpha)
            self.animation_steps.append(interpolated)
            
        self.current_step = 0
        self.timer.start(duration_ms // steps)
    
    def _interpolate_grids(self, grid1, grid2, alpha):
        """Interpolate between two grids"""
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)
        
        # Simple alpha blending for demonstration
        result = arr1 * (1 - alpha) + arr2 * alpha
        return result.astype(int)
    
    def _animate_step(self):
        """Process one animation step"""
        if self.current_step < len(self.animation_steps):
            self.parent._update_animated_grid(self.animation_steps[self.current_step])
            self.current_step += 1
        else:
            self.timer.stop()
            self.parent.animation_complete.emit()


class ExportManager:
    """Feature 2: Advanced export capabilities"""
    
    @staticmethod
    def export_to_formats(figure, base_filename, formats=['png', 'svg', 'pdf']):
        """Export visualization to multiple formats"""
        exported_files = []
        
        for fmt in formats:
            filename = f"{base_filename}.{fmt}"
            try:
                figure.savefig(filename, format=fmt, dpi=300, 
                             bbox_inches='tight', facecolor='#1a1a1a')
                exported_files.append(filename)
            except Exception as e:
                print(f"Error exporting to {fmt}: {e}")
                
        return exported_files
    
    @staticmethod
    def export_grid_data(grid_data, filename):
        """Export grid data to CSV format"""
        try:
            np.savetxt(filename, grid_data, delimiter=',', fmt='%d')
            return True
        except Exception as e:
            print(f"Error exporting grid data: {e}")
            return False


class ComparisonEngine:
    """Feature 3: Advanced grid comparison with metrics"""
    
    @staticmethod
    def compute_similarity_metrics(grid1, grid2):
        """Compute various similarity metrics between two grids"""
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)
        
        # Ensure same shape
        if arr1.shape != arr2.shape:
            max_h = max(arr1.shape[0], arr2.shape[0])
            max_w = max(arr1.shape[1], arr2.shape[1])
            
            padded1 = np.zeros((max_h, max_w))
            padded2 = np.zeros((max_h, max_w))
            
            padded1[:arr1.shape[0], :arr1.shape[1]] = arr1
            padded2[:arr2.shape[0], :arr2.shape[1]] = arr2
            
            arr1, arr2 = padded1, padded2
        
        metrics = {
            'pixel_accuracy': np.mean(arr1 == arr2),
            'hamming_distance': np.sum(arr1 != arr2),
            'mean_absolute_error': np.mean(np.abs(arr1 - arr2)),
            'structural_similarity': ComparisonEngine._compute_ssim(arr1, arr2),
            'pattern_similarity': ComparisonEngine._compute_pattern_similarity(arr1, arr2)
        }
        
        return metrics
    
    @staticmethod
    def _compute_ssim(img1, img2):
        """Simplified SSIM computation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    @staticmethod
    def _compute_pattern_similarity(grid1, grid2):
        """Compute pattern-based similarity"""
        # Simple pattern detection using gradient
        grad1_x = np.gradient(grid1, axis=1)
        grad1_y = np.gradient(grid1, axis=0)
        grad2_x = np.gradient(grid2, axis=1)
        grad2_y = np.gradient(grid2, axis=0)
        
        correlation_x = np.corrcoef(grad1_x.flatten(), grad2_x.flatten())[0, 1]
        correlation_y = np.corrcoef(grad1_y.flatten(), grad2_y.flatten())[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation_x):
            correlation_x = 0
        if np.isnan(correlation_y):
            correlation_y = 0
            
        return (correlation_x + correlation_y) / 2


class InteractiveAnnotator:
    """Feature 4: Interactive annotation system"""
    
    def __init__(self):
        self.annotations = {}
        self.annotation_mode = False
    
    def add_annotation(self, position, text, annotation_type='note'):
        """Add annotation at specific position"""
        self.annotations[position] = {
            'text': text,
            'type': annotation_type,
            'timestamp': np.datetime64('now')
        }
    
    def remove_annotation(self, position):
        """Remove annotation at position"""
        if position in self.annotations:
            del self.annotations[position]
    
    def get_annotations_for_display(self):
        """Get formatted annotations for display"""
        return self.annotations
    
    def export_annotations(self, filename):
        """Export annotations to file"""
        try:
            with open(filename, 'w') as f:
                for pos, annotation in self.annotations.items():
                    f.write(f"{pos}: {annotation['text']} ({annotation['type']})\n")
            return True
        except Exception as e:
            print(f"Error exporting annotations: {e}")
            return False


class StatisticsCalculator:
    """Feature 5: Comprehensive statistics calculator"""
    
    @staticmethod
    def compute_grid_statistics(grid):
        """Compute comprehensive statistics for a grid"""
        arr = np.array(grid)
        
        stats = {
            'shape': arr.shape,
            'total_cells': arr.size,
            'unique_values': len(np.unique(arr)),
            'value_counts': dict(zip(*np.unique(arr, return_counts=True))),
            'mean': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'entropy': StatisticsCalculator._compute_entropy(arr),
            'complexity': StatisticsCalculator._compute_complexity(arr),
            'symmetry_scores': StatisticsCalculator._compute_symmetry(arr)
        }
        
        return stats
    
    @staticmethod
    def _compute_entropy(arr):
        """Compute Shannon entropy"""
        values, counts = np.unique(arr, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    @staticmethod
    def _compute_complexity(arr):
        """Compute visual complexity metric"""
        # Count transitions between different values
        transitions = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1] - 1):
                if arr[i, j] != arr[i, j + 1]:
                    transitions += 1
        
        for i in range(arr.shape[0] - 1):
            for j in range(arr.shape[1]):
                if arr[i, j] != arr[i + 1, j]:
                    transitions += 1
        
        max_transitions = 2 * arr.shape[0] * arr.shape[1] - arr.shape[0] - arr.shape[1]
        complexity = transitions / max_transitions if max_transitions > 0 else 0
        
        return complexity
    
    @staticmethod
    def _compute_symmetry(arr):
        """Compute symmetry scores"""
        scores = {
            'horizontal': np.mean(arr == np.fliplr(arr)),
            'vertical': np.mean(arr == np.flipud(arr)),
            'diagonal': np.mean(arr == arr.T) if arr.shape[0] == arr.shape[1] else 0,
            'anti_diagonal': 0
        }
        
        if arr.shape[0] == arr.shape[1]:
            scores['anti_diagonal'] = np.mean(arr == np.fliplr(arr.T))
            
        return scores


class VoxSigilVisualizationInterface(QWidget):
    """Enhanced Qt5 Visualization interface for VoxSigil Dynamic GridFormer"""
    
    # Signals
    animation_complete = pyqtSignal()
    visualization_updated = pyqtSignal()
    
    def __init__(self, parent_gui, tab_widget):
        super().__init__()
        self.parent_gui = parent_gui
        self.tab_widget = tab_widget
        
        # Enhanced features
        self.animation_controller = AnimationController(self)
        self.export_manager = ExportManager()
        self.comparison_engine = ComparisonEngine()
        self.annotator = InteractiveAnnotator()
        self.stats_calculator = StatisticsCalculator()
        
        # Visualization state
        self.current_sample = None
        self.current_prediction = None
        self.figure = None
        self.canvas = None
        
        # UI components
        self.sample_spinbox = None
        self.show_grid_lines = None
        self.show_differences = None
        self.show_confidence = None
        self.show_annotations = None
        self.animation_speed = None
        self.comparison_mode = None
        
        # Initialize UI
        self._setup_ui()
        self._connect_signals()
        
        # Add to tab widget
        self.tab_widget.addTab(self, "📊 Visualization")
    
    def _setup_ui(self):
        """Setup the user interface"""
        VoxSigilStyles.apply_dark_theme(self)
        
        layout = QVBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Control panel
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # Visualization area
        viz_area = self._create_visualization_area()
        splitter.addWidget(viz_area)
        
        # Statistics panel
        stats_panel = self._create_statistics_panel()
        splitter.addWidget(stats_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 0)  # Control panel - fixed height
        splitter.setStretchFactor(1, 1)  # Visualization - expandable
        splitter.setStretchFactor(2, 0)  # Statistics - fixed height
    
    def _create_control_panel(self):
        """Create the enhanced control panel"""
        panel = QGroupBox("Visualization Controls")
        layout = QVBoxLayout(panel)
        
        # Main controls
        main_controls = QHBoxLayout()
        
        btn_visualize = QPushButton("🎨 Visualize Sample")
        btn_visualize.clicked.connect(self.visualize_sample)
        main_controls.addWidget(btn_visualize)
        
        btn_performance = QPushButton("📈 Show Performance")
        btn_performance.clicked.connect(self.show_performance)
        main_controls.addWidget(btn_performance)
        
        btn_analyze = QPushButton("🔍 Analyze Predictions")
        btn_analyze.clicked.connect(self.analyze_predictions)
        main_controls.addWidget(btn_analyze)
        
        btn_export = QPushButton("💾 Export")
        btn_export.clicked.connect(self.export_visualization)
        main_controls.addWidget(btn_export)
        
        btn_clear = QPushButton("🔄 Clear Display")
        btn_clear.clicked.connect(self.clear_display)
        main_controls.addWidget(btn_clear)
        
        layout.addLayout(main_controls)
        
        # Sample selection and options
        options_layout = QGridLayout()
        
        # Sample selection
        options_layout.addWidget(QLabel("Sample Index:"), 0, 0)
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setRange(0, 1000)
        self.sample_spinbox.setValue(0)
        options_layout.addWidget(self.sample_spinbox, 0, 1)
        
        # Comparison mode
        options_layout.addWidget(QLabel("Comparison Mode:"), 0, 2)
        self.comparison_mode = QComboBox()
        self.comparison_mode.addItems(['None', 'Side-by-side', 'Overlay', 'Difference'])
        options_layout.addWidget(self.comparison_mode, 0, 3)
        
        # Animation speed
        options_layout.addWidget(QLabel("Animation Speed:"), 1, 0)
        self.animation_speed = QSlider(Qt.Horizontal)
        self.animation_speed.setRange(1, 10)
        self.animation_speed.setValue(5)
        options_layout.addWidget(self.animation_speed, 1, 1)
        
        # Checkboxes
        self.show_grid_lines = QCheckBox("Grid Lines")
        self.show_grid_lines.setChecked(True)
        options_layout.addWidget(self.show_grid_lines, 1, 2)
        
        self.show_differences = QCheckBox("Show Differences")
        self.show_differences.setChecked(True)
        options_layout.addWidget(self.show_differences, 1, 3)
        
        self.show_confidence = QCheckBox("Confidence Map")
        options_layout.addWidget(self.show_confidence, 2, 0)
        
        self.show_annotations = QCheckBox("Show Annotations")
        self.show_annotations.setChecked(True)
        options_layout.addWidget(self.show_annotations, 2, 1)
        
        # Animation controls
        btn_animate = QPushButton("▶️ Animate Transition")
        btn_animate.clicked.connect(self.animate_transition)
        options_layout.addWidget(btn_animate, 2, 2)
        
        btn_annotate = QPushButton("📝 Toggle Annotation Mode")
        btn_annotate.clicked.connect(self.toggle_annotation_mode)
        options_layout.addWidget(btn_annotate, 2, 3)
        
        layout.addLayout(options_layout)
        
        return panel
    
    def _create_visualization_area(self):
        """Create the main visualization area"""
        panel = QGroupBox("Grid Visualization")
        layout = QVBoxLayout(panel)
        
        # Setup matplotlib with dark theme
        plt.style.use('dark_background')
        
        self.figure = Figure(figsize=(12, 8), 
                           facecolor=VoxSigilStyles.COLORS['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(self.canvas)
        
        # Initialize with placeholder
        self._create_placeholder_plot()
        
        return panel
    
    def _create_statistics_panel(self):
        """Create the statistics display panel"""
        panel = QGroupBox("Statistics & Analysis")
        layout = QHBoxLayout(panel)
        
        # Statistics text area
        self.stats_display = QTextEdit()
        self.stats_display.setMaximumHeight(150)
        self.stats_display.setPlainText("Load a sample to view statistics...")
        layout.addWidget(self.stats_display)
        
        # Comparison metrics area
        self.comparison_display = QTextEdit()
        self.comparison_display.setMaximumHeight(150)
        self.comparison_display.setPlainText("Run inference to view comparison metrics...")
        layout.addWidget(self.comparison_display)
        
        return panel
    
    def _connect_signals(self):
        """Connect Qt signals"""
        self.animation_complete.connect(self._on_animation_complete)
        self.visualization_updated.connect(self._update_statistics)
    
    def _create_placeholder_plot(self):
        """Create a placeholder visualization"""
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 
                "📊 VoxSigil Visualization\n\nLoad data and select a sample to visualize",
                ha='center', va='center', fontsize=16,
                color=VoxSigilStyles.COLORS['accent_cyan'],
                transform=ax.transAxes)
        
        ax.set_facecolor(VoxSigilStyles.COLORS['bg_secondary'])
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.canvas.draw()
    
    def visualize_sample(self):
        """Visualize a specific sample with enhanced features"""
        try:
            # Get test data
            if (hasattr(self.parent_gui, 'testing_interface') and 
                self.parent_gui.testing_interface.test_data):
                test_data = self.parent_gui.testing_interface.test_data
            else:
                QMessageBox.warning(self, "No Data", 
                                  "Please load test data in the Testing tab first.")
                return
            
            sample_idx = self.sample_spinbox.value()
            if sample_idx >= len(test_data):
                QMessageBox.warning(self, "Invalid Index", 
                                  f"Sample index must be less than {len(test_data)}")
                return
            
            sample = test_data[sample_idx]
            self.current_sample = sample
            
            # Get prediction if model is loaded
            prediction = None
            if self.parent_gui.current_model:
                try:
                    result = self.parent_gui.inference_engine.run_single_inference(sample)
                    if result and 'prediction' in result:
                        prediction = result['prediction']
                        self.current_prediction = prediction
                except Exception as e:
                    print(f"Error getting prediction: {e}")
            
            # Create enhanced visualization
            self._create_enhanced_sample_visualization(sample, prediction)
            self.visualization_updated.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", 
                               f"Failed to visualize sample: {str(e)}")
    
    def _create_enhanced_sample_visualization(self, sample, prediction=None):
        """Create enhanced sample visualization with new features"""
        self.figure.clear()
        
        # Extract grids
        input_grids = sample.get('input', [])
        output_grid = sample.get('output')
        
        if not input_grids:
            self._create_error_plot("No input grids found in sample")
            return
        
        # Determine layout based on comparison mode
        comparison_mode = self.comparison_mode.currentText()
        
        if comparison_mode == 'Side-by-side' and prediction is not None and output_grid is not None:
            self._create_side_by_side_visualization(input_grids, output_grid, prediction)
        elif comparison_mode == 'Overlay' and prediction is not None and output_grid is not None:
            self._create_overlay_visualization(input_grids, output_grid, prediction)
        elif comparison_mode == 'Difference' and prediction is not None and output_grid is not None:
            self._create_difference_visualization(input_grids, output_grid, prediction)
        else:
            self._create_standard_visualization(input_grids, output_grid, prediction)
        
        # Add annotations if enabled
        if self.show_annotations.isChecked():
            self._add_annotations_to_plot()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _create_side_by_side_visualization(self, input_grids, output_grid, prediction):
        """Create side-by-side comparison visualization"""
        # Implementation for side-by-side layout
        cols = len(input_grids) + 2  # inputs + target + prediction
        
        cmap = plt.get_cmap('Set3')
        
        # Plot inputs
        for i, input_grid in enumerate(input_grids):
            ax = self.figure.add_subplot(1, cols, i + 1)
            self._plot_enhanced_grid(ax, input_grid, f'Input {i + 1}', cmap)
        
        # Plot target
        ax = self.figure.add_subplot(1, cols, cols - 1)
        self._plot_enhanced_grid(ax, output_grid, 'Target', cmap)
        
        # Plot prediction
        ax = self.figure.add_subplot(1, cols, cols)
        is_correct = np.array_equal(prediction, output_grid)
        title_color = VoxSigilStyles.COLORS['accent_mint'] if is_correct else VoxSigilStyles.COLORS['accent_coral']
        self._plot_enhanced_grid(ax, prediction, 'Prediction', cmap, title_color)
    
    def _create_standard_visualization(self, input_grids, output_grid, prediction):
        """Create standard grid visualization"""
        # Similar to original implementation but with enhancements
        num_inputs = len(input_grids)
        cols = min(4, num_inputs + (2 if output_grid is not None else 1) + 
                  (1 if prediction is not None else 0))
        rows = max(1, (num_inputs + cols - 1) // cols)
        
        if output_grid is not None or prediction is not None:
            rows = max(rows, 2)
        
        cmap = plt.get_cmap('Set3')
        
        # Plot input grids
        for i, input_grid in enumerate(input_grids):
            if i >= 8:  # Limit to prevent overcrowding
                break
            
            ax = self.figure.add_subplot(rows, cols, i + 1)
            self._plot_enhanced_grid(ax, input_grid, f'Input {i + 1}', cmap)
        
        # Plot target and prediction as before, but with enhancements
        if output_grid is not None:
            ax = self.figure.add_subplot(rows, cols, cols * (rows - 1) + 1)
            self._plot_enhanced_grid(ax, output_grid, 'Expected Output', cmap)
        
        if prediction is not None:
            ax_pos = cols * (rows - 1) + (2 if output_grid is not None else 1)
            if ax_pos <= rows * cols:
                ax = self.figure.add_subplot(rows, cols, ax_pos)
                is_correct = np.array_equal(prediction, output_grid) if output_grid is not None else False
                title = 'Prediction ✓' if is_correct else 'Prediction ✗'
                title_color = VoxSigilStyles.COLORS['accent_mint'] if is_correct else VoxSigilStyles.COLORS['accent_coral']
                self._plot_enhanced_grid(ax, prediction, title, cmap, title_color)
    
    def _plot_enhanced_grid(self, ax, grid, title, cmap, title_color=None):
        """Enhanced grid plotting with additional features"""
        if title_color is None:
            title_color = VoxSigilStyles.COLORS['accent_cyan']
        
        grid_array = np.array(grid)
        im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)
        
        ax.set_title(title, color=title_color, fontsize=12, weight='bold')
        ax.set_facecolor(VoxSigilStyles.COLORS['bg_secondary'])
        
        # Enhanced grid lines
        if self.show_grid_lines.isChecked():
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(which='minor', color=VoxSigilStyles.COLORS['border_inactive'],
                   linestyle='-', linewidth=1, alpha=0.7)
        
        # Enhanced value annotations for small grids
        if grid_array.shape[0] <= 10 and grid_array.shape[1] <= 10:
            for i in range(grid_array.shape[0]):
                for j in range(grid_array.shape[1]):
                    text_color = 'white' if grid_array[i, j] < 5 else 'black'
                    ax.text(j, i, str(grid_array[i, j]),
                           ha='center', va='center', color=text_color,
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='black', alpha=0.3))
        
        ax.tick_params(colors=VoxSigilStyles.COLORS['accent_mint'], labelsize=8)
    
    def _update_statistics(self):
        """Update statistics display"""
        if self.current_sample is None:
            return
        
        # Compute statistics for current sample
        input_grids = self.current_sample.get('input', [])
        output_grid = self.current_sample.get('output')
        
        stats_text = "=== Sample Statistics ===\n\n"
        
        if input_grids:
            for i, grid in enumerate(input_grids):
                stats = self.stats_calculator.compute_grid_statistics(grid)
                stats_text += f"Input {i + 1}:\n"
                stats_text += f"  Shape: {stats['shape']}\n"
                stats_text += f"  Unique values: {stats['unique_values']}\n"
                stats_text += f"  Entropy: {stats['entropy']:.3f}\n"
                stats_text += f"  Complexity: {stats['complexity']:.3f}\n\n"
        
        if output_grid is not None:
            stats = self.stats_calculator.compute_grid_statistics(output_grid)
            stats_text += f"Target Output:\n"
            stats_text += f"  Shape: {stats['shape']}\n"
            stats_text += f"  Unique values: {stats['unique_values']}\n"
            stats_text += f"  Entropy: {stats['entropy']:.3f}\n"
            stats_text += f"  Complexity: {stats['complexity']:.3f}\n"
        
        self.stats_display.setPlainText(stats_text)
        
        # Update comparison metrics if prediction exists
        if self.current_prediction is not None and output_grid is not None:
            metrics = self.comparison_engine.compute_similarity_metrics(
                output_grid, self.current_prediction)
            
            comparison_text = "=== Comparison Metrics ===\n\n"
            comparison_text += f"Pixel Accuracy: {metrics['pixel_accuracy']:.3f}\n"
            comparison_text += f"Hamming Distance: {metrics['hamming_distance']}\n"
            comparison_text += f"Mean Absolute Error: {metrics['mean_absolute_error']:.3f}\n"
            comparison_text += f"Structural Similarity: {metrics['structural_similarity']:.3f}\n"
            comparison_text += f"Pattern Similarity: {metrics['pattern_similarity']:.3f}\n"
            
            self.comparison_display.setPlainText(comparison_text)
    
    def animate_transition(self):
        """Animate transition between input and prediction"""
        if self.current_sample is None or self.current_prediction is None:
            QMessageBox.warning(self, "No Data", "Need both sample and prediction for animation")
            return
        
        input_grids = self.current_sample.get('input', [])
        if not input_grids:
            return
        
        # Animate from first input to prediction
        speed_factor = self.animation_speed.value()
        duration = 2000 // speed_factor  # 2 seconds base, adjusted by speed
        
        self.animation_controller.animate_grid_transition(
            input_grids[0], self.current_prediction, duration)
    
    def _update_animated_grid(self, grid):
        """Update display with animated grid frame"""
        # This would update a specific subplot with the animated frame
        # Implementation depends on how you want to display the animation
        pass
    
    def _on_animation_complete(self):
        """Handle animation completion"""
        # Restore normal visualization
        if self.current_sample:
            self.visualize_sample()
    
    def toggle_annotation_mode(self):
        """Toggle annotation mode"""
        self.annotator.annotation_mode = not self.annotator.annotation_mode
        status = "enabled" if self.annotator.annotation_mode else "disabled"
        QMessageBox.information(self, "Annotation Mode", f"Annotation mode {status}")
    
    def export_visualization(self):
        """Export current visualization"""
        if self.figure is None:
            QMessageBox.warning(self, "No Visualization", "Nothing to export")
            return
        
        base_filename = f"voxsigil_sample_{self.sample_spinbox.value()}"
        formats = ['png', 'svg', 'pdf']
        
        exported = self.export_manager.export_to_formats(self.figure, base_filename, formats)
        
        if exported:
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported to: {', '.join(exported)}")
        else:
            QMessageBox.warning(self, "Export Failed", "Failed to export visualization")
    
    def show_performance(self):
        """Show enhanced performance visualization"""
        # Implementation similar to original but with Qt5 widgets
        # This would create performance plots with enhanced features
        pass
    
    def analyze_predictions(self):
        """Analyze predictions with enhanced metrics"""
        # Implementation with enhanced analysis features
        pass
    
    def clear_display(self):
        """Clear the visualization display"""
        self._create_placeholder_plot()
        self.current_sample = None
        self.current_prediction = None
        self.stats_display.setPlainText("Load a sample to view statistics...")
        self.comparison_display.setPlainText("Run inference to view comparison metrics...")
    
    def _create_error_plot(self, error_message):
        """Create an error visualization"""
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f"❌ Error\n\n{error_message}",
                ha='center', va='center', fontsize=14,
                color=VoxSigilStyles.COLORS['accent_coral'],
                transform=ax.transAxes)
        
        ax.set_facecolor(VoxSigilStyles.COLORS['bg_secondary'])
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.canvas.draw()
    
    def _add_annotations_to_plot(self):
        """Add annotations to the current plot"""
        if not self.annotator.annotations:
            return
        
        # Add annotations to matplotlib plots
        for position, annotation in self.annotator.get_annotations_for_display().items():
            # Implementation would add text annotations to specific positions
            pass
    
    def _create_overlay_visualization(self, input_grids, output_grid, prediction):
        """Create overlay comparison visualization"""
        # Implementation for overlay mode
        pass
    
    def _create_difference_visualization(self, input_grids, output_grid, prediction):
        """Create difference comparison visualization"""
        # Implementation for difference mode
        pass
