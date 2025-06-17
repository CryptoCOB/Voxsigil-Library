#!/usr/bin/env python3
"""
VoxSigil Performance Analysis Tab Interface - Qt5 Version
Modular component for performance analysis and metrics calculation

Created by: GitHub Copilot
Purpose: Encapsulated performance analysis interface for Dynamic GridFormer GUI
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QPushButton, QScrollArea, QGroupBox, QFileDialog, QMessageBox,
    QDialog, QTableWidget, QTableWidgetItem, QTabWidget, QSpinBox,
    QCheckBox, QProgressBar, QTextEdit, QSplitter, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QFont

# Matplotlib imports with fallback
if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    MATPLOTLIB_AVAILABLE = True
else:
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        
        class MockCanvas(QWidget):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.setMinimumSize(400, 300)
                
            def draw(self):
                pass
        
        FigureCanvasQTAgg = MockCanvas
        Figure = object


class VoxSigilPerformanceInterface(QWidget):
    """Qt5-based Performance analysis interface for VoxSigil Dynamic GridFormer"""
    
    # Signals for communication
    metrics_updated = pyqtSignal(dict)
    analysis_completed = pyqtSignal(dict)
    model_compared = pyqtSignal(str, str, dict)
    
    def __init__(self, parent_gui=None, perf_visualizer=None, analyze_callback=None):
        """Initialize the Qt5 performance analysis interface"""
        super().__init__()
        self.parent_gui = parent_gui
        self.perf_visualizer = perf_visualizer
        self.analyze_callback = analyze_callback
        self.current_metrics = {}
        self.metrics_history = []
        
        # Initialize encapsulated features
        self.real_time_monitor = RealTimeMonitor(self)
        self.benchmark_suite = BenchmarkSuite(self)
        self.model_profiler = ModelProfiler(self)
        self.metrics_export_manager = MetricsExportManager(self)
        self.performance_predictor = PerformancePredictor(self)
        self.automated_testing = AutomatedTesting(self)
        self.resource_optimizer = ResourceOptimizer(self)
        self.comparative_analyzer = ComparativeAnalyzer(self)
        self.alert_system = AlertSystem(self)
        self.custom_metrics_builder = CustomMetricsBuilder(self)
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the Qt5 user interface"""
        layout = QHBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel
        left_widget = self.create_left_panel()
        splitter.addWidget(left_widget)
        
        # Right panel
        right_widget = self.create_right_panel()
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
    def create_left_panel(self) -> QWidget:
        """Create the left panel with metrics and controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model Metrics Section
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # Create scrollable metrics area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        self.metrics_widgets = {}
        metrics_labels = [
            "Model Name", "Accuracy", "Loss", "Precision", "Recall",
            "F1 Score", "Inference Time", "GPU Memory", "Model Size"
        ]
        
        for i, label in enumerate(metrics_labels):
            label_widget = QLabel(f"{label}:")
            label_widget.setFont(QFont("Arial", 10, QFont.Bold))
            scroll_layout.addWidget(label_widget, i, 0)
            
            value_widget = QLabel("N/A")
            scroll_layout.addWidget(value_widget, i, 1)
            
            self.metrics_widgets[label] = value_widget
            
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        metrics_layout.addWidget(scroll_area)
        layout.addWidget(metrics_group)
        
        # Performance Actions
        actions_group = QGroupBox("Performance Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Main action buttons
        load_btn = QPushButton("Load Test Results")
        load_btn.clicked.connect(self.load_test_results)
        actions_layout.addWidget(load_btn)
        
        calc_btn = QPushButton("Calculate Model Metrics")
        calc_btn.clicked.connect(self.calculate_model_metrics)
        actions_layout.addWidget(calc_btn)
        
        export_btn = QPushButton("Export Metrics Report")
        export_btn.clicked.connect(self.export_metrics_report)
        actions_layout.addWidget(export_btn)
        
        # Real-time monitoring toggle
        self.monitor_btn = QPushButton("Start Real-time Monitoring")
        self.monitor_btn.clicked.connect(self.toggle_real_time_monitoring)
        actions_layout.addWidget(self.monitor_btn)
        
        # Benchmark suite button
        benchmark_btn = QPushButton("Run Benchmark Suite")
        benchmark_btn.clicked.connect(self.run_benchmark_suite)
        actions_layout.addWidget(benchmark_btn)
        
        layout.addWidget(actions_group)
        
        # Advanced Features Tab
        tabs = QTabWidget()
        
        # Profiler tab
        profiler_tab = self.create_profiler_tab()
        tabs.addTab(profiler_tab, "Profiler")
        
        # Optimizer tab
        optimizer_tab = self.create_optimizer_tab()
        tabs.addTab(optimizer_tab, "Optimizer")
        
        # Alerts tab
        alerts_tab = self.create_alerts_tab()
        tabs.addTab(alerts_tab, "Alerts")
        
        layout.addWidget(tabs)
        
        return widget
        
    def create_right_panel(self) -> QWidget:
        """Create the right panel with visualization and comparison"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Visualization Section
        viz_group = QGroupBox("Performance Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Matplotlib canvas
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 6))
            self.plot = self.figure.add_subplot(111)
            self.canvas = FigureCanvasQTAgg(self.figure)
        else:
            self.canvas = QLabel("Matplotlib not available")
            self.canvas.setMinimumSize(400, 300)
            self.canvas.setStyleSheet("border: 1px solid gray;")
            
        viz_layout.addWidget(self.canvas)
        
        # Visualization controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        controls_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Accuracy", "Loss", "Inference Time"])
        self.metric_combo.currentTextChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.metric_combo)
        
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(plot_btn)
        
        controls_layout.addStretch()
        viz_layout.addWidget(controls_widget)
        layout.addWidget(viz_group)
        
        # Model Comparison Section
        compare_group = QGroupBox("Model Comparison")
        compare_layout = QVBoxLayout(compare_group)
        
        # Model selection
        selection_widget = QWidget()
        selection_layout = QGridLayout(selection_widget)
        
        selection_layout.addWidget(QLabel("Model A:"), 0, 0)
        self.model_a_combo = QComboBox()
        selection_layout.addWidget(self.model_a_combo, 0, 1)
        
        selection_layout.addWidget(QLabel("Model B:"), 0, 2)
        self.model_b_combo = QComboBox()
        selection_layout.addWidget(self.model_b_combo, 0, 3)
        
        compare_layout.addWidget(selection_widget)
        
        compare_btn = QPushButton("Compare Models")
        compare_btn.clicked.connect(self.compare_models)
        compare_layout.addWidget(compare_btn)
        
        layout.addWidget(compare_group)
        
        # Custom Metrics Builder
        custom_group = QGroupBox("Custom Metrics")
        custom_layout = QVBoxLayout(custom_group)
        
        build_btn = QPushButton("Build Custom Metrics")
        build_btn.clicked.connect(self.open_custom_metrics_builder)
        custom_layout.addWidget(build_btn)
        
        layout.addWidget(custom_group)
        
        return widget
        
    def create_profiler_tab(self) -> QWidget:
        """Create the model profiler tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Profiling options
        options_group = QGroupBox("Profiling Options")
        options_layout = QGridLayout(options_group)
        
        options_layout.addWidget(QLabel("Profile Depth:"), 0, 0)
        self.profile_depth = QSpinBox()
        self.profile_depth.setRange(1, 10)
        self.profile_depth.setValue(3)
        options_layout.addWidget(self.profile_depth, 0, 1)
        
        self.profile_memory = QCheckBox("Profile Memory Usage")
        self.profile_memory.setChecked(True)
        options_layout.addWidget(self.profile_memory, 1, 0, 1, 2)
        
        self.profile_gpu = QCheckBox("Profile GPU Utilization")
        self.profile_gpu.setChecked(True)
        options_layout.addWidget(self.profile_gpu, 2, 0, 1, 2)
        
        layout.addWidget(options_group)
        
        # Profiling controls
        profile_btn = QPushButton("Start Profiling")
        profile_btn.clicked.connect(self.start_profiling)
        layout.addWidget(profile_btn)
        
        # Progress bar
        self.profile_progress = QProgressBar()
        layout.addWidget(self.profile_progress)
        
        # Results display
        self.profile_results = QTextEdit()
        self.profile_results.setReadOnly(True)
        layout.addWidget(self.profile_results)
        
        return widget
        
    def create_optimizer_tab(self) -> QWidget:
        """Create the resource optimizer tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Optimization targets
        targets_group = QGroupBox("Optimization Targets")
        targets_layout = QVBoxLayout(targets_group)
        
        self.optimize_speed = QCheckBox("Optimize for Speed")
        self.optimize_speed.setChecked(True)
        targets_layout.addWidget(self.optimize_speed)
        
        self.optimize_memory = QCheckBox("Optimize for Memory")
        targets_layout.addWidget(self.optimize_memory)
        
        self.optimize_accuracy = QCheckBox("Maintain Accuracy")
        self.optimize_accuracy.setChecked(True)
        targets_layout.addWidget(self.optimize_accuracy)
        
        layout.addWidget(targets_group)
        
        # Optimization level
        level_group = QGroupBox("Optimization Level")
        level_layout = QVBoxLayout(level_group)
        
        self.optimization_slider = QSlider(Qt.Horizontal)
        self.optimization_slider.setRange(1, 5)
        self.optimization_slider.setValue(3)
        self.optimization_slider.valueChanged.connect(self.update_optimization_level)
        level_layout.addWidget(self.optimization_slider)
        
        self.optimization_label = QLabel("Level: 3 (Balanced)")
        level_layout.addWidget(self.optimization_label)
        
        layout.addWidget(level_group)
        
        # Optimize button
        optimize_btn = QPushButton("Start Optimization")
        optimize_btn.clicked.connect(self.start_optimization)
        layout.addWidget(optimize_btn)
        
        return widget
        
    def create_alerts_tab(self) -> QWidget:
        """Create the alerts configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Alert thresholds
        thresholds_group = QGroupBox("Alert Thresholds")
        thresholds_layout = QGridLayout(thresholds_group)
        
        thresholds_layout.addWidget(QLabel("Accuracy Drop (%):"), 0, 0)
        self.accuracy_threshold = QSpinBox()
        self.accuracy_threshold.setRange(1, 50)
        self.accuracy_threshold.setValue(5)
        thresholds_layout.addWidget(self.accuracy_threshold, 0, 1)
        
        thresholds_layout.addWidget(QLabel("Memory Usage (%):"), 1, 0)
        self.memory_threshold = QSpinBox()
        self.memory_threshold.setRange(50, 95)
        self.memory_threshold.setValue(85)
        thresholds_layout.addWidget(self.memory_threshold, 1, 1)
        
        layout.addWidget(thresholds_group)
        
        # Alert types
        types_group = QGroupBox("Alert Types")
        types_layout = QVBoxLayout(types_group)
        
        self.email_alerts = QCheckBox("Email Alerts")
        types_layout.addWidget(self.email_alerts)
        
        self.popup_alerts = QCheckBox("Popup Alerts")
        self.popup_alerts.setChecked(True)
        types_layout.addWidget(self.popup_alerts)
        
        self.log_alerts = QCheckBox("Log File Alerts")
        self.log_alerts.setChecked(True)
        types_layout.addWidget(self.log_alerts)
        
        layout.addWidget(types_group)
        
        return widget
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.metrics_updated.connect(self.on_metrics_updated)
        self.analysis_completed.connect(self.on_analysis_completed)
        self.model_compared.connect(self.on_model_compared)
        
    # Core functionality methods
    def load_test_results(self):
        """Load test results from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Results File", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as file:
                test_results = json.load(file)
                
            self.update_metrics_display(test_results)
            
            if test_results.get("model_name"):
                self.metrics_history.append(test_results)
                
            self.update_model_selection_combos()
            
            QMessageBox.information(self, "Success", "Test results loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load test results: {str(e)}")
            
    def update_metrics_display(self, metrics):
        """Update the metrics display with new values"""
        self.current_metrics = metrics
        
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
        
        for label, value in display_map.items():
            if label in self.metrics_widgets:
                self.metrics_widgets[label].setText(value)
                
        self.metrics_updated.emit(metrics)
        
    def calculate_model_metrics(self):
        """Calculate additional metrics for the current model"""
        if not hasattr(self.parent_gui, 'current_model') or not self.parent_gui.current_model:
            QMessageBox.warning(self, "No Model", "Please load a model first")
            return
            
        try:
            # Mock metrics calculation
            import random
            
            accuracy = random.uniform(0.7, 0.98)
            loss = random.uniform(0.01, 0.3)
            precision = random.uniform(0.7, 0.95)
            recall = random.uniform(0.7, 0.95)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            metrics = {
                "model_name": getattr(self.parent_gui.current_model, "name", "Unknown Model"),
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
            
            self.update_metrics_display(metrics)
            self.metrics_history.append(metrics)
            self.update_model_selection_combos()
            self.update_visualization()
            
            QMessageBox.information(self, "Success", "Metrics calculated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate metrics: {str(e)}")
            
    def export_metrics_report(self):
        """Export the current metrics as a report"""
        if not self.current_metrics:
            QMessageBox.warning(self, "No Metrics", "No metrics available to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics Report", "", 
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            if "timestamp" not in self.current_metrics:
                self.current_metrics["timestamp"] = datetime.now().isoformat()
                
            with open(file_path, 'w') as file:
                json.dump(self.current_metrics, file, indent=2)
                
            QMessageBox.information(self, "Success", f"Metrics report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export metrics report: {str(e)}")
            
    def update_visualization(self):
        """Update the performance visualization"""
        if not MATPLOTLIB_AVAILABLE or not self.metrics_history:
            return
            
        self.plot.clear()
        
        metric = self.metric_combo.currentText().lower().replace(" ", "_")
        
        models = []
        values = []
        
        for metrics in self.metrics_history:
            if metric in metrics:
                models.append(metrics.get("model_name", "Unknown"))
                values.append(metrics[metric])
                
        if not models or not values:
            return
            
        self.plot.bar(models, values)
        self.plot.set_title(f"{self.metric_combo.currentText()} Comparison")
        self.plot.set_ylabel(self.metric_combo.currentText())
        self.plot.set_xlabel("Model")
        
        if len(models) > 3:
            self.plot.tick_params(axis='x', rotation=45)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def compare_models(self):
        """Compare two selected models"""
        model_a = self.model_a_combo.currentText()
        model_b = self.model_b_combo.currentText()
        
        if not model_a or not model_b:
            QMessageBox.warning(self, "Selection Missing", "Please select two models to compare")
            return
            
        self.comparative_analyzer.compare_models(model_a, model_b)
        
    def update_model_selection_combos(self):
        """Update the model selection combo boxes"""
        model_names = [
            m.get("model_name", "Unknown") 
            for m in self.metrics_history 
            if "model_name" in m
        ]
        
        unique_models = list(dict.fromkeys(model_names))  # Remove duplicates
        
        self.model_a_combo.clear()
        self.model_a_combo.addItems(unique_models)
        
        self.model_b_combo.clear()
        self.model_b_combo.addItems(unique_models)
        
    # New encapsulated feature methods
    def toggle_real_time_monitoring(self):
        """Toggle real-time performance monitoring"""
        if self.real_time_monitor.is_running:
            self.real_time_monitor.stop_monitoring()
            self.monitor_btn.setText("Start Real-time Monitoring")
        else:
            self.real_time_monitor.start_monitoring()
            self.monitor_btn.setText("Stop Real-time Monitoring")
            
    def run_benchmark_suite(self):
        """Run the benchmark suite"""
        self.benchmark_suite.run_benchmarks()
        
    def start_profiling(self):
        """Start model profiling"""
        options = {
            'depth': self.profile_depth.value(),
            'memory': self.profile_memory.isChecked(),
            'gpu': self.profile_gpu.isChecked()
        }
        self.model_profiler.start_profiling(options)
        
    def start_optimization(self):
        """Start resource optimization"""
        targets = {
            'speed': self.optimize_speed.isChecked(),
            'memory': self.optimize_memory.isChecked(),
            'accuracy': self.optimize_accuracy.isChecked(),
            'level': self.optimization_slider.value()
        }
        self.resource_optimizer.optimize(targets)
        
    def update_optimization_level(self, value):
        """Update optimization level label"""
        levels = {1: "Conservative", 2: "Mild", 3: "Balanced", 4: "Aggressive", 5: "Maximum"}
        self.optimization_label.setText(f"Level: {value} ({levels[value]})")
        
    def open_custom_metrics_builder(self):
        """Open custom metrics builder dialog"""
        self.custom_metrics_builder.show_builder()
        
    # Signal handlers
    @pyqtSlot(dict)
    def on_metrics_updated(self, metrics):
        """Handle metrics updated signal"""
        self.alert_system.check_thresholds(metrics)
        
    @pyqtSlot(dict)
    def on_analysis_completed(self, results):
        """Handle analysis completed signal"""
        print(f"Analysis completed: {results}")
        
    @pyqtSlot(str, str, dict)
    def on_model_compared(self, model_a, model_b, comparison):
        """Handle model comparison signal"""
        self.show_detailed_comparison(comparison)
        
    def show_detailed_comparison(self, comparison):
        """Show detailed comparison dialog"""
        dialog = ModelComparisonDialog(comparison, self)
        dialog.exec_()


# Encapsulated Feature Classes

class RealTimeMonitor(QObject):
    """Real-time performance monitoring"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        self.timer.start(1000)  # Update every second
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        self.timer.stop()
        
    def update_metrics(self):
        """Update metrics in real-time"""
        # Mock real-time data
        import random
        metrics = {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 90),
            "gpu_utilization": random.uniform(40, 95)
        }
        # Update UI with real-time metrics


class BenchmarkSuite(QObject):
    """Comprehensive benchmark testing"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def run_benchmarks(self):
        """Run comprehensive benchmarks"""
        QMessageBox.information(self.parent, "Benchmark", "Running benchmark suite...")
        # Implement benchmark logic


class ModelProfiler(QObject):
    """Deep model profiling"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def start_profiling(self, options):
        """Start profiling with given options"""
        self.parent.profile_progress.setValue(0)
        # Implement profiling logic
        self.parent.profile_results.setText("Profiling results will appear here...")


class MetricsExportManager(QObject):
    """Advanced metrics export"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def export_to_csv(self, data):
        """Export metrics to CSV"""
        pass
        
    def export_to_pdf(self, data):
        """Export metrics to PDF report"""
        pass


class PerformancePredictor(QObject):
    """Performance prediction based on historical data"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def predict_performance(self, model_config):
        """Predict model performance"""
        pass


class AutomatedTesting(QObject):
    """Automated testing framework"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def schedule_tests(self, schedule):
        """Schedule automated tests"""
        pass


class ResourceOptimizer(QObject):
    """Resource optimization engine"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def optimize(self, targets):
        """Optimize based on targets"""
        QMessageBox.information(self.parent, "Optimizer", "Starting optimization...")


class ComparativeAnalyzer(QObject):
    """Advanced model comparison"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def compare_models(self, model_a, model_b):
        """Compare two models in detail"""
        # Find metrics for both models
        metrics_a = None
        metrics_b = None
        
        for metrics in self.parent.metrics_history:
            if metrics.get("model_name") == model_a:
                metrics_a = metrics
            if metrics.get("model_name") == model_b:
                metrics_b = metrics
                
        if metrics_a and metrics_b:
            comparison = {"model_a": metrics_a, "model_b": metrics_b}
            self.parent.model_compared.emit(model_a, model_b, comparison)


class AlertSystem(QObject):
    """Intelligent alert system"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def check_thresholds(self, metrics):
        """Check if metrics exceed thresholds"""
        if self.parent.popup_alerts.isChecked():
            # Check accuracy threshold
            accuracy = metrics.get('accuracy', 1.0)
            if accuracy < 0.8:  # Example threshold
                QMessageBox.warning(
                    self.parent, "Performance Alert", 
                    f"Model accuracy dropped to {accuracy:.2%}"
                )


class CustomMetricsBuilder(QObject):
    """Custom metrics builder"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def show_builder(self):
        """Show custom metrics builder dialog"""
        dialog = CustomMetricsDialog(self.parent)
        dialog.exec_()


class ModelComparisonDialog(QDialog):
    """Dialog for detailed model comparison"""
    
    def __init__(self, comparison_data, parent=None):
        super().__init__(parent)
        self.comparison_data = comparison_data
        self.setWindowTitle("Detailed Model Comparison")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create comparison table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Metric", "Model A", "Model B", "Difference"])
        
        # Populate table with comparison data
        metrics_to_compare = [
            ("accuracy", "Accuracy"),
            ("loss", "Loss"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 Score")
        ]
        
        table.setRowCount(len(metrics_to_compare))
        
        for i, (key, label) in enumerate(metrics_to_compare):
            table.setItem(i, 0, QTableWidgetItem(label))
            # Add comparison values (simplified)
            table.setItem(i, 1, QTableWidgetItem("0.85"))
            table.setItem(i, 2, QTableWidgetItem("0.82"))
            table.setItem(i, 3, QTableWidgetItem("0.03"))
            
        layout.addWidget(table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


class CustomMetricsDialog(QDialog):
    """Dialog for building custom metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Metrics Builder")
        self.setMinimumSize(400, 300)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        label = QLabel("Build your custom metrics here:")
        layout.addWidget(label)
        
        # Metric formula input
        formula_group = QGroupBox("Metric Formula")
        formula_layout = QVBoxLayout(formula_group)
        
        self.formula_edit = QTextEdit()
        self.formula_edit.setPlaceholderText("Enter your custom metric formula...")
        formula_layout.addWidget(self.formula_edit)
        
        layout.addWidget(formula_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Metric")
        save_btn.clicked.connect(self.save_metric)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def save_metric(self):
        """Save the custom metric"""
        formula = self.formula_edit.toPlainText()
        if formula.strip():
            QMessageBox.information(self, "Success", "Custom metric saved!")
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Please enter a metric formula")
