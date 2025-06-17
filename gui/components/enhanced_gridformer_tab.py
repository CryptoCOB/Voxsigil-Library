"""
Enhanced GridFormer Tab with Development Mode Controls
Comprehensive GridFormer interface with configurable dev mode options.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config
from gui.components.dev_mode_panel import DevModeControlPanel

logger = logging.getLogger("EnhancedGridFormerTab")


class GridFormerWorker(QThread):
    """Worker thread for GridFormer operations."""

    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(bool, str, dict)  # success, message, results
    progress_updated = pyqtSignal(int)  # progress percentage
    iteration_completed = pyqtSignal(int, dict)  # iteration, metrics
    grid_updated = pyqtSignal(list)  # grid_state

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.should_stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self.should_stop = True

    def run(self):
        """Run GridFormer processing in background thread."""
        try:
            self.processing_started.emit()
            self.progress_updated.emit(5)

            grid_size = self.config.get("grid_size", 64)
            max_iterations = self.config.get("max_iterations", 1000)
            learning_rate = self.config.get("learning_rate", 0.01)

            # Simulate GridFormer processing
            import random
            import time

            # Initialize grid
            grid_state = [[random.random() for _ in range(grid_size)] for _ in range(grid_size)]

            results = {
                "iterations_completed": 0,
                "final_convergence": 0.0,
                "grid_size": grid_size,
                "processing_time": 0.0,
            }

            start_time = time.time()

            for iteration in range(max_iterations):
                if self.should_stop:
                    break

                time.sleep(0.01)  # Simulate processing time

                # Simulate iteration metrics
                convergence = max(0, 1.0 - (iteration / max_iterations) + random.uniform(-0.1, 0.1))
                loss = max(
                    0, 1.0 - (iteration / max_iterations) * 0.9 + random.uniform(-0.05, 0.05)
                )

                metrics = {
                    "convergence": convergence,
                    "loss": loss,
                    "learning_rate": learning_rate * (0.99 ** (iteration // 100)),
                    "active_nodes": random.randint(
                        int(grid_size * grid_size * 0.6), grid_size * grid_size
                    ),
                }

                self.iteration_completed.emit(iteration, metrics)

                # Update grid state occasionally
                if iteration % 10 == 0:
                    # Simulate grid evolution
                    for i in range(min(5, grid_size)):
                        for j in range(min(5, grid_size)):
                            grid_state[i][j] = max(
                                0, min(1, grid_state[i][j] + random.uniform(-0.1, 0.1))
                            )

                    self.grid_updated.emit(grid_state)

                # Update progress
                progress = 10 + int((iteration / max_iterations) * 85)
                self.progress_updated.emit(progress)

                # Check convergence
                if convergence < 0.01:
                    break

            processing_time = time.time() - start_time
            results["iterations_completed"] = iteration + 1
            results["final_convergence"] = convergence
            results["processing_time"] = processing_time

            self.progress_updated.emit(100)
            self.processing_finished.emit(True, "GridFormer processing completed", results)

        except Exception as e:
            logger.error(f"GridFormer processing error: {e}")
            self.processing_finished.emit(False, f"Error: {e}", {})


class EnhancedGridFormerTab(QWidget):
    """
    Enhanced GridFormer tab with comprehensive controls and dev mode support.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_dev_config()
        self.current_worker = None
        self.processing_history = []
        self.current_grid_state = None

        self._init_ui()
        self._setup_timers()
        self._connect_signals()
        self._load_presets()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Development Mode Panel
        self.dev_panel = DevModeControlPanel("gridformer", self)
        layout.addWidget(self.dev_panel)

        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        controls_widget = self._create_controls_panel()
        main_splitter.addWidget(controls_widget)

        # Right panel - Visualization and Results
        results_widget = self._create_results_panel()
        main_splitter.addWidget(results_widget)

        main_splitter.setSizes([400, 600])
        layout.addWidget(main_splitter)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ccc;")
        layout.addWidget(self.status_label)

    def _create_controls_panel(self) -> QWidget:
        """Create the GridFormer controls panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model Configuration
        model_group = QGroupBox("🔄 GridFormer Configuration")
        model_layout = QGridLayout(model_group)

        row = 0

        # Grid size
        model_layout.addWidget(QLabel("Grid Size:"), row, 0)
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(8, 256)
        self.grid_size_spin.setValue(self.config.gridformer.default_grid_size)
        model_layout.addWidget(self.grid_size_spin, row, 1)
        row += 1

        # Processing mode
        model_layout.addWidget(QLabel("Mode:"), row, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["Training", "Inference", "Fine-tuning", "Evaluation", "Exploration"]
        )
        model_layout.addWidget(self.mode_combo, row, 1)
        row += 1

        # Algorithm variant
        model_layout.addWidget(QLabel("Algorithm:"), row, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(
            [
                "Standard GridFormer",
                "Attention-Enhanced",
                "Hierarchical",
                "Sparse GridFormer",
                "Quantum-Inspired",
            ]
        )
        model_layout.addWidget(self.algorithm_combo, row, 1)
        row += 1

        # Learning parameters
        learning_group = QGroupBox("📚 Learning Parameters")
        learning_layout = QGridLayout(learning_group)

        # Learning rate
        learning_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setSingleStep(0.001)
        learning_layout.addWidget(self.learning_rate_spin, 0, 1)

        # Max iterations
        learning_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 50000)
        self.max_iter_spin.setValue(self.config.gridformer.max_iterations)
        learning_layout.addWidget(self.max_iter_spin, 1, 1)

        # Batch size
        learning_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(32)
        learning_layout.addWidget(self.batch_size_spin, 2, 1)

        # Advanced parameters (shown in dev mode)
        self.advanced_group = QGroupBox("⚙️ Advanced Parameters")
        advanced_layout = QGridLayout(self.advanced_group)

        # Attention heads
        advanced_layout.addWidget(QLabel("Attention Heads:"), 0, 0)
        self.attention_heads_spin = QSpinBox()
        self.attention_heads_spin.setRange(1, 32)
        self.attention_heads_spin.setValue(8)
        advanced_layout.addWidget(self.attention_heads_spin, 0, 1)

        # Hidden dimensions
        advanced_layout.addWidget(QLabel("Hidden Dim:"), 1, 0)
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(64, 2048)
        self.hidden_dim_spin.setValue(512)
        advanced_layout.addWidget(self.hidden_dim_spin, 1, 1)

        # Dropout rate
        advanced_layout.addWidget(QLabel("Dropout Rate:"), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.1)
        advanced_layout.addWidget(self.dropout_spin, 2, 1)

        # Enable checkboxes
        self.real_time_checkbox = QCheckBox("Real-time Updates")
        self.real_time_checkbox.setChecked(self.config.gridformer.real_time_updates)
        advanced_layout.addWidget(self.real_time_checkbox, 3, 0, 1, 2)

        self.visualization_checkbox = QCheckBox("Enable Visualization")
        self.visualization_checkbox.setChecked(self.config.gridformer.visualization_enabled)
        advanced_layout.addWidget(self.visualization_checkbox, 4, 0, 1, 2)

        # Presets list
        presets_group = QGroupBox("📋 Configuration Presets")
        presets_layout = QVBoxLayout(presets_group)

        self.presets_list = QListWidget()
        self.presets_list.setMaximumHeight(100)
        presets_layout.addWidget(self.presets_list)

        preset_buttons = QHBoxLayout()
        self.load_preset_btn = QPushButton("Load")
        self.save_preset_btn = QPushButton("Save")
        preset_buttons.addWidget(self.load_preset_btn)
        preset_buttons.addWidget(self.save_preset_btn)
        presets_layout.addLayout(preset_buttons)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.clicked.connect(self._start_processing)
        buttons_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self._reset_grid)
        buttons_layout.addWidget(self.reset_btn)

        layout.addWidget(model_group)
        layout.addWidget(learning_group)
        layout.addWidget(self.advanced_group)
        layout.addWidget(presets_group)
        layout.addLayout(buttons_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return widget

    def _create_results_panel(self) -> QWidget:
        """Create the results and visualization panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabs for different views
        tabs = QTabWidget()

        # Grid visualization tab
        grid_tab = QWidget()
        grid_layout = QVBoxLayout(grid_tab)

        self.grid_display = QLabel("🔄 GridFormer visualization will appear here")
        self.grid_display.setStyleSheet(
            "border: 1px solid #ccc; padding: 20px; text-align: center; background-color: #f8f8f8;"
        )
        self.grid_display.setMinimumHeight(300)
        grid_layout.addWidget(self.grid_display)

        # Grid controls
        grid_controls = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        grid_controls.addWidget(QLabel("Zoom:"))
        grid_controls.addWidget(self.zoom_slider)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Viridis", "Plasma", "Coolwarm", "Grayscale"])
        grid_controls.addWidget(QLabel("Colormap:"))
        grid_controls.addWidget(self.colormap_combo)

        grid_layout.addLayout(grid_controls)
        tabs.addTab(grid_tab, "Grid Visualization")

        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.metrics_table)

        tabs.addTab(metrics_tab, "Training Metrics")

        # Processing log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.processing_log = QTextEdit()
        self.processing_log.setPlaceholderText("GridFormer processing logs will appear here...")
        self.processing_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.processing_log)
        tabs.addTab(log_tab, "Processing Log")

        # Performance analysis (shown in dev mode)
        self.performance_tab = QWidget()
        performance_layout = QVBoxLayout(self.performance_tab)

        # Performance charts placeholder
        self.performance_display = QLabel("📊 Performance analysis charts will appear here")
        self.performance_display.setStyleSheet(
            "border: 1px solid #ccc; padding: 20px; text-align: center;"
        )
        self.performance_display.setMinimumHeight(200)
        performance_layout.addWidget(self.performance_display)

        # Real-time performance metrics
        self.realtime_performance = QTextEdit()
        self.realtime_performance.setMaximumHeight(100)
        self.realtime_performance.setPlaceholderText("Real-time performance metrics...")
        performance_layout.addWidget(self.realtime_performance)

        tabs.addTab(self.performance_tab, "🔧 Performance")

        layout.addWidget(tabs)

        # Status panel
        status_group = QGroupBox("📊 Processing Status")
        status_layout = QGridLayout(status_group)

        self.iteration_label = QLabel("Iteration: --")
        self.convergence_label = QLabel("Convergence: --")
        self.loss_label = QLabel("Loss: --")
        self.time_label = QLabel("Time: --")

        status_layout.addWidget(self.iteration_label, 0, 0)
        status_layout.addWidget(self.convergence_label, 0, 1)
        status_layout.addWidget(self.loss_label, 1, 0)
        status_layout.addWidget(self.time_label, 1, 1)

        layout.addWidget(status_group)

        return widget

    def _setup_timers(self):
        """Setup update timers."""
        # Performance metrics timer (for dev mode)
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_metrics)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(self.config.gridformer.update_interval)

    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect dev panel signals
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_toggled)
        self.dev_panel.config_changed.connect(self._on_config_changed)

        # Connect control signals
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        self.grid_size_spin.valueChanged.connect(self._on_grid_size_changed)

        # Connect preset signals
        self.load_preset_btn.clicked.connect(self._load_preset)
        self.save_preset_btn.clicked.connect(self._save_preset)

        # Connect visualization signals
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)

    def _load_presets(self):
        """Load configuration presets."""
        presets = [
            "Default Configuration",
            "High Performance",
            "Memory Efficient",
            "Experimental",
            "Debug Mode",
        ]

        for preset in presets:
            self.presets_list.addItem(preset)

        # Update UI based on current configuration
        self._update_advanced_visibility()
        self._update_performance_visibility()

    def _update_advanced_visibility(self):
        """Update visibility of advanced controls based on dev mode."""
        is_dev = self.config.get_tab_config("gridformer").dev_mode
        self.advanced_group.setVisible(
            is_dev or self.config.get_tab_config("gridformer").show_advanced_controls
        )

    def _update_performance_visibility(self):
        """Update visibility of performance metrics based on dev mode."""
        is_dev = self.config.get_tab_config("gridformer").dev_mode
        show_internal = self.config.gridformer.dev_show_internal_state

        self.performance_tab.setVisible(is_dev and show_internal)

        if is_dev and show_internal:
            if not self.performance_timer.isActive():
                self.performance_timer.start(2000)  # Update every 2 seconds
        else:
            self.performance_timer.stop()

    @pyqtSlot()
    def _start_processing(self):
        """Start GridFormer processing with current parameters."""
        if self.current_worker and self.current_worker.isRunning():
            return

        # Get current configuration
        processing_config = {
            "grid_size": self.grid_size_spin.value(),
            "mode": self.mode_combo.currentText(),
            "algorithm": self.algorithm_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value(),
            "max_iterations": self.max_iter_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "attention_heads": self.attention_heads_spin.value(),
            "hidden_dim": self.hidden_dim_spin.value(),
            "dropout_rate": self.dropout_spin.value(),
            "real_time_updates": self.real_time_checkbox.isChecked(),
            "visualization_enabled": self.visualization_checkbox.isChecked(),
        }

        self.current_worker = GridFormerWorker(processing_config)
        self.current_worker.processing_started.connect(self._on_processing_started)
        self.current_worker.processing_finished.connect(self._on_processing_finished)
        self.current_worker.progress_updated.connect(self._on_progress_updated)
        self.current_worker.iteration_completed.connect(self._on_iteration_completed)
        self.current_worker.grid_updated.connect(self._on_grid_updated)
        self.current_worker.start()

        # Log the processing request
        self.processing_log.append(f"[{self._get_timestamp()}] Starting GridFormer processing...")
        self.processing_log.append(f"Configuration: {processing_config}")
        self.processing_log.append("=" * 50)

    @pyqtSlot()
    def _stop_processing(self):
        """Stop GridFormer processing."""
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker.wait()
            self._on_processing_finished(False, "Processing stopped by user", {})

    @pyqtSlot()
    def _reset_grid(self):
        """Reset the grid to initial state."""
        self.processing_log.append(f"[{self._get_timestamp()}] Grid reset to initial state")
        self.grid_display.setText("🔄 GridFormer visualization will appear here")

        # Reset status displays
        self.iteration_label.setText("Iteration: --")
        self.convergence_label.setText("Convergence: --")
        self.loss_label.setText("Loss: --")
        self.time_label.setText("Time: --")

        # Clear metrics
        self.metrics_table.setRowCount(0)

    @pyqtSlot()
    def _load_preset(self):
        """Load selected preset configuration."""
        current_item = self.presets_list.currentItem()
        if current_item:
            preset_name = current_item.text()
            self.processing_log.append(f"[{self._get_timestamp()}] Loading preset: {preset_name}")

            # Apply preset-specific configurations
            if preset_name == "High Performance":
                self.grid_size_spin.setValue(128)
                self.learning_rate_spin.setValue(0.001)
                self.max_iter_spin.setValue(5000)
                self.batch_size_spin.setValue(64)
            elif preset_name == "Memory Efficient":
                self.grid_size_spin.setValue(32)
                self.learning_rate_spin.setValue(0.01)
                self.max_iter_spin.setValue(1000)
                self.batch_size_spin.setValue(16)
            elif preset_name == "Debug Mode":
                self.real_time_checkbox.setChecked(True)
                self.visualization_checkbox.setChecked(True)

    @pyqtSlot()
    def _save_preset(self):
        """Save current configuration as preset."""
        # In a real implementation, this would save to a file
        self.processing_log.append(f"[{self._get_timestamp()}] Preset saved (placeholder)")

    # Event handlers
    @pyqtSlot()
    def _on_processing_started(self):
        """Handle processing started."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing GridFormer...")

        # Clear previous metrics
        self.metrics_table.setRowCount(0)

    @pyqtSlot(bool, str, dict)
    def _on_processing_finished(self, success: bool, message: str, results: Dict):
        """Handle processing finished."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        if success:
            self.status_label.setText("Processing complete")
            self.processing_log.append(f"✅ {message}")

            # Update status displays
            iterations = results.get("iterations_completed", 0)
            convergence = results.get("final_convergence", 0.0)
            proc_time = results.get("processing_time", 0.0)

            self.iteration_label.setText(f"Iteration: {iterations}")
            self.convergence_label.setText(f"Convergence: {convergence:.4f}")
            self.time_label.setText(f"Time: {proc_time:.1f}s")

            # Store in history
            self.processing_history.append(
                {"timestamp": self._get_timestamp(), "results": results, "success": success}
            )

        else:
            self.status_label.setText("Processing failed")
            self.processing_log.append(f"❌ {message}")

    @pyqtSlot(int)
    def _on_progress_updated(self, progress: int):
        """Handle progress update."""
        self.progress_bar.setValue(progress)

    @pyqtSlot(int, dict)
    def _on_iteration_completed(self, iteration: int, metrics: Dict):
        """Handle iteration completion."""
        # Update status
        self.iteration_label.setText(f"Iteration: {iteration}")
        self.convergence_label.setText(f"Convergence: {metrics.get('convergence', 0):.4f}")
        self.loss_label.setText(f"Loss: {metrics.get('loss', 0):.4f}")

        # Update metrics table
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)

        self.metrics_table.setItem(row, 0, QTableWidgetItem(f"Iteration {iteration}"))
        self.metrics_table.setItem(
            row,
            1,
            QTableWidgetItem(
                f"Conv: {metrics.get('convergence', 0):.4f}, Loss: {metrics.get('loss', 0):.4f}"
            ),
        )

        # Keep only last 20 entries
        if self.metrics_table.rowCount() > 20:
            self.metrics_table.removeRow(0)

        # Auto-scroll to bottom
        self.metrics_table.scrollToBottom()

    @pyqtSlot(list)
    def _on_grid_updated(self, grid_state: list):
        """Handle grid state update."""
        self.current_grid_state = grid_state

        if self.visualization_checkbox.isChecked():
            # Update grid visualization
            grid_size = len(grid_state)
            self.grid_display.setText(
                f"🔄 Grid Updated ({grid_size}x{grid_size})\n📊 Active nodes: {sum(sum(row) for row in grid_state):.0f}"
            )

    @pyqtSlot(bool)
    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        self._update_advanced_visibility()
        self._update_performance_visibility()

        if enabled:
            self.processing_log.append(f"[{self._get_timestamp()}] 🔧 Developer mode enabled")
        else:
            self.processing_log.append(f"[{self._get_timestamp()}] 🔧 Developer mode disabled")

    @pyqtSlot(str, object)
    def _on_config_changed(self, setting_name: str, value):
        """Handle configuration changes."""
        self.processing_log.append(
            f"[{self._get_timestamp()}] ⚙️ Config updated: {setting_name} = {value}"
        )

    @pyqtSlot(str)
    def _on_mode_changed(self, mode: str):
        """Handle mode changes."""
        self.processing_log.append(f"[{self._get_timestamp()}] Mode changed to: {mode}")

    @pyqtSlot(str)
    def _on_algorithm_changed(self, algorithm: str):
        """Handle algorithm changes."""
        self.processing_log.append(f"[{self._get_timestamp()}] Algorithm changed to: {algorithm}")

    @pyqtSlot(int)
    def _on_grid_size_changed(self, size: int):
        """Handle grid size changes."""
        self.processing_log.append(f"[{self._get_timestamp()}] Grid size changed to: {size}x{size}")

    @pyqtSlot(int)
    def _on_zoom_changed(self, zoom: int):
        """Handle zoom changes."""
        if self.current_grid_state:
            self.processing_log.append(f"[{self._get_timestamp()}] Zoom changed to: {zoom}%")

    @pyqtSlot(str)
    def _on_colormap_changed(self, colormap: str):
        """Handle colormap changes."""
        if self.current_grid_state:
            self.processing_log.append(f"[{self._get_timestamp()}] Colormap changed to: {colormap}")

    @pyqtSlot()
    def _update_performance_metrics(self):
        """Update dev performance metrics display."""
        if not self.config.gridformer.dev_show_internal_state:
            return

        # Simulate performance metrics
        import random
        import time

        timestamp = time.strftime("%H:%M:%S")
        cpu_usage = random.uniform(20, 80)
        memory_usage = random.uniform(200, 1000)
        gpu_usage = random.uniform(10, 95)
        throughput = random.uniform(50, 500)

        performance_text = f"[{timestamp}] CPU: {cpu_usage:.1f}% | Mem: {memory_usage:.0f}MB | GPU: {gpu_usage:.1f}% | Throughput: {throughput:.0f} ops/s\n"

        # Keep only last 5 lines
        current_text = self.realtime_performance.toPlainText()
        lines = current_text.split("\n")
        if len(lines) > 5:
            lines = lines[-4:]
            self.realtime_performance.setPlainText("\n".join(lines))

        self.realtime_performance.append(performance_text.strip())

    @pyqtSlot()
    def _update_status(self):
        """Update status information."""
        tab_config = self.config.get_tab_config("gridformer")

        if tab_config.dev_mode and tab_config.debug_logging:
            # Show detailed status in dev mode
            if (
                hasattr(self, "current_worker")
                and self.current_worker
                and self.current_worker.isRunning()
            ):
                self.status_label.setText("Status: Processing | Dev Mode: ON | Debug: ON")
            else:
                self.status_label.setText("Status: Ready | Dev Mode: ON | Debug: ON")

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import time

        return time.strftime("%H:%M:%S")
