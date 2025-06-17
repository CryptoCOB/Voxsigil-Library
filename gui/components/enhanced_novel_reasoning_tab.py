"""
Enhanced Novel Reasoning Tab with Development Mode Controls
Comprehensive reasoning interface with configurable dev mode options.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config
from gui.components.dev_mode_panel import DevModeControlPanel

logger = logging.getLogger("EnhancedNovelReasoningTab")


class ReasoningWorker(QThread):
    """Worker thread for reasoning operations."""

    reasoning_started = pyqtSignal()
    reasoning_finished = pyqtSignal(bool, str, dict)  # success, message, results
    progress_updated = pyqtSignal(int)  # progress percentage
    step_completed = pyqtSignal(str, dict)  # step_name, step_results

    def __init__(self, task_data: Dict[str, Any], method: str):
        super().__init__()
        self.task_data = task_data
        self.method = method

    def run(self):
        """Run reasoning process in background thread."""
        try:
            self.reasoning_started.emit()
            self.progress_updated.emit(10)

            # Simulate reasoning steps
            steps = [
                ("Problem Analysis", {"complexity": "medium", "patterns": 3}),
                ("Pattern Recognition", {"patterns_found": 5, "confidence": 0.85}),
                ("Rule Extraction", {"rules": 8, "accuracy": 0.92}),
                ("Solution Generation", {"candidates": 12, "best_score": 0.94}),
                ("Validation", {"passed": True, "final_confidence": 0.91}),
            ]

            results = {"steps": [], "overall_accuracy": 0.0}

            for i, (step_name, step_data) in enumerate(steps):
                import time

                time.sleep(0.8)  # Simulate processing time

                self.step_completed.emit(step_name, step_data)
                results["steps"].append({"name": step_name, "data": step_data})

                progress = 20 + (i + 1) * 15
                self.progress_updated.emit(progress)

            results["overall_accuracy"] = 0.91
            results["method"] = self.method
            results["task_type"] = self.task_data.get("type", "unknown")

            self.progress_updated.emit(100)
            self.reasoning_finished.emit(True, "Reasoning completed successfully", results)

        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            self.reasoning_finished.emit(False, f"Error: {e}", {})


class EnhancedNovelReasoningTab(QWidget):
    """
    Enhanced Novel Reasoning tab with comprehensive controls and dev mode support.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_dev_config()
        self.current_worker = None
        self.reasoning_history = []

        self._init_ui()
        self._setup_timers()
        self._connect_signals()
        self._load_sample_tasks()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Development Mode Panel
        self.dev_panel = DevModeControlPanel("novel_reasoning", self)
        layout.addWidget(self.dev_panel)

        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls and Tasks
        controls_widget = self._create_controls_panel()
        main_splitter.addWidget(controls_widget)

        # Right panel - Results and Visualization
        results_widget = self._create_results_panel()
        main_splitter.addWidget(results_widget)

        main_splitter.setSizes([450, 550])
        layout.addWidget(main_splitter)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ccc;")
        layout.addWidget(self.status_label)

    def _create_controls_panel(self) -> QWidget:
        """Create the reasoning controls panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Task Selection
        task_group = QGroupBox("🧠 Reasoning Task")
        task_layout = QGridLayout(task_group)

        row = 0

        # Task type selection
        task_layout.addWidget(QLabel("Task Type:"), row, 0)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(
            [
                "ARC Pattern Recognition",
                "Logical Puzzle Solving",
                "Abstract Reasoning",
                "Causal Inference",
                "Analogical Reasoning",
                "Custom Task",
            ]
        )
        task_layout.addWidget(self.task_type_combo, row, 1)
        row += 1

        # Method selection
        task_layout.addWidget(QLabel("Method:"), row, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "Logical Neural Units",
                "Kuramoto Oscillators",
                "Spiking Neural Networks",
                "Hybrid Approach",
                "Ensemble Method",
            ]
        )
        task_layout.addWidget(self.method_combo, row, 1)
        row += 1

        # Difficulty level
        task_layout.addWidget(QLabel("Difficulty:"), row, 0)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Easy", "Medium", "Hard", "Expert"])
        self.difficulty_combo.setCurrentText("Medium")
        task_layout.addWidget(self.difficulty_combo, row, 1)
        row += 1

        # Advanced parameters (shown in dev mode)
        self.advanced_group = QGroupBox("⚙️ Advanced Parameters")
        advanced_layout = QGridLayout(self.advanced_group)

        # Learning rate
        advanced_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.learning_rate_spin = QSpinBox()
        self.learning_rate_spin.setRange(1, 100)
        self.learning_rate_spin.setValue(10)
        self.learning_rate_spin.setSuffix(" × 0.001")
        advanced_layout.addWidget(self.learning_rate_spin, 0, 1)

        # Max iterations
        advanced_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 10000)
        self.max_iter_spin.setValue(1000)
        advanced_layout.addWidget(self.max_iter_spin, 1, 1)

        # Convergence threshold
        advanced_layout.addWidget(QLabel("Convergence:"), 2, 0)
        self.convergence_spin = QSpinBox()
        self.convergence_spin.setRange(1, 100)
        self.convergence_spin.setValue(5)
        self.convergence_spin.setSuffix(" × 0.0001")
        advanced_layout.addWidget(self.convergence_spin, 2, 1)

        # Enable debug output
        self.debug_checkbox = QCheckBox("Enable Debug Output")
        advanced_layout.addWidget(self.debug_checkbox, 3, 0, 1, 2)

        # Task list
        self.task_list = QTreeWidget()
        self.task_list.setHeaderLabels(["Available Tasks", "Type", "Difficulty"])
        self.task_list.setMaximumHeight(150)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.start_btn = QPushButton("🚀 Start Reasoning")
        self.start_btn.clicked.connect(self._start_reasoning)
        buttons_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._stop_reasoning)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self._clear_results)
        buttons_layout.addWidget(self.clear_btn)

        layout.addWidget(task_group)
        layout.addWidget(self.advanced_group)
        layout.addWidget(QLabel("📋 Sample Tasks:"))
        layout.addWidget(self.task_list)
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

        # Step-by-step results tab
        steps_tab = QWidget()
        steps_layout = QVBoxLayout(steps_tab)

        self.steps_table = QTableWidget()
        self.steps_table.setColumnCount(3)
        self.steps_table.setHorizontalHeaderLabels(["Step", "Status", "Details"])
        self.steps_table.setAlternatingRowColors(True)
        steps_layout.addWidget(self.steps_table)
        tabs.addTab(steps_tab, "Reasoning Steps")

        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        self.visualization_area = QLabel("🧠 Reasoning visualization will appear here")
        self.visualization_area.setStyleSheet(
            "border: 1px solid #ccc; padding: 20px; text-align: center;"
        )
        self.visualization_area.setMinimumHeight(250)
        viz_layout.addWidget(self.visualization_area)
        tabs.addTab(viz_tab, "Visualization")

        # Results log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.results_log = QTextEdit()
        self.results_log.setPlaceholderText("Reasoning process logs will appear here...")
        self.results_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.results_log)
        tabs.addTab(log_tab, "Detailed Log")

        # Performance metrics (shown in dev mode)
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.metrics_table)

        # Real-time metrics
        self.realtime_metrics = QTextEdit()
        self.realtime_metrics.setMaximumHeight(100)
        self.realtime_metrics.setPlaceholderText("Real-time performance metrics...")
        metrics_layout.addWidget(self.realtime_metrics)

        tabs.addTab(self.metrics_tab, "🔧 Performance")

        layout.addWidget(tabs)

        # Summary panel
        summary_group = QGroupBox("📊 Results Summary")
        summary_layout = QGridLayout(summary_group)

        self.accuracy_label = QLabel("Accuracy: --")
        self.time_label = QLabel("Time: --")
        self.steps_label = QLabel("Steps: --")
        self.confidence_label = QLabel("Confidence: --")

        summary_layout.addWidget(self.accuracy_label, 0, 0)
        summary_layout.addWidget(self.time_label, 0, 1)
        summary_layout.addWidget(self.steps_label, 1, 0)
        summary_layout.addWidget(self.confidence_label, 1, 1)

        layout.addWidget(summary_group)

        return widget

    def _setup_timers(self):
        """Setup update timers."""
        # Metrics update timer (for dev mode)
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_metrics)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Update every second

    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect dev panel signals
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_toggled)
        self.dev_panel.config_changed.connect(self._on_config_changed)

        # Connect control signals
        self.task_type_combo.currentTextChanged.connect(self._on_task_type_changed)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.task_list.itemClicked.connect(self._on_task_selected)

    def _load_sample_tasks(self):
        """Load sample reasoning tasks."""
        sample_tasks = [
            ("Pattern Matrix 1", "ARC Pattern", "Easy"),
            ("Logic Puzzle A", "Logical", "Medium"),
            ("Abstract Grid", "Abstract", "Hard"),
            ("Causal Chain", "Causal", "Medium"),
            ("Analogy Test", "Analogical", "Hard"),
            ("Complex Reasoning", "Hybrid", "Expert"),
        ]

        for task_name, task_type, difficulty in sample_tasks:
            item = QTreeWidgetItem([task_name, task_type, difficulty])
            self.task_list.addTopLevelItem(item)

        # Update UI based on current configuration
        self._update_advanced_visibility()
        self._update_metrics_visibility()

    def _update_advanced_visibility(self):
        """Update visibility of advanced controls based on dev mode."""
        is_dev = self.config.get_tab_config("novel_reasoning").dev_mode
        self.advanced_group.setVisible(
            is_dev or self.config.get_tab_config("novel_reasoning").show_advanced_controls
        )

    def _update_metrics_visibility(self):
        """Update visibility of metrics based on dev mode."""
        is_dev = self.config.get_tab_config("novel_reasoning").dev_mode
        self.metrics_tab.setVisible(is_dev)

        if is_dev:
            if not self.metrics_timer.isActive():
                self.metrics_timer.start(1500)  # Update every 1.5 seconds
        else:
            self.metrics_timer.stop()

    @pyqtSlot()
    def _start_reasoning(self):
        """Start reasoning process with current parameters."""
        if self.current_worker and self.current_worker.isRunning():
            return

        # Get current task data
        task_data = {
            "type": self.task_type_combo.currentText(),
            "difficulty": self.difficulty_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value() * 0.001,
            "max_iterations": self.max_iter_spin.value(),
            "convergence_threshold": self.convergence_spin.value() * 0.0001,
            "debug_enabled": self.debug_checkbox.isChecked(),
        }

        method = self.method_combo.currentText()

        self.current_worker = ReasoningWorker(task_data, method)
        self.current_worker.reasoning_started.connect(self._on_reasoning_started)
        self.current_worker.reasoning_finished.connect(self._on_reasoning_finished)
        self.current_worker.progress_updated.connect(self._on_progress_updated)
        self.current_worker.step_completed.connect(self._on_step_completed)
        self.current_worker.start()

        # Log the reasoning request
        self.results_log.append(f"[{self._get_timestamp()}] Starting reasoning process...")
        self.results_log.append(f"Task: {task_data['type']} ({task_data['difficulty']})")
        self.results_log.append(f"Method: {method}")
        self.results_log.append(f"Parameters: {task_data}")
        self.results_log.append("=" * 50)

    @pyqtSlot()
    def _stop_reasoning(self):
        """Stop reasoning process."""
        if self.current_worker:
            self.current_worker.terminate()
            self.current_worker.wait()
            self._on_reasoning_finished(False, "Reasoning stopped by user", {})

    @pyqtSlot()
    def _clear_results(self):
        """Clear all results."""
        self.results_log.clear()
        self.steps_table.setRowCount(0)
        self.metrics_table.setRowCount(0)
        self.realtime_metrics.clear()

        # Reset summary
        self.accuracy_label.setText("Accuracy: --")
        self.time_label.setText("Time: --")
        self.steps_label.setText("Steps: --")
        self.confidence_label.setText("Confidence: --")

        self.visualization_area.setText("🧠 Reasoning visualization will appear here")

    @pyqtSlot()
    def _on_reasoning_started(self):
        """Handle reasoning started."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing reasoning task...")

        # Clear previous results
        self.steps_table.setRowCount(0)

    @pyqtSlot(bool, str, dict)
    def _on_reasoning_finished(self, success: bool, message: str, results: Dict):
        """Handle reasoning finished."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        if success:
            self.status_label.setText("Reasoning complete")
            self.results_log.append(f"✅ {message}")

            # Update summary
            accuracy = results.get("overall_accuracy", 0.0)
            steps_count = len(results.get("steps", []))

            self.accuracy_label.setText(f"Accuracy: {accuracy:.1%}")
            self.time_label.setText(f"Time: {3.2:.1f}s")  # Simulated
            self.steps_label.setText(f"Steps: {steps_count}")
            self.confidence_label.setText(f"Confidence: {accuracy:.1%}")

            # Update visualization
            self.visualization_area.setText(
                f"🧠 Reasoning completed with {accuracy:.1%} accuracy\n📊 {steps_count} processing steps\n🎯 Method: {results.get('method', 'Unknown')}"
            )

            # Store in history
            self.reasoning_history.append(
                {"timestamp": self._get_timestamp(), "results": results, "success": success}
            )

        else:
            self.status_label.setText("Reasoning failed")
            self.results_log.append(f"❌ {message}")

    @pyqtSlot(int)
    def _on_progress_updated(self, progress: int):
        """Handle progress update."""
        self.progress_bar.setValue(progress)

    @pyqtSlot(str, dict)
    def _on_step_completed(self, step_name: str, step_data: Dict):
        """Handle step completion."""
        # Add to steps table
        row = self.steps_table.rowCount()
        self.steps_table.insertRow(row)

        self.steps_table.setItem(row, 0, QTableWidgetItem(step_name))
        self.steps_table.setItem(row, 1, QTableWidgetItem("✅ Complete"))
        self.steps_table.setItem(row, 2, QTableWidgetItem(str(step_data)))

        # Log the step
        self.results_log.append(f"[{self._get_timestamp()}] {step_name}: {step_data}")

        # Auto-scroll to bottom
        self.results_log.moveCursor(self.results_log.textCursor().End)
        self.steps_table.scrollToBottom()

    @pyqtSlot(bool)
    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        self._update_advanced_visibility()
        self._update_metrics_visibility()

        if enabled:
            self.results_log.append(f"[{self._get_timestamp()}] 🔧 Developer mode enabled")
        else:
            self.results_log.append(f"[{self._get_timestamp()}] 🔧 Developer mode disabled")

    @pyqtSlot(str, object)
    def _on_config_changed(self, setting_name: str, value):
        """Handle configuration changes."""
        self.results_log.append(
            f"[{self._get_timestamp()}] ⚙️ Config updated: {setting_name} = {value}"
        )

    @pyqtSlot(str)
    def _on_task_type_changed(self, task_type: str):
        """Handle task type changes."""
        # Could update available methods or parameters
        self.results_log.append(f"[{self._get_timestamp()}] Task type changed to: {task_type}")

    @pyqtSlot(str)
    def _on_method_changed(self, method: str):
        """Handle method changes."""
        self.results_log.append(f"[{self._get_timestamp()}] Method changed to: {method}")

    @pyqtSlot()
    def _on_task_selected(self):
        """Handle task selection from list."""
        current = self.task_list.currentItem()
        if current:
            task_name = current.text(0)
            task_type = current.text(1)
            difficulty = current.text(2)

            # Update UI based on selection
            if "Pattern" in task_type:
                self.task_type_combo.setCurrentText("ARC Pattern Recognition")
            elif "Logical" in task_type:
                self.task_type_combo.setCurrentText("Logical Puzzle Solving")
            elif "Abstract" in task_type:
                self.task_type_combo.setCurrentText("Abstract Reasoning")
            elif "Causal" in task_type:
                self.task_type_combo.setCurrentText("Causal Inference")
            elif "Analogical" in task_type:
                self.task_type_combo.setCurrentText("Analogical Reasoning")

            self.difficulty_combo.setCurrentText(difficulty)

            self.results_log.append(f"[{self._get_timestamp()}] Selected task: {task_name}")

    @pyqtSlot()
    def _update_metrics(self):
        """Update dev metrics display."""
        if not self.config.get_tab_config("novel_reasoning").dev_mode:
            return

        # Simulate reasoning metrics
        import random
        import time

        timestamp = time.strftime("%H:%M:%S")
        cpu_usage = random.uniform(15, 60)
        memory_usage = random.uniform(100, 500)
        neural_activity = random.uniform(0.3, 0.9)
        convergence_rate = random.uniform(0.01, 0.05)

        # Update metrics table
        metrics = [
            ("CPU Usage", f"{cpu_usage:.1f}%"),
            ("Memory Usage", f"{memory_usage:.1f} MB"),
            ("Neural Activity", f"{neural_activity:.2f}"),
            ("Convergence Rate", f"{convergence_rate:.4f}"),
            ("Processing Units", f"{random.randint(8, 32)}"),
            ("Active Patterns", f"{random.randint(50, 200)}"),
        ]

        self.metrics_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

        # Update real-time metrics
        realtime_text = f"[{timestamp}] CPU: {cpu_usage:.1f}% | Mem: {memory_usage:.0f}MB | Activity: {neural_activity:.2f}\n"

        # Keep only last 5 lines
        current_text = self.realtime_metrics.toPlainText()
        lines = current_text.split("\n")
        if len(lines) > 5:
            lines = lines[-4:]
            self.realtime_metrics.setPlainText("\n".join(lines))

        self.realtime_metrics.append(realtime_text.strip())

    @pyqtSlot()
    def _update_status(self):
        """Update status information."""
        tab_config = self.config.get_tab_config("novel_reasoning")

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
