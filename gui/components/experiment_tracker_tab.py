#!/usr/bin/env python3
"""
Experiment Tracker Tab - ML Experiment Monitoring
Provides comprehensive tracking and monitoring of machine learning experiments.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class ExperimentTableWidget(QTableWidget):
    """Table widget for displaying experiment data"""

    def __init__(self):
        super().__init__()
        self.init_table()

    def init_table(self):
        """Initialize the experiment table"""
        headers = ["ID", "Name", "Status", "Accuracy", "Loss", "Started", "Duration"]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        # Style the table
        self.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())
        self.setAlternatingRowColors(True)

        # Configure header
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Add sample data
        self.add_sample_experiments()

    def add_sample_experiments(self):
        """Add sample experiment data"""
        sample_experiments = [
            {
                "id": "exp_001",
                "name": "GridFormer_v1_baseline",
                "status": "✅ Completed",
                "accuracy": "94.2%",
                "loss": "0.058",
                "started": "2025-06-13 09:30",
                "duration": "2h 15m",
            },
            {
                "id": "exp_002",
                "name": "GridFormer_v2_optimized",
                "status": "🟡 Running",
                "accuracy": "96.1%",
                "loss": "0.039",
                "started": "2025-06-13 11:45",
                "duration": "45m",
            },
            {
                "id": "exp_003",
                "name": "ARC_solver_enhanced",
                "status": "⏸️ Paused",
                "accuracy": "89.7%",
                "loss": "0.103",
                "started": "2025-06-13 08:15",
                "duration": "1h 30m",
            },
        ]

        self.setRowCount(len(sample_experiments))

        for row, exp in enumerate(sample_experiments):
            self.setItem(row, 0, QTableWidgetItem(exp["id"]))
            self.setItem(row, 1, QTableWidgetItem(exp["name"]))
            self.setItem(row, 2, QTableWidgetItem(exp["status"]))
            self.setItem(row, 3, QTableWidgetItem(exp["accuracy"]))
            self.setItem(row, 4, QTableWidgetItem(exp["loss"]))
            self.setItem(row, 5, QTableWidgetItem(exp["started"]))
            self.setItem(row, 6, QTableWidgetItem(exp["duration"]))

    def add_experiment(self, experiment_data: Dict[str, Any]):
        """Add a new experiment to the table"""
        row = self.rowCount()
        self.insertRow(row)

        self.setItem(row, 0, QTableWidgetItem(experiment_data.get("id", "")))
        self.setItem(row, 1, QTableWidgetItem(experiment_data.get("name", "")))
        self.setItem(row, 2, QTableWidgetItem(experiment_data.get("status", "")))
        self.setItem(row, 3, QTableWidgetItem(experiment_data.get("accuracy", "")))
        self.setItem(row, 4, QTableWidgetItem(experiment_data.get("loss", "")))
        self.setItem(row, 5, QTableWidgetItem(experiment_data.get("started", "")))
        self.setItem(row, 6, QTableWidgetItem(experiment_data.get("duration", "")))


class ExperimentMetricsWidget(QWidget):
    """Widget displaying experiment metrics and plots"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("📊 Experiment Metrics", "section")
        layout.addWidget(title)

        # Metrics grid
        metrics_grid = QGridLayout()

        # Total experiments
        self.total_label = VoxSigilWidgetFactory.create_label("Total Experiments:", "info")
        self.total_value = VoxSigilWidgetFactory.create_label("127", "normal")
        metrics_grid.addWidget(self.total_label, 0, 0)
        metrics_grid.addWidget(self.total_value, 0, 1)

        # Running experiments
        self.running_label = VoxSigilWidgetFactory.create_label("Currently Running:", "info")
        self.running_value = VoxSigilWidgetFactory.create_label("3", "normal")
        metrics_grid.addWidget(self.running_label, 1, 0)
        metrics_grid.addWidget(self.running_value, 1, 1)

        # Success rate
        self.success_label = VoxSigilWidgetFactory.create_label("Success Rate:", "info")
        self.success_value = VoxSigilWidgetFactory.create_label("89.7%", "normal")
        metrics_grid.addWidget(self.success_label, 2, 0)
        metrics_grid.addWidget(self.success_value, 2, 1)

        # Best accuracy
        self.best_label = VoxSigilWidgetFactory.create_label("Best Accuracy:", "info")
        self.best_value = VoxSigilWidgetFactory.create_label("97.3%", "normal")
        metrics_grid.addWidget(self.best_label, 3, 0)
        metrics_grid.addWidget(self.best_value, 3, 1)

        layout.addLayout(metrics_grid)

        # Recent activity log
        activity_title = VoxSigilWidgetFactory.create_label("📝 Recent Activity", "section")
        layout.addWidget(activity_title)

        self.activity_log = QTextEdit()
        self.activity_log.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setReadOnly(True)
        layout.addWidget(self.activity_log)

        # Add sample activity
        self.add_sample_activity()

    def add_sample_activity(self):
        """Add sample activity log entries"""
        activities = [
            "🟢 exp_002: Epoch 45 completed, accuracy: 96.1%",
            "🔄 exp_004: Started new hyperparameter search",
            "✅ exp_001: Training completed successfully",
            "⚠️ exp_003: Training paused due to resource constraints",
            "📊 Generated performance report for exp_001",
        ]

        for activity in activities:
            timestamp = datetime.now().strftime("%H:%M")
            self.activity_log.append(f"[{timestamp}] {activity}")

    def add_activity(self, message: str):
        """Add a new activity log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append(f"[{timestamp}] {message}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update displayed metrics"""
        if "total_experiments" in metrics:
            self.total_value.setText(str(metrics["total_experiments"]))
        if "running_experiments" in metrics:
            self.running_value.setText(str(metrics["running_experiments"]))
        if "success_rate" in metrics:
            self.success_value.setText(f"{metrics['success_rate']:.1f}%")
        if "best_accuracy" in metrics:
            self.best_value.setText(f"{metrics['best_accuracy']:.1f}%")


class ExperimentTrackerTab(QWidget):
    """Main experiment tracker tab"""

    # Signals
    experiment_started = pyqtSignal(dict)
    experiment_completed = pyqtSignal(dict)
    metrics_updated = pyqtSignal(dict)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.init_ui()
        self.setup_streaming()
        self.setup_timers()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("🧪 Experiment Tracker", "title")
        layout.addWidget(title)

        # Main splitter
        splitter = VoxSigilWidgetFactory.create_splitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Experiments table
        experiments_group = QGroupBox("Active Experiments")
        experiments_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        experiments_layout = QVBoxLayout(experiments_group)

        self.experiments_table = ExperimentTableWidget()
        experiments_layout.addWidget(self.experiments_table)

        # Right panel - Metrics and activity
        metrics_group = QGroupBox("Metrics & Activity")
        metrics_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_widget = ExperimentMetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)

        # Add to splitter
        splitter.addWidget(experiments_group)
        splitter.addWidget(metrics_group)
        splitter.setSizes([600, 400])

        # Bottom status
        self.status_label = VoxSigilWidgetFactory.create_label(
            "📊 Monitoring 3 active experiments", "info"
        )
        layout.addWidget(self.status_label)

    def setup_streaming(self):
        """Setup event bus streaming"""
        if self.event_bus:
            # Subscribe to experiment events
            self.event_bus.subscribe("experiment.started", self.on_experiment_started)
            self.event_bus.subscribe("experiment.completed", self.on_experiment_completed)
            self.event_bus.subscribe("experiment.metrics", self.on_metrics_updated)
            self.event_bus.subscribe("training.status", self.on_training_status)

    def setup_timers(self):
        """Setup update timers"""
        # Simulation timer for demo
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulate_experiment_updates)
        self.sim_timer.start(10000)  # 10 second intervals

    def on_experiment_started(self, data: Dict[str, Any]):
        """Handle experiment started events"""
        try:
            self.experiments_table.add_experiment(data)
            self.metrics_widget.add_activity(f"🟢 Started: {data.get('name', 'Unknown')}")
            self.experiment_started.emit(data)

        except Exception as e:
            logger.error(f"Error handling experiment start: {e}")

    def on_experiment_completed(self, data: Dict[str, Any]):
        """Handle experiment completed events"""
        try:
            exp_name = data.get("name", "Unknown")
            accuracy = data.get("final_accuracy", 0)
            self.metrics_widget.add_activity(
                f"✅ Completed: {exp_name} (Accuracy: {accuracy:.1f}%)"
            )
            self.experiment_completed.emit(data)

        except Exception as e:
            logger.error(f"Error handling experiment completion: {e}")

    def on_metrics_updated(self, data: Dict[str, Any]):
        """Handle metrics update events"""
        try:
            self.metrics_widget.update_metrics(data)
            self.metrics_updated.emit(data)

        except Exception as e:
            logger.error(f"Error handling metrics update: {e}")

    def on_training_status(self, data: Dict[str, Any]):
        """Handle training status updates"""
        try:
            exp_id = data.get("experiment_id", "Unknown")
            status = data.get("status", "Unknown")
            epoch = data.get("epoch", 0)
            accuracy = data.get("accuracy", 0)

            if epoch > 0:
                self.metrics_widget.add_activity(
                    f"📊 {exp_id}: Epoch {epoch}, Accuracy: {accuracy:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error handling training status: {e}")

    def simulate_experiment_updates(self):
        """Simulate experiment updates for demonstration"""
        import random

        # Simulate random experiment activity
        activities = [
            "🔄 exp_002: Epoch progress, current accuracy: 96.3%",
            "📈 exp_005: Hyperparameter optimization in progress",
            "⚡ exp_002: GPU utilization optimized, training speed increased",
            "📊 Generated performance metrics for active experiments",
            "🔧 Auto-tuning learning rate for exp_002",
        ]

        if random.random() < 0.7:  # 70% chance
            activity = random.choice(activities)
            self.metrics_widget.add_activity(activity)

        # Update status
        running_count = random.randint(2, 5)
        self.status_label.setText(f"📊 Monitoring {running_count} active experiments")


# Backward compatibility
ExperimentTab = ExperimentTrackerTab
