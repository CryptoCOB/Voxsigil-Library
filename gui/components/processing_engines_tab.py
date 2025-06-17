#!/usr/bin/env python3
"""
Processing Engines Tab - Real-time engine monitoring and control
===============================================================

Provides comprehensive monitoring of all processing engines including async engines,
training engines, compression engines, and other processing components.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class EngineMonitorWidget(QWidget):
    """Widget for monitoring a specific processing engine"""

    def __init__(self, engine_name: str):
        super().__init__()
        self.engine_name = engine_name
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Engine info
        info_label = QLabel(f"Engine: {self.engine_name}")
        info_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(info_label)

        # Status section
        status_group = QGroupBox("Status & Health")
        status_layout = QGridLayout()

        status_layout.addWidget(QLabel("Status:"), 0, 0)
        self.status_label = QLabel("Unknown")
        status_layout.addWidget(self.status_label, 0, 1)

        status_layout.addWidget(QLabel("Health:"), 1, 0)
        self.health_label = QLabel("Unknown")
        status_layout.addWidget(self.health_label, 1, 1)

        status_layout.addWidget(QLabel("Uptime:"), 2, 0)
        self.uptime_label = QLabel("0h 0m")
        status_layout.addWidget(self.uptime_label, 2, 1)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout()

        # Throughput
        perf_layout.addWidget(QLabel("Throughput:"), 0, 0)
        self.throughput_label = QLabel("0/s")
        perf_layout.addWidget(self.throughput_label, 0, 1)

        # Queue size
        perf_layout.addWidget(QLabel("Queue Size:"), 1, 0)
        self.queue_progress = QProgressBar()
        self.queue_progress.setRange(0, 100)
        perf_layout.addWidget(self.queue_progress, 1, 1)

        # Error rate
        perf_layout.addWidget(QLabel("Error Rate:"), 2, 0)
        self.error_rate_label = QLabel("0%")
        perf_layout.addWidget(self.error_rate_label, 2, 1)

        # Average processing time
        perf_layout.addWidget(QLabel("Avg Process Time:"), 3, 0)
        self.avg_time_label = QLabel("0ms")
        perf_layout.addWidget(self.avg_time_label, 3, 1)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.restart_btn = QPushButton("Restart")
        self.clear_queue_btn = QPushButton("Clear Queue")

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.restart_btn)
        controls_layout.addWidget(self.clear_queue_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Recent activity
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout()

        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(100)
        self.activity_log.setReadOnly(True)
        activity_layout.addWidget(self.activity_log)

        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group)

        self.setLayout(layout)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update engine metrics with streaming data"""
        try:
            # Status updates
            status = metrics.get("status", "Unknown")
            health = metrics.get("health", "Unknown")
            uptime = metrics.get("uptime", "0h 0m")

            self.status_label.setText(status)
            self.health_label.setText(health)
            self.uptime_label.setText(uptime)

            # Performance metrics
            throughput = metrics.get("throughput", 0)
            queue_size = metrics.get("queue_size", 0)
            queue_capacity = metrics.get("queue_capacity", 100)
            error_rate = metrics.get("error_rate", 0)
            avg_time = metrics.get("avg_processing_time", 0)

            self.throughput_label.setText(f"{throughput:.1f}/s")

            # Queue progress bar
            queue_percentage = (queue_size / queue_capacity * 100) if queue_capacity > 0 else 0
            self.queue_progress.setValue(int(queue_percentage))
            self.queue_progress.setFormat(f"{queue_size}/{queue_capacity}")

            self.error_rate_label.setText(f"{error_rate:.2f}%")
            self.avg_time_label.setText(f"{avg_time:.1f}ms")

            # Color coding for health status
            if health == "healthy":
                self.health_label.setStyleSheet("color: green;")
            elif health == "warning":
                self.health_label.setStyleSheet("color: orange;")
            elif health == "error":
                self.health_label.setStyleSheet("color: red;")
            else:
                self.health_label.setStyleSheet("color: gray;")

            # Status color coding
            if status == "running":
                self.status_label.setStyleSheet("color: green;")
            elif status == "paused":
                self.status_label.setStyleSheet("color: orange;")
            elif status == "stopped":
                self.status_label.setStyleSheet("color: red;")
            else:
                self.status_label.setStyleSheet("color: gray;")

        except Exception as e:
            logger.error(f"Error updating metrics for {self.engine_name}: {e}")


class ProcessingEnginesTab(QWidget):
    """Comprehensive processing engine monitoring tab"""

    engine_command_sent = pyqtSignal(str, str)  # engine_name, command

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.engine_widgets: Dict[str, EngineMonitorWidget] = {}
        self.engine_metrics: Dict[str, Dict[str, Any]] = {}

        # Known engines in the system
        self.known_engines = [
            "async_processing_engine",
            "async_stt_engine",
            "async_training_engine",
            "async_tts_engine",
            "cat_engine",
            "hybrid_cognition_engine",
            "rag_compression_engine",
            "tot_engine",
        ]

        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()

        # Header
        header = QLabel("Processing Engines Monitoring & Control")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Main content
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Engine overview
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Engine details
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        self.setLayout(layout)

    def create_left_panel(self) -> QWidget:
        """Create the left panel with engine overview"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Engine overview table
        overview_group = QGroupBox("Engine Overview")
        overview_layout = QVBoxLayout()

        self.engine_table = QTableWidget()
        self.engine_table.setColumnCount(4)
        self.engine_table.setHorizontalHeaderLabels(["Engine", "Status", "Queue", "Throughput"])
        self.engine_table.itemClicked.connect(self.on_engine_selected)
        overview_layout.addWidget(self.engine_table)

        # Control buttons
        controls_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_engines)
        controls_layout.addWidget(refresh_btn)

        start_all_btn = QPushButton("▶️ Start All")
        start_all_btn.clicked.connect(lambda: self.send_command_to_all("start"))
        controls_layout.addWidget(start_all_btn)

        pause_all_btn = QPushButton("⏸️ Pause All")
        pause_all_btn.clicked.connect(lambda: self.send_command_to_all("pause"))
        controls_layout.addWidget(pause_all_btn)

        overview_layout.addLayout(controls_layout)
        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)

        # System metrics
        system_group = QGroupBox("System Metrics")
        system_layout = QVBoxLayout()

        self.total_throughput_label = QLabel("Total Throughput: 0/s")
        self.active_engines_label = QLabel("Active Engines: 0")
        self.total_queue_size_label = QLabel("Total Queue Size: 0")
        self.system_health_label = QLabel("System Health: Unknown")

        system_layout.addWidget(self.total_throughput_label)
        system_layout.addWidget(self.active_engines_label)
        system_layout.addWidget(self.total_queue_size_label)
        system_layout.addWidget(self.system_health_label)

        system_group.setLayout(system_layout)
        layout.addWidget(system_group)

        widget.setLayout(layout)
        return widget

    def create_right_panel(self) -> QWidget:
        """Create the right panel with engine details"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Tab widget for individual engines
        self.engine_tabs = QTabWidget()
        layout.addWidget(self.engine_tabs)

        widget.setLayout(layout)
        return widget

    def setup_streaming(self):
        """Setup real-time streaming for engine data"""
        # Event bus subscriptions
        if self.event_bus:
            self.event_bus.subscribe("engine_status_update", self.on_engine_status_update)
            self.event_bus.subscribe("engine_metrics_update", self.on_engine_metrics_update)
            self.event_bus.subscribe("engine_activity", self.on_engine_activity)

        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(1000)  # Update every second

        # Initialize engines
        self.refresh_engines()

    def refresh_engines(self):
        """Refresh the engine list"""
        try:
            self.engine_table.setRowCount(len(self.known_engines))
            self.engine_tabs.clear()
            self.engine_widgets.clear()

            for i, engine_name in enumerate(self.known_engines):
                self.add_engine(engine_name, i)

            self.update_system_metrics()

        except Exception as e:
            logger.error(f"Error refreshing engines: {e}")

    def add_engine(self, engine_name: str, row: int):
        """Add an engine to the monitoring interface"""
        # Add to table
        self.engine_table.setItem(row, 0, QTableWidgetItem(engine_name))
        self.engine_table.setItem(row, 1, QTableWidgetItem("Unknown"))
        self.engine_table.setItem(row, 2, QTableWidgetItem("0"))
        self.engine_table.setItem(row, 3, QTableWidgetItem("0/s"))

        # Create engine widget
        engine_widget = EngineMonitorWidget(engine_name)
        self.engine_widgets[engine_name] = engine_widget

        # Add tab
        self.engine_tabs.addTab(engine_widget, engine_name)

        # Initialize metrics
        self.engine_metrics[engine_name] = {
            "status": "Unknown",
            "health": "Unknown",
            "throughput": 0,
            "queue_size": 0,
            "queue_capacity": 100,
            "error_rate": 0,
            "avg_processing_time": 0,
            "uptime": "0h 0m",
        }

    def on_engine_selected(self, item):
        """Handle engine selection in table"""
        row = item.row()
        engine_name = self.engine_table.item(row, 0).text()
        if engine_name in self.engine_widgets:
            widget = self.engine_widgets[engine_name]
            index = self.engine_tabs.indexOf(widget)
            if index >= 0:
                self.engine_tabs.setCurrentIndex(index)

    def send_command_to_all(self, command: str):
        """Send command to all engines"""
        for engine_name in self.known_engines:
            self.engine_command_sent.emit(engine_name, command)

    def on_engine_status_update(self, event):
        """Handle engine status updates from event bus"""
        try:
            data = event.get("data", {})
            engine_name = data.get("engine_name")
            if engine_name and engine_name in self.engine_widgets:
                self.engine_metrics[engine_name].update(data)
                self.engine_widgets[engine_name].update_metrics(self.engine_metrics[engine_name])
                self.update_table_row(engine_name)
        except Exception as e:
            logger.error(f"Error handling engine status update: {e}")

    def on_engine_metrics_update(self, event):
        """Handle engine metrics updates"""
        try:
            data = event.get("data", {})
            engine_name = data.get("engine_name")
            if engine_name and engine_name in self.engine_widgets:
                self.engine_metrics[engine_name].update(data)
                self.engine_widgets[engine_name].update_metrics(self.engine_metrics[engine_name])
                self.update_table_row(engine_name)
        except Exception as e:
            logger.error(f"Error handling engine metrics update: {e}")

    def on_engine_activity(self, event):
        """Handle engine activity updates"""
        try:
            data = event.get("data", {})
            engine_name = data.get("engine_name")
            activity = data.get("activity", "")
            if engine_name and engine_name in self.engine_widgets:
                self.engine_widgets[engine_name].activity_log.append(f"Activity: {activity}")
        except Exception as e:
            logger.error(f"Error handling engine activity: {e}")

    def update_table_row(self, engine_name: str):
        """Update table row for specific engine"""
        try:
            metrics = self.engine_metrics[engine_name]

            # Find row
            for row in range(self.engine_table.rowCount()):
                if self.engine_table.item(row, 0).text() == engine_name:
                    self.engine_table.setItem(
                        row, 1, QTableWidgetItem(metrics.get("status", "Unknown"))
                    )
                    self.engine_table.setItem(
                        row, 2, QTableWidgetItem(str(metrics.get("queue_size", 0)))
                    )
                    self.engine_table.setItem(
                        row, 3, QTableWidgetItem(f"{metrics.get('throughput', 0):.1f}/s")
                    )
                    break
        except Exception as e:
            logger.error(f"Error updating table row for {engine_name}: {e}")

    def update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            total_throughput = sum(
                metrics.get("throughput", 0) for metrics in self.engine_metrics.values()
            )
            active_engines = sum(
                1 for metrics in self.engine_metrics.values() if metrics.get("status") == "running"
            )
            total_queue_size = sum(
                metrics.get("queue_size", 0) for metrics in self.engine_metrics.values()
            )

            # Calculate system health
            healthy_engines = sum(
                1 for metrics in self.engine_metrics.values() if metrics.get("health") == "healthy"
            )
            total_engines = len(self.engine_metrics)
            health_percentage = (healthy_engines / total_engines * 100) if total_engines > 0 else 0

            if health_percentage >= 80:
                system_health = "Healthy"
                health_color = "green"
            elif health_percentage >= 60:
                system_health = "Warning"
                health_color = "orange"
            else:
                system_health = "Critical"
                health_color = "red"

            self.total_throughput_label.setText(f"Total Throughput: {total_throughput:.1f}/s")
            self.active_engines_label.setText(f"Active Engines: {active_engines}")
            self.total_queue_size_label.setText(f"Total Queue Size: {total_queue_size}")
            self.system_health_label.setText(f"System Health: {system_health}")
            self.system_health_label.setStyleSheet(f"color: {health_color};")

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def periodic_update(self):
        """Periodic update for simulated data"""
        try:
            # Simulate engine metrics for testing
            import random

            for engine_name in self.engine_metrics:
                # Simulate changing metrics
                self.engine_metrics[engine_name].update(
                    {
                        "status": random.choice(["running", "paused", "stopped"]),
                        "health": random.choice(["healthy", "warning", "error"]),
                        "throughput": random.uniform(0, 50),
                        "queue_size": random.randint(0, 100),
                        "error_rate": random.uniform(0, 5),
                        "avg_processing_time": random.uniform(1, 500),
                    }
                )

                if engine_name in self.engine_widgets:
                    self.engine_widgets[engine_name].update_metrics(
                        self.engine_metrics[engine_name]
                    )
                    self.update_table_row(engine_name)

            self.update_system_metrics()

        except Exception as e:
            logger.error(f"Error in periodic update: {e}")


# Factory function
def create_processing_engines_tab(event_bus=None) -> ProcessingEnginesTab:
    """Factory function to create the processing engines tab"""
    return ProcessingEnginesTab(event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    tab = create_processing_engines_tab()
    tab.show()
    sys.exit(app.exec_())
