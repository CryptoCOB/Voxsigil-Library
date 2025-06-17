#!/usr/bin/env python3
"""
System Health Dashboard Tab - Overall system monitoring and alerts
================================================================

Provides comprehensive system health monitoring with real-time metrics,
alerts, resource usage tracking, and system-wide status overview.
"""

import logging
from typing import Any, Dict, List

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class HealthMetricWidget(QWidget):
    """Widget for displaying a single health metric"""

    def __init__(
        self,
        metric_name: str,
        metric_unit: str = "",
        warning_threshold: float = 80,
        critical_threshold: float = 95,
    ):
        super().__init__()
        self.metric_name = metric_name
        self.metric_unit = metric_unit
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Metric name
        self.name_label = QLabel(self.metric_name)
        self.name_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Value label
        self.value_label = QLabel(f"0{self.metric_unit}")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

        self.setLayout(layout)
        self.setMaximumHeight(100)

    def update_value(self, value: float, max_value: float = 100):
        """Update the metric value"""
        try:
            percentage = (value / max_value * 100) if max_value > 0 else 0
            self.progress_bar.setValue(int(percentage))
            self.value_label.setText(f"{value:.1f}{self.metric_unit}")

            # Color coding based on thresholds
            if percentage >= self.critical_threshold:
                self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
                self.value_label.setStyleSheet("color: red; font-weight: bold;")
            elif percentage >= self.warning_threshold:
                self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
                self.value_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
                self.value_label.setStyleSheet("color: green;")

        except Exception as e:
            logger.error(f"Error updating health metric {self.metric_name}: {e}")


class AlertsWidget(QWidget):
    """Widget for displaying system alerts"""

    def __init__(self):
        super().__init__()
        self.alerts: List[Dict[str, Any]] = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("System Alerts")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)

        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(4)
        self.alerts_table.setHorizontalHeaderLabels(["Severity", "Component", "Message", "Time"])
        layout.addWidget(self.alerts_table)

        # Clear alerts button
        clear_btn = QPushButton("Clear All Alerts")
        clear_btn.clicked.connect(self.clear_alerts)
        layout.addWidget(clear_btn)

        self.setLayout(layout)

    def add_alert(self, severity: str, component: str, message: str, timestamp: str):
        """Add a new alert"""
        try:
            alert = {
                "severity": severity,
                "component": component,
                "message": message,
                "timestamp": timestamp,
            }
            self.alerts.append(alert)

            # Add to table
            row = self.alerts_table.rowCount()
            self.alerts_table.insertRow(row)

            severity_item = QTableWidgetItem(severity)
            if severity == "CRITICAL":
                severity_item.setBackground(QColor(255, 0, 0, 100))
            elif severity == "WARNING":
                severity_item.setBackground(QColor(255, 165, 0, 100))
            elif severity == "INFO":
                severity_item.setBackground(QColor(0, 255, 0, 100))

            self.alerts_table.setItem(row, 0, severity_item)
            self.alerts_table.setItem(row, 1, QTableWidgetItem(component))
            self.alerts_table.setItem(row, 2, QTableWidgetItem(message))
            self.alerts_table.setItem(row, 3, QTableWidgetItem(timestamp))

            # Auto-scroll to latest
            self.alerts_table.scrollToBottom()

        except Exception as e:
            logger.error(f"Error adding alert: {e}")

    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        self.alerts_table.setRowCount(0)


class SystemHealthDashboard(QWidget):
    """Comprehensive system health dashboard tab"""

    alert_generated = pyqtSignal(str, str, str, str)  # severity, component, message, timestamp

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.health_metrics: Dict[str, HealthMetricWidget] = {}
        self.system_stats: Dict[str, Any] = {}

        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()

        # Header
        header = QLabel("System Health Dashboard")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Overall system status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_layout = QHBoxLayout()

        self.overall_status_label = QLabel("System Status: Initializing...")
        self.overall_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.overall_status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.overall_status_label)

        self.uptime_label = QLabel("Uptime: 0h 0m")
        self.uptime_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.uptime_label)

        status_frame.setLayout(status_layout)
        layout.addWidget(status_frame)

        # Main content
        main_splitter = QSplitter(Qt.Vertical)

        # Top section - Health metrics
        top_section = self.create_health_metrics_section()
        main_splitter.addWidget(top_section)

        # Bottom section - Alerts and logs
        bottom_section = self.create_alerts_section()
        main_splitter.addWidget(bottom_section)

        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter)

        self.setLayout(layout)

    def create_health_metrics_section(self) -> QWidget:
        """Create the health metrics section"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Resource usage metrics
        resources_group = QGroupBox("Resource Usage")
        resources_layout = QGridLayout()

        # CPU Usage
        self.health_metrics["cpu"] = HealthMetricWidget("CPU Usage", "%", 70, 90)
        resources_layout.addWidget(self.health_metrics["cpu"], 0, 0)

        # Memory Usage
        self.health_metrics["memory"] = HealthMetricWidget("Memory Usage", "%", 80, 95)
        resources_layout.addWidget(self.health_metrics["memory"], 0, 1)

        # Disk Usage
        self.health_metrics["disk"] = HealthMetricWidget("Disk Usage", "%", 85, 95)
        resources_layout.addWidget(self.health_metrics["disk"], 0, 2)

        # Network I/O
        self.health_metrics["network"] = HealthMetricWidget("Network I/O", " MB/s", 80, 95)
        resources_layout.addWidget(self.health_metrics["network"], 0, 3)

        resources_group.setLayout(resources_layout)
        layout.addWidget(resources_group)

        # System component health
        components_group = QGroupBox("Component Health")
        components_layout = QGridLayout()

        # Agents Health
        self.health_metrics["agents"] = HealthMetricWidget("Agents Health", "%", 70, 50)
        components_layout.addWidget(self.health_metrics["agents"], 0, 0)

        # Engines Health
        self.health_metrics["engines"] = HealthMetricWidget("Engines Health", "%", 70, 50)
        components_layout.addWidget(self.health_metrics["engines"], 0, 1)

        # Training Health
        self.health_metrics["training"] = HealthMetricWidget("Training Health", "%", 70, 50)
        components_layout.addWidget(self.health_metrics["training"], 0, 2)

        # Integration Health
        self.health_metrics["integration"] = HealthMetricWidget("Integration Health", "%", 70, 50)
        components_layout.addWidget(self.health_metrics["integration"], 0, 3)

        components_group.setLayout(components_layout)
        layout.addWidget(components_group)

        # Performance metrics
        performance_group = QGroupBox("Performance Metrics")
        performance_layout = QGridLayout()

        # Response time
        performance_layout.addWidget(QLabel("Avg Response Time:"), 0, 0)
        self.response_time_label = QLabel("0ms")
        performance_layout.addWidget(self.response_time_label, 0, 1)

        # Throughput
        performance_layout.addWidget(QLabel("System Throughput:"), 1, 0)
        self.throughput_label = QLabel("0/s")
        performance_layout.addWidget(self.throughput_label, 1, 1)

        # Error Rate
        performance_layout.addWidget(QLabel("System Error Rate:"), 2, 0)
        self.error_rate_label = QLabel("0%")
        performance_layout.addWidget(self.error_rate_label, 2, 1)

        # Active Connections
        performance_layout.addWidget(QLabel("Active Connections:"), 3, 0)
        self.connections_label = QLabel("0")
        performance_layout.addWidget(self.connections_label, 3, 1)

        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)

        widget.setLayout(layout)
        return widget

    def create_alerts_section(self) -> QWidget:
        """Create the alerts and logs section"""
        widget = QWidget()
        layout = QHBoxLayout()

        # Alerts widget
        self.alerts_widget = AlertsWidget()
        layout.addWidget(self.alerts_widget)

        # System logs
        logs_group = QGroupBox("System Logs")
        logs_layout = QVBoxLayout()

        self.system_logs = QTextEdit()
        self.system_logs.setReadOnly(True)
        self.system_logs.setMaximumHeight(200)
        logs_layout.addWidget(self.system_logs)

        # Log controls
        log_controls = QHBoxLayout()

        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.clear_logs)
        log_controls.addWidget(clear_logs_btn)

        export_logs_btn = QPushButton("Export Logs")
        export_logs_btn.clicked.connect(self.export_logs)
        log_controls.addWidget(export_logs_btn)

        logs_layout.addLayout(log_controls)
        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)

        widget.setLayout(layout)
        return widget

    def setup_streaming(self):
        """Setup real-time streaming for system health data"""
        # Event bus subscriptions
        if self.event_bus:
            self.event_bus.subscribe("system_health_update", self.on_system_health_update)
            self.event_bus.subscribe("system_alert", self.on_system_alert)
            self.event_bus.subscribe("system_log", self.on_system_log)
            self.event_bus.subscribe("resource_usage_update", self.on_resource_usage_update)

        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(2000)  # Update every 2 seconds

        # Initialize system stats
        self.initialize_system_stats()

    def initialize_system_stats(self):
        """Initialize system statistics"""
        self.system_stats = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "network_io": 0,
            "agents_health": 100,
            "engines_health": 100,
            "training_health": 100,
            "integration_health": 100,
            "response_time": 0,
            "throughput": 0,
            "error_rate": 0,
            "active_connections": 0,
            "uptime_seconds": 0,
        }

    def on_system_health_update(self, event):
        """Handle system health updates from event bus"""
        try:
            data = event.get("data", {})
            self.system_stats.update(data)
            self.update_health_metrics()
        except Exception as e:
            logger.error(f"Error handling system health update: {e}")

    def on_system_alert(self, event):
        """Handle system alerts"""
        try:
            data = event.get("data", {})
            severity = data.get("severity", "INFO")
            component = data.get("component", "Unknown")
            message = data.get("message", "No message")
            timestamp = data.get("timestamp", "Unknown")

            self.alerts_widget.add_alert(severity, component, message, timestamp)
            self.alert_generated.emit(severity, component, message, timestamp)
        except Exception as e:
            logger.error(f"Error handling system alert: {e}")

    def on_system_log(self, event):
        """Handle system log messages"""
        try:
            data = event.get("data", {})
            log_message = data.get("message", "")
            timestamp = data.get("timestamp", "")

            self.system_logs.append(f"[{timestamp}] {log_message}")
        except Exception as e:
            logger.error(f"Error handling system log: {e}")

    def on_resource_usage_update(self, event):
        """Handle resource usage updates"""
        try:
            data = event.get("data", {})
            self.system_stats.update(data)
            self.update_health_metrics()
        except Exception as e:
            logger.error(f"Error handling resource usage update: {e}")

    def update_health_metrics(self):
        """Update all health metrics displays"""
        try:
            # Update resource metrics
            self.health_metrics["cpu"].update_value(self.system_stats.get("cpu_usage", 0))
            self.health_metrics["memory"].update_value(self.system_stats.get("memory_usage", 0))
            self.health_metrics["disk"].update_value(self.system_stats.get("disk_usage", 0))
            self.health_metrics["network"].update_value(self.system_stats.get("network_io", 0), 100)

            # Update component health
            self.health_metrics["agents"].update_value(self.system_stats.get("agents_health", 100))
            self.health_metrics["engines"].update_value(
                self.system_stats.get("engines_health", 100)
            )
            self.health_metrics["training"].update_value(
                self.system_stats.get("training_health", 100)
            )
            self.health_metrics["integration"].update_value(
                self.system_stats.get("integration_health", 100)
            )

            # Update performance labels
            self.response_time_label.setText(f"{self.system_stats.get('response_time', 0):.1f}ms")
            self.throughput_label.setText(f"{self.system_stats.get('throughput', 0):.1f}/s")
            self.error_rate_label.setText(f"{self.system_stats.get('error_rate', 0):.2f}%")
            self.connections_label.setText(str(self.system_stats.get("active_connections", 0)))

            # Update overall status
            self.update_overall_status()

            # Update uptime
            uptime_seconds = self.system_stats.get("uptime_seconds", 0)
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            self.uptime_label.setText(f"Uptime: {hours}h {minutes}m")

        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")

    def update_overall_status(self):
        """Update the overall system status"""
        try:
            # Calculate overall health score
            component_healths = [
                self.system_stats.get("agents_health", 100),
                self.system_stats.get("engines_health", 100),
                self.system_stats.get("training_health", 100),
                self.system_stats.get("integration_health", 100),
            ]

            resource_usage = [
                self.system_stats.get("cpu_usage", 0),
                self.system_stats.get("memory_usage", 0),
                self.system_stats.get("disk_usage", 0),
            ]

            avg_component_health = sum(component_healths) / len(component_healths)
            max_resource_usage = max(resource_usage)

            # Determine status
            if avg_component_health >= 80 and max_resource_usage < 80:
                status = "Healthy"
                color = "green"
            elif avg_component_health >= 60 and max_resource_usage < 90:
                status = "Warning"
                color = "orange"
            else:
                status = "Critical"
                color = "red"

            self.overall_status_label.setText(f"System Status: {status}")
            self.overall_status_label.setStyleSheet(f"color: {color};")

        except Exception as e:
            logger.error(f"Error updating overall status: {e}")

    def clear_logs(self):
        """Clear system logs"""
        self.system_logs.clear()

    def export_logs(self):
        """Export system logs to file"""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_logs_{timestamp}.txt"

            with open(filename, "w") as f:
                f.write(self.system_logs.toPlainText())

            self.system_logs.append(f"Logs exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")

    def periodic_update(self):
        """Periodic update for simulated data"""
        try:
            # Simulate system metrics for testing
            import random
            import time

            # Update simulated metrics
            self.system_stats.update(
                {
                    "cpu_usage": random.uniform(10, 95),
                    "memory_usage": random.uniform(20, 90),
                    "disk_usage": random.uniform(30, 85),
                    "network_io": random.uniform(0, 50),
                    "agents_health": random.uniform(70, 100),
                    "engines_health": random.uniform(80, 100),
                    "training_health": random.uniform(60, 100),
                    "integration_health": random.uniform(75, 100),
                    "response_time": random.uniform(1, 200),
                    "throughput": random.uniform(10, 100),
                    "error_rate": random.uniform(0, 3),
                    "active_connections": random.randint(0, 50),
                    "uptime_seconds": self.system_stats.get("uptime_seconds", 0) + 2,
                }
            )

            # Generate occasional alerts
            if random.random() < 0.1:  # 10% chance per update
                severities = ["INFO", "WARNING", "CRITICAL"]
                components = ["Agents", "Engines", "Training", "Integration", "System"]
                severity = random.choice(severities)
                component = random.choice(components)
                message = f"Simulated {severity.lower()} event in {component}"
                timestamp = time.strftime("%H:%M:%S")

                self.alerts_widget.add_alert(severity, component, message, timestamp)

            self.update_health_metrics()

        except Exception as e:
            logger.error(f"Error in periodic update: {e}")


# Factory function
def create_system_health_dashboard(event_bus=None) -> SystemHealthDashboard:
    """Factory function to create the system health dashboard"""
    return SystemHealthDashboard(event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dashboard = create_system_health_dashboard()
    dashboard.show()
    sys.exit(app.exec_())
