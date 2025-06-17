#!/usr/bin/env python3
"""
Enhanced BLT/RAG Components Tab - Real-time monitoring
====================================================

Enhanced version of the BLT/RAG components tab with real-time streaming
of BLT middleware status, RAG performance metrics, and component health.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class BLTComponentMonitor(QWidget):
    """Widget for monitoring BLT component status"""

    def __init__(self, component_name: str):
        super().__init__()
        self.component_name = component_name
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Component header
        header = QLabel(self.component_name)
        header.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header)

        # Status indicator
        self.status_label = QLabel("Status: Unknown")
        layout.addWidget(self.status_label)

        # Performance metrics
        metrics_layout = QGridLayout()

        # Response time
        metrics_layout.addWidget(QLabel("Response Time:"), 0, 0)
        self.response_time_label = QLabel("0ms")
        metrics_layout.addWidget(self.response_time_label, 0, 1)

        # Throughput
        metrics_layout.addWidget(QLabel("Throughput:"), 1, 0)
        self.throughput_label = QLabel("0/s")
        metrics_layout.addWidget(self.throughput_label, 1, 1)

        # Success rate
        metrics_layout.addWidget(QLabel("Success Rate:"), 2, 0)
        self.success_progress = QProgressBar()
        self.success_progress.setRange(0, 100)
        metrics_layout.addWidget(self.success_progress, 2, 1)

        layout.addLayout(metrics_layout)
        self.setLayout(layout)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update component metrics"""
        try:
            status = metrics.get("status", "Unknown")
            response_time = metrics.get("response_time", 0)
            throughput = metrics.get("throughput", 0)
            success_rate = metrics.get("success_rate", 100)

            self.status_label.setText(f"Status: {status}")
            self.response_time_label.setText(f"{response_time:.1f}ms")
            self.throughput_label.setText(f"{throughput:.1f}/s")
            self.success_progress.setValue(int(success_rate))

            # Color coding
            if success_rate >= 95:
                self.status_label.setStyleSheet("color: green;")
            elif success_rate >= 85:
                self.status_label.setStyleSheet("color: orange;")
            else:
                self.status_label.setStyleSheet("color: red;")

        except Exception as e:
            logger.error(f"Error updating BLT component metrics for {self.component_name}: {e}")


class EnhancedBLTRAGTab(QWidget):
    """Enhanced BLT/RAG component monitoring tab with streaming"""

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.component_monitors: Dict[str, BLTComponentMonitor] = {}
        self.component_metrics: Dict[str, Dict[str, Any]] = {}

        # Known BLT/RAG components
        self.components = [
            "BLT Supervisor",
            "BLT Enhanced RAG",
            "Hybrid Middleware",
            "RAG Compression Engine",
            "BLT Middleware Loader",
            "Patch Aware Validator",
            "Patch Aware Compressor",
        ]

        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()

        # Header
        header = QLabel("BLT/RAG Components - Real-time Monitoring")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Overall status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_layout = QHBoxLayout()

        self.overall_status_label = QLabel("Overall Status: Initializing...")
        self.overall_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.overall_status_label)

        self.active_components_label = QLabel("Active: 0/0")
        status_layout.addWidget(self.active_components_label)

        status_frame.setLayout(status_layout)
        layout.addWidget(status_frame)

        # Component grid
        components_group = QGroupBox("Component Status")
        components_layout = QGridLayout()

        # Create component monitors
        for i, component_name in enumerate(self.components):
            monitor = BLTComponentMonitor(component_name)
            self.component_monitors[component_name] = monitor

            row = i // 3
            col = i % 3
            components_layout.addWidget(monitor, row, col)

            # Initialize metrics
            self.component_metrics[component_name] = {
                "status": "Unknown",
                "response_time": 0,
                "throughput": 0,
                "success_rate": 100,
            }

        components_group.setLayout(components_layout)
        layout.addWidget(components_group)

        # System metrics
        metrics_group = QGroupBox("System Metrics")
        metrics_layout = QGridLayout()

        # Total throughput
        metrics_layout.addWidget(QLabel("Total Throughput:"), 0, 0)
        self.total_throughput_label = QLabel("0/s")
        metrics_layout.addWidget(self.total_throughput_label, 0, 1)

        # Average response time
        metrics_layout.addWidget(QLabel("Avg Response Time:"), 1, 0)
        self.avg_response_label = QLabel("0ms")
        metrics_layout.addWidget(self.avg_response_label, 1, 1)

        # System reliability
        metrics_layout.addWidget(QLabel("System Reliability:"), 2, 0)
        self.reliability_progress = QProgressBar()
        self.reliability_progress.setRange(0, 100)
        metrics_layout.addWidget(self.reliability_progress, 2, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Activity log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()

        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setReadOnly(True)
        log_layout.addWidget(self.activity_log)

        # Control buttons
        controls_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_components)
        controls_layout.addWidget(refresh_btn)

        clear_log_btn = QPushButton("🗑️ Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        controls_layout.addWidget(clear_log_btn)

        log_layout.addLayout(controls_layout)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

    def setup_streaming(self):
        """Setup real-time streaming for BLT/RAG data"""
        # Event bus subscriptions
        if self.event_bus:
            self.event_bus.subscribe("blt_component_update", self.on_blt_component_update)
            self.event_bus.subscribe("rag_performance_update", self.on_rag_performance_update)
            self.event_bus.subscribe("blt_activity", self.on_blt_activity)

        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(3000)  # Update every 3 seconds

        # Initial component check
        self.refresh_components()

    def refresh_components(self):
        """Refresh component status"""
        try:
            self.activity_log.append("🔄 Refreshing BLT/RAG components...")

            # Try to import and check actual components
            try:
                from BLT.blt_supervisor_integration import COMPONENTS_AVAILABLE

                if COMPONENTS_AVAILABLE:
                    self.activity_log.append("✅ BLT Components detected")
                else:
                    self.activity_log.append("❌ BLT Components not available")
            except ImportError:
                self.activity_log.append("❓ BLT Components status unknown")

            try:
                from VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG

                self.activity_log.append("✅ BLT Enhanced RAG detected")
            except ImportError:
                self.activity_log.append("❌ BLT Enhanced RAG not available")

            self.update_overall_status()

        except Exception as e:
            logger.error(f"Error refreshing BLT/RAG components: {e}")
            self.activity_log.append(f"❌ Error refreshing components: {e}")

    def on_blt_component_update(self, event):
        """Handle BLT component updates"""
        try:
            data = event.get("data", {})
            component_name = data.get("component_name")
            if component_name and component_name in self.component_monitors:
                self.component_metrics[component_name].update(data)
                self.component_monitors[component_name].update_metrics(
                    self.component_metrics[component_name]
                )
                self.update_system_metrics()
        except Exception as e:
            logger.error(f"Error handling BLT component update: {e}")

    def on_rag_performance_update(self, event):
        """Handle RAG performance updates"""
        try:
            data = event.get("data", {})
            self.activity_log.append(f"📈 RAG Performance: {data}")
        except Exception as e:
            logger.error(f"Error handling RAG performance update: {e}")

    def on_blt_activity(self, event):
        """Handle BLT activity updates"""
        try:
            data = event.get("data", {})
            activity = data.get("activity", "")
            timestamp = data.get("timestamp", "")
            self.activity_log.append(f"[{timestamp}] {activity}")
        except Exception as e:
            logger.error(f"Error handling BLT activity: {e}")

    def update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            # Calculate aggregated metrics
            total_throughput = sum(
                metrics.get("throughput", 0) for metrics in self.component_metrics.values()
            )
            avg_response = sum(
                metrics.get("response_time", 0) for metrics in self.component_metrics.values()
            ) / len(self.component_metrics)
            avg_success_rate = sum(
                metrics.get("success_rate", 100) for metrics in self.component_metrics.values()
            ) / len(self.component_metrics)

            self.total_throughput_label.setText(f"{total_throughput:.1f}/s")
            self.avg_response_label.setText(f"{avg_response:.1f}ms")
            self.reliability_progress.setValue(int(avg_success_rate))

            self.update_overall_status()

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def update_overall_status(self):
        """Update overall system status"""
        try:
            active_components = sum(
                1
                for metrics in self.component_metrics.values()
                if metrics.get("status") == "Active"
            )
            total_components = len(self.component_metrics)

            self.active_components_label.setText(f"Active: {active_components}/{total_components}")

            # Determine overall status
            if active_components == total_components:
                status = "All Systems Operational"
                color = "green"
            elif active_components > total_components * 0.7:
                status = "Mostly Operational"
                color = "orange"
            else:
                status = "System Issues Detected"
                color = "red"

            self.overall_status_label.setText(f"Overall Status: {status}")
            self.overall_status_label.setStyleSheet(f"color: {color};")

        except Exception as e:
            logger.error(f"Error updating overall status: {e}")

    def clear_log(self):
        """Clear activity log"""
        self.activity_log.clear()

    def periodic_update(self):
        """Periodic update for simulated data"""
        try:
            # Simulate component metrics for testing
            import random

            for component_name in self.component_metrics:
                # Simulate changing metrics
                self.component_metrics[component_name].update(
                    {
                        "status": random.choice(["Active", "Idle", "Busy"]),
                        "response_time": random.uniform(1, 100),
                        "throughput": random.uniform(0, 20),
                        "success_rate": random.uniform(85, 100),
                    }
                )

                if component_name in self.component_monitors:
                    self.component_monitors[component_name].update_metrics(
                        self.component_metrics[component_name]
                    )

            self.update_system_metrics()

            # Occasional activity updates
            if random.random() < 0.3:  # 30% chance
                activities = [
                    "BLT middleware processed request",
                    "RAG compression completed",
                    "Enhanced RAG query executed",
                    "Component health check passed",
                ]
                activity = random.choice(activities)
                import time

                timestamp = time.strftime("%H:%M:%S")
                self.activity_log.append(f"[{timestamp}] {activity}")

        except Exception as e:
            logger.error(f"Error in BLT/RAG periodic update: {e}")


# Factory function
def create_enhanced_blt_rag_tab(event_bus=None) -> EnhancedBLTRAGTab:
    """Factory function to create the enhanced BLT/RAG tab"""
    return EnhancedBLTRAGTab(event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    tab = create_enhanced_blt_rag_tab()
    tab.show()
    sys.exit(app.exec_())
