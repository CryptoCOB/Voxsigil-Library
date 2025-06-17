#!/usr/bin/env python3
"""
Heartbeat Monitor Tab - Real-time System Pulse
Provides live monitoring of system vital signs and performance metrics.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class SystemPulseWidget(QWidget):
    """Widget displaying real-time system pulse metrics"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)

        # TPS (Transactions Per Second)
        self.tps_label = QLabel("TPS: 0")
        self.tps_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
        self.tps_bar = VoxSigilWidgetFactory.create_progress_bar()
        self.tps_bar.setRange(0, 1000)  # GPU Utilization - Support multiple GPUs
        self.gpu_labels = []
        self.gpu_bars = []
        self.gpu_count = 0

        # Detect available GPUs
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_count = torch.cuda.device_count()
            else:
                self.gpu_count = 0
        except Exception:
            self.gpu_count = 0

        if self.gpu_count == 0:
            # No GPU available
            self.gpu_label = QLabel("GPU: Not Available")
            self.gpu_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
            self.gpu_bar = VoxSigilWidgetFactory.create_progress_bar()
            self.gpu_bar.setRange(0, 100)
            self.gpu_bar.setValue(0)
        else:
            # Multiple GPUs available
            for i in range(self.gpu_count):
                gpu_label = QLabel(f"GPU {i}: 0%")
                gpu_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
                gpu_bar = VoxSigilWidgetFactory.create_progress_bar()
                gpu_bar.setRange(0, 100)
                self.gpu_labels.append(gpu_label)
                self.gpu_bars.append(gpu_bar)

        # CPU Utilization
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
        self.cpu_bar = VoxSigilWidgetFactory.create_progress_bar()
        self.cpu_bar.setRange(0, 100)

        # Memory Usage
        self.memory_label = QLabel("Memory: 0%")
        self.memory_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
        self.memory_bar = VoxSigilWidgetFactory.create_progress_bar()
        self.memory_bar.setRange(0, 100)

        # Error Rate
        self.error_label = QLabel("Errors: 0/min")
        self.error_label.setStyleSheet(VoxSigilStyles.get_label_stylesheet("section"))
        self.error_bar = VoxSigilWidgetFactory.create_progress_bar()
        self.error_bar.setRange(0, 100)  # Layout
        row = 0
        layout.addWidget(self.tps_label, row, 0)
        layout.addWidget(self.tps_bar, row, 1)
        row += 1

        # Add GPU widgets (multiple if available)
        if self.gpu_count == 0:
            layout.addWidget(self.gpu_label, row, 0)
            layout.addWidget(self.gpu_bar, row, 1)
            row += 1
        else:
            for i in range(self.gpu_count):
                layout.addWidget(self.gpu_labels[i], row, 0)
                layout.addWidget(self.gpu_bars[i], row, 1)
                row += 1

        layout.addWidget(self.cpu_label, row, 0)
        layout.addWidget(self.cpu_bar, row, 1)
        row += 1
        layout.addWidget(self.memory_label, row, 0)
        layout.addWidget(self.memory_bar, row, 1)
        row += 1
        layout.addWidget(self.error_label, row, 0)
        layout.addWidget(self.error_bar, row, 1)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update displayed metrics"""
        if "tps" in metrics:
            tps = metrics["tps"]
            self.tps_label.setText(f"TPS: {tps}")
            self.tps_bar.setValue(
                min(tps, 1000)
            )  # Handle GPU usage (multiple GPUs or single/no GPU)
        if "gpu_usage" in metrics:
            gpu_data = metrics["gpu_usage"]
            if self.gpu_count == 0:
                # No GPU available
                self.gpu_label.setText("GPU: Not Available")
                self.gpu_bar.setValue(0)
            elif isinstance(gpu_data, list) and len(gpu_data) == self.gpu_count:
                # Multiple GPU data
                for i, gpu_usage in enumerate(gpu_data):
                    self.gpu_labels[i].setText(f"GPU {i}: {gpu_usage}%")
                    self.gpu_bars[i].setValue(gpu_usage)
            elif isinstance(gpu_data, (int, float)):
                # Single GPU value - apply to all GPUs
                for i in range(self.gpu_count):
                    self.gpu_labels[i].setText(f"GPU {i}: {gpu_data}%")
                    self.gpu_bars[i].setValue(gpu_data)
        elif self.gpu_count == 0:
            # Fallback for no GPU
            if hasattr(self, "gpu_label"):
                self.gpu_label.setText("GPU: Not Available")
                self.gpu_bar.setValue(0)

        if "cpu_usage" in metrics:
            cpu = metrics["cpu_usage"]
            self.cpu_label.setText(f"CPU: {cpu}%")
            self.cpu_bar.setValue(cpu)

        if "memory_usage" in metrics:
            memory = metrics["memory_usage"]
            self.memory_label.setText(f"Memory: {memory}%")
            self.memory_bar.setValue(memory)

        if "error_rate" in metrics:
            errors = metrics["error_rate"]
            self.error_label.setText(f"Errors: {errors}/min")
            self.error_bar.setValue(min(errors, 100))

    def get_real_gpu_stats(self):
        """Get real GPU statistics if available"""
        gpu_stats = []
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(
                    torch.cuda.device_count()
                ):  # Get GPU memory usage as a proxy for utilization
                    memory_reserved = torch.cuda.memory_reserved(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory

                    # Calculate usage percentage
                    usage_percent = int((memory_reserved / total_memory) * 100)
                    gpu_stats.append(min(100, max(0, usage_percent)))

            return gpu_stats if gpu_stats else None
        except Exception as e:
            logger.warning(f"Could not get real GPU stats: {e}")
            return None

    def get_real_system_stats(self):
        """Get real system statistics for monitoring"""
        try:
            import psutil

            # Get real GPU stats if available
            gpu_usage = self.get_real_gpu_stats()
            if gpu_usage is None:
                # Fall back to simulated data for GPU
                import random

                if self.gpu_count > 0:
                    gpu_usage = [random.randint(20, 80) for _ in range(self.gpu_count)]
                else:
                    gpu_usage = 0

            stats = {
                "tps": random.randint(50, 200),  # TPS still simulated for now
                "gpu_usage": gpu_usage,
                "cpu_usage": int(psutil.cpu_percent(interval=0.1)),
                "memory_usage": int(psutil.virtual_memory().percent),
                "error_rate": random.randint(0, 5),  # Error rate still simulated
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"[SYSTEM STATS] {stats}")
            return stats
        except ImportError:
            logger.warning("psutil not available, using simulated data")
            return None
        except Exception as e:
            logger.error(f"Error getting real system stats: {e}")
            return None


class AlertsWidget(QWidget):
    """Widget displaying system alerts and notifications"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("🚨 System Alerts", "title")
        layout.addWidget(title)

        # Alerts display
        self.alerts_text = QTextEdit()
        self.alerts_text.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.alerts_text.setMaximumHeight(150)
        self.alerts_text.setReadOnly(True)
        layout.addWidget(self.alerts_text)

    def add_alert(self, level: str, message: str):
        """Add a new alert"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon_map = {"critical": "🔴", "warning": "🟡", "info": "🔵", "success": "🟢"}
        icon = icon_map.get(level, "ℹ️")

        self.alerts_text.append(f"[{timestamp}] {icon} {message}")


class HeartbeatMonitorTab(QWidget):
    """Main heartbeat monitor tab providing real-time system pulse monitoring"""

    # Signals
    pulse_detected = pyqtSignal(dict)
    alert_triggered = pyqtSignal(str, str)

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
        title = VoxSigilWidgetFactory.create_label(
            "❤️ System Heartbeat Monitor", "title"
        )
        layout.addWidget(title)

        # Main splitter
        splitter = VoxSigilWidgetFactory.create_splitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - System pulse
        pulse_group = QGroupBox("System Vital Signs")
        pulse_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        pulse_layout = QVBoxLayout(pulse_group)

        self.pulse_widget = SystemPulseWidget()
        pulse_layout.addWidget(self.pulse_widget)

        # Status indicator
        self.status_label = VoxSigilWidgetFactory.create_label(
            "💚 System Healthy", "section"
        )
        pulse_layout.addWidget(self.status_label)

        # Right panel - Alerts and logs
        alerts_group = QGroupBox("System Alerts & Status")
        alerts_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        alerts_layout = QVBoxLayout(alerts_group)

        self.alerts_widget = AlertsWidget()
        alerts_layout.addWidget(self.alerts_widget)

        # Add to splitter
        splitter.addWidget(pulse_group)
        splitter.addWidget(alerts_group)
        splitter.setSizes([400, 400])

        # Bottom status
        self.last_update_label = VoxSigilWidgetFactory.create_label(
            "Last update: Never", "info"
        )
        layout.addWidget(self.last_update_label)

    def setup_streaming(self):
        """Setup event bus streaming"""
        if self.event_bus:
            # Subscribe to heartbeat topic
            self.event_bus.subscribe("heartbeat", self.on_heartbeat_received)
            self.event_bus.subscribe("system.alert", self.on_alert_received)

    def setup_timers(self):
        """Setup update timers"""
        # Pulse simulation timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.simulate_pulse)
        self.pulse_timer.start(1000)  # 1 second intervals

        # Health check timer
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_system_health)
        self.health_timer.start(5000)  # 5 second intervals

    def on_heartbeat_received(self, data: Dict[str, Any]):
        """Handle incoming heartbeat data"""
        try:
            self.pulse_widget.update_metrics(data)
            self.last_update_label.setText(
                f"Last update: {datetime.now().strftime('%H:%M:%S')}"
            )
            self.pulse_detected.emit(data)

            # Update system status based on metrics
            self.update_system_status(data)

        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")

    def update_stats(self, stats: Dict[str, Any]):
        """Alias to update metrics from external stream."""
        self.on_heartbeat_received(stats)

    def on_alert_received(self, alert_data: Dict[str, Any]):
        """Handle incoming system alerts"""
        try:
            level = alert_data.get("level", "info")
            message = alert_data.get("message", "System alert")
            self.alerts_widget.add_alert(level, message)
            self.alert_triggered.emit(level, message)
        except Exception as e:
            logger.error(f"Error processing alert: {e}")

    def simulate_pulse(self):
        """Simulate heartbeat data for demonstration (with real GPU data when available)"""
        import random  # Ensure random is always available

        # Try to get real system stats first
        real_stats = self.pulse_widget.get_real_system_stats()
        if real_stats:
            simulated_data = real_stats
        else:
            # Fall back to pure simulation
            # Generate GPU usage data based on detected GPU count
            gpu_usage_data = []
            gpu_count = self.pulse_widget.gpu_count
            if gpu_count > 0:
                for i in range(gpu_count):
                    # Simulate realistic GPU usage patterns
                    base_usage = random.randint(20, 80)
                    # Add some variation between GPUs
                    variation = random.randint(-10, 10)
                    gpu_usage = max(0, min(100, base_usage + variation))
                    gpu_usage_data.append(gpu_usage)
            else:
                gpu_usage_data = []
            simulated_data = {
                "tps": random.randint(100, 500),
                "gpu_usage": gpu_usage_data,
                "timestamp": datetime.now().isoformat(),
            }
        self.pulse_widget.update_metrics(simulated_data)
        self.last_update_label.setText(
            f"Last update: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.pulse_detected.emit(simulated_data)
        self.update_system_status(simulated_data)

    def update_system_status(self, metrics: Dict[str, Any]):
        """Update overall system status based on metrics"""
        try:
            # Calculate health score
            cpu = metrics.get("cpu_usage", 0)
            memory = metrics.get("memory_usage", 0)
            errors = metrics.get("error_rate", 0)

            if errors > 10 or cpu > 90 or memory > 90:
                self.status_label.setText("🔴 System Under Stress")
                self.status_label.setStyleSheet(
                    f"color: {VoxSigilStyles.COLORS['error']};"
                )
            elif errors > 5 or cpu > 70 or memory > 80:
                self.status_label.setText("🟡 System Busy")
                self.status_label.setStyleSheet(
                    f"color: {VoxSigilStyles.COLORS['warning']};"
                )
            else:
                self.status_label.setText("💚 System Healthy")
                self.status_label.setStyleSheet(
                    f"color: {VoxSigilStyles.COLORS['success']};"
                )

        except Exception as e:
            logger.error(f"Error updating system status: {e}")

    def check_system_health(self):
        """Perform periodic system health checks"""
        try:
            # This would integrate with actual system monitoring
            # For now, just update the timestamp
            pass

        except Exception as e:
            logger.error(f"Error in health check: {e}")


# Backward compatibility
HeartbeatTab = HeartbeatMonitorTab
