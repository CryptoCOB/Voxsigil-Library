"""
Real-time System Status Dashboard
===============================

Comprehensive streaming data display showing all system metrics,
UnifiedVantaCore status, and live component activity.
"""

import datetime
import logging
from typing import Any, Dict, List

import psutil
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.components.real_time_data_provider import get_all_metrics

logger = logging.getLogger("SystemDashboard")


class LiveMetricsWidget(QWidget):
    """Widget showing live streaming metrics."""

    def __init__(self, title: str, metric_keys: List[str]):
        super().__init__()
        self.title = title
        self.metric_keys = metric_keys
        self.metrics = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2E86AB; padding: 5px;")
        layout.addWidget(title_label)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(200)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #555;
            }
        """)
        layout.addWidget(self.metrics_text)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update displayed metrics."""
        self.metrics = metrics

        display_text = (
            f"🔴 LIVE {self.title} | {datetime.datetime.now().strftime('%H:%M:%S')}\n"
        )
        display_text += "=" * 60 + "\n"

        for key in self.metric_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    display_text += f"{key:25}: {value:8.2f}\n"
                elif isinstance(value, bool):
                    display_text += f"{key:25}: {'🟢 YES' if value else '🔴 NO'}\n"
                else:
                    display_text += f"{key:25}: {str(value)[:30]}\n"

        # Keep only last few updates
        current_text = self.metrics_text.toPlainText()
        lines = current_text.split("\n")
        if len(lines) > 100:
            lines = lines[-50:]
            self.metrics_text.setPlainText("\n".join(lines))

        self.metrics_text.append(display_text)

        # Auto-scroll to bottom
        scrollbar = self.metrics_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class UnifiedVantaCoreStatus(QWidget):
    """Real-time UnifiedVantaCore status display."""

    def __init__(self):
        super().__init__()
        self.vanta_core = None
        self._init_ui()
        self._try_connect_vanta()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🧠 UnifiedVantaCore Live Status")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #8B0000; padding: 5px;")
        layout.addWidget(title)

        # Connection status
        self.connection_status = QLabel("🔴 Connecting...")
        self.connection_status.setStyleSheet("font-weight: bold; color: red;")
        layout.addWidget(self.connection_status)

        # Component registry
        components_group = QGroupBox("Active Components")
        components_layout = QVBoxLayout()

        self.components_list = QListWidget()
        self.components_list.setMaximumHeight(150)
        components_layout.addWidget(self.components_list)

        components_group.setLayout(components_layout)
        layout.addWidget(components_group)

        # Agent status
        agents_group = QGroupBox("Agent Registry")
        agents_layout = QVBoxLayout()

        self.agents_status = QLabel("Agents: 0 active")
        agents_layout.addWidget(self.agents_status)

        self.agents_list = QListWidget()
        self.agents_list.setMaximumHeight(100)
        agents_layout.addWidget(self.agents_list)

        agents_group.setLayout(agents_layout)
        layout.addWidget(agents_group)

        # Metrics
        self.metrics_label = QLabel("Cognitive Load: 0% | Events: 0 | Memory: 0MB")
        self.metrics_label.setStyleSheet("font-family: monospace; padding: 5px;")
        layout.addWidget(self.metrics_label)

    def _try_connect_vanta(self):
        """Try to connect to UnifiedVantaCore."""
        try:
            # Use real-time data provider instead of direct VantaCore calls
            from .real_time_data_provider import RealTimeDataProvider

            data_provider = RealTimeDataProvider()
            vanta_metrics = data_provider.get_vanta_core_metrics()

            if vanta_metrics["vanta_core_connected"]:
                self.connection_status.setText("🟢 Connected to UnifiedVantaCore")
                self.connection_status.setStyleSheet("font-weight: bold; color: green;")
                return True
        except ImportError as e:
            logger.debug(f"Could not import RealTimeDataProvider: {e}")
        except AttributeError as e:
            logger.debug(f"VantaCore connection failed - missing method: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error connecting to VantaCore: {e}")

        self.connection_status.setText("🟡 VantaCore unavailable - using simulation")
        self.connection_status.setStyleSheet("font-weight: bold; color: orange;")
        return False

    def update_status(self):
        """Update VantaCore status display."""
        # Try to use real data first, fall back to simulation if unavailable
        if self._try_connect_vanta():
            self._update_real_status()
        else:
            self._update_simulated_status()

    def _update_real_status(self):
        """Update with real VantaCore data."""
        try:
            # Use real-time data provider instead of direct VantaCore calls
            from .real_time_data_provider import RealTimeDataProvider

            data_provider = RealTimeDataProvider()
            vanta_metrics = data_provider.get_vanta_core_metrics()

            # Update components list
            components = vanta_metrics.get("components", {})
            self.components_list.clear()
            for name in components:
                status = "🟢"  # Assume healthy for now
                item = QListWidgetItem(f"{status} {name}")
                self.components_list.addItem(item)  # Update agent count
                agent_count = vanta_metrics.get("total_agents", 0)
            self.agents_status.setText(f"Agents: {agent_count} active")

            # Update metrics using real data (reuse existing data_provider)
            all_metrics = data_provider.get_all_metrics()

            cognitive_load = all_metrics["cognitive_load"]
            event_count = all_metrics["events_processed"]
            memory_usage = all_metrics["memory_usage_mb"]

            self.metrics_label.setText(
                f"Cognitive Load: {cognitive_load:.1%} | Events: {event_count} | Memory: {memory_usage:.0f}MB"
            )

        except Exception as e:
            logger.debug(f"Error updating real VantaCore status: {e}")
            self._update_simulated_status()

    def _update_simulated_status(self):
        """Update with simulated VantaCore data."""
        # Simulate active components
        simulated_components = [
            "🟢 BLTEncoder",
            "🟢 VoxSigilMesh",
            "🟡 GridFormer",
            "🟢 AgentRegistry",
            "🟢 AsyncBus",
            "🟡 SupervisorConnector",
        ]

        self.components_list.clear()
        for comp in simulated_components:
            item = QListWidgetItem(comp)
            self.components_list.addItem(item)

        # Simulate agents
        simulated_agents = [
            "🤖 MusicComposerAgent",
            "🤖 TrainingAgent",
            "🤖 VisualizationAgent",
        ]

        self.agents_status.setText(f"Agents: {len(simulated_agents)} simulated")

        self.agents_list.clear()
        for agent in simulated_agents:
            item = QListWidgetItem(agent)
            self.agents_list.addItem(
                item
            )  # Use fallback real metrics instead of pure simulation
        try:
            # Try to get at least some real metrics
            all_metrics = get_all_metrics()
            cognitive_load = all_metrics["cognitive_load"]
            event_count = all_metrics["events_processed"]
            memory_usage = all_metrics["memory_usage_mb"]
        except Exception as e:
            logger.debug(f"Could not get real metrics for fallback: {e}")
            # Use minimal real system data as final fallback
            import psutil

            memory = psutil.virtual_memory()
            cognitive_load = 0.5  # 50% default
            event_count = 100  # Default
            memory_usage = memory.used / (1024**2)  # Real memory usage in MB

        self.metrics_label.setText(
            f"Cognitive Load: {cognitive_load:.1%} | Events: {event_count} | Memory: {memory_usage:.0f}MB (Fallback)"
        )


class StreamingDashboard(QWidget):
    """Main streaming dashboard showing all real-time data."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxSigil Real-time System Dashboard")
        self.resize(1200, 800)

        self._init_ui()
        self._setup_timers()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("🚀 VoxSigil Real-time System Dashboard")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setStyleSheet("color: #2E86AB; padding: 10px; text-align: center;")
        layout.addWidget(header)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - System metrics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # System metrics
        self.system_metrics = LiveMetricsWidget(
            "💻 System Metrics",
            [
                "cpu_percent",
                "memory_percent",
                "disk_usage",
                "network_bytes_sent",
                "process_count",
                "load_average",
            ],
        )
        left_layout.addWidget(self.system_metrics)

        # Training metrics
        self.training_metrics = LiveMetricsWidget(
            "🎯 Training Metrics",
            [
                "training_loss",
                "validation_accuracy",
                "learning_rate",
                "batch_processing_time",
                "model_parameters",
            ],
        )
        left_layout.addWidget(self.training_metrics)

        splitter.addWidget(left_panel)

        # Right panel - VantaCore and other components
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # VantaCore status
        self.vanta_status = UnifiedVantaCoreStatus()
        right_layout.addWidget(self.vanta_status)

        # Audio metrics
        self.audio_metrics = LiveMetricsWidget(
            "🎵 Audio Engine",
            ["audio_level", "sample_rate", "audio_latency", "buffer_status"],
        )
        right_layout.addWidget(self.audio_metrics)

        splitter.addWidget(right_panel)

        # Bottom status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("🔴 LIVE STREAMING | Updating every second")
        self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
        status_layout.addWidget(self.status_label)

        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        status_layout.addWidget(self.pause_btn)

        layout.addLayout(status_layout)

    def _setup_timers(self):
        """Setup update timers."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_all_metrics)
        self.update_timer.start(1000)  # Update every second

        self.vanta_timer = QTimer()
        self.vanta_timer.timeout.connect(self.vanta_status.update_status)
        self.vanta_timer.start(2000)  # Update VantaCore every 2 seconds

    def _update_all_metrics(self):
        """Update all streaming metrics using real data."""
        try:
            # Get all real metrics from the data provider
            all_metrics = get_all_metrics()

            # Structure metrics for display
            system_metrics = {
                "cpu_percent": all_metrics["cpu_percent"],
                "memory_percent": all_metrics["memory_percent"],
                "disk_usage": all_metrics["disk_usage_percent"],
                "network_bytes_sent": all_metrics["network_bytes_sent"],
                "process_count": all_metrics["process_count"],
                "load_average": all_metrics["cpu_load_avg"],
            }

            # Real training metrics
            training_metrics = {
                "training_loss": all_metrics["training_loss"],
                "validation_accuracy": all_metrics["validation_accuracy"],
                "learning_rate": all_metrics["learning_rate"],
                "batch_processing_time": all_metrics["batch_processing_time"],
                "model_parameters": all_metrics["model_parameters"],
            }

            # Real audio metrics
            audio_metrics = {
                "audio_level": all_metrics["audio_level"],
                "sample_rate": all_metrics["sample_rate"],
                "audio_latency": all_metrics["audio_latency"],
                "buffer_status": "OK" if all_metrics["audio_latency"] < 30 else "HIGH",
            }

            # Update displays
            self.system_metrics.update_metrics(system_metrics)
            self.training_metrics.update_metrics(training_metrics)
            self.audio_metrics.update_metrics(audio_metrics)

        except Exception as e:
            logger.error(f"Error updating streaming metrics: {e}")
            # Fallback to basic system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_usage": 50.0,  # Default fallback
                "network_bytes_sent": 0.0,
                "process_count": len(psutil.pids()),
                "load_average": cpu_percent / 100.0,
            }

            # Update only system metrics in fallback mode
            self.system_metrics.update_metrics(system_metrics)

        # Update status
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"🔴 LIVE STREAMING | Last update: {timestamp}")

    def _toggle_pause(self):
        """Toggle pause/resume streaming."""
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.vanta_timer.stop()
            self.pause_btn.setText("▶️ Resume")
            self.status_label.setText("⏸️ PAUSED")
            self.status_label.setStyleSheet(
                "color: orange; font-weight: bold; padding: 5px;"
            )
        else:
            self.update_timer.start(1000)
            self.vanta_timer.start(2000)
            self.pause_btn.setText("⏸️ Pause")
            self.status_label.setText("🔴 LIVE STREAMING")
            self.status_label.setStyleSheet(
                "color: red; font-weight: bold; padding: 5px;"
            )


def show_streaming_dashboard():
    """Show the streaming dashboard."""
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dashboard = StreamingDashboard()
    dashboard.show()

    return dashboard


if __name__ == "__main__":
    dashboard = show_streaming_dashboard()
    import sys

    sys.exit(dashboard.exec_())
