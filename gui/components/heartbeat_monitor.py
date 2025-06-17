#!/usr/bin/env python3
"""
Heartbeat Monitor - Real-time oscilloscope-style system health visualization

Provides a live ECG-style monitor showing:
- Event throughput (events/sec)
- GPU VRAM usage %
- CPU usage %
- Error count per minute

Colors: Green=normal, Yellow=high load, Red=error spike
"""

import collections
import logging
import time
from typing import Any, Dict

import psutil

try:
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtWidgets

    HAVE_PYQTGRAPH = True
except ImportError:
    HAVE_PYQTGRAPH = False

    # Fallback for testing
    class QtWidgets:
        class QWidget:
            pass

        class QVBoxLayout:
            pass

        class QHBoxLayout:
            pass

        class QLabel:
            pass


from core.base_core import BaseCore
from Vanta.core.vanta_registration import vanta_core_module
from Vanta.interfaces.cognitive_mesh import CognitiveMeshRole

logger = logging.getLogger(__name__)


def get_gpu_memory_percent() -> float:
    """Get GPU memory usage percentage."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_stats(0)
            reserved = gpu_mem.get("reserved_bytes.all.current", 0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (reserved / total) * 100 if total > 0 else 0.0
    except Exception:
        pass
    return 0.0


@vanta_core_module(
    name="heartbeat_monitor",
    subsystem="ui",
    mesh_role=CognitiveMeshRole.MONITOR,
    description="Real-time oscilloscope-style system health monitor",
    capabilities=["system_health", "real_time_monitoring", "performance_visualization"],
)
class HeartbeatMonitor(BaseCore, QtWidgets.QWidget):
    """
    Real-time system heartbeat monitor with oscilloscope-style visualization.

    Shows live metrics in ECG-style waveforms:
    - Event throughput (green)
    - GPU memory usage (blue)
    - CPU usage (orange)
    - Error rate (red)
    """

    def __init__(self, bus=None, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        BaseCore.__init__(self)

        self.bus = bus
        self.logger = logging.getLogger("VoxSigil.GUI.Heartbeat")

        # Data storage - 600 points = 150 seconds at 4Hz
        self.data_points = 600
        self.data = {
            "tps": collections.deque(maxlen=self.data_points),
            "gpu": collections.deque(maxlen=self.data_points),
            "cpu": collections.deque(maxlen=self.data_points),
            "errors": collections.deque(maxlen=self.data_points),
        }

        # Track last update time for rate calculation
        self.last_update = time.time()
        self.event_count = 0
        self.error_count = 0
        self.last_error_check = time.time()

        # Agent-specific overlay data
        self.agent_overlay_data = collections.deque(maxlen=20)  # 5 seconds at 4Hz
        self.selected_agent_id = None

        self._init_ui()

        # Subscribe to events
        if self.bus:
            self.bus.subscribe("heartbeat", self.update_plot)
            self.bus.subscribe("agent.selected", self.on_agent_selected)
            self.bus.subscribe("*", self.count_event)  # Count all events for TPS

        # Initialize with some data
        self._populate_initial_data()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QtWidgets.QLabel("System Heartbeat Monitor")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00;")
        layout.addWidget(title)

        if not HAVE_PYQTGRAPH:
            # Fallback display
            fallback = QtWidgets.QLabel(
                "PyQtGraph not available\nInstall with: pip install pyqtgraph"
            )
            fallback.setStyleSheet("color: #ff6666; font-size: 14px;")
            layout.addWidget(fallback)
            return

        # Create the plot widget
        self.plot = pg.PlotWidget(background="#111111")
        self.plot.setLabel("left", "Value", units="%")
        self.plot.setLabel("bottom", "Time", units="samples")
        self.plot.setYRange(0, 100)
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Create curves for each metric
        self.curves = {
            "tps": self.plot.plot(pen=pg.mkPen("#44ff44", width=2), name="Events/sec"),
            "gpu": self.plot.plot(pen=pg.mkPen("#4299e1", width=2), name="GPU Memory %"),
            "cpu": self.plot.plot(pen=pg.mkPen("#e19642", width=2), name="CPU %"),
            "errors": self.plot.plot(pen=pg.mkPen("#ff4444", width=2), name="Errors/min"),
        }

        # Agent overlay curve (dotted)
        self.agent_curve = self.plot.plot(
            pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DotLine), name="Agent Activity"
        )

        layout.addWidget(self.plot)

        # Status bar
        status_layout = QtWidgets.QHBoxLayout()

        self.status_labels = {}
        for metric, color in [
            ("TPS", "#44ff44"),
            ("GPU", "#4299e1"),
            ("CPU", "#e19642"),
            ("ERR", "#ff4444"),
        ]:
            label = QtWidgets.QLabel(f"{metric}: --")
            label.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 12px;")
            self.status_labels[metric.lower()] = label
            status_layout.addWidget(label)
            status_layout.addStretch()

        # Agent selection display
        self.agent_label = QtWidgets.QLabel("Agent: None")
        self.agent_label.setStyleSheet("color: #ffffff; font-family: monospace;")
        status_layout.addWidget(self.agent_label)

        layout.addLayout(status_layout)

        # Add legend
        self.plot.addLegend()

        self.logger.info("Heartbeat monitor UI initialized")

    def _populate_initial_data(self):
        """Populate with initial data points."""
        for _ in range(10):
            self.data["tps"].append(0)
            self.data["gpu"].append(get_gpu_memory_percent())
            self.data["cpu"].append(psutil.cpu_percent())
            self.data["errors"].append(0)

    async def initialize_subsystem(self, core):
        """Initialize the heartbeat monitor subsystem."""
        self.core = core
        self.logger.info("Heartbeat monitor subsystem initialized")

    def count_event(self, topic: str, payload: Any):
        """Count events for TPS calculation."""
        self.event_count += 1

        # Check for errors
        if "error" in topic.lower() or (isinstance(payload, dict) and payload.get("error")):
            self.error_count += 1

    def update_plot(self, payload: Dict[str, Any]):
        """Update the plot with new heartbeat data."""
        if not HAVE_PYQTGRAPH:
            return

        try:
            # Update data
            for key in ["tps", "gpu", "cpu", "errors"]:
                value = payload.get(key, 0)
                self.data[key].append(value)

                # Determine color based on value
                color = self._get_color_for_metric(key, value)

                # Update curve data
                self.curves[key].setData(list(self.data[key]))
                self.curves[key].setPen(pg.mkPen(color, width=2))

                # Update status label
                if key == "tps":
                    self.status_labels["tps"].setText(f"TPS: {value:.1f}")
                elif key == "gpu":
                    self.status_labels["gpu"].setText(f"GPU: {value:.1f}%")
                elif key == "cpu":
                    self.status_labels["cpu"].setText(f"CPU: {value:.1f}%")
                elif key == "errors":
                    self.status_labels["err"].setText(f"ERR: {value:.0f}/min")

            self.logger.debug(f"Updated heartbeat plot: {payload}")

        except Exception as e:
            self.logger.error(f"Error updating heartbeat plot: {e}")

    def _get_color_for_metric(self, metric: str, value: float) -> str:
        """Get color based on metric value and thresholds."""
        if metric == "errors" and value > 1:
            return "#ff4444"  # Red for errors
        elif metric in ["gpu", "cpu"] and value > 90:
            return "#ffff44"  # Yellow for high load
        elif metric in ["gpu", "cpu"] and value > 75:
            return "#ffa500"  # Orange for medium load
        else:
            # Default colors
            colors = {"tps": "#44ff44", "gpu": "#4299e1", "cpu": "#e19642", "errors": "#ff4444"}
            return colors.get(metric, "#ffffff")

    def on_agent_selected(self, payload: Dict[str, Any]):
        """Handle agent selection for overlay display."""
        agent_id = payload.get("agent_id")
        self.selected_agent_id = agent_id

        if agent_id:
            self.agent_label.setText(f"Agent: {agent_id}")
            # Subscribe to agent-specific heartbeat
            if self.bus:
                self.bus.subscribe(f"agent.{agent_id}.heartbeat", self.update_agent_overlay)
        else:
            self.agent_label.setText("Agent: None")
            self.agent_overlay_data.clear()
            if HAVE_PYQTGRAPH:
                self.agent_curve.setData([])

    def update_agent_overlay(self, payload: Dict[str, Any]):
        """Update agent-specific overlay curve."""
        if not HAVE_PYQTGRAPH or not self.selected_agent_id:
            return

        try:
            # Add agent activity level (0-100)
            activity = payload.get("activity_level", 0)
            self.agent_overlay_data.append(activity)

            # Update overlay curve
            self.agent_curve.setData(list(self.agent_overlay_data))

        except Exception as e:
            self.logger.error(f"Error updating agent overlay: {e}")

    def get_ui_spec(self) -> Dict[str, Any]:
        """Get UI specification for bridge registration."""
        return {
            "tab": "System Health",
            "widget": "HeartbeatMonitor",
            "stream": True,
            "stream_topic": "heartbeat",
            "priority": 2,
        }


def test_heartbeat_monitor():
    """Test the heartbeat monitor functionality."""
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)  # noqa: F841

    # Create mock bus
    class MockBus:
        def subscribe(self, topic, callback):
            pass

    monitor = HeartbeatMonitor(bus=MockBus())
    monitor.show()

    # Simulate some data
    test_data = {"tps": 15.5, "gpu": 67.3, "cpu": 45.2, "errors": 0}
    monitor.update_plot(test_data)

    print("Heartbeat monitor test completed!")
    return monitor


if __name__ == "__main__":
    test_heartbeat_monitor()
