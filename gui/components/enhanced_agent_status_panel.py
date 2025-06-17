"""
Enhanced Agent Status Panel with Development Mode Controls
Comprehensive agent monitoring interface with configurable dev mode options.
"""

import logging

from core.dev_config_manager import get_dev_config
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.components.dev_mode_panel import DevModeControlPanel

logger = logging.getLogger("EnhancedAgentStatusPanel")


class EnhancedAgentStatusPanel(QWidget):
    """
    Enhanced Agent Status panel with comprehensive controls and dev mode support.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_dev_config()
        self.agents_data = {}
        self.status_history = []

        self._init_ui()
        self._setup_timers()
        self._connect_signals()
        self._load_initial_data()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Development Mode Panel
        self.dev_panel = DevModeControlPanel("agent_status", self)
        layout.addWidget(self.dev_panel)

        # Main content
        main_splitter = QSplitter(Qt.Vertical)

        # Controls panel
        controls_widget = self._create_controls_panel()
        main_splitter.addWidget(controls_widget)

        # Status display
        status_widget = self._create_status_panel()
        main_splitter.addWidget(status_widget)

        main_splitter.setSizes([120, 400])
        layout.addWidget(main_splitter)

        # Status bar
        self.status_label = QLabel("Agent Status Monitor Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ccc;")
        layout.addWidget(self.status_label)

    def _create_controls_panel(self) -> QWidget:
        """Create the controls panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Monitor Settings
        monitor_group = QGroupBox("📡 Monitor Settings")
        monitor_layout = QGridLayout(monitor_group)

        # Update interval
        monitor_layout.addWidget(QLabel("Update (sec):"), 0, 0)
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(1, 60)
        self.update_interval_spin.setValue(5)
        monitor_layout.addWidget(self.update_interval_spin, 0, 1)

        # Agent filter
        monitor_layout.addWidget(QLabel("Agent Filter:"), 0, 2)
        self.agent_filter_combo = QComboBox()
        self.agent_filter_combo.addItems(
            ["ALL", "Nova", "Aria", "Kai", "Echo", "Sage", "ONLINE", "OFFLINE"]
        )
        monitor_layout.addWidget(self.agent_filter_combo, 0, 3)

        # Auto-refresh
        self.auto_refresh_checkbox = QCheckBox("Auto Refresh")
        self.auto_refresh_checkbox.setChecked(True)
        monitor_layout.addWidget(self.auto_refresh_checkbox, 1, 0, 1, 2)

        # Show inactive
        self.show_inactive_checkbox = QCheckBox("Show Inactive")
        self.show_inactive_checkbox.setChecked(True)
        monitor_layout.addWidget(self.show_inactive_checkbox, 1, 2, 1, 2)

        layout.addWidget(monitor_group)

        # Actions
        actions_group = QGroupBox("⚡ Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self._refresh_status)
        actions_layout.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("🗑️ Clear History")
        self.clear_btn.clicked.connect(self._clear_history)
        actions_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("📊 Export")
        self.export_btn.clicked.connect(self._export_data)
        actions_layout.addWidget(self.export_btn)

        layout.addWidget(actions_group)

        # Dev Options (shown in dev mode)
        self.dev_options_group = QGroupBox("🔧 Dev Options")
        dev_layout = QGridLayout(self.dev_options_group)

        # Detailed metrics
        self.detailed_metrics_checkbox = QCheckBox("Detailed Metrics")
        dev_layout.addWidget(self.detailed_metrics_checkbox, 0, 0)

        # Debug logging
        self.debug_logging_checkbox = QCheckBox("Debug Logging")
        dev_layout.addWidget(self.debug_logging_checkbox, 0, 1)

        # Performance monitoring
        self.performance_monitoring_checkbox = QCheckBox("Performance Monitor")
        dev_layout.addWidget(self.performance_monitoring_checkbox, 1, 0)

        # Voice status tracking
        self.voice_tracking_checkbox = QCheckBox("Voice Tracking")
        dev_layout.addWidget(self.voice_tracking_checkbox, 1, 1)

        layout.addWidget(self.dev_options_group)

        return widget

    def _create_status_panel(self) -> QWidget:
        """Create the status display panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabs for different views
        tabs = QTabWidget()

        # Agent Overview Tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(6)
        self.agents_table.setHorizontalHeaderLabels(
            ["Agent", "Status", "Voice", "Tasks", "CPU", "Memory"]
        )
        self.agents_table.setAlternatingRowColors(True)
        overview_layout.addWidget(self.agents_table)

        tabs.addTab(overview_tab, "Agent Overview")

        # Status History Tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        self.history_display = QTextEdit()
        self.history_display.setPlaceholderText(
            "Agent status history will appear here..."
        )
        self.history_display.setFont(QFont("Consolas", 9))
        self.history_display.setReadOnly(True)
        history_layout.addWidget(self.history_display)

        tabs.addTab(history_tab, "Status History")

        # Performance Tab (shown in dev mode)
        self.performance_tab = QWidget()
        performance_layout = QVBoxLayout(self.performance_tab)

        # Performance metrics table
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(4)
        self.performance_table.setHorizontalHeaderLabels(
            ["Agent", "Response Time", "Success Rate", "Load"]
        )
        self.performance_table.setAlternatingRowColors(True)
        performance_layout.addWidget(self.performance_table)

        # Real-time metrics
        self.realtime_metrics = QTextEdit()
        self.realtime_metrics.setMaximumHeight(100)
        self.realtime_metrics.setPlaceholderText("Real-time agent metrics...")
        performance_layout.addWidget(self.realtime_metrics)

        tabs.addTab(self.performance_tab, "🔧 Performance")

        # Voice Status Tab
        voice_tab = QWidget()
        voice_layout = QVBoxLayout(voice_tab)

        self.voice_table = QTableWidget()
        self.voice_table.setColumnCount(4)
        self.voice_table.setHorizontalHeaderLabels(
            ["Agent", "Voice Profile", "TTS Engine", "Last Speech"]
        )
        self.voice_table.setAlternatingRowColors(True)
        voice_layout.addWidget(self.voice_table)

        tabs.addTab(voice_tab, "🎙️ Voice Status")

        layout.addWidget(tabs)

        # Summary statistics
        summary_group = QGroupBox("📊 Summary")
        summary_layout = QGridLayout(summary_group)

        self.total_agents_label = QLabel("Total: 0")
        self.online_agents_label = QLabel("Online: 0")
        self.active_tasks_label = QLabel("Tasks: 0")
        self.avg_response_label = QLabel("Avg Response: 0ms")

        summary_layout.addWidget(self.total_agents_label, 0, 0)
        summary_layout.addWidget(self.online_agents_label, 0, 1)
        summary_layout.addWidget(self.active_tasks_label, 1, 0)
        summary_layout.addWidget(self.avg_response_label, 1, 1)

        layout.addWidget(summary_group)

        return widget

    def _setup_timers(self):
        """Setup update timers."""
        # Main status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_agent_status)

        # Performance metrics timer (for dev mode)
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_performance_metrics)

        # Start with initial interval
        self._update_timer_interval()

    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect dev panel signals
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_toggled)
        self.dev_panel.config_changed.connect(self._on_config_changed)

        # Connect control signals
        self.update_interval_spin.valueChanged.connect(self._update_timer_interval)
        self.auto_refresh_checkbox.toggled.connect(self._on_auto_refresh_toggled)
        self.agent_filter_combo.currentTextChanged.connect(self._apply_agent_filter)

        # Connect dev option signals
        self.detailed_metrics_checkbox.toggled.connect(
            self._on_detailed_metrics_toggled
        )
        self.performance_monitoring_checkbox.toggled.connect(
            self._on_performance_monitoring_toggled
        )

    def _load_initial_data(self):
        """Load initial agent data."""
        # Initialize with default agents
        default_agents = ["Nova", "Aria", "Kai", "Echo", "Sage"]

        for agent_name in default_agents:
            self.agents_data[agent_name] = {
                "status": "offline",
                "voice_enabled": True,
                "active_tasks": 0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "last_activity": "Never",
                "voice_profile": f"{agent_name}_voice",
                "tts_engine": "SpeechT5",
                "response_time": 0,
                "success_rate": 100.0,
                "load": 0.0,
            }

        # Update UI based on current configuration
        self._update_dev_options_visibility()
        self._update_performance_visibility()
        self._refresh_status()

    def _update_dev_options_visibility(self):
        """Update visibility of dev options based on dev mode."""
        is_dev = self.config.get_tab_config("agent_status").dev_mode
        self.dev_options_group.setVisible(
            is_dev or self.config.get_tab_config("agent_status").show_advanced_controls
        )

    def _update_performance_visibility(self):
        """Update visibility of performance metrics based on dev mode."""
        is_dev = self.config.get_tab_config("agent_status").dev_mode
        show_detailed = self.detailed_metrics_checkbox.isChecked()

        self.performance_tab.setVisible(is_dev and show_detailed)

        if (
            is_dev
            and show_detailed
            and self.performance_monitoring_checkbox.isChecked()
        ):
            if not self.metrics_timer.isActive():
                self.metrics_timer.start(2000)  # Update every 2 seconds
        else:
            self.metrics_timer.stop()

    def add_status(self, status_message: str):
        """Add a status message to the history."""
        import time

        timestamp = time.strftime("%H:%M:%S")

        formatted_msg = f"[{timestamp}] {status_message}"
        self.history_display.append(formatted_msg)

        # Store in history
        self.status_history.append({"timestamp": timestamp, "message": status_message})

        # Limit history size
        if len(self.status_history) > 1000:
            self.status_history = self.status_history[-500:]

        # Auto-scroll
        scrollbar = self.history_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Try to extract agent info from status message
        self._parse_status_message(status_message)

    def _parse_status_message(self, message: str):
        """Parse status message to update agent data."""
        message_lower = message.lower()

        # Update agent status based on message content
        for agent_name in self.agents_data.keys():
            if agent_name.lower() in message_lower:
                if "online" in message_lower or "active" in message_lower:
                    self.agents_data[agent_name]["status"] = "online"
                elif "offline" in message_lower or "disconnected" in message_lower:
                    self.agents_data[agent_name]["status"] = "offline"
                elif "speaking" in message_lower or "voice" in message_lower:
                    self.agents_data[agent_name]["voice_enabled"] = True

                # Update last activity
                import time

                self.agents_data[agent_name]["last_activity"] = time.strftime(
                    "%H:%M:%S"
                )
                break

    @pyqtSlot()
    def _refresh_status(self):
        """Refresh the agent status display."""
        # Simulate status updates
        import random

        for agent_name, agent_data in self.agents_data.items():
            # Simulate status changes
            if random.random() < 0.1:  # 10% chance of status change
                agent_data["status"] = random.choice(
                    ["online", "offline", "busy", "idle"]
                )

            # Simulate metrics
            if agent_data["status"] == "online":
                agent_data["cpu_usage"] = random.uniform(5, 30)
                agent_data["memory_usage"] = random.uniform(50, 200)
                agent_data["active_tasks"] = random.randint(0, 5)
                agent_data["response_time"] = random.randint(50, 300)
                agent_data["success_rate"] = random.uniform(85, 100)
                agent_data["load"] = random.uniform(0, 1)
            else:
                agent_data["cpu_usage"] = 0
                agent_data["memory_usage"] = 0
                agent_data["active_tasks"] = 0

        self._update_agents_table()
        self._update_voice_table()
        self._update_performance_table()
        self._update_summary()

    def _update_agents_table(self):
        """Update the agents overview table."""
        # Filter agents based on current filter
        filter_text = self.agent_filter_combo.currentText()
        filtered_agents = {}

        for name, data in self.agents_data.items():
            if filter_text == "ALL":
                filtered_agents[name] = data
            elif filter_text == "ONLINE" and data["status"] == "online":
                filtered_agents[name] = data
            elif filter_text == "OFFLINE" and data["status"] == "offline":
                filtered_agents[name] = data
            elif filter_text == name:
                filtered_agents[name] = data
            elif (
                not self.show_inactive_checkbox.isChecked()
                and data["status"] == "offline"
            ):
                continue
            else:
                filtered_agents[name] = data

        # Update table
        self.agents_table.setRowCount(len(filtered_agents))

        for row, (name, data) in enumerate(filtered_agents.items()):
            # Agent name
            self.agents_table.setItem(row, 0, QTableWidgetItem(name))

            # Status with color coding
            status_item = QTableWidgetItem(data["status"].title())
            if data["status"] == "online":
                status_item.setBackground(QColor(144, 238, 144))  # Light green
            elif data["status"] == "offline":
                status_item.setBackground(QColor(255, 182, 193))  # Light red
            elif data["status"] == "busy":
                status_item.setBackground(QColor(255, 255, 0))  # Yellow
            self.agents_table.setItem(row, 1, status_item)

            # Voice status
            voice_status = "Enabled" if data["voice_enabled"] else "Disabled"
            self.agents_table.setItem(row, 2, QTableWidgetItem(voice_status))

            # Active tasks
            self.agents_table.setItem(
                row, 3, QTableWidgetItem(str(data["active_tasks"]))
            )

            # CPU usage
            cpu_item = QTableWidgetItem(f"{data['cpu_usage']:.1f}%")
            self.agents_table.setItem(row, 4, cpu_item)

            # Memory usage
            memory_item = QTableWidgetItem(f"{data['memory_usage']:.0f}MB")
            self.agents_table.setItem(row, 5, memory_item)

        # Resize columns to content
        self.agents_table.resizeColumnsToContents()

    def _update_voice_table(self):
        """Update the voice status table."""
        self.voice_table.setRowCount(len(self.agents_data))

        for row, (name, data) in enumerate(self.agents_data.items()):
            self.voice_table.setItem(row, 0, QTableWidgetItem(name))
            self.voice_table.setItem(row, 1, QTableWidgetItem(data["voice_profile"]))
            self.voice_table.setItem(row, 2, QTableWidgetItem(data["tts_engine"]))
            self.voice_table.setItem(row, 3, QTableWidgetItem(data["last_activity"]))

        self.voice_table.resizeColumnsToContents()

    def _update_performance_table(self):
        """Update the performance metrics table."""
        if not self.performance_tab.isVisible():
            return

        self.performance_table.setRowCount(len(self.agents_data))

        for row, (name, data) in enumerate(self.agents_data.items()):
            self.performance_table.setItem(row, 0, QTableWidgetItem(name))
            self.performance_table.setItem(
                row, 1, QTableWidgetItem(f"{data['response_time']}ms")
            )
            self.performance_table.setItem(
                row, 2, QTableWidgetItem(f"{data['success_rate']:.1f}%")
            )
            self.performance_table.setItem(
                row, 3, QTableWidgetItem(f"{data['load']:.2f}")
            )

        self.performance_table.resizeColumnsToContents()

    def _update_summary(self):
        """Update summary statistics."""
        total_agents = len(self.agents_data)
        online_agents = sum(
            1 for data in self.agents_data.values() if data["status"] == "online"
        )
        total_tasks = sum(data["active_tasks"] for data in self.agents_data.values())
        # Calculate average response time for online agents
        online_response_times = [
            data["response_time"]
            for data in self.agents_data.values()
            if data["status"] == "online"
        ]
        avg_response = (
            sum(online_response_times) / len(online_response_times)
            if online_response_times
            else 0
        )

        self.total_agents_label.setText(f"Total: {total_agents}")
        self.online_agents_label.setText(f"Online: {online_agents}")
        self.active_tasks_label.setText(f"Tasks: {total_tasks}")
        self.avg_response_label.setText(f"Avg Response: {avg_response:.0f}ms")

    @pyqtSlot()
    def _clear_history(self):
        """Clear the status history."""
        self.history_display.clear()
        self.status_history.clear()
        self.status_label.setText("Status history cleared")

    @pyqtSlot()
    def _export_data(self):
        """Export agent data to file."""
        import json
        import time

        from PyQt5.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Agent Data",
            "agent_data.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if file_path:
            try:
                export_data = {
                    "agents": self.agents_data,
                    "history": self.status_history,
                    "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Write to temporary file first, then rename for atomic operation

                temp_file = file_path + ".tmp"
                try:
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(export_data, f, indent=2)

                    # Atomic rename
                    import os

                    if os.path.exists(file_path):
                        os.replace(temp_file, file_path)
                    else:
                        os.rename(temp_file, file_path)

                    self.status_label.setText(f"Data exported to {file_path}")
                except Exception as e:  # Clean up temp file if it exists
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except (OSError, PermissionError) as cleanup_error:
                            logger.warning(
                                f"Failed to clean up temp file {temp_file}: {cleanup_error}"
                            )
                        except Exception as cleanup_error:
                            logger.error(
                                f"Unexpected error cleaning up temp file {temp_file}: {cleanup_error}"
                            )
                    raise e

            except PermissionError as e:
                self.status_label.setText(f"Export error - permission denied: {e}")
            except OSError as e:
                self.status_label.setText(f"Export error - file system error: {e}")
            except Exception as e:
                self.status_label.setText(f"Export error - unexpected: {e}")

    @pyqtSlot()
    def _update_timer_interval(self):
        """Update the timer interval."""
        interval = self.update_interval_spin.value() * 1000  # Convert to milliseconds
        if self.status_timer.isActive():
            self.status_timer.setInterval(interval)
        self.status_label.setText(
            f"Update interval: {self.update_interval_spin.value()}s"
        )

    @pyqtSlot(bool)
    def _on_auto_refresh_toggled(self, checked: bool):
        """Handle auto-refresh toggle."""
        if checked:
            interval = self.update_interval_spin.value() * 1000
            self.status_timer.start(interval)
            self.status_label.setText("Auto-refresh enabled")
        else:
            self.status_timer.stop()
            self.status_label.setText("Auto-refresh disabled")

    @pyqtSlot(str)
    def _apply_agent_filter(self, filter_text: str):
        """Apply agent filter."""
        self._update_agents_table()
        self.status_label.setText(f"Filter applied: {filter_text}")

    @pyqtSlot(bool)
    def _on_detailed_metrics_toggled(self, checked: bool):
        """Handle detailed metrics toggle."""
        self._update_performance_visibility()
        self.status_label.setText(
            f"Detailed metrics: {'Enabled' if checked else 'Disabled'}"
        )

    @pyqtSlot(bool)
    def _on_performance_monitoring_toggled(self, checked: bool):
        """Handle performance monitoring toggle."""
        self._update_performance_visibility()
        self.status_label.setText(
            f"Performance monitoring: {'Enabled' if checked else 'Disabled'}"
        )

    @pyqtSlot(bool)
    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        self._update_dev_options_visibility()
        self._update_performance_visibility()

        self.add_status(f"Developer mode {'enabled' if enabled else 'disabled'}")

    @pyqtSlot(str, object)
    def _on_config_changed(self, setting_name: str, value):
        """Handle configuration changes."""
        self.add_status(f"Config updated: {setting_name} = {value}")

    @pyqtSlot()
    def _update_agent_status(self):
        """Timer callback to update agent status."""
        if self.auto_refresh_checkbox.isChecked():
            self._refresh_status()

    @pyqtSlot()
    def _update_performance_metrics(self):
        """Update real-time performance metrics."""
        if not self.performance_monitoring_checkbox.isChecked():
            return

        import time

        timestamp = time.strftime("%H:%M:%S")

        # Calculate system-wide metrics
        total_cpu = sum(
            data["cpu_usage"]
            for data in self.agents_data.values()
            if data["status"] == "online"
        )
        total_memory = sum(data["memory_usage"] for data in self.agents_data.values())
        avg_load = sum(data["load"] for data in self.agents_data.values()) / len(
            self.agents_data
        )

        metrics_text = f"[{timestamp}] System CPU: {total_cpu:.1f}% | Memory: {total_memory:.0f}MB | Avg Load: {avg_load:.2f}\n"

        # Keep only last 5 lines
        current_text = self.realtime_metrics.toPlainText()
        lines = current_text.split("\n")
        if len(lines) > 5:
            lines = lines[-4:]
            self.realtime_metrics.setPlainText("\n".join(lines))

        self.realtime_metrics.append(metrics_text.strip())
