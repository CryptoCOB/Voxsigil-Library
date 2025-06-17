#!/usr/bin/env python3
"""
Control Center Tab - Master control interface
==============================================

Central command interface providing chat-based interaction, system flags,
trace viewer, and universal controls for the entire VoxSigil system.
"""

import logging

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ControlCenterTab(QWidget):
    """
    Master control center tab providing:
    - Chat-based command interface
    - System flags and toggles
    - Event trace viewer
    - System overview dashboard
    """

    # Signals
    command_submitted = pyqtSignal(str)
    flag_changed = pyqtSignal(str, object)

    def __init__(self, event_bus=None, training_engine=None):
        super().__init__()
        self.event_bus = event_bus
        self.training_engine = training_engine
        self.command_history = []
        self.command_index = -1
        self.active_traces = {}

        self.setup_ui()
        self.setup_connections()

        # Start update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second

    def setup_ui(self):
        """Setup the control center UI"""
        main_layout = QVBoxLayout(self)

        # Header
        header = QLabel("VoxSigil Control Center")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel - Chat interface
        chat_panel = self.create_chat_panel()
        main_splitter.addWidget(chat_panel)

        # Right panel - Controls and status
        right_panel = QSplitter(Qt.Vertical)

        # System flags
        flags_panel = self.create_flags_panel()
        right_panel.addWidget(flags_panel)

        # Trace viewer
        trace_panel = self.create_trace_panel()
        right_panel.addWidget(trace_panel)

        # System overview
        overview_panel = self.create_overview_panel()
        right_panel.addWidget(overview_panel)

        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 300])
        right_panel.setSizes([150, 200, 150])

    def create_chat_panel(self) -> QWidget:
        """Create the chat command interface"""
        panel = QGroupBox("Command Interface")
        layout = QVBoxLayout(panel)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
            }
        """)
        layout.addWidget(self.chat_display)

        # Command input
        input_layout = QHBoxLayout()

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText(
            "Enter command (e.g., /ping, /help, /flag VANTA_MUSIC on)"
        )
        self.command_input.setFont(QFont("Consolas", 10))
        self.command_input.returnPressed.connect(self.submit_command)
        input_layout.addWidget(self.command_input)

        submit_btn = QPushButton("Submit")
        submit_btn.clicked.connect(self.submit_command)
        input_layout.addWidget(submit_btn)

        layout.addLayout(input_layout)

        # Add welcome message
        self.add_chat_message(
            "system", "VoxSigil Control Center initialized. Type /help for available commands."
        )

        return panel

    def create_flags_panel(self) -> QWidget:
        """Create the system flags control panel"""
        panel = QGroupBox("System Flags")
        layout = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Common flags
        self.flags = {}
        common_flags = [
            ("VANTA_DEBUG", False, "Enable debug mode"),
            ("VANTA_MUSIC", False, "Enable music generation"),
            ("VANTA_TRAINING", False, "Enable training mode"),
            ("VANTA_STREAMING", True, "Enable live streaming"),
            ("VANTA_COMPRESSION", True, "Enable compression"),
            ("GUI_VERBOSE", False, "Verbose GUI logging"),
            ("ARC_SOLVER_ACTIVE", True, "ARC solver enabled"),
            ("BLT_RAG_ACTIVE", True, "BLT RAG active"),
        ]

        for flag_name, default_value, description in common_flags:
            flag_widget = self.create_flag_widget(flag_name, default_value, description)
            scroll_layout.addWidget(flag_widget)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return panel

    def create_flag_widget(self, name: str, default_value: bool, description: str) -> QWidget:
        """Create a widget for a single flag"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox(name)
        checkbox.setChecked(default_value)
        checkbox.stateChanged.connect(lambda state, n=name: self.on_flag_changed(n, bool(state)))
        layout.addWidget(checkbox)

        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(desc_label)

        self.flags[name] = checkbox

        return widget

    def create_trace_panel(self) -> QWidget:
        """Create the event trace viewer"""
        panel = QGroupBox("Event Traces")
        layout = QVBoxLayout(panel)

        # Trace tree
        self.trace_tree = QTreeWidget()
        self.trace_tree.setHeaderLabels(["Event ID", "Type", "Status", "Time"])
        layout.addWidget(self.trace_tree)

        # Trace details
        self.trace_details = QTextEdit()
        self.trace_details.setMaximumHeight(100)
        self.trace_details.setFont(QFont("Consolas", 9))
        layout.addWidget(self.trace_details)

        return panel

    def create_overview_panel(self) -> QWidget:
        """Create the system overview panel"""
        panel = QGroupBox("System Overview")
        layout = QGridLayout(panel)

        # Status indicators
        layout.addWidget(QLabel("Status:"), 0, 0)
        self.system_status = QLabel("Initializing...")
        layout.addWidget(self.system_status, 0, 1)

        layout.addWidget(QLabel("Active Tabs:"), 1, 0)
        self.active_tabs_count = QLabel("0")
        layout.addWidget(self.active_tabs_count, 1, 1)

        layout.addWidget(QLabel("Streaming:"), 2, 0)
        self.streaming_count = QLabel("0")
        layout.addWidget(self.streaming_count, 2, 1)

        layout.addWidget(QLabel("Memory:"), 3, 0)
        self.memory_usage = QLabel("Unknown")
        layout.addWidget(self.memory_usage, 3, 1)

        layout.addWidget(QLabel("CPU:"), 4, 0)
        self.cpu_usage = QLabel("Unknown")
        layout.addWidget(self.cpu_usage, 4, 1)

        return panel

    def setup_connections(self):
        """Setup event bus connections"""
        if self.event_bus:
            # Subscribe to relevant events
            try:
                self.event_bus.subscribe("command.reply", self.on_command_reply)
                self.event_bus.subscribe("flag.changed", self.on_flag_update)
                self.event_bus.subscribe("trace.event", self.on_trace_event)
                self.event_bus.subscribe("system.status", self.on_system_status)
            except Exception as e:
                logger.warning(f"Could not setup event bus connections: {e}")

    @pyqtSlot()
    def submit_command(self):
        """Submit a command from the input field"""
        command = self.command_input.text().strip()
        if not command:
            return

        # Add to history
        self.command_history.append(command)
        self.command_index = len(self.command_history)

        # Clear input
        self.command_input.clear()

        # Display command
        self.add_chat_message("user", command)

        # Process command
        self.process_command(command)

    def process_command(self, command: str):
        """Process a command and generate response"""
        try:
            if command.startswith("/"):
                # Built-in commands
                parts = command[1:].split()
                cmd = parts[0].lower() if parts else ""

                if cmd == "ping":
                    self.add_chat_message("system", "pong")

                elif cmd == "help":
                    help_text = """Available commands:
/ping - Test connectivity
/help - Show this help
/flag <name> <on|off> - Set system flag
/status - Show system status
/clear - Clear chat
/tabs - List all tabs
/restart <component> - Restart component
/trace <event_id> - Show trace details"""
                    self.add_chat_message("system", help_text)

                elif cmd == "flag" and len(parts) >= 3:
                    flag_name = parts[1].upper()
                    flag_value = parts[2].lower() in ["on", "true", "1", "yes"]
                    self.set_flag(flag_name, flag_value)
                    self.add_chat_message("system", f"Flag {flag_name} set to {flag_value}")

                elif cmd == "status":
                    self.show_system_status()

                elif cmd == "clear":
                    self.chat_display.clear()

                elif cmd == "tabs":
                    self.list_tabs()

                else:
                    self.add_chat_message(
                        "system", f"Unknown command: {cmd}. Type /help for available commands."
                    )

            else:
                # Regular text - could be processed by AI or other handlers
                self.add_chat_message("system", f"Processing: {command}")
                if self.event_bus:
                    self.event_bus.publish("user.command", {"text": command})

        except Exception as e:
            self.add_chat_message("system", f"Error processing command: {e}")

    def add_chat_message(self, sender: str, message: str):
        """Add a message to the chat display"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Format message
        if sender == "user":
            formatted = f"<span style='color: #4FC3F7;'>&gt; {message}</span><br>"
        elif sender == "system":
            formatted = f"<span style='color: #81C784;'>[SYSTEM] {message}</span><br>"
        else:
            formatted = f"<span style='color: #FFB74D;'>[{sender.upper()}] {message}</span><br>"

        cursor.insertHtml(formatted)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_flag_changed(self, flag_name: str, value: bool):
        """Handle flag changes from UI"""
        self.set_flag(flag_name, value)

    def set_flag(self, flag_name: str, value: bool):
        """Set a system flag"""
        try:
            # Update UI if needed
            if flag_name in self.flags:
                self.flags[flag_name].setChecked(value)

            # Emit signal
            self.flag_changed.emit(flag_name, value)

            # Publish to event bus
            if self.event_bus:
                self.event_bus.publish("flag.set", {"flag": flag_name, "value": value})

            logger.info(f"Flag {flag_name} set to {value}")

        except Exception as e:
            logger.error(f"Error setting flag {flag_name}: {e}")

    def show_system_status(self):
        """Show detailed system status"""
        status_info = {
            "GUI": "Active",
            "Event Bus": "Connected" if self.event_bus else "Disconnected",
            "Training Engine": "Available" if self.training_engine else "Not Available",
            "Flags": len(self.flags),
            "Active Traces": len(self.active_traces),
        }

        status_text = "System Status:\n"
        for key, value in status_info.items():
            status_text += f"  {key}: {value}\n"

        self.add_chat_message("system", status_text)

    def list_tabs(self):
        """List all available tabs"""
        # This would be filled by the main GUI
        self.add_chat_message("system", "Tab listing not yet implemented")

    @pyqtSlot()
    def update_display(self):
        """Update the display with current system info"""
        try:
            # Update system status
            self.system_status.setText("Running")

            # Update memory usage (basic estimation)
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.setText(f"{memory_mb:.1f} MB")

            # Update CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.setText(f"{cpu_percent:.1f}%")

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Error updating display: {e}")

    def on_command_reply(self, event):
        """Handle command reply events"""
        try:
            reply = event.get("data", {})
            if isinstance(reply, dict):
                text = reply.get("text", str(reply))
            else:
                text = str(reply)
            self.add_chat_message("system", text)
        except Exception as e:
            logger.error(f"Error handling command reply: {e}")

    def on_flag_update(self, event):
        """Handle flag update events"""
        try:
            data = event.get("data", {})
            flag_name = data.get("flag")
            value = data.get("value")
            if flag_name and flag_name in self.flags:
                self.flags[flag_name].setChecked(bool(value))
        except Exception as e:
            logger.error(f"Error handling flag update: {e}")

    def on_trace_event(self, event):
        """Handle trace events"""
        try:
            data = event.get("data", {})
            event_id = data.get("event_id", "unknown")
            event_type = data.get("type", "unknown")
            status = data.get("status", "unknown")
            timestamp = data.get("timestamp", "unknown")

            # Add to trace tree
            item = QTreeWidgetItem([event_id, event_type, status, str(timestamp)])
            self.trace_tree.addTopLevelItem(item)

            # Keep only last 100 traces
            if self.trace_tree.topLevelItemCount() > 100:
                self.trace_tree.takeTopLevelItem(0)

            self.active_traces[event_id] = data

        except Exception as e:
            logger.error(f"Error handling trace event: {e}")

    def on_system_status(self, event):
        """Handle system status events"""
        try:
            data = event.get("data", {})
            active_tabs = data.get("active_tabs", 0)
            streaming_count = data.get("streaming_count", 0)

            self.active_tabs_count.setText(str(active_tabs))
            self.streaming_count.setText(str(streaming_count))

        except Exception as e:
            logger.error(f"Error handling system status: {e}")


# UI specification for auto-registration
ui_spec = {
    "tab": "Control-Center",
    "type": "control_center",
    "stream": False,  # chat is bidirectional, not passive
    "priority": 0,  # First tab
    "capabilities": ["command_interface", "flag_control", "trace_viewer", "system_overview"],
}
