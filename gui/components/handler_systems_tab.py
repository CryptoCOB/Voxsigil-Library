#!/usr/bin/env python3
"""
Handler Systems Tab - Real-time Handler Performance Monitoring
Provides live monitoring of VoxSigil handler systems, event handlers, and message processing.
"""

import logging
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class HandlerPerformanceWidget(QWidget):
    """Widget displaying handler performance metrics"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_metrics)
        self.update_timer.start(1500)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Handler Performance Overview
        perf_group = QGroupBox("Handler Performance Overview")
        perf_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        perf_layout = QGridLayout(perf_group)

        # Performance metrics
        self.active_handlers_label = VoxSigilWidgetFactory.create_label(
            "Active Handlers: --", "info"
        )
        self.total_throughput_label = VoxSigilWidgetFactory.create_label(
            "Total Throughput: -- msg/s", "info"
        )
        self.avg_latency_label = VoxSigilWidgetFactory.create_label("Avg Latency: -- ms", "info")
        self.error_rate_label = VoxSigilWidgetFactory.create_label("Error Rate: --%", "info")

        # Queue and processing stats
        self.queue_size_label = VoxSigilWidgetFactory.create_label("Queue Size: --", "info")
        self.processed_today_label = VoxSigilWidgetFactory.create_label(
            "Processed Today: --", "info"
        )

        self.throughput_progress = VoxSigilWidgetFactory.create_progress_bar()
        self.latency_progress = VoxSigilWidgetFactory.create_progress_bar()

        perf_layout.addWidget(self.active_handlers_label, 0, 0)
        perf_layout.addWidget(self.total_throughput_label, 0, 1)
        perf_layout.addWidget(self.avg_latency_label, 1, 0)
        perf_layout.addWidget(self.error_rate_label, 1, 1)
        perf_layout.addWidget(self.queue_size_label, 2, 0)
        perf_layout.addWidget(self.processed_today_label, 2, 1)
        perf_layout.addWidget(self.throughput_progress, 3, 0)
        perf_layout.addWidget(self.latency_progress, 3, 1)

        layout.addWidget(perf_group)

        # Message Types Statistics
        msg_group = QGroupBox("Message Type Statistics")
        msg_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        msg_layout = QGridLayout(msg_group)

        # Message type counters
        message_types = ["User Interactions", "System Events", "Agent Messages", "Error Reports"]
        self.message_counters = {}

        for i, msg_type in enumerate(message_types):
            label = VoxSigilWidgetFactory.create_label(f"{msg_type}:", "info")
            count_label = VoxSigilWidgetFactory.create_label("-- / min", "normal")
            progress = VoxSigilWidgetFactory.create_progress_bar()

            self.message_counters[msg_type] = {"count": count_label, "progress": progress}

            msg_layout.addWidget(label, i, 0)
            msg_layout.addWidget(count_label, i, 1)
            msg_layout.addWidget(progress, i, 2)

        layout.addWidget(msg_group)

    def refresh_metrics(self):
        """Refresh handler performance metrics"""
        try:
            import random

            # Simulate handler metrics
            active_handlers = random.randint(5, 15)
            throughput = random.randint(50, 500)
            latency = random.randint(10, 200)
            error_rate = random.uniform(0.0, 5.0)
            queue_size = random.randint(0, 100)
            processed_today = random.randint(10000, 50000)

            self.active_handlers_label.setText(f"Active Handlers: {active_handlers}")
            self.total_throughput_label.setText(f"Total Throughput: {throughput} msg/s")
            self.avg_latency_label.setText(f"Avg Latency: {latency} ms")
            self.error_rate_label.setText(f"Error Rate: {error_rate:.1f}%")
            self.queue_size_label.setText(f"Queue Size: {queue_size}")
            self.processed_today_label.setText(f"Processed Today: {processed_today:,}")

            # Update progress bars (normalized)
            self.throughput_progress.setValue(min(100, throughput // 5))
            self.latency_progress.setValue(min(100, latency // 2))

            # Update message type counters
            for msg_type, widgets in self.message_counters.items():
                count = random.randint(10, 200)
                widgets["count"].setText(f"{count} / min")
                widgets["progress"].setValue(min(100, count // 2))

        except Exception as e:
            logger.error(f"Error updating handler metrics: {e}")


class HandlerTree(QWidget):
    """Tree view of handler systems and their status"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_tree)
        self.refresh_timer.start(3000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ["Handler", "Status", "Messages/s", "Latency", "Error Rate", "Last Check"]
        )
        self.tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        layout.addWidget(self.tree)
        self.refresh_tree()

    def refresh_tree(self):
        """Refresh the handler tree"""
        self.tree.clear()

        # Handler categories
        categories = {
            "Core Handlers": [
                ("UserInteractionHandler", "Active", "45 msg/s", "12 ms", "0.1%"),
                ("AgentMessageHandler", "Active", "123 msg/s", "8 ms", "0.3%"),
                ("SystemEventHandler", "Active", "67 msg/s", "15 ms", "0.0%"),
                ("ErrorHandler", "Active", "3 msg/s", "25 ms", "0.0%"),
            ],
            "Processing Handlers": [
                ("NLPProcessingHandler", "Active", "34 msg/s", "45 ms", "0.5%"),
                ("DataTransformHandler", "Active", "89 msg/s", "23 ms", "0.2%"),
                ("ValidationHandler", "Active", "156 msg/s", "7 ms", "1.2%"),
                ("CacheHandler", "Active", "234 msg/s", "3 ms", "0.1%"),
            ],
            "Integration Handlers": [
                ("DatabaseHandler", "Warning", "67 msg/s", "89 ms", "2.1%"),
                ("APIHandler", "Active", "45 msg/s", "34 ms", "0.8%"),
                ("FileSystemHandler", "Active", "23 msg/s", "12 ms", "0.0%"),
                ("NetworkHandler", "Active", "78 msg/s", "56 ms", "1.5%"),
            ],
            "Monitoring Handlers": [
                ("MetricsHandler", "Active", "12 msg/s", "5 ms", "0.0%"),
                ("LoggingHandler", "Active", "345 msg/s", "2 ms", "0.0%"),
                ("AlertHandler", "Active", "8 msg/s", "18 ms", "0.0%"),
                ("HealthCheckHandler", "Active", "15 msg/s", "10 ms", "0.1%"),
            ],
        }

        for category, handlers in categories.items():
            parent = QTreeWidgetItem([category, "", "", "", "", ""])
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            parent.setExpanded(True)

            for name, status, throughput, latency, error_rate in handlers:
                child = QTreeWidgetItem(
                    [
                        name,
                        status,
                        throughput,
                        latency,
                        error_rate,
                        datetime.now().strftime("%H:%M:%S"),
                    ]
                )

                # Color code by status
                if status == "Active":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["success"]))
                elif status == "Warning":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["warning"]))
                elif status == "Error":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["error"]))
                else:
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["text_secondary"]))

                parent.addChild(child)

            self.tree.addTopLevelItem(parent)


class MessageFlowLog(QWidget):
    """Log of message flow and handler processing events"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.add_sample_messages()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = VoxSigilWidgetFactory.create_checkbox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)

        self.show_debug_checkbox = VoxSigilWidgetFactory.create_checkbox("Show Debug")
        self.show_debug_checkbox.setChecked(False)

        clear_btn = VoxSigilWidgetFactory.create_button("Clear Log", "default")
        clear_btn.clicked.connect(self.clear_log)

        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.show_debug_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.log_display.setReadOnly(True)

        layout.addWidget(self.log_display)

    def add_sample_messages(self):
        """Add sample message flow events"""
        messages = [
            "[RECEIVED] UserInteractionHandler: New user message (ID: msg_12345)",
            "[PROCESSING] NLPProcessingHandler: Tokenizing and analyzing message",
            "[CACHE_HIT] CacheHandler: Retrieved similar query from cache (45ms saved)",
            "[FORWARDED] AgentMessageHandler: Routing to VoxAgent for response",
            "[COMPLETED] UserInteractionHandler: Response sent (total: 127ms)",
            "[RECEIVED] SystemEventHandler: Training pipeline status update",
            "[VALIDATION] ValidationHandler: Checking message format and schema",
            "[DATABASE] DatabaseHandler: Storing event data (latency: 23ms)",
            "[ALERT] ErrorHandler: Processing rate exceeded threshold (>500 msg/s)",
            "[METRICS] MetricsHandler: Updated throughput statistics",
        ]

        for message in messages:
            self.add_log_entry(message)

    def add_log_entry(self, message: str):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_message = f"[{timestamp}] {message}"

        # Color code by operation type
        if "[ERROR]" in message or "error" in message.lower():
            color = VoxSigilStyles.COLORS["error"]
        elif "[ALERT]" in message or "threshold" in message.lower():
            color = VoxSigilStyles.COLORS["warning"]
        elif "[COMPLETED]" in message or "completed" in message.lower():
            color = VoxSigilStyles.COLORS["success"]
        elif "[CACHE_HIT]" in message:
            color = VoxSigilStyles.COLORS["accent_mint"]
        elif "[PROCESSING]" in message:
            color = VoxSigilStyles.COLORS["accent_cyan"]
        elif "[FORWARDED]" in message:
            color = VoxSigilStyles.COLORS["accent_gold"]
        elif "[METRICS]" in message:
            color = VoxSigilStyles.COLORS["accent_purple"]
        else:
            color = VoxSigilStyles.COLORS["text_primary"]

        self.log_display.append(f'<span style="color: {color}">{formatted_message}</span>')

        if self.auto_scroll_checkbox.isChecked():
            self.log_display.moveCursor(self.log_display.textCursor().End)

    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()


class HandlerQueues(QWidget):
    """Display of handler queues and message backlogs"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_queues)
        self.refresh_timer.start(2000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Queue status tree
        self.queue_tree = QTreeWidget()
        self.queue_tree.setHeaderLabels(
            ["Queue", "Size", "Max Size", "Utilization", "Oldest Message"]
        )
        self.queue_tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        layout.addWidget(self.queue_tree)
        self.refresh_queues()

    def refresh_queues(self):
        """Refresh queue information"""
        self.queue_tree.clear()

        import random

        # Queue categories
        queue_categories = {
            "Input Queues": [
                ("user_interaction_queue", random.randint(0, 50), 1000),
                ("system_event_queue", random.randint(0, 30), 500),
                ("agent_message_queue", random.randint(0, 100), 2000),
            ],
            "Processing Queues": [
                ("nlp_processing_queue", random.randint(0, 25), 200),
                ("validation_queue", random.randint(0, 15), 100),
                ("transformation_queue", random.randint(0, 40), 300),
            ],
            "Output Queues": [
                ("database_write_queue", random.randint(0, 20), 150),
                ("api_response_queue", random.randint(0, 10), 50),
                ("notification_queue", random.randint(0, 35), 200),
            ],
        }

        for category, queues in queue_categories.items():
            parent = QTreeWidgetItem([category, "", "", "", ""])
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            parent.setExpanded(True)

            for queue_name, size, max_size in queues:
                utilization = (size / max_size) * 100
                oldest_time = f"{random.randint(1, 30)}s ago"

                child = QTreeWidgetItem(
                    [
                        queue_name.replace("_", " ").title(),
                        str(size),
                        str(max_size),
                        f"{utilization:.1f}%",
                        oldest_time,
                    ]
                )

                # Color code by utilization
                if utilization > 80:
                    child.setForeground(3, QColor(VoxSigilStyles.COLORS["error"]))
                elif utilization > 60:
                    child.setForeground(3, QColor(VoxSigilStyles.COLORS["warning"]))
                else:
                    child.setForeground(3, QColor(VoxSigilStyles.COLORS["success"]))

                parent.addChild(child)

            self.queue_tree.addTopLevelItem(parent)


class HandlerSystemsTab(QWidget):
    """Main Handler Systems monitoring tab with streaming support"""

    # Signals for streaming data
    handler_update = pyqtSignal(dict)
    message_flow = pyqtSignal(str)
    queue_update = pyqtSignal(dict)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("🔌 Handler Systems Monitor", "title")
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(VoxSigilStyles.get_splitter_stylesheet())

        # Left panel - Performance and tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Handler performance
        self.performance_widget = HandlerPerformanceWidget()
        left_layout.addWidget(self.performance_widget)

        # Handler tree
        self.tree_widget = HandlerTree()
        left_layout.addWidget(self.tree_widget)

        splitter.addWidget(left_panel)

        # Right panel - Message flow and queues
        right_panel = QTabWidget()
        right_panel.setStyleSheet(VoxSigilStyles.get_tab_widget_stylesheet())

        # Message flow tab
        self.message_log = MessageFlowLog()
        right_panel.addTab(self.message_log, "📨 Message Flow")

        # Queue status tab
        self.queue_widget = HandlerQueues()
        right_panel.addTab(self.queue_widget, "📋 Queues")

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([500, 400])

        layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()

        self.connection_status = VoxSigilWidgetFactory.create_label(
            "🔌 Connected to Handler Monitor", "info"
        )
        self.last_update = VoxSigilWidgetFactory.create_label("Last update: --:--:--", "info")

        status_layout.addWidget(self.connection_status)
        status_layout.addStretch()
        status_layout.addWidget(self.last_update)

        layout.addLayout(status_layout)

        # Apply dark theme
        self.setStyleSheet(VoxSigilStyles.get_base_stylesheet())

    def setup_streaming(self):
        """Setup event bus streaming subscriptions"""
        if self.event_bus:
            # Subscribe to handler-related events
            self.event_bus.subscribe("handler.performance", self.on_handler_performance)
            self.event_bus.subscribe("handler.message", self.on_message_flow)
            self.event_bus.subscribe("handler.queue", self.on_queue_update)

            # Connect internal signals
            self.handler_update.connect(self.update_handler_display)
            self.message_flow.connect(self.message_log.add_log_entry)
            self.queue_update.connect(self.update_queue_display)

            logger.info("Handler Systems tab connected to streaming events")
            self.connection_status.setText("🔌 Connected to Handler Monitor")
        else:
            logger.warning("Handler Systems tab: No event bus available")
            self.connection_status.setText("⚠️ No Event Bus Connection")

    def on_handler_performance(self, data):
        """Handle handler performance updates"""
        try:
            self.handler_update.emit(data)
            self.last_update.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error processing handler performance: {e}")

    def on_message_flow(self, message):
        """Handle message flow events"""
        try:
            if isinstance(message, dict):
                msg = message.get("message", str(message))
            else:
                msg = str(message)
            self.message_flow.emit(msg)
        except Exception as e:
            logger.error(f"Error processing message flow: {e}")

    def on_queue_update(self, data):
        """Handle queue update events"""
        try:
            self.queue_update.emit(data)
        except Exception as e:
            logger.error(f"Error processing queue update: {e}")

    def update_handler_display(self, data):
        """Update handler display with new data"""
        try:
            # Update would be handled by the performance widget
            pass
        except Exception as e:
            logger.error(f"Error updating handler display: {e}")

    def update_queue_display(self, data):
        """Update queue display with new data"""
        try:
            # Update would be handled by the queue widget
            pass
        except Exception as e:
            logger.error(f"Error updating queue display: {e}")


def create_handler_systems_tab(event_bus=None) -> HandlerSystemsTab:
    """Factory function to create Handler Systems tab"""
    return HandlerSystemsTab(event_bus=event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    VoxSigilStyles.apply_dark_theme(app)

    tab = HandlerSystemsTab()
    tab.show()

    sys.exit(app.exec_())
