#!/usr/bin/env python3
"""
Real-time Logs Tab - Live Log Streaming and Analysis
Provides live streaming of system logs with filtering, search, and analysis capabilities.
"""

import logging
from datetime import datetime

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class LogStreamWidget(QWidget):
    """Main log streaming display widget"""

    def __init__(self):
        super().__init__()
        self.log_buffer = []
        self.max_buffer_size = 1000
        self.setup_ui()
        self.setup_sample_stream()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        # Auto-scroll checkbox
        self.auto_scroll_checkbox = VoxSigilWidgetFactory.create_checkbox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)

        # Log level filter
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_filter.setStyleSheet(VoxSigilStyles.get_combo_box_stylesheet())

        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search logs...")
        self.search_box.setStyleSheet(VoxSigilStyles.get_line_edit_stylesheet())

        # Control buttons
        pause_btn = VoxSigilWidgetFactory.create_button("⏸️ Pause", "default")
        clear_btn = VoxSigilWidgetFactory.create_button("🗑️ Clear", "default")
        export_btn = VoxSigilWidgetFactory.create_button("💾 Export", "default")

        pause_btn.clicked.connect(self.toggle_pause)
        clear_btn.clicked.connect(self.clear_logs)
        export_btn.clicked.connect(self.export_logs)

        controls_layout.addWidget(VoxSigilWidgetFactory.create_label("Level:", "info"))
        controls_layout.addWidget(self.level_filter)
        controls_layout.addWidget(VoxSigilWidgetFactory.create_label("Search:", "info"))
        controls_layout.addWidget(self.search_box)
        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(pause_btn)
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(export_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))

        layout.addWidget(self.log_display)

        # Status bar
        status_layout = QHBoxLayout()

        self.log_count_label = VoxSigilWidgetFactory.create_label("Logs: 0", "info")
        self.stream_status_label = VoxSigilWidgetFactory.create_label("🟢 Streaming", "info")
        self.last_log_time = VoxSigilWidgetFactory.create_label("Last: --:--:--", "info")

        status_layout.addWidget(self.log_count_label)
        status_layout.addWidget(self.stream_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.last_log_time)

        layout.addLayout(status_layout)

        self.paused = False

    def setup_sample_stream(self):
        """Setup sample log streaming"""
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.add_sample_log)
        self.stream_timer.start(1500)  # Add log every 1.5 seconds

    def add_sample_log(self):
        """Add a sample log entry"""
        if self.paused:
            return

        import random

        # Sample log messages
        components = ["API", "AUTH", "ML", "DATA", "CACHE", "DB", "QUEUE", "SCHEDULER", "MONITOR"]
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        component = random.choice(components)
        level = random.choice(levels)

        messages = {
            "DEBUG": [
                f"[{component}] Processing request with ID: req_{random.randint(10000, 99999)}",
                f"[{component}] Cache lookup for key: {random.choice(['user_123', 'session_456', 'model_789'])}",
                f"[{component}] Configuration loaded: {random.randint(45, 120)} settings",
            ],
            "INFO": [
                f"[{component}] Service started successfully on port {random.randint(8000, 9999)}",
                f"[{component}] Processing batch of {random.randint(10, 500)} items",
                f"[{component}] Connection established with upstream service",
                f"[{component}] Health check passed - all systems operational",
            ],
            "WARNING": [
                f"[{component}] High memory usage detected: {random.randint(80, 95)}%",
                f"[{component}] Slow query detected: {random.randint(500, 2000)}ms",
                f"[{component}] Rate limit threshold approaching: {random.randint(80, 99)}% of limit",
                f"[{component}] Connection pool {random.randint(85, 95)}% utilized",
            ],
            "ERROR": [
                f"[{component}] Failed to connect to database after 3 retries",
                f"[{component}] Invalid request format: missing required field 'user_id'",
                f"[{component}] External API returned HTTP {random.choice([404, 500, 503])}",
                f"[{component}] Authentication failed for user: invalid credentials",
            ],
        }

        message = random.choice(messages[level])
        self.add_log_entry(level, message)

    def add_log_entry(self, level: str, message: str):
        """Add a log entry to the display"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level:>8}] {message}"

        # Add to buffer
        self.log_buffer.append((timestamp, level, message))
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)

        # Apply filters
        current_filter = self.level_filter.currentText()
        search_text = self.search_box.text().lower()

        if current_filter != "ALL" and level != current_filter:
            return

        if search_text and search_text not in message.lower():
            return

        # Color code by log level
        if level == "ERROR":
            color = VoxSigilStyles.COLORS["error"]
        elif level == "WARNING":
            color = VoxSigilStyles.COLORS["warning"]
        elif level == "INFO":
            color = VoxSigilStyles.COLORS["accent_cyan"]
        elif level == "DEBUG":
            color = VoxSigilStyles.COLORS["text_secondary"]
        else:
            color = VoxSigilStyles.COLORS["text_primary"]

        formatted_entry = f'<span style="color: {color}">{log_entry}</span>'
        self.log_display.append(formatted_entry)

        # Auto-scroll
        if self.auto_scroll_checkbox.isChecked():
            self.log_display.moveCursor(self.log_display.textCursor().End)

        # Update status
        self.log_count_label.setText(f"Logs: {len(self.log_buffer)}")
        self.last_log_time.setText(f"Last: {datetime.now().strftime('%H:%M:%S')}")

    def toggle_pause(self):
        """Toggle log streaming pause"""
        self.paused = not self.paused
        if self.paused:
            self.stream_status_label.setText("⏸️ Paused")
        else:
            self.stream_status_label.setText("🟢 Streaming")

    def clear_logs(self):
        """Clear the log display"""
        self.log_display.clear()
        self.log_buffer.clear()
        self.log_count_label.setText("Logs: 0")

    def export_logs(self):
        """Export logs to file"""
        # Placeholder - would implement actual export
        logger.info("Log export requested")


class LogAnalyticsWidget(QWidget):
    """Widget for log analytics and statistics"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_analytics)
        self.update_timer.start(5000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Analytics overview
        analytics_group = QGroupBox("Log Analytics (Last Hour)")
        analytics_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        analytics_layout = QGridLayout(analytics_group)

        # Log level counters
        self.debug_count_label = VoxSigilWidgetFactory.create_label("DEBUG: --", "info")
        self.info_count_label = VoxSigilWidgetFactory.create_label("INFO: --", "info")
        self.warning_count_label = VoxSigilWidgetFactory.create_label("WARNING: --", "info")
        self.error_count_label = VoxSigilWidgetFactory.create_label("ERROR: --", "info")

        # Rate metrics
        self.logs_per_minute_label = VoxSigilWidgetFactory.create_label("Logs/min: --", "info")
        self.error_rate_label = VoxSigilWidgetFactory.create_label("Error Rate: --%", "info")

        analytics_layout.addWidget(self.debug_count_label, 0, 0)
        analytics_layout.addWidget(self.info_count_label, 0, 1)
        analytics_layout.addWidget(self.warning_count_label, 1, 0)
        analytics_layout.addWidget(self.error_count_label, 1, 1)
        analytics_layout.addWidget(self.logs_per_minute_label, 2, 0)
        analytics_layout.addWidget(self.error_rate_label, 2, 1)

        layout.addWidget(analytics_group)

        # Top errors
        errors_group = QGroupBox("Top Error Patterns")
        errors_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        errors_layout = QVBoxLayout(errors_group)

        self.errors_tree = QTreeWidget()
        self.errors_tree.setHeaderLabels(["Error Pattern", "Count", "Last Seen", "Component"])
        self.errors_tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        errors_layout.addWidget(self.errors_tree)
        layout.addWidget(errors_group)

        # Component activity
        components_group = QGroupBox("Component Activity")
        components_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        components_layout = QVBoxLayout(components_group)

        self.components_tree = QTreeWidget()
        self.components_tree.setHeaderLabels(
            ["Component", "Log Count", "Error Count", "Last Activity"]
        )
        self.components_tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        components_layout.addWidget(self.components_tree)
        layout.addWidget(components_group)

        self.refresh_analytics()

    def refresh_analytics(self):
        """Refresh analytics data"""
        import random

        # Update counters
        debug_count = random.randint(500, 2000)
        info_count = random.randint(300, 1500)
        warning_count = random.randint(20, 200)
        error_count = random.randint(5, 50)

        self.debug_count_label.setText(f"DEBUG: {debug_count}")
        self.info_count_label.setText(f"INFO: {info_count}")
        self.warning_count_label.setText(f"WARNING: {warning_count}")
        self.error_count_label.setText(f"ERROR: {error_count}")

        total_logs = debug_count + info_count + warning_count + error_count
        logs_per_minute = random.randint(50, 300)
        error_rate = (error_count / total_logs) * 100

        self.logs_per_minute_label.setText(f"Logs/min: {logs_per_minute}")
        self.error_rate_label.setText(f"Error Rate: {error_rate:.1f}%")

        # Update error patterns
        self.errors_tree.clear()

        error_patterns = [
            ("Database connection timeout", random.randint(5, 25), "2m ago", "DB"),
            ("Authentication failed", random.randint(8, 40), "5m ago", "AUTH"),
            ("External API unreachable", random.randint(3, 15), "1m ago", "API"),
            ("Memory allocation failed", random.randint(1, 8), "8m ago", "ML"),
        ]

        for pattern, count, last_seen, component in error_patterns:
            item = QTreeWidgetItem([pattern, str(count), last_seen, component])
            item.setForeground(0, QColor(VoxSigilStyles.COLORS["error"]))
            self.errors_tree.addTopLevelItem(item)

        # Update component activity
        self.components_tree.clear()

        components = [
            ("API Gateway", random.randint(200, 800), random.randint(2, 15), "30s ago"),
            ("Authentication Service", random.randint(100, 400), random.randint(1, 8), "1m ago"),
            ("ML Inference", random.randint(150, 600), random.randint(3, 12), "45s ago"),
            ("Database Service", random.randint(300, 900), random.randint(5, 20), "15s ago"),
            ("Cache Service", random.randint(80, 300), random.randint(0, 5), "2m ago"),
        ]

        for component, log_count, error_count, last_activity in components:
            item = QTreeWidgetItem([component, str(log_count), str(error_count), last_activity])

            # Color code by error rate
            if error_count > 10:
                item.setForeground(2, QColor(VoxSigilStyles.COLORS["error"]))
            elif error_count > 5:
                item.setForeground(2, QColor(VoxSigilStyles.COLORS["warning"]))
            else:
                item.setForeground(2, QColor(VoxSigilStyles.COLORS["success"]))

            self.components_tree.addTopLevelItem(item)


class LogSearchWidget(QWidget):
    """Advanced log search and filtering widget"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Search controls
        search_group = QGroupBox("Advanced Search")
        search_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        search_layout = QGridLayout(search_group)

        # Search parameters
        self.search_text = QLineEdit()
        self.search_text.setPlaceholderText("Search pattern or regex...")
        self.search_text.setStyleSheet(VoxSigilStyles.get_line_edit_stylesheet())

        self.component_filter = QComboBox()
        self.component_filter.addItems(
            ["All Components", "API", "AUTH", "ML", "DATA", "CACHE", "DB"]
        )
        self.component_filter.setStyleSheet(VoxSigilStyles.get_combo_box_stylesheet())

        self.time_range = QComboBox()
        self.time_range.addItems(["Last 15 minutes", "Last hour", "Last 6 hours", "Last 24 hours"])
        self.time_range.setStyleSheet(VoxSigilStyles.get_combo_box_stylesheet())

        search_btn = VoxSigilWidgetFactory.create_button("🔍 Search", "primary")

        search_layout.addWidget(VoxSigilWidgetFactory.create_label("Pattern:", "info"), 0, 0)
        search_layout.addWidget(self.search_text, 0, 1)
        search_layout.addWidget(VoxSigilWidgetFactory.create_label("Component:", "info"), 1, 0)
        search_layout.addWidget(self.component_filter, 1, 1)
        search_layout.addWidget(VoxSigilWidgetFactory.create_label("Time Range:", "info"), 2, 0)
        search_layout.addWidget(self.time_range, 2, 1)
        search_layout.addWidget(search_btn, 3, 1)

        layout.addWidget(search_group)

        # Search results
        results_group = QGroupBox("Search Results")
        results_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        results_layout = QVBoxLayout(results_group)

        self.results_display = QTextEdit()
        self.results_display.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.results_display.setReadOnly(True)
        self.results_display.setFont(QFont("Consolas", 9))

        results_layout.addWidget(self.results_display)
        layout.addWidget(results_group)


class RealtimeLogsTab(QWidget):
    """Main Real-time Logs monitoring tab with streaming support"""

    # Signals for streaming data
    log_received = pyqtSignal(str, str)  # level, message
    analytics_update = pyqtSignal(dict)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("📜 Real-time Logs Monitor", "title")
        layout.addWidget(title)

        # Main content with tabs
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(VoxSigilStyles.get_tab_widget_stylesheet())

        # Log stream tab
        self.stream_widget = LogStreamWidget()
        tab_widget.addTab(self.stream_widget, "📡 Live Stream")

        # Analytics tab
        self.analytics_widget = LogAnalyticsWidget()
        tab_widget.addTab(self.analytics_widget, "📊 Analytics")

        # Search tab
        self.search_widget = LogSearchWidget()
        tab_widget.addTab(self.search_widget, "🔍 Search")

        layout.addWidget(tab_widget)

        # Status bar
        status_layout = QHBoxLayout()

        self.connection_status = VoxSigilWidgetFactory.create_label(
            "🔌 Connected to Log Stream", "info"
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
            # Subscribe to log-related events
            self.event_bus.subscribe("logs.stream", self.on_log_stream)
            self.event_bus.subscribe("logs.analytics", self.on_analytics_update)

            # Connect internal signals
            self.log_received.connect(self.stream_widget.add_log_entry)
            self.analytics_update.connect(self.update_analytics_display)

            logger.info("Real-time Logs tab connected to streaming events")
            self.connection_status.setText("🔌 Connected to Log Stream")
        else:
            logger.warning("Real-time Logs tab: No event bus available")
            self.connection_status.setText("⚠️ No Event Bus Connection")

    def on_log_stream(self, log_data):
        """Handle incoming log stream data"""
        try:
            if isinstance(log_data, dict):
                level = log_data.get("level", "INFO")
                message = log_data.get("message", str(log_data))
            else:
                level = "INFO"
                message = str(log_data)

            self.log_received.emit(level, message)
            self.last_update.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error processing log stream: {e}")

    def on_analytics_update(self, data):
        """Handle analytics update events"""
        try:
            self.analytics_update.emit(data)
        except Exception as e:
            logger.error(f"Error processing analytics update: {e}")

    def update_analytics_display(self, data):
        """Update analytics display with new data"""
        try:
            # Update would be handled by the analytics widget
            pass
        except Exception as e:
            logger.error(f"Error updating analytics display: {e}")


def create_realtime_logs_tab(event_bus=None) -> RealtimeLogsTab:
    """Factory function to create Real-time Logs tab"""
    return RealtimeLogsTab(event_bus=event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    VoxSigilStyles.apply_dark_theme(app)

    tab = RealtimeLogsTab()
    tab.show()

    sys.exit(app.exec_())
