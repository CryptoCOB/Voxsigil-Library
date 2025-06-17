#!/usr/bin/env python3
"""
Notification Center Tab - System Alerts and Notifications
Centralized management and display of all system notifications and alerts.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class NotificationItem(QListWidgetItem):
    """Custom notification list item"""

    def __init__(self, notification_data: Dict[str, Any]):
        super().__init__()
        self.notification_data = notification_data
        self.setup_item()

    def setup_item(self):
        """Setup the notification item display"""
        level = self.notification_data.get("level", "info")
        message = self.notification_data.get("message", "No message")
        timestamp = self.notification_data.get("timestamp", datetime.now())

        # Icon mapping
        icon_map = {
            "critical": "🔴",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅",
            "debug": "🔧",
        }

        icon = icon_map.get(level, "ℹ️")

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

        time_str = timestamp.strftime("%H:%M:%S")

        self.setText(f"{icon} [{time_str}] {message}")  # Set color based on level
        color_map = {
            "critical": VoxSigilStyles.COLORS["error"],
            "error": VoxSigilStyles.COLORS["error"],
            "warning": VoxSigilStyles.COLORS["warning"],
            "info": VoxSigilStyles.COLORS["info"],
            "success": VoxSigilStyles.COLORS["success"],
            "debug": VoxSigilStyles.COLORS["text_muted"],
        }

        if level in color_map:
            self.setForeground(QColor(color_map[level]))


class NotificationListWidget(QListWidget):
    """Widget for displaying notifications"""

    notification_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_widget()
        self.add_sample_notifications()

    def init_widget(self):
        """Initialize the notification list"""
        self.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())
        self.itemClicked.connect(self.on_item_clicked)

    def add_sample_notifications(self):
        """Add sample notifications"""
        sample_notifications = [
            {
                "level": "success",
                "message": "System startup completed successfully",
                "timestamp": datetime.now(),
                "source": "System",
            },
            {
                "level": "info",
                "message": "New experiment started: GridFormer_v2_optimized",
                "timestamp": datetime.now(),
                "source": "Experiment Tracker",
            },
            {
                "level": "warning",
                "message": "High GPU utilization detected (87%)",
                "timestamp": datetime.now(),
                "source": "Performance Monitor",
            },
            {
                "level": "error",
                "message": "Failed to load configuration file: config.yaml",
                "timestamp": datetime.now(),
                "source": "Config Manager",
            },
            {
                "level": "critical",
                "message": "Memory usage critical: 95% utilized",
                "timestamp": datetime.now(),
                "source": "System Monitor",
            },
        ]

        for notification in sample_notifications:
            self.add_notification(notification)

    def add_notification(self, notification_data: Dict[str, Any]):
        """Add a new notification"""
        item = NotificationItem(notification_data)
        self.insertItem(0, item)  # Insert at top

        # Limit to 100 notifications
        if self.count() > 100:
            self.takeItem(self.count() - 1)

    def on_item_clicked(self, item: NotificationItem):
        """Handle notification item click"""
        if hasattr(item, "notification_data"):
            self.notification_selected.emit(item.notification_data)


class NotificationDetailsWidget(QWidget):
    """Widget for displaying notification details"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        self.title_label = VoxSigilWidgetFactory.create_label("📋 Notification Details", "section")
        layout.addWidget(self.title_label)

        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)

        # Statistics
        stats_title = VoxSigilWidgetFactory.create_label("📊 Notification Statistics", "section")
        layout.addWidget(stats_title)

        # Stats grid
        stats_grid = QGridLayout()

        self.total_label = VoxSigilWidgetFactory.create_label("Total Today:", "info")
        self.total_value = VoxSigilWidgetFactory.create_label("0", "normal")
        stats_grid.addWidget(self.total_label, 0, 0)
        stats_grid.addWidget(self.total_value, 0, 1)

        self.critical_label = VoxSigilWidgetFactory.create_label("Critical:", "info")
        self.critical_value = VoxSigilWidgetFactory.create_label("0", "normal")
        stats_grid.addWidget(self.critical_label, 1, 0)
        stats_grid.addWidget(self.critical_value, 1, 1)

        self.warnings_label = VoxSigilWidgetFactory.create_label("Warnings:", "info")
        self.warnings_value = VoxSigilWidgetFactory.create_label("0", "normal")
        stats_grid.addWidget(self.warnings_label, 2, 0)
        stats_grid.addWidget(self.warnings_value, 2, 1)

        self.errors_label = VoxSigilWidgetFactory.create_label("Errors:", "info")
        self.errors_value = VoxSigilWidgetFactory.create_label("0", "normal")
        stats_grid.addWidget(self.errors_label, 3, 0)
        stats_grid.addWidget(self.errors_value, 3, 1)

        layout.addLayout(stats_grid)

        # Initialize stats
        self.update_stats()

    def show_notification_details(self, notification_data: Dict[str, Any]):
        """Display details for a selected notification"""
        level = notification_data.get("level", "info")
        message = notification_data.get("message", "No message")
        timestamp = notification_data.get("timestamp", datetime.now())
        source = notification_data.get("source", "Unknown")

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

        details = f"""
<h3>Notification Details</h3>
<p><strong>Level:</strong> {level.upper()}</p>
<p><strong>Message:</strong> {message}</p>
<p><strong>Source:</strong> {source}</p>
<p><strong>Timestamp:</strong> {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><strong>Additional Info:</strong></p>
<ul>
"""

        # Add any additional data
        for key, value in notification_data.items():
            if key not in ["level", "message", "timestamp", "source"]:
                details += f"<li><strong>{key}:</strong> {value}</li>"

        details += "</ul>"

        self.details_text.setHtml(details)

    def update_stats(self):
        """Update notification statistics"""
        # This would be updated with real data
        self.total_value.setText("47")
        self.critical_value.setText("2")
        self.warnings_value.setText("8")
        self.errors_value.setText("5")


class NotificationCenterTab(QWidget):
    """Main notification center tab"""

    # Signals
    notification_received = pyqtSignal(dict)
    notification_acknowledged = pyqtSignal(str)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.notification_count = 0
        self.init_ui()
        self.setup_streaming()
        self.setup_timers()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("🔔 Notification Center", "title")
        layout.addWidget(title)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Main splitter
        splitter = VoxSigilWidgetFactory.create_splitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Notifications list
        notifications_group = QGroupBox("Notifications")
        notifications_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        notifications_layout = QVBoxLayout(notifications_group)

        self.notifications_list = NotificationListWidget()
        self.notifications_list.notification_selected.connect(self.on_notification_selected)
        notifications_layout.addWidget(self.notifications_list)

        # Right panel - Details and stats
        details_group = QGroupBox("Details & Statistics")
        details_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        details_layout = QVBoxLayout(details_group)

        self.details_widget = NotificationDetailsWidget()
        details_layout.addWidget(self.details_widget)

        # Add to splitter
        splitter.addWidget(notifications_group)
        splitter.addWidget(details_group)
        splitter.setSizes([500, 400])

        # Status bar
        self.status_label = VoxSigilWidgetFactory.create_label("📊 5 notifications today", "info")
        layout.addWidget(self.status_label)

    def create_toolbar(self):
        """Create notification toolbar"""
        toolbar = VoxSigilWidgetFactory.create_frame()
        layout = QHBoxLayout(toolbar)

        # Clear all button
        self.clear_btn = VoxSigilWidgetFactory.create_button("🗑️ Clear All", "danger")
        self.clear_btn.clicked.connect(self.clear_all_notifications)
        layout.addWidget(self.clear_btn)

        # Mark read button
        self.mark_read_btn = VoxSigilWidgetFactory.create_button("✅ Mark Read", "success")
        self.mark_read_btn.clicked.connect(self.mark_all_read)
        layout.addWidget(self.mark_read_btn)

        # Filter buttons
        self.critical_filter_btn = VoxSigilWidgetFactory.create_button("🔴 Critical", "default")
        self.critical_filter_btn.clicked.connect(lambda: self.filter_notifications("critical"))
        layout.addWidget(self.critical_filter_btn)

        self.warning_filter_btn = VoxSigilWidgetFactory.create_button("⚠️ Warnings", "default")
        self.warning_filter_btn.clicked.connect(lambda: self.filter_notifications("warning"))
        layout.addWidget(self.warning_filter_btn)

        self.all_filter_btn = VoxSigilWidgetFactory.create_button("📋 All", "primary")
        self.all_filter_btn.clicked.connect(lambda: self.filter_notifications("all"))
        layout.addWidget(self.all_filter_btn)

        layout.addStretch()
        return toolbar

    def setup_streaming(self):
        """Setup event bus streaming"""
        if self.event_bus:
            # Subscribe to all notification topics
            notification_topics = [
                "system.alert",
                "security.alert",
                "performance.alert",
                "experiment.alert",
                "config.alert",
                "error.alert",
            ]

            for topic in notification_topics:
                self.event_bus.subscribe(topic, self.on_notification_received)

    def setup_timers(self):
        """Setup timers for demo notifications"""
        # Demo notification timer
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.generate_demo_notification)
        self.demo_timer.start(15000)  # 15 second intervals

    def on_notification_received(self, notification_data: Dict[str, Any]):
        """Handle incoming notifications"""
        try:
            self.notifications_list.add_notification(notification_data)
            self.notification_count += 1
            self.update_status()
            self.notification_received.emit(notification_data)

        except Exception as e:
            logger.error(f"Error processing notification: {e}")

    def on_notification_selected(self, notification_data: Dict[str, Any]):
        """Handle notification selection"""
        self.details_widget.show_notification_details(notification_data)

    def clear_all_notifications(self):
        """Clear all notifications"""
        self.notifications_list.clear()
        self.notification_count = 0
        self.update_status()

    def mark_all_read(self):
        """Mark all notifications as read"""
        # This would update notification status
        self.status_label.setText("📊 All notifications marked as read")

    def filter_notifications(self, filter_type: str):
        """Filter notifications by type"""
        # This would implement filtering logic
        if filter_type == "all":
            self.status_label.setText("📊 Showing all notifications")
        else:
            self.status_label.setText(f"📊 Showing {filter_type} notifications")

    def update_status(self):
        """Update status information"""
        self.status_label.setText(f"📊 {self.notification_count} notifications today")
        self.details_widget.update_stats()

    def generate_demo_notification(self):
        """Generate demo notifications"""
        import random

        demo_notifications = [
            {
                "level": "info",
                "message": "Heartbeat check completed successfully",
                "source": "Heartbeat Monitor",
                "timestamp": datetime.now(),
            },
            {
                "level": "warning",
                "message": "Disk space running low on /data partition (85% full)",
                "source": "System Monitor",
                "timestamp": datetime.now(),
            },
            {
                "level": "success",
                "message": "Configuration backup completed",
                "source": "Config Manager",
                "timestamp": datetime.now(),
            },
            {
                "level": "error",
                "message": "Connection timeout to external API",
                "source": "API Gateway",
                "timestamp": datetime.now(),
            },
        ]

        if random.random() < 0.6:  # 60% chance
            notification = random.choice(demo_notifications)
            self.on_notification_received(notification)


# Backward compatibility
NotificationTab = NotificationCenterTab
