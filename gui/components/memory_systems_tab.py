#!/usr/bin/env python3
"""
Memory Systems Tab - Real-time Memory State Monitoring
Provides live monitoring of VoxSigil memory systems, caches, and state management.
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
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class MemoryStatsWidget(QWidget):
    """Widget displaying memory statistics and metrics"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_stats)
        self.update_timer.start(2000)  # Update every 2 seconds

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Memory usage overview
        stats_group = QGroupBox("Memory Usage Overview")
        stats_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        stats_layout = QGridLayout(stats_group)

        # Memory metrics
        self.total_memory_label = VoxSigilWidgetFactory.create_label("Total Memory: --", "info")
        self.available_memory_label = VoxSigilWidgetFactory.create_label("Available: --", "info")
        self.used_memory_label = VoxSigilWidgetFactory.create_label("Used: --", "info")
        self.memory_percent_label = VoxSigilWidgetFactory.create_label("Usage: --%", "info")

        self.memory_progress = VoxSigilWidgetFactory.create_progress_bar()
        self.memory_progress.setMaximum(100)

        stats_layout.addWidget(self.total_memory_label, 0, 0)
        stats_layout.addWidget(self.available_memory_label, 0, 1)
        stats_layout.addWidget(self.used_memory_label, 1, 0)
        stats_layout.addWidget(self.memory_percent_label, 1, 1)
        stats_layout.addWidget(self.memory_progress, 2, 0, 1, 2)

        layout.addWidget(stats_group)

        # Cache statistics
        cache_group = QGroupBox("Cache Systems")
        cache_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        cache_layout = QGridLayout(cache_group)

        self.cache_stats = {}
        cache_types = ["Vector Cache", "Model Cache", "Dataset Cache", "Query Cache"]

        for i, cache_type in enumerate(cache_types):
            label = VoxSigilWidgetFactory.create_label(f"{cache_type}:", "info")
            value_label = VoxSigilWidgetFactory.create_label("-- / --", "normal")
            progress = VoxSigilWidgetFactory.create_progress_bar()
            progress.setMaximum(100)

            self.cache_stats[cache_type] = {"label": value_label, "progress": progress}

            cache_layout.addWidget(label, i, 0)
            cache_layout.addWidget(value_label, i, 1)
            cache_layout.addWidget(progress, i, 2)

        layout.addWidget(cache_group)

    def refresh_stats(self):
        """Refresh memory statistics"""
        try:
            import psutil

            # Get system memory
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            used_gb = memory.used / (1024**3)
            percent = memory.percent

            self.total_memory_label.setText(f"Total Memory: {total_gb:.1f} GB")
            self.available_memory_label.setText(f"Available: {available_gb:.1f} GB")
            self.used_memory_label.setText(f"Used: {used_gb:.1f} GB")
            self.memory_percent_label.setText(f"Usage: {percent:.1f}%")
            self.memory_progress.setValue(int(percent))

            # Update cache stats (placeholder - would connect to actual cache systems)
            import random

            for cache_type, widgets in self.cache_stats.items():
                usage = random.randint(20, 80)
                widgets["progress"].setValue(usage)
                widgets["label"].setText(f"{usage}% / 100 MB")

        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")


class MemoryComponentTree(QWidget):
    """Tree view of memory components and their states"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_tree)
        self.refresh_timer.start(3000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Component", "State", "Memory Usage", "Last Updated"])
        self.tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        layout.addWidget(self.tree)
        self.refresh_tree()

    def refresh_tree(self):
        """Refresh the component tree"""
        self.tree.clear()

        # Memory subsystems
        subsystems = {
            "Core Memory": [
                ("Conversation Store", "Active", "45.2 MB"),
                ("Agent State Cache", "Active", "23.1 MB"),
                ("Task Memory", "Active", "12.7 MB"),
            ],
            "Vector Stores": [
                ("Document Embeddings", "Active", "156.8 MB"),
                ("Query Cache", "Active", "34.5 MB"),
                ("Similarity Index", "Active", "89.2 MB"),
            ],
            "Model Caches": [
                ("Transformer Cache", "Active", "512.1 MB"),
                ("Tokenizer Cache", "Active", "78.3 MB"),
                ("Inference Cache", "Active", "234.7 MB"),
            ],
            "Training Memory": [
                ("Gradient Cache", "Active", "289.4 MB"),
                ("Batch Memory", "Active", "123.6 MB"),
                ("Checkpoint Store", "Active", "678.9 MB"),
            ],
        }

        for subsystem, components in subsystems.items():
            parent = QTreeWidgetItem([subsystem, "", "", ""])
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            parent.setExpanded(True)

            for name, state, memory in components:
                child = QTreeWidgetItem([name, state, memory, datetime.now().strftime("%H:%M:%S")])

                # Color code by state
                if state == "Active":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["success"]))
                elif state == "Idle":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["warning"]))
                else:
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["error"]))

                parent.addChild(child)

            self.tree.addTopLevelItem(parent)


class MemoryEventsLog(QWidget):
    """Log of memory-related events and operations"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = VoxSigilWidgetFactory.create_checkbox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)

        clear_btn = VoxSigilWidgetFactory.create_button("Clear Log", "default")
        clear_btn.clicked.connect(self.clear_log)

        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.log_display.setReadOnly(True)

        layout.addWidget(self.log_display)

        # Add some sample events
        self.add_sample_events()

    def add_sample_events(self):
        """Add sample memory events"""
        events = [
            "[12:34:56] Memory allocation: Vector store expanded to 256MB",
            "[12:35:12] Cache miss: Loading model weights from disk",
            "[12:35:45] Garbage collection: Freed 45MB from conversation cache",
            "[12:36:01] Memory warning: Usage above 80% threshold",
            "[12:36:23] Cache hit: Retrieved cached embedding for query",
            "[12:36:47] Memory optimization: Compressed old conversation data",
        ]

        for event in events:
            self.add_log_entry(event)

    def add_log_entry(self, message: str):
        """Add a log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        self.log_display.append(formatted_message)

        if self.auto_scroll_checkbox.isChecked():
            self.log_display.moveCursor(self.log_display.textCursor().End)

    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()


class MemorySystemsTab(QWidget):
    """Main Memory Systems monitoring tab with streaming support"""

    # Signals for streaming data
    memory_update = pyqtSignal(dict)
    cache_update = pyqtSignal(dict)
    event_received = pyqtSignal(str)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("🧠 Memory Systems Monitor", "title")
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(VoxSigilStyles.get_splitter_stylesheet())

        # Left panel - Statistics and tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Memory stats
        self.stats_widget = MemoryStatsWidget()
        left_layout.addWidget(self.stats_widget)

        # Component tree
        self.tree_widget = MemoryComponentTree()
        left_layout.addWidget(self.tree_widget)

        splitter.addWidget(left_panel)

        # Right panel - Events log
        self.events_log = MemoryEventsLog()
        splitter.addWidget(self.events_log)

        # Set splitter proportions
        splitter.setSizes([400, 300])

        layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()

        self.connection_status = VoxSigilWidgetFactory.create_label(
            "🔌 Connected to Memory Monitor", "info"
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
            # Subscribe to memory-related events
            self.event_bus.subscribe("memory.stats", self.on_memory_stats)
            self.event_bus.subscribe("memory.cache", self.on_cache_update)
            self.event_bus.subscribe("memory.event", self.on_memory_event)

            # Connect internal signals
            self.memory_update.connect(self.update_memory_display)
            self.cache_update.connect(self.update_cache_display)
            self.event_received.connect(self.events_log.add_log_entry)

            logger.info("Memory Systems tab connected to streaming events")
            self.connection_status.setText("🔌 Connected to Memory Monitor")
        else:
            logger.warning("Memory Systems tab: No event bus available")
            self.connection_status.setText("⚠️ No Event Bus Connection")

    def on_memory_stats(self, data):
        """Handle memory statistics updates"""
        try:
            self.memory_update.emit(data)
            self.last_update.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error processing memory stats: {e}")

    def on_cache_update(self, data):
        """Handle cache update events"""
        try:
            self.cache_update.emit(data)
        except Exception as e:
            logger.error(f"Error processing cache update: {e}")

    def on_memory_event(self, event):
        """Handle memory event notifications"""
        try:
            if isinstance(event, dict):
                message = event.get("message", str(event))
            else:
                message = str(event)
            self.event_received.emit(message)
        except Exception as e:
            logger.error(f"Error processing memory event: {e}")

    def update_memory_display(self, data):
        """Update memory display with new data"""
        try:
            # Update would be handled by the stats widget
            pass
        except Exception as e:
            logger.error(f"Error updating memory display: {e}")

    def update_cache_display(self, data):
        """Update cache display with new data"""
        try:
            # Update would be handled by the stats widget
            pass
        except Exception as e:
            logger.error(f"Error updating cache display: {e}")


def create_memory_systems_tab(event_bus=None) -> MemorySystemsTab:
    """Factory function to create Memory Systems tab"""
    return MemorySystemsTab(event_bus=event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    VoxSigilStyles.apply_dark_theme(app)

    tab = MemorySystemsTab()
    tab.show()

    sys.exit(app.exec_())
