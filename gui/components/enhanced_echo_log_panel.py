"""
Enhanced Echo Log Panel with Development Mode Controls
Comprehensive logging interface with configurable dev mode options.
"""

import logging

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config
from gui.components.dev_mode_panel import DevModeControlPanel

logger = logging.getLogger("EnhancedEchoLogPanel")


class EnhancedEchoLogPanel(QWidget):
    """
    Enhanced Echo Log panel with comprehensive controls and dev mode support.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_dev_config()
        self.message_count = 0
        self.filtered_messages = []

        self._init_ui()
        self._setup_timers()
        self._connect_signals()
        self._apply_config()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Development Mode Panel
        self.dev_panel = DevModeControlPanel("echo_log", self)
        layout.addWidget(self.dev_panel)

        # Main content splitter
        main_splitter = QSplitter(Qt.Vertical)

        # Controls panel
        controls_widget = self._create_controls_panel()
        main_splitter.addWidget(controls_widget)

        # Log display
        log_widget = self._create_log_panel()
        main_splitter.addWidget(log_widget)

        main_splitter.setSizes([150, 400])
        layout.addWidget(main_splitter)

        # Status bar
        self.status_label = QLabel("Echo Log Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ccc;")
        layout.addWidget(self.status_label)

    def _create_controls_panel(self) -> QWidget:
        """Create the log controls panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Log Level Filter
        filter_group = QGroupBox("📋 Log Filters")
        filter_layout = QGridLayout(filter_group)

        # Log level filter
        filter_layout.addWidget(QLabel("Level:"), 0, 0)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.setCurrentText("INFO")
        filter_layout.addWidget(self.level_combo, 0, 1)

        # Source filter
        filter_layout.addWidget(QLabel("Source:"), 0, 2)
        self.source_combo = QComboBox()
        self.source_combo.addItems(
            ["ALL", "GUI", "TTS", "Training", "GridFormer", "Music", "Neural"]
        )
        filter_layout.addWidget(self.source_combo, 0, 3)

        # Search filter
        filter_layout.addWidget(QLabel("Search:"), 1, 0)
        self.search_combo = QComboBox()
        self.search_combo.setEditable(True)
        self.search_combo.addItems(["", "error", "warning", "completed", "started"])
        filter_layout.addWidget(self.search_combo, 1, 1, 1, 3)

        layout.addWidget(filter_group)

        # Display Options
        display_group = QGroupBox("🎨 Display Options")
        display_layout = QGridLayout(display_group)

        # Auto-scroll
        self.autoscroll_checkbox = QCheckBox("Auto Scroll")
        self.autoscroll_checkbox.setChecked(True)
        display_layout.addWidget(self.autoscroll_checkbox, 0, 0)

        # Timestamps
        self.timestamps_checkbox = QCheckBox("Show Timestamps")
        self.timestamps_checkbox.setChecked(True)
        display_layout.addWidget(self.timestamps_checkbox, 0, 1)

        # Line numbers
        self.linenums_checkbox = QCheckBox("Line Numbers")
        display_layout.addWidget(self.linenums_checkbox, 1, 0)

        # Word wrap
        self.wordwrap_checkbox = QCheckBox("Word Wrap")
        self.wordwrap_checkbox.setChecked(True)
        display_layout.addWidget(self.wordwrap_checkbox, 1, 1)

        layout.addWidget(display_group)

        # Actions
        actions_group = QGroupBox("⚡ Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self._clear_log)
        actions_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._save_log)
        actions_layout.addWidget(self.save_btn)

        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        actions_layout.addWidget(self.pause_btn)

        layout.addWidget(actions_group)

        # Dev options (shown in dev mode)
        self.dev_options_group = QGroupBox("🔧 Dev Options")
        dev_layout = QGridLayout(self.dev_options_group)

        # Max messages
        dev_layout.addWidget(QLabel("Max Messages:"), 0, 0)
        self.max_messages_spin = QSpinBox()
        self.max_messages_spin.setRange(100, 10000)
        self.max_messages_spin.setValue(1000)
        dev_layout.addWidget(self.max_messages_spin, 0, 1)

        # Update interval
        dev_layout.addWidget(QLabel("Update (ms):"), 1, 0)
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(100, 5000)
        self.update_interval_spin.setValue(500)
        dev_layout.addWidget(self.update_interval_spin, 1, 1)

        # Debug logging
        self.debug_logging_checkbox = QCheckBox("Enable Debug Logging")
        dev_layout.addWidget(self.debug_logging_checkbox, 2, 0, 1, 2)

        layout.addWidget(self.dev_options_group)

        return widget

    def _create_log_panel(self) -> QWidget:
        """Create the log display panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Main log display
        self.log_display = QTextEdit()
        self.log_display.setPlaceholderText("Echo log messages will appear here...")
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Stats panel
        stats_layout = QHBoxLayout()

        self.message_count_label = QLabel("Messages: 0")
        self.filtered_count_label = QLabel("Filtered: 0")
        self.rate_label = QLabel("Rate: 0 msg/s")

        stats_layout.addWidget(self.message_count_label)
        stats_layout.addWidget(self.filtered_count_label)
        stats_layout.addWidget(self.rate_label)
        stats_layout.addStretch()

        layout.addLayout(stats_layout)

        return widget

    def _setup_timers(self):
        """Setup update timers."""
        # Stats update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)  # Update every second

        # Message rate calculation
        self.last_message_count = 0
        self.paused = False

    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect dev panel signals
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_toggled)
        self.dev_panel.config_changed.connect(self._on_config_changed)

        # Connect filter signals
        self.level_combo.currentTextChanged.connect(self._apply_filters)
        self.source_combo.currentTextChanged.connect(self._apply_filters)
        self.search_combo.currentTextChanged.connect(self._apply_filters)

        # Connect display option signals
        self.autoscroll_checkbox.toggled.connect(self._on_autoscroll_toggled)
        self.timestamps_checkbox.toggled.connect(self._on_timestamps_toggled)
        self.wordwrap_checkbox.toggled.connect(self._on_wordwrap_toggled)

        # Connect dev option signals
        self.max_messages_spin.valueChanged.connect(self._on_max_messages_changed)
        self.update_interval_spin.valueChanged.connect(self._on_update_interval_changed)

    def _apply_config(self):
        """Apply configuration settings."""
        tab_config = self.config.get_tab_config("echo_log")

        # Update UI based on dev mode
        self._update_dev_options_visibility()

        # Apply auto-refresh setting
        if tab_config.auto_refresh:
            self.stats_timer.start(tab_config.refresh_interval)

    def _update_dev_options_visibility(self):
        """Update visibility of dev options based on dev mode."""
        is_dev = self.config.get_tab_config("echo_log").dev_mode
        self.dev_options_group.setVisible(
            is_dev or self.config.get_tab_config("echo_log").show_advanced_controls
        )

    def add_message(self, message: str, level: str = "INFO", source: str = "UNKNOWN"):
        """Add a message to the echo log."""
        if self.paused:
            return

        import time

        timestamp = time.strftime("%H:%M:%S")

        # Format message
        if self.timestamps_checkbox.isChecked():
            formatted_msg = f"[{timestamp}] [{level}] [{source}] {message}"
        else:
            formatted_msg = f"[{level}] [{source}] {message}"

        # Apply line numbers if enabled
        if self.linenums_checkbox.isChecked():
            line_num = self.message_count + 1
            formatted_msg = f"{line_num:04d}: {formatted_msg}"

        # Check if message passes filters
        if self._message_passes_filters(formatted_msg, level, source):
            self.log_display.append(formatted_msg)
            self.filtered_messages.append(formatted_msg)

            # Auto-scroll if enabled
            if self.autoscroll_checkbox.isChecked():
                scrollbar = self.log_display.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())

        self.message_count += 1

        # Trim messages if over limit
        max_messages = self.max_messages_spin.value()
        if len(self.filtered_messages) > max_messages:
            # Remove oldest messages
            excess = len(self.filtered_messages) - max_messages
            self.filtered_messages = self.filtered_messages[excess:]

            # Rebuild display
            self._rebuild_display()

    def _message_passes_filters(self, message: str, level: str, source: str) -> bool:
        """Check if message passes current filters."""
        # Level filter
        level_filter = self.level_combo.currentText()
        if level_filter != "ALL":
            level_hierarchy = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in level_hierarchy:
                return False
            if level_hierarchy.index(level) < level_hierarchy.index(level_filter):
                return False

        # Source filter
        source_filter = self.source_combo.currentText()
        if source_filter != "ALL" and source != source_filter:
            return False

        # Search filter
        search_filter = self.search_combo.currentText().strip().lower()
        if search_filter and search_filter not in message.lower():
            return False

        return True

    def _rebuild_display(self):
        """Rebuild the entire log display."""
        self.log_display.clear()
        for message in self.filtered_messages:
            self.log_display.append(message)

        # Auto-scroll if enabled
        if self.autoscroll_checkbox.isChecked():
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot()
    def _clear_log(self):
        """Clear the log display."""
        self.log_display.clear()
        self.filtered_messages.clear()
        self.message_count = 0
        self.status_label.setText("Log cleared")

    @pyqtSlot()
    def _save_log(self):
        """Save the log to a file."""
        from PyQt5.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Echo Log", "echo_log.txt", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_display.toPlainText())
                self.status_label.setText(f"Log saved to {file_path}")
            except Exception as e:
                self.status_label.setText(f"Error saving log: {e}")

    @pyqtSlot()
    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.setText("▶️ Resume")
            self.status_label.setText("Log paused")
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.status_label.setText("Log resumed")

    @pyqtSlot()
    def _apply_filters(self):
        """Apply current filters to the log."""
        # This would re-filter all stored messages
        # For now, just update status
        level_filter = self.level_combo.currentText()
        source_filter = self.source_combo.currentText()
        search_filter = self.search_combo.currentText()

        filter_desc = f"Filters: Level={level_filter}, Source={source_filter}"
        if search_filter:
            filter_desc += f", Search='{search_filter}'"

        self.status_label.setText(filter_desc)

    @pyqtSlot(bool)
    def _on_autoscroll_toggled(self, checked: bool):
        """Handle autoscroll toggle."""
        self.status_label.setText(f"Auto-scroll: {'Enabled' if checked else 'Disabled'}")

    @pyqtSlot(bool)
    def _on_timestamps_toggled(self, checked: bool):
        """Handle timestamps toggle."""
        self.status_label.setText(f"Timestamps: {'Enabled' if checked else 'Disabled'}")

    @pyqtSlot(bool)
    def _on_wordwrap_toggled(self, checked: bool):
        """Handle word wrap toggle."""
        if checked:
            self.log_display.setLineWrapMode(QTextEdit.WidgetWidth)
        else:
            self.log_display.setLineWrapMode(QTextEdit.NoWrap)
        self.status_label.setText(f"Word wrap: {'Enabled' if checked else 'Disabled'}")

    @pyqtSlot(int)
    def _on_max_messages_changed(self, value: int):
        """Handle max messages change."""
        self.status_label.setText(f"Max messages set to: {value}")

    @pyqtSlot(int)
    def _on_update_interval_changed(self, value: int):
        """Handle update interval change."""
        self.stats_timer.setInterval(value)
        self.status_label.setText(f"Update interval set to: {value}ms")

    @pyqtSlot(bool)
    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        self._update_dev_options_visibility()

        if enabled:
            self.add_message("Developer mode enabled", "INFO", "GUI")
        else:
            self.add_message("Developer mode disabled", "INFO", "GUI")

    @pyqtSlot(str, object)
    def _on_config_changed(self, setting_name: str, value):
        """Handle configuration changes."""
        self.add_message(f"Config updated: {setting_name} = {value}", "INFO", "CONFIG")

    @pyqtSlot()
    def _update_stats(self):
        """Update statistics display."""
        # Update message counts
        self.message_count_label.setText(f"Messages: {self.message_count}")
        self.filtered_count_label.setText(f"Filtered: {len(self.filtered_messages)}")

        # Calculate message rate
        current_count = self.message_count
        rate = current_count - self.last_message_count
        self.rate_label.setText(f"Rate: {rate} msg/s")
        self.last_message_count = current_count
