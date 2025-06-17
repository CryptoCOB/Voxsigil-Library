#!/usr/bin/env python3
"""
Config Editor Tab - Live Configuration Management
Provides real-time editing and monitoring of VoxSigil configuration files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class ConfigEditorWidget(QWidget):
    """Widget for editing configuration files"""

    config_changed = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Editor
        self.editor = QTextEdit()
        self.editor.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.editor.setFont(VoxSigilStyles.FONTS["code"])
        self.editor.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.editor)

        # Status
        self.status_label = VoxSigilWidgetFactory.create_label("No file loaded", "info")
        layout.addWidget(self.status_label)

    def create_toolbar(self):
        """Create configuration toolbar"""
        toolbar = VoxSigilWidgetFactory.create_frame()
        layout = QHBoxLayout(toolbar)

        # Load config button
        self.load_btn = VoxSigilWidgetFactory.create_button("📂 Load Config", "default")
        self.load_btn.clicked.connect(self.load_config)
        layout.addWidget(self.load_btn)

        # Save config button
        self.save_btn = VoxSigilWidgetFactory.create_button("💾 Save Config", "primary")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        # Validate button
        self.validate_btn = VoxSigilWidgetFactory.create_button("✅ Validate", "success")
        self.validate_btn.clicked.connect(self.validate_config)
        self.validate_btn.setEnabled(False)
        layout.addWidget(self.validate_btn)

        # Reload button
        self.reload_btn = VoxSigilWidgetFactory.create_button("🔄 Reload", "default")
        self.reload_btn.clicked.connect(self.reload_config)
        self.reload_btn.setEnabled(False)
        layout.addWidget(self.reload_btn)

        layout.addStretch()
        return toolbar

    def load_config(self):
        """Load a configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration File",
            ".",
            "Config Files (*.json *.yaml *.yml);;All Files (*)",
        )

        if file_path:
            try:
                self.current_file = Path(file_path)
                content = self.current_file.read_text(encoding="utf-8")
                self.editor.setPlainText(content)
                self.status_label.setText(f"Loaded: {self.current_file.name}")

                # Enable buttons
                self.save_btn.setEnabled(True)
                self.validate_btn.setEnabled(True)
                self.reload_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def save_config(self):
        """Save the current configuration"""
        if not self.current_file:
            return

        try:
            content = self.editor.toPlainText()
            self.current_file.write_text(content, encoding="utf-8")
            self.status_label.setText(
                f"Saved: {self.current_file.name} at {datetime.now().strftime('%H:%M:%S')}"
            )            # Emit change signal
            try:
                if self.current_file.suffix.lower() in [".json"]:
                    config_data = json.loads(content)
                elif self.current_file.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(content)
                else:
                    config_data = {"raw_content": content}

                self.config_changed.emit(str(self.current_file), config_data)
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                logger.debug(f"Config parsing error for signal emission: {e}")
                # Still emit signal with raw content for non-parseable configs
                self.config_changed.emit(str(self.current_file), {"raw_content": content, "parse_error": str(e)})
            except Exception as e:
                logger.warning(f"Unexpected error parsing config for signal: {e}")
                self.config_changed.emit(str(self.current_file), {"raw_content": content, "error": str(e)})

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def validate_config(self):
        """Validate the current configuration"""
        if not self.current_file:
            return

        try:
            content = self.editor.toPlainText()

            if self.current_file.suffix.lower() == ".json":
                json.loads(content)
                QMessageBox.information(self, "Validation", "✅ Valid JSON configuration")
            elif self.current_file.suffix.lower() in [".yaml", ".yml"]:
                yaml.safe_load(content)
                QMessageBox.information(self, "Validation", "✅ Valid YAML configuration")
            else:
                QMessageBox.information(self, "Validation", "✅ File syntax appears valid")

        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Validation Error", f"❌ Invalid JSON:\n{e}")
        except yaml.YAMLError as e:
            QMessageBox.warning(self, "Validation Error", f"❌ Invalid YAML:\n{e}")
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", f"❌ Validation error:\n{e}")

    def reload_config(self):
        """Reload the configuration from disk"""
        if not self.current_file or not self.current_file.exists():
            return

        try:
            content = self.current_file.read_text(encoding="utf-8")
            self.editor.setPlainText(content)
            self.status_label.setText(f"Reloaded: {self.current_file.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reload file:\n{e}")

    def on_text_changed(self):
        """Handle text changes"""
        if self.current_file:
            self.status_label.setText(f"Modified: {self.current_file.name} (unsaved)")


class ConfigMonitorWidget(QWidget):
    """Widget for monitoring configuration changes"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("📊 Configuration Monitor", "section")
        layout.addWidget(title)

        # Recent changes log
        self.changes_log = QTextEdit()
        self.changes_log.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.changes_log.setMaximumHeight(200)
        self.changes_log.setReadOnly(True)
        layout.addWidget(self.changes_log)

        # Active configs display
        active_title = VoxSigilWidgetFactory.create_label("🔧 Active Configurations", "section")
        layout.addWidget(active_title)

        self.active_configs = QTextEdit()
        self.active_configs.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.active_configs.setReadOnly(True)
        layout.addWidget(self.active_configs)

        # Load initial config info
        self.load_config_info()

    def add_change_log(self, file_path: str, action: str):
        """Add a change to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.changes_log.append(f"[{timestamp}] {action}: {Path(file_path).name}")

    def load_config_info(self):
        """Load information about active configurations"""
        try:
            config_info = "Configuration files in project:\n\n"

            # Look for common config files
            config_files = [
                "pyproject.toml",
                "requirements.in",
                "requirements.lock",
                "agents.json",
                "config/*.yaml",
                "config/*.json",
            ]

            for pattern in config_files:
                files = list(Path(".").glob(pattern))
                for file in files:
                    if file.exists():
                        size = file.stat().st_size
                        modified = datetime.fromtimestamp(file.stat().st_mtime)
                        config_info += f"📄 {file.name} ({size} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')})\n"

            self.active_configs.setPlainText(config_info)

        except Exception as e:
            logger.error(f"Error loading config info: {e}")


class ConfigEditorTab(QWidget):
    """Main configuration editor tab"""

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.init_ui()
        self.setup_streaming()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("⚙️ Configuration Editor", "title")
        layout.addWidget(title)

        # Main splitter
        splitter = VoxSigilWidgetFactory.create_splitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Editor
        editor_group = QGroupBox("Configuration Editor")
        editor_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        editor_layout = QVBoxLayout(editor_group)

        self.config_editor = ConfigEditorWidget()
        self.config_editor.config_changed.connect(self.on_config_changed)
        editor_layout.addWidget(self.config_editor)

        # Right panel - Monitor
        monitor_group = QGroupBox("Configuration Monitor")
        monitor_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        monitor_layout = QVBoxLayout(monitor_group)

        self.config_monitor = ConfigMonitorWidget()
        monitor_layout.addWidget(self.config_monitor)

        # Add to splitter
        splitter.addWidget(editor_group)
        splitter.addWidget(monitor_group)
        splitter.setSizes([500, 300])

    def setup_streaming(self):
        """Setup event bus streaming"""
        if self.event_bus:
            # Subscribe to config change events
            self.event_bus.subscribe("config.changed", self.on_external_config_change)

    def on_config_changed(self, file_path: str, config_data: Dict[str, Any]):
        """Handle configuration changes"""
        try:
            self.config_monitor.add_change_log(file_path, "Saved")

            # Publish config change event
            if self.event_bus:
                self.event_bus.publish(
                    "config.changed",
                    {
                        "file_path": file_path,
                        "config_data": config_data,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error handling config change: {e}")

    def on_external_config_change(self, data: Dict[str, Any]):
        """Handle external configuration changes"""
        try:
            file_path = data.get("file_path", "Unknown")
            self.config_monitor.add_change_log(file_path, "External change detected")

        except Exception as e:
            logger.error(f"Error handling external config change: {e}")


# Backward compatibility
ConfigTab = ConfigEditorTab
