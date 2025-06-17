"""
Universal Development Mode Control Panel
Provides standardized dev mode controls for all VoxSigil GUI tabs.
"""

import logging
from typing import Any, Callable, Dict

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config, is_dev_mode

logger = logging.getLogger("DevModePanel")


class DevModeControlPanel(QWidget):
    """
    Universal development mode control panel that can be embedded in any tab.
    Provides standardized controls for dev mode settings.
    """  # Signals

    config_changed = pyqtSignal(str, object)  # setting_name, value
    dev_mode_toggled = pyqtSignal(bool)  # enabled
    refresh_triggered = pyqtSignal()  # refresh requested

    def __init__(self, component_name: str, parent=None):
        super().__init__(parent)
        self.component_name = component_name
        self.config = get_dev_config()
        self.controls: Dict[str, QWidget] = {}
        self.callbacks: Dict[str, Callable] = {}

        self._init_ui()
        self._connect_signals()
        self._update_from_config()

    def _init_ui(self):
        """Initialize the development mode UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create collapsible dev mode section
        self.dev_group = QGroupBox(f"🔧 Development Mode - {self.component_name.title()}")
        self.dev_group.setCheckable(True)
        self.dev_group.setChecked(is_dev_mode(self.component_name))

        dev_layout = QVBoxLayout(self.dev_group)

        # Create tabbed interface for different dev controls
        self.dev_tabs = QTabWidget()

        # Basic Controls Tab
        basic_tab = self._create_basic_controls_tab()
        self.dev_tabs.addTab(basic_tab, "Basic")

        # Advanced Controls Tab
        advanced_tab = self._create_advanced_controls_tab()
        self.dev_tabs.addTab(advanced_tab, "Advanced")

        # Debugging Tab
        debug_tab = self._create_debugging_tab()
        self.dev_tabs.addTab(debug_tab, "Debug")

        # Configuration Tab
        config_tab = self._create_configuration_tab()
        self.dev_tabs.addTab(config_tab, "Config")

        dev_layout.addWidget(self.dev_tabs)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        dev_layout.addWidget(self.status_label)

        layout.addWidget(self.dev_group)

        # Hide by default if not in dev mode
        if not is_dev_mode(self.component_name):
            self.dev_group.setVisible(False)

    def _create_basic_controls_tab(self) -> QWidget:
        """Create basic development controls."""
        widget = QWidget()
        layout = QGridLayout(widget)

        row = 0

        # Auto-refresh toggle
        self.controls["auto_refresh"] = QCheckBox("Auto Refresh")
        layout.addWidget(QLabel("Auto Refresh:"), row, 0)
        layout.addWidget(self.controls["auto_refresh"], row, 1)
        row += 1

        # Refresh interval
        self.controls["refresh_interval"] = QSpinBox()
        self.controls["refresh_interval"].setRange(100, 30000)
        self.controls["refresh_interval"].setSuffix(" ms")
        self.controls["refresh_interval"].setValue(5000)
        layout.addWidget(QLabel("Refresh Interval:"), row, 0)
        layout.addWidget(self.controls["refresh_interval"], row, 1)
        row += 1

        # Debug logging
        self.controls["debug_logging"] = QCheckBox("Debug Logging")
        layout.addWidget(QLabel("Debug Logging:"), row, 0)
        layout.addWidget(self.controls["debug_logging"], row, 1)
        row += 1

        # Advanced controls toggle
        self.controls["show_advanced"] = QCheckBox("Show Advanced Controls")
        layout.addWidget(QLabel("Advanced UI:"), row, 0)
        layout.addWidget(self.controls["show_advanced"], row, 1)
        row += 1

        # Quick actions
        actions_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 Refresh Now")
        self.reset_btn = QPushButton("🔄 Reset")
        self.export_btn = QPushButton("💾 Export Config")

        actions_layout.addWidget(self.refresh_btn)
        actions_layout.addWidget(self.reset_btn)
        actions_layout.addWidget(self.export_btn)

        layout.addLayout(actions_layout, row, 0, 1, 2)

        return widget

    def _create_advanced_controls_tab(self) -> QWidget:
        """Create advanced development controls."""
        widget = QWidget()
        layout = QGridLayout(widget)

        row = 0

        # Performance monitoring
        perf_group = QGroupBox("Performance Monitoring")
        perf_layout = QGridLayout(perf_group)

        self.controls["monitor_performance"] = QCheckBox("Enable Performance Monitoring")
        perf_layout.addWidget(self.controls["monitor_performance"], 0, 0, 1, 2)

        self.controls["perf_sample_rate"] = QSpinBox()
        self.controls["perf_sample_rate"].setRange(100, 10000)
        self.controls["perf_sample_rate"].setSuffix(" ms")
        self.controls["perf_sample_rate"].setValue(1000)
        perf_layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        perf_layout.addWidget(self.controls["perf_sample_rate"], 1, 1)

        layout.addWidget(perf_group, row, 0, 1, 2)
        row += 1

        # Memory management
        memory_group = QGroupBox("Memory Management")
        memory_layout = QGridLayout(memory_group)

        self.controls["auto_gc"] = QCheckBox("Auto Garbage Collection")
        memory_layout.addWidget(self.controls["auto_gc"], 0, 0, 1, 2)

        self.controls["max_cache_size"] = QSpinBox()
        self.controls["max_cache_size"].setRange(10, 1000)
        self.controls["max_cache_size"].setSuffix(" MB")
        self.controls["max_cache_size"].setValue(100)
        memory_layout.addWidget(QLabel("Max Cache Size:"), 1, 0)
        memory_layout.addWidget(self.controls["max_cache_size"], 1, 1)

        layout.addWidget(memory_group, row, 0, 1, 2)
        row += 1

        # Threading controls
        thread_group = QGroupBox("Threading")
        thread_layout = QGridLayout(thread_group)

        self.controls["max_threads"] = QSpinBox()
        self.controls["max_threads"].setRange(1, 16)
        self.controls["max_threads"].setValue(4)
        thread_layout.addWidget(QLabel("Max Threads:"), 0, 0)
        thread_layout.addWidget(self.controls["max_threads"], 0, 1)

        self.controls["thread_priority"] = QComboBox()
        self.controls["thread_priority"].addItems(["Low", "Normal", "High"])
        self.controls["thread_priority"].setCurrentText("Normal")
        thread_layout.addWidget(QLabel("Thread Priority:"), 1, 0)
        thread_layout.addWidget(self.controls["thread_priority"], 1, 1)

        layout.addWidget(thread_group, row, 0, 1, 2)

        return widget

    def _create_debugging_tab(self) -> QWidget:
        """Create debugging controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Debug options
        debug_group = QGroupBox("Debug Options")
        debug_layout = QGridLayout(debug_group)

        self.controls["verbose_logging"] = QCheckBox("Verbose Logging")
        debug_layout.addWidget(self.controls["verbose_logging"], 0, 0)

        self.controls["trace_execution"] = QCheckBox("Trace Execution")
        debug_layout.addWidget(self.controls["trace_execution"], 0, 1)

        self.controls["profile_performance"] = QCheckBox("Profile Performance")
        debug_layout.addWidget(self.controls["profile_performance"], 1, 0)

        self.controls["dump_state"] = QCheckBox("Dump State on Error")
        debug_layout.addWidget(self.controls["dump_state"], 1, 1)

        layout.addWidget(debug_group)

        # Log viewer
        log_group = QGroupBox("Live Log Viewer")
        log_layout = QVBoxLayout(log_group)

        self.log_viewer = QTextEdit()
        self.log_viewer.setMaximumHeight(150)
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet(
            "background: #1e1e1e; color: #00ff00; font-family: 'Courier New';"
        )
        log_layout.addWidget(self.log_viewer)

        log_controls = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear Log")
        self.export_log_btn = QPushButton("Export Log")
        log_controls.addWidget(self.clear_log_btn)
        log_controls.addWidget(self.export_log_btn)
        log_controls.addStretch()
        log_layout.addLayout(log_controls)

        layout.addWidget(log_group)

        return widget

    def _create_configuration_tab(self) -> QWidget:
        """Create configuration management controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Configuration actions
        config_actions = QGroupBox("Configuration Management")
        config_layout = QGridLayout(config_actions)

        self.save_config_btn = QPushButton("💾 Save Configuration")
        self.load_config_btn = QPushButton("📂 Load Configuration")
        self.reset_config_btn = QPushButton("🔄 Reset to Defaults")
        self.export_config_btn = QPushButton("📤 Export Configuration")

        config_layout.addWidget(self.save_config_btn, 0, 0)
        config_layout.addWidget(self.load_config_btn, 0, 1)
        config_layout.addWidget(self.reset_config_btn, 1, 0)
        config_layout.addWidget(self.export_config_btn, 1, 1)

        layout.addWidget(config_actions)

        # Configuration viewer/editor
        config_viewer_group = QGroupBox("Current Configuration")
        config_viewer_layout = QVBoxLayout(config_viewer_group)

        self.config_viewer = QTextEdit()
        self.config_viewer.setMaximumHeight(200)
        self.config_viewer.setReadOnly(True)
        config_viewer_layout.addWidget(self.config_viewer)

        layout.addWidget(config_viewer_group)

        # Auto-save toggle
        self.controls["auto_save"] = QCheckBox("Auto-save configuration changes")
        self.controls["auto_save"].setChecked(True)
        layout.addWidget(self.controls["auto_save"])

        return widget

    def _connect_signals(self):
        """Connect all signals and slots."""
        # Dev mode toggle
        self.dev_group.toggled.connect(self._on_dev_mode_toggled)

        # Basic controls
        if "auto_refresh" in self.controls:
            self.controls["auto_refresh"].toggled.connect(
                lambda checked: self._emit_config_change("auto_refresh", checked)
            )

        if "refresh_interval" in self.controls:
            self.controls["refresh_interval"].valueChanged.connect(
                lambda value: self._emit_config_change("refresh_interval", value)
            )

        if "debug_logging" in self.controls:
            self.controls["debug_logging"].toggled.connect(
                lambda checked: self._emit_config_change("debug_logging", checked)
            )

        # Action buttons
        if hasattr(self, "refresh_btn"):
            self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        if hasattr(self, "reset_btn"):
            self.reset_btn.clicked.connect(self._on_reset_clicked)
        if hasattr(self, "clear_log_btn"):
            self.clear_log_btn.clicked.connect(self._on_clear_log_clicked)
        if hasattr(self, "save_config_btn"):
            self.save_config_btn.clicked.connect(self._on_save_config_clicked)

    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        self.config.update_tab_config(self.component_name, dev_mode=enabled)
        self.dev_mode_toggled.emit(enabled)
        self._update_status(f"Dev mode {'enabled' if enabled else 'disabled'}")

    def _emit_config_change(self, setting_name: str, value: Any):
        """Emit configuration change signal."""
        self.config.update_tab_config(self.component_name, **{setting_name: value})
        self.config_changed.emit(setting_name, value)
        self._update_status(f"Updated {setting_name}: {value}")

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        self._update_status("Refreshing...")
        # Emit signal that parent can connect to
        self.refresh_triggered.emit()
        self.config_changed.emit("refresh_requested", True)

    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.config.reset_to_defaults()
        self._update_from_config()
        self._update_status("Reset to defaults")

    def _on_clear_log_clicked(self):
        """Handle clear log button click."""
        if hasattr(self, "log_viewer"):
            self.log_viewer.clear()
        self._update_status("Log cleared")

    def _on_save_config_clicked(self):
        """Handle save config button click."""
        self.config.save_config()
        self._update_status("Configuration saved")

    def _update_from_config(self):
        """Update UI controls from current configuration."""
        tab_config = self.config.get_tab_config(self.component_name)

        # Update controls based on current config
        if "auto_refresh" in self.controls:
            self.controls["auto_refresh"].setChecked(tab_config.auto_refresh)
        if "refresh_interval" in self.controls:
            self.controls["refresh_interval"].setValue(tab_config.refresh_interval)
        if "debug_logging" in self.controls:
            self.controls["debug_logging"].setChecked(tab_config.debug_logging)
        if "show_advanced" in self.controls:
            self.controls["show_advanced"].setChecked(tab_config.show_advanced_controls)
        if "auto_save" in self.controls:
            self.controls["auto_save"].setChecked(self.config.auto_save_config)

        # Update dev group
        self.dev_group.setChecked(tab_config.dev_mode)

        # Update config viewer
        if hasattr(self, "config_viewer"):
            import json

            config_data = {
                "tab_config": tab_config.__dict__,
                "global_dev_mode": self.config.global_dev_mode,
            }
            self.config_viewer.setText(json.dumps(config_data, indent=2))

    def _update_status(self, message: str):
        """Update status label."""
        self.status_label.setText(f"Status: {message}")
        # Auto-clear status after 3 seconds
        QTimer.singleShot(3000, lambda: self.status_label.setText("Ready"))

    def add_custom_control(self, name: str, control: QWidget, tab: str = "Advanced"):
        """Add a custom control to the specified tab."""
        self.controls[name] = control

        # Find the tab and add the control
        for i in range(self.dev_tabs.count()):
            if self.dev_tabs.tabText(i) == tab:
                tab_widget = self.dev_tabs.widget(i)
                if hasattr(tab_widget, "layout") and tab_widget.layout():
                    tab_widget.layout().addWidget(control)
                break

    def set_callback(self, setting_name: str, callback: Callable):
        """Set a callback for when a specific setting changes."""
        self.callbacks[setting_name] = callback

    def log_message(self, message: str):
        """Add a message to the log viewer."""
        if hasattr(self, "log_viewer"):
            self.log_viewer.append(f"[{self.component_name}] {message}")

    def is_dev_mode_enabled(self) -> bool:
        """Check if dev mode is currently enabled."""
        return self.dev_group.isChecked()


class SimpleDevModeToggle(QWidget):
    """
    Simplified dev mode toggle for tabs that don't need full controls.
    """

    dev_mode_toggled = pyqtSignal(bool)

    def __init__(self, component_name: str, parent=None):
        super().__init__(parent)
        self.component_name = component_name
        self.config = get_dev_config()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.dev_toggle = QCheckBox(f"🔧 Dev Mode ({component_name})")
        self.dev_toggle.setChecked(is_dev_mode(component_name))
        self.dev_toggle.toggled.connect(self._on_toggled)

        layout.addWidget(self.dev_toggle)
        layout.addStretch()

    def _on_toggled(self, enabled: bool):
        """Handle toggle."""
        self.config.update_tab_config(self.component_name, dev_mode=enabled)
        self.dev_mode_toggled.emit(enabled)

    def set_dev_mode(self, enabled: bool):
        """Programmatically set dev mode."""
        self.dev_toggle.setChecked(enabled)
