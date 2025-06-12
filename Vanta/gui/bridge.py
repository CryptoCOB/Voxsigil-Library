#!/usr/bin/env python3
"""
VoxSigil GUI Bridge - Auto-Widget Generation System
Automatically generates GUI widgets based on module ui_spec configurations

This bridge connects the registered modules to the GUI by creating toggles,
controls, and interface elements based on each module's capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, 
    QPushButton, QLabel, QSlider, QSpinBox, QComboBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)

class ModuleWidget(QFrame):
    """Individual widget for a registered module with auto-generated controls"""
    
    toggled = pyqtSignal(str, bool)  # module_id, enabled
    configured = pyqtSignal(str, dict)  # module_id, config
    
    def __init__(self, module_id: str, ui_spec: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self.ui_spec = ui_spec or {}
        self.config = {}
        self.is_enabled = False
        
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #4ecdc4;
                border-radius: 6px;
                background-color: #2a2a4e;
                margin: 2px;
                padding: 4px;
            }
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI based on the module's ui_spec"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header with toggle
        header_layout = QHBoxLayout()
        
        # Main toggle checkbox
        self.toggle_checkbox = QCheckBox(self.ui_spec.get("name", self.module_id))
        self.toggle_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-weight: bold;
                font-size: 10pt;
            }
            QCheckBox::indicator:checked {
                background-color: #4ecdc4;
                border: 1px solid #4ecdc4;
            }
        """)
        self.toggle_checkbox.toggled.connect(self._on_toggle)
        header_layout.addWidget(self.toggle_checkbox)
        
        # Status indicator
        self.status_label = QLabel("Disabled")
        self.status_label.setStyleSheet("color: #888888; font-size: 8pt;")
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Description
        description = self.ui_spec.get("description", "")
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #cccccc; font-size: 9pt; font-style: italic;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # Auto-generate controls based on ui_spec
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self._generate_controls()
        
        layout.addWidget(self.controls_widget)
        self.controls_widget.setVisible(False)  # Hidden until enabled
    
    def _generate_controls(self):
        """Generate controls based on ui_spec configuration"""
        controls = self.ui_spec.get("controls", {})
        
        for control_name, control_spec in controls.items():
            control_widget = self._create_control(control_name, control_spec)
            if control_widget:
                self.controls_layout.addWidget(control_widget)
    
    def _create_control(self, name: str, spec: Dict[str, Any]) -> Optional[QWidget]:
        """Create a control widget based on specification"""
        control_type = spec.get("type", "text")
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Label
        label = QLabel(f"{spec.get('label', name)}:")
        label.setStyleSheet("color: #ffffff; font-size: 9pt;")
        label.setMinimumWidth(80)
        layout.addWidget(label)
        
        # Control based on type
        control = None
        
        if control_type == "slider":
            control = QSlider(Qt.Horizontal)
            control.setRange(spec.get("min", 0), spec.get("max", 100))
            control.setValue(spec.get("default", 50))
            control.valueChanged.connect(lambda v, n=name: self._update_config(n, v))
            
        elif control_type == "spinbox":
            control = QSpinBox()
            control.setRange(spec.get("min", 0), spec.get("max", 100))
            control.setValue(spec.get("default", 0))
            control.valueChanged.connect(lambda v, n=name: self._update_config(n, v))
            
        elif control_type == "checkbox":
            control = QCheckBox()
            control.setChecked(spec.get("default", False))
            control.toggled.connect(lambda v, n=name: self._update_config(n, v))
            
        elif control_type == "combo":
            control = QComboBox()
            options = spec.get("options", [])
            control.addItems(options)
            if spec.get("default") in options:
                control.setCurrentText(spec.get("default"))
            control.currentTextChanged.connect(lambda v, n=name: self._update_config(n, v))
        
        if control:
            control.setStyleSheet("""
                QSlider::groove:horizontal { background: #3d3d6b; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #4ecdc4; width: 12px; border-radius: 6px; }
                QSpinBox, QComboBox { background: #3d3d6b; color: #ffffff; border: 1px solid #4ecdc4; padding: 2px; }
                QCheckBox::indicator { width: 12px; height: 12px; }
                QCheckBox::indicator:checked { background-color: #4ecdc4; border: 1px solid #4ecdc4; }
            """)
            layout.addWidget(control)
            
            # Store default value
            if control_type == "slider" or control_type == "spinbox":
                self.config[name] = spec.get("default", 0)
            elif control_type == "checkbox":
                self.config[name] = spec.get("default", False)
            elif control_type == "combo":
                self.config[name] = spec.get("default", "")
        
        return widget if control else None
    
    def _update_config(self, name: str, value: Any):
        """Update configuration when control value changes"""
        self.config[name] = value
        self.configured.emit(self.module_id, self.config.copy())
    
    def _on_toggle(self, enabled: bool):
        """Handle module enable/disable toggle"""
        self.is_enabled = enabled
        self.controls_widget.setVisible(enabled)
        self.status_label.setText("Enabled" if enabled else "Disabled")
        self.status_label.setStyleSheet(
            f"color: {'#4ecdc4' if enabled else '#888888'}; font-size: 8pt;"
        )
        self.toggled.emit(self.module_id, enabled)
    
    def set_enabled(self, enabled: bool):
        """Programmatically enable/disable the module"""
        self.toggle_checkbox.setChecked(enabled)


class GUIBridge(QWidget):
    """Main bridge widget that contains all auto-generated module widgets"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.module_widgets = {}
        self.enabled_modules = set()
        self.module_configs = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main bridge UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🎛️ Module Controls")
        header.setStyleSheet("""
            QLabel {
                color: #4ecdc4;
                font-size: 14pt;
                font-weight: bold;
                padding: 8px;
                border-bottom: 2px solid #4ecdc4;
                margin-bottom: 8px;
            }
        """)
        layout.addWidget(header)
        
        # Modules container
        self.modules_container = QWidget()
        self.modules_layout = QVBoxLayout(self.modules_container)
        self.modules_layout.setSpacing(4)
        
        layout.addWidget(self.modules_container)
        layout.addStretch()
    
    def add_to_gui(self, module_id: str, ui_spec: Dict[str, Any]) -> bool:
        """
        Add a module to the GUI based on its ui_spec
        
        Args:
            module_id: Unique identifier for the module
            ui_spec: UI specification containing controls, name, description etc.
            
        Returns:
            bool: True if successfully added
        """
        try:
            if module_id in self.module_widgets:
                logger.warning(f"Module {module_id} already exists in GUI")
                return False
            
            # Create module widget
            module_widget = ModuleWidget(module_id, ui_spec)
            module_widget.toggled.connect(self._on_module_toggled)
            module_widget.configured.connect(self._on_module_configured)
            
            # Add to layout
            self.modules_layout.addWidget(module_widget)
            self.module_widgets[module_id] = module_widget
            
            logger.info(f"Added module {module_id} to GUI with {len(ui_spec.get('controls', {}))} controls")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add module {module_id} to GUI: {e}")
            return False
    
    def remove_from_gui(self, module_id: str) -> bool:
        """Remove a module from the GUI"""
        try:
            if module_id not in self.module_widgets:
                logger.warning(f"Module {module_id} not found in GUI")
                return False
            
            widget = self.module_widgets[module_id]
            self.modules_layout.removeWidget(widget)
            widget.deleteLater()
            
            del self.module_widgets[module_id]
            self.enabled_modules.discard(module_id)
            self.module_configs.pop(module_id, None)
            
            logger.info(f"Removed module {module_id} from GUI")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove module {module_id} from GUI: {e}")
            return False
    
    def enable_module(self, module_id: str, enabled: bool = True):
        """Enable or disable a module"""
        if module_id in self.module_widgets:
            self.module_widgets[module_id].set_enabled(enabled)
    
    def get_enabled_modules(self) -> List[str]:
        """Get list of currently enabled modules"""
        return list(self.enabled_modules)
    
    def get_module_config(self, module_id: str) -> Dict[str, Any]:
        """Get configuration for a specific module"""
        return self.module_configs.get(module_id, {})
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all modules"""
        return self.module_configs.copy()
    
    def _on_module_toggled(self, module_id: str, enabled: bool):
        """Handle when a module is toggled on/off"""
        if enabled:
            self.enabled_modules.add(module_id)
        else:
            self.enabled_modules.discard(module_id)
        
        logger.info(f"Module {module_id} {'enabled' if enabled else 'disabled'}")
    
    def _on_module_configured(self, module_id: str, config: Dict[str, Any]):
        """Handle when a module's configuration changes"""
        self.module_configs[module_id] = config
        logger.debug(f"Module {module_id} configuration updated: {config}")


# Global bridge instance for easy access
_bridge_instance = None

def get_gui_bridge() -> GUIBridge:
    """Get the global GUI bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = GUIBridge()
    return _bridge_instance

def add_to_gui(module_id: str, ui_spec: Dict[str, Any]) -> bool:
    """
    Convenience function to add a module to the GUI
    
    Args:
        module_id: Unique identifier for the module
        ui_spec: UI specification dictionary containing:
            - name: Display name for the module
            - description: Brief description
            - controls: Dict of control specifications
                - type: "slider", "spinbox", "checkbox", "combo"
                - label: Display label
                - min/max: For numeric controls
                - default: Default value
                - options: For combo boxes
    
    Returns:
        bool: True if successfully added
    
    Example ui_spec:
    {
        "name": "My Module",
        "description": "Does something cool",
        "controls": {
            "intensity": {
                "type": "slider",
                "label": "Intensity",
                "min": 0,
                "max": 100,
                "default": 50
            },
            "enabled": {
                "type": "checkbox", 
                "label": "Enable Feature",
                "default": True
            }
        }
    }
    """
    bridge = get_gui_bridge()
    return bridge.add_to_gui(module_id, ui_spec)

def remove_from_gui(module_id: str) -> bool:
    """Remove a module from the GUI"""
    bridge = get_gui_bridge()
    return bridge.remove_from_gui(module_id)

def get_enabled_modules() -> List[str]:
    """Get list of enabled modules"""
    bridge = get_gui_bridge()
    return bridge.get_enabled_modules()

def get_module_config(module_id: str) -> Dict[str, Any]:
    """Get configuration for a module"""
    bridge = get_gui_bridge()
    return bridge.get_module_config(module_id)
