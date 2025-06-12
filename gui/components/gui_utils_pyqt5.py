
import json
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QLabel, QFrame, QHBoxLayout, QPushButton, QToolTip
)
from PyQt5.QtCore import Qt, QEvent, QPoint
from PyQt5.QtGui import QFont, QEnterEvent


class ToolTip(QWidget):
    """Simple tooltip for PyQt5 widgets."""

    def __init__(self, widget: QWidget, text: str) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.tooltip_widget = None
        
        # Install event filter on the widget
        widget.installEventFilter(self)
        
        # Set up tooltip widget
        self.setWindowFlags(Qt.ToolTip)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create tooltip label
        self.label = QLabel(text)
        self.label.setStyleSheet("""
            QLabel {
                background-color: #ffffe0;
                color: black;
                border: 1px solid #000000;
                padding: 4px;
                font-family: 'Consolas';
                font-size: 8pt;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def eventFilter(self, obj, event):
        """Handle mouse events for tooltip display"""
        if obj == self.widget:
            if event.type() == QEvent.Enter:
                self.show_tooltip(event)
                return True
            elif event.type() == QEvent.Leave:
                self.hide_tooltip()
                return True
        return False

    def show_tooltip(self, event=None):
        """Show the tooltip"""
        if not self.text:
            return
            
        # Position tooltip near the widget
        widget_pos = self.widget.mapToGlobal(QPoint(0, 0))
        x = widget_pos.x() + 20
        y = widget_pos.y() + 20
        
        # Adjust size to content
        self.adjustSize()
        self.move(x, y)
        self.show()

    def hide_tooltip(self):
        """Hide the tooltip"""
        self.hide()


def bind_agent_buttons(parent_widget: QWidget, registry, meta_path: str = "agents.json") -> None:
    """Bind agent on_gui_call buttons to a PyQt5 widget."""
    if not parent_widget or not registry:
        return

    meta: dict[str, dict] = {}
    try:
        with open(meta_path, "r") as f:
            for entry in json.load(f):
                meta[entry.get("name")] = entry
    except Exception:
        meta = {}

    # Create frame for agent buttons
    frame = QFrame(parent_widget)
    frame.setStyleSheet("""
        QFrame {
            background-color: #2a2a4e;
            border: none;
        }
    """)
    
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(5, 5, 5, 5)
    
    # Add "Agents" label
    agents_label = QLabel("Agents")
    agents_label.setStyleSheet("""
        QLabel {
            background-color: #2a2a4e;
            color: #00ff88;
            font-family: 'Consolas';
            font-size: 10pt;
            font-weight: bold;
        }
    """)
    layout.addWidget(agents_label)
    
    # Add agent buttons
    for agent_name, agent in registry.get_all_agents():
        if not hasattr(agent, "on_gui_call"):
            continue

        meta_entry = meta.get(agent_name, {})
        invocations = getattr(agent, "invocations", [])
        label = invocations[0] if invocations else f"Invoke {agent_name}"

        btn = QPushButton(label)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                font-family: 'Consolas';
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #4ecdc4;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #45b7aa;
                border-color: #45b7aa;
            }
            QPushButton:pressed {
                background-color: #3da399;
            }
        """)
        
        # Connect button to agent
        btn.clicked.connect(lambda checked, a=agent: a.on_gui_call())
        
        # Add tooltip
        tooltip_text = f"{meta_entry.get('class', '')} | {', '.join(getattr(agent, 'tags', []))}"
        if tooltip_text.strip(" |"):
            btn.setToolTip(tooltip_text)
        
        layout.addWidget(btn)
    
    # Add stretch to push buttons to the left
    layout.addStretch()
    
    # Add frame to parent widget's layout if it has one
    if hasattr(parent_widget, 'layout') and parent_widget.layout():
        parent_widget.layout().addWidget(frame)


class GUIUtils:
    """PyQt5 GUI utility functions for VoxSigil"""
    
    @staticmethod
    def create_tooltip(widget: QWidget, text: str) -> ToolTip:
        """Create a tooltip for a widget"""
        return ToolTip(widget, text)
    
    @staticmethod
    def set_widget_tooltip(widget: QWidget, text: str):
        """Set a simple tooltip for a widget using PyQt5's built-in tooltip"""
        widget.setToolTip(text)
    
    @staticmethod
    def apply_voxsigil_button_style(button: QPushButton):
        """Apply VoxSigil styling to a button"""
        button.setStyleSheet("""
            QPushButton {
                background-color: #4ecdc4;
                color: white;
                font-family: 'Consolas';
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #4ecdc4;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #45b7aa;
                border-color: #45b7aa;
            }
            QPushButton:pressed {
                background-color: #3da399;
            }
        """)
    
    @staticmethod
    def create_voxsigil_frame(parent=None) -> QFrame:
        """Create a frame with VoxSigil styling"""
        frame = QFrame(parent)
        frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a4e;
                border: 1px solid #3a4750;
                border-radius: 4px;
            }
        """)
        return frame
    
    @staticmethod
    def create_voxsigil_label(text: str, parent=None) -> QLabel:
        """Create a label with VoxSigil styling"""
        label = QLabel(text, parent)
        label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #ffffff;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
        """)
        return label
