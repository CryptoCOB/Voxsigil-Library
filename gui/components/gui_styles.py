#!/usr/bin/env python3
"""
VoxSigil GUI Styles Module (PyQt5)
Enhanced centralized styling management for the Dynamic GridFormer GUI

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Provide consistent VoxSigil aesthetic across all GUI components
Using PyQt5 stylesheets for modern GUI styling with enhanced features

Features:
- Advanced animated tooltips with fade effects and positioning
- Cross-platform styling with enhanced widget factory
- Dynamic theme management and custom theme registration
- Responsive layout utilities and hover effects
- Widget state management and accessibility features
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout, 
    QPushButton, QToolTip, QScrollArea, QSplitter, QStatusBar,
    QProgressBar, QSlider, QCheckBox, QRadioButton, QSpinBox,
    QDoubleSpinBox, QDateEdit, QTimeEdit, QCalendarWidget
)
from PyQt5.QtCore import Qt, QEvent, QPoint, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QPainter, QLinearGradient


class AnimatedToolTip(QWidget):
    """Enhanced animated tooltip for PyQt5 widgets with fade effects."""

    def __init__(self, widget: QWidget, text: str, delay: int = 1000) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.delay = delay
        self.show_timer = QTimer()
        self.hide_timer = QTimer()
        
        # Install event filter on the widget
        widget.installEventFilter(self)
        
        # Set up tooltip widget
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Create tooltip label with enhanced styling
        self.label = QLabel(text)
        self.label.setStyleSheet(f"""
            QLabel {{
                background-color: {VoxSigilStyles.COLORS["bg_tertiary"]};
                color: {VoxSigilStyles.COLORS["text_primary"]};
                border: 1px solid {VoxSigilStyles.COLORS["accent_cyan"]};
                border-radius: 6px;
                padding: 8px 12px;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 500;
            }}
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        # Setup timers
        self.show_timer.setSingleShot(True)
        self.show_timer.timeout.connect(self.show_tooltip)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide_tooltip)

    def eventFilter(self, obj, event):
        """Handle mouse events for tooltip display with delay"""
        if obj == self.widget:
            if event.type() == QEvent.Enter:
                self.hide_timer.stop()
                self.show_timer.start(self.delay)
                return True
            elif event.type() == QEvent.Leave:
                self.show_timer.stop()
                self.hide_timer.start(100)
                return True
        return False

    def show_tooltip(self):
        """Show the tooltip with positioning"""
        if not self.text:
            return
            
        widget_pos = self.widget.mapToGlobal(QPoint(0, 0))
        widget_size = self.widget.size()
        
        # Position tooltip below the widget, centered
        self.adjustSize()
        tooltip_size = self.size()
        
        x = widget_pos.x() + (widget_size.width() - tooltip_size.width()) // 2
        y = widget_pos.y() + widget_size.height() + 5
        
        self.move(x, y)
        self.show()
        self.raise_()

    def hide_tooltip(self):
        """Hide the tooltip"""
        self.hide()


class VoxSigilStyles:
    """Enhanced VoxSigil GUI styling management for PyQt5"""

    # VoxSigil Color Palette - Enhanced
    COLORS = {
        "bg_primary": "#1a1a2e",  # Deep navy background
        "bg_secondary": "#16213e",  # Slightly lighter navy
        "bg_tertiary": "#0f1419",  # Dark code background
        "bg_quaternary": "#2a2a3e",  # Alternative row color
        "accent_cyan": "#00d4ff",  # Bright cyan
        "accent_coral": "#ff6b6b",  # Coral red
        "accent_mint": "#4ecdc4",  # Mint green
        "accent_gold": "#ffd93d",  # Gold highlights
        "accent_purple": "#a855f7",  # Purple accent
        "accent_orange": "#fb923c",  # Orange accent
        "text_primary": "#ffffff",  # White text
        "text_secondary": "#b0bec5",  # Light gray text
        "text_muted": "#64748b",  # Muted text
        "border_active": "#00d4ff",  # Active border color
        "border_inactive": "#3a4750",  # Inactive border color
        "success": "#10b981",  # Success green
        "warning": "#f59e0b",  # Warning amber
        "error": "#ef4444",  # Error red
        "info": "#3b82f6",  # Info blue
    }

    # Enhanced color constants
    ALT_ROW_COLOR = "#2a2a3e"
    HOVER_COLOR = "#363652"
    PRESSED_COLOR = "#2d2d45"

    # Enhanced font configurations
    FONTS = {
        "primary": QFont("Segoe UI", 10),
        "primary_bold": QFont("Segoe UI", 10, QFont.Bold),
        "header": QFont("Segoe UI", 14, QFont.Bold),
        "title": QFont("Segoe UI", 16, QFont.Bold),
        "code": QFont("Consolas", 10),
        "code_small": QFont("Consolas", 9),
        "small": QFont("Segoe UI", 8),
        "large": QFont("Segoe UI", 12),
    }

    @classmethod
    def get_animation_stylesheet(cls):
        """Get stylesheet with CSS animations support"""
        return f"""
        QWidget {{
            background-color: {cls.COLORS["bg_primary"]};
            color: {cls.COLORS["text_primary"]};
            font-family: "Segoe UI";
            font-size: 10pt;
        }}
        QWidget:focus {{
            outline: 2px solid {cls.COLORS["accent_cyan"]};
            outline-offset: 2px;
        }}
        """

    @classmethod
    def get_enhanced_button_stylesheet(cls, variant="default"):
        """Get enhanced button stylesheet with variants"""
        base = f"""
        QPushButton {{
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 10pt;
            font-weight: 500;
            min-height: 24px;
            text-align: center;
        }}
        QPushButton:disabled {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_muted"]};
            border-color: {cls.COLORS["border_inactive"]};
        }}
        """
        
        if variant == "primary":
            return base + f"""
            QPushButton {{
                background-color: {cls.COLORS["accent_cyan"]};
                color: {cls.COLORS["bg_primary"]};
                border-color: {cls.COLORS["accent_cyan"]};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #00a6c7;
            }}
            QPushButton:pressed {{
                background-color: #008fb0;
            }}
            """
        elif variant == "success":
            return base + f"""
            QPushButton {{
                background-color: {cls.COLORS["success"]};
                color: {cls.COLORS["text_primary"]};
                border-color: {cls.COLORS["success"]};
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
            QPushButton:pressed {{
                background-color: #047857;
            }}
            """
        elif variant == "danger":
            return base + f"""
            QPushButton {{
                background-color: {cls.COLORS["error"]};
                color: {cls.COLORS["text_primary"]};
                border-color: {cls.COLORS["error"]};
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
            QPushButton:pressed {{
                background-color: #b91c1c;
            }}
            """
        else:  # default
            return base + f"""
            QPushButton {{
                background-color: {cls.COLORS["bg_secondary"]};
                color: {cls.COLORS["text_primary"]};
            }}
            QPushButton:hover {{
                background-color: {cls.HOVER_COLOR};
                border-color: {cls.COLORS["accent_cyan"]};
            }}
            QPushButton:pressed {{
                background-color: {cls.PRESSED_COLOR};
            }}
            """

    @classmethod
    def get_enhanced_input_stylesheet(cls):
        """Get enhanced input field stylesheet"""
        return f"""
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_primary"]};
            selection-background-color: {cls.COLORS["accent_cyan"]};
            selection-color: {cls.COLORS["bg_primary"]};
            border: 2px solid {cls.COLORS["border_inactive"]};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 10pt;
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, 
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {cls.COLORS["border_active"]};
            background-color: {cls.COLORS["bg_secondary"]};
        }}
        QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
        QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {cls.COLORS["accent_mint"]};
        }}
        """

    @classmethod
    def get_progress_bar_stylesheet(cls):
        """Get progress bar stylesheet"""
        return f"""
        QProgressBar {{
            background-color: {cls.COLORS["bg_tertiary"]};
            border: 2px solid {cls.COLORS["border_inactive"]};
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {cls.COLORS["accent_cyan"]}, 
                stop:1 {cls.COLORS["accent_mint"]});
            border-radius: 6px;
        }}
        """

    @classmethod
    def get_slider_stylesheet(cls):
        """Get slider stylesheet"""
        return f"""
        QSlider::groove:horizontal {{
            background-color: {cls.COLORS["bg_tertiary"]};
            height: 8px;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background-color: {cls.COLORS["accent_cyan"]};
            border: 2px solid {cls.COLORS["accent_cyan"]};
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QSlider::handle:horizontal:hover {{
            background-color: {cls.COLORS["accent_mint"]};
            border-color: {cls.COLORS["accent_mint"]};
        }}
        """

    @classmethod
    def get_checkbox_stylesheet(cls):
        """Get checkbox stylesheet"""
        return f"""
        QCheckBox {{
            color: {cls.COLORS["text_primary"]};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {cls.COLORS["border_inactive"]};
            border-radius: 4px;
            background-color: {cls.COLORS["bg_tertiary"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {cls.COLORS["accent_cyan"]};
            border-color: {cls.COLORS["accent_cyan"]};
        }}
        QCheckBox::indicator:hover {{
            border-color: {cls.COLORS["accent_mint"]};
        }}
        """

    @classmethod
    def get_scrollbar_stylesheet(cls):
        """Get scrollbar stylesheet"""
        return f"""
        QScrollBar:vertical {{
            background-color: {cls.COLORS["bg_secondary"]};
            width: 12px;
            border: none;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {cls.COLORS["accent_cyan"]};
            border-radius: 6px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {cls.COLORS["accent_mint"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """

    @classmethod
    def get_status_bar_stylesheet(cls):
        """Get status bar stylesheet"""
        return f"""
        QStatusBar {{
            background-color: {cls.COLORS["bg_secondary"]};
            color: {cls.COLORS["text_secondary"]};
            border-top: 1px solid {cls.COLORS["border_inactive"]};
            font-size: 9pt;
        }}
        QStatusBar::item {{
            border: none;
        }}
        """

    @classmethod
    def get_calendar_stylesheet(cls):
        """Get calendar widget stylesheet"""
        return f"""
        QCalendarWidget {{
            background-color: {cls.COLORS["bg_primary"]};
            color: {cls.COLORS["text_primary"]};
        }}
        QCalendarWidget QToolButton {{
            background-color: {cls.COLORS["bg_secondary"]};
            color: {cls.COLORS["text_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 4px;
            padding: 4px;
        }}
        QCalendarWidget QToolButton:hover {{
            background-color: {cls.COLORS["accent_cyan"]};
            color: {cls.COLORS["bg_primary"]};
        }}
        """

    @classmethod
    def get_splitter_stylesheet(cls):
        """Get splitter stylesheet"""
        return f"""
        QSplitter::handle {{
            background-color: {cls.COLORS["border_inactive"]};
        }}
        QSplitter::handle:hover {{
            background-color: {cls.COLORS["accent_cyan"]};
        }}
        QSplitter::handle:horizontal {{
            width: 3px;
        }}
        QSplitter::handle:vertical {{
            height: 3px;
        }}
        """

    # Original methods preserved for compatibility
    @classmethod
    def heading_font(cls):
        """Heading font configuration"""
        return cls.FONTS["header"]

    @classmethod
    def label_font(cls):
        """Label font configuration"""
        return cls.FONTS["primary"]

    @classmethod
    def text_font(cls):
        """Text widget font configuration"""
        return cls.FONTS["code"]

    @classmethod
    def get_base_stylesheet(cls):
        """Get base stylesheet for the application"""
        return f"""
        QWidget {{
            background-color: {cls.COLORS["bg_primary"]};
            color: {cls.COLORS["text_primary"]};
            font-family: "Segoe UI";
            font-size: 10pt;
        }}
        """

    @classmethod
    def get_frame_stylesheet(cls):
        """Get standard frame stylesheet"""
        return f"""
        QFrame {{
            background-color: {cls.COLORS["bg_primary"]};
            border: none;
        }}
        """

    @classmethod
    def get_group_box_stylesheet(cls):
        """Get group box (labeled frame) stylesheet"""
        return f"""
        QGroupBox {{
            background-color: {cls.COLORS["bg_secondary"]};
            color: {cls.COLORS["text_primary"]};
            font-weight: bold;
            font-size: 14pt;
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 3px;
            margin-top: 10px;
            padding-top: 5px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 10px 0 10px;
        }}
        """

    @classmethod
    def get_text_edit_stylesheet(cls):
        """Get text edit widget stylesheet"""
        return f"""
        QTextEdit, QPlainTextEdit {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_primary"]};
            selection-background-color: {cls.COLORS["accent_cyan"]};
            selection-color: {cls.COLORS["bg_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 3px;
            font-family: "Consolas";
            font-size: 10pt;
        }}
        QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {cls.COLORS["border_active"]};
        }}
        """

    @classmethod
    def get_button_stylesheet(cls):
        """Get button stylesheet"""
        return cls.get_enhanced_button_stylesheet("default")

    @classmethod
    def get_action_button_stylesheet(cls):
        """Get action button stylesheet (for confirm/proceed actions)"""
        return cls.get_enhanced_button_stylesheet("primary")

    @classmethod
    def get_danger_button_stylesheet(cls):
        """Get danger button stylesheet (for destructive actions)"""
        return cls.get_enhanced_button_stylesheet("danger")

    @classmethod
    def get_line_edit_stylesheet(cls):
        """Get line edit (entry) widget stylesheet"""
        return cls.get_enhanced_input_stylesheet()

    @classmethod
    def get_list_widget_stylesheet(cls):
        """Get list widget stylesheet"""
        return f"""
        QListWidget {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_primary"]};
            selection-background-color: {cls.COLORS["accent_cyan"]};
            selection-color: {cls.COLORS["bg_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 3px;
            font-size: 10pt;
        }}
        QListWidget::item {{
            padding: 3px;
            border-bottom: 1px solid {cls.COLORS["border_inactive"]};
        }}
        QListWidget::item:selected {{
            background-color: {cls.COLORS["accent_cyan"]};
            color: {cls.COLORS["bg_primary"]};
        }}
        """

    @classmethod
    def get_combo_box_stylesheet(cls):
        """Get combo box stylesheet"""
        return f"""
        QComboBox {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 3px;
            padding: 5px;
            font-size: 10pt;
            min-width: 6em;
        }}
        QComboBox:focus {{
            border-color: {cls.COLORS["border_active"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {cls.COLORS["accent_cyan"]};
            margin-right: 5px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {cls.COLORS["bg_tertiary"]};
            color: {cls.COLORS["text_primary"]};
            selection-background-color: {cls.COLORS["accent_cyan"]};
            selection-color: {cls.COLORS["bg_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
        }}
        """

    @classmethod
    def get_tab_widget_stylesheet(cls):
        """Get tab widget stylesheet"""
        return f"""
        QTabWidget::pane {{
            background-color: {cls.COLORS["bg_primary"]};
            border: 1px solid {cls.COLORS["border_inactive"]};
            border-radius: 3px;
        }}
        QTabBar::tab {{
            background-color: {cls.COLORS["bg_secondary"]};
            color: {cls.COLORS["text_secondary"]};
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }}
        QTabBar::tab:selected {{
            background-color: {cls.COLORS["accent_cyan"]};
            color: {cls.COLORS["bg_primary"]};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: #15a6c7;
            color: {cls.COLORS["text_primary"]};
        }}
        """

    @classmethod
    def get_label_stylesheet(cls, style_type="normal"):
        """Get label stylesheet"""
        if style_type == "title":
            return f"""
            QLabel {{
                color: {cls.COLORS["accent_cyan"]};
                font-size: 16pt;
                font-weight: bold;
                background-color: transparent;
            }}
            """
        elif style_type == "section":
            return f"""
            QLabel {{
                color: {cls.COLORS["accent_coral"]};
                font-size: 12pt;
                font-weight: bold;
                background-color: transparent;
            }}
            """
        elif style_type == "info":
            return f"""
            QLabel {{
                color: {cls.COLORS["text_secondary"]};
                font-size: 10pt;
                background-color: transparent;
            }}
            """
        else:
            return f"""
            QLabel {{
                color: {cls.COLORS["text_primary"]};
                background-color: transparent;
            }}
            """

    @classmethod
    def apply_dark_theme(cls, app: QApplication):
        """Apply dark theme to the entire application"""
        app.setStyle('Fusion')
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(cls.COLORS["bg_primary"]))
        palette.setColor(QPalette.WindowText, QColor(cls.COLORS["text_primary"]))
        palette.setColor(QPalette.Base, QColor(cls.COLORS["bg_tertiary"]))
        palette.setColor(QPalette.AlternateBase, QColor(cls.COLORS["bg_secondary"]))
        palette.setColor(QPalette.ToolTipBase, QColor(cls.COLORS["bg_tertiary"]))
        palette.setColor(QPalette.ToolTipText, QColor(cls.COLORS["text_primary"]))
        palette.setColor(QPalette.Text, QColor(cls.COLORS["text_primary"]))
        palette.setColor(QPalette.Button, QColor(cls.COLORS["bg_secondary"]))
        palette.setColor(QPalette.ButtonText, QColor(cls.COLORS["text_primary"]))
        palette.setColor(QPalette.BrightText, QColor(cls.COLORS["accent_coral"]))
        palette.setColor(QPalette.Link, QColor(cls.COLORS["accent_cyan"]))
        palette.setColor(QPalette.Highlight, QColor(cls.COLORS["accent_cyan"]))
        palette.setColor(QPalette.HighlightedText, QColor(cls.COLORS["bg_primary"]))
        
        app.setPalette(palette)

    @classmethod
    def apply_icon(cls, window: QWidget):
        """Apply VoxSigil icon to window"""
        try:
            icon_path = Path(__file__).parent / "voxsigil.ico"
            if icon_path.exists():
                from PyQt5.QtGui import QIcon
                window.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass

    @classmethod
    def apply_theme(cls, widget: QWidget):
        """Apply VoxSigil theme to a specific widget"""
        try:
            widget.setStyleSheet(cls.get_complete_stylesheet())
        except Exception:
            pass

    @classmethod
    def get_complete_stylesheet(cls):
        """Get complete application stylesheet with all enhancements"""
        return f"""
        {cls.get_base_stylesheet()}
        {cls.get_frame_stylesheet()}
        {cls.get_group_box_stylesheet()}
        {cls.get_text_edit_stylesheet()}
        {cls.get_enhanced_input_stylesheet()}
        {cls.get_enhanced_button_stylesheet()}
        {cls.get_list_widget_stylesheet()}
        {cls.get_combo_box_stylesheet()}
        {cls.get_tab_widget_stylesheet()}
        {cls.get_progress_bar_stylesheet()}
        {cls.get_slider_stylesheet()}
        {cls.get_checkbox_stylesheet()}
        {cls.get_scrollbar_stylesheet()}
        {cls.get_status_bar_stylesheet()}
        {cls.get_calendar_stylesheet()}
        {cls.get_splitter_stylesheet()}
        """


class VoxSigilWidgetFactory:
    """Factory class for creating pre-styled VoxSigil widgets"""
    
    @staticmethod
    def create_button(text: str, variant: str = "default", parent=None) -> QPushButton:
        """Create a styled button"""
        btn = QPushButton(text, parent)
        btn.setStyleSheet(VoxSigilStyles.get_enhanced_button_stylesheet(variant))
        return btn
    
    @staticmethod
    def create_label(text: str, style_type: str = "normal", parent=None) -> QLabel:
        """Create a styled label"""
        label = QLabel(text, parent)
        label.setStyleSheet(VoxSigilStyles.get_label_stylesheet(style_type))
        return label
    
    @staticmethod
    def create_frame(parent=None) -> QFrame:
        """Create a styled frame"""
        frame = QFrame(parent)
        frame.setStyleSheet(VoxSigilStyles.get_frame_stylesheet())
        return frame
    
    @staticmethod
    def create_progress_bar(parent=None) -> QProgressBar:
        """Create a styled progress bar"""
        progress = QProgressBar(parent)
        progress.setStyleSheet(VoxSigilStyles.get_progress_bar_stylesheet())
        return progress
    
    @staticmethod
    def create_slider(orientation=Qt.Horizontal, parent=None) -> QSlider:
        """Create a styled slider"""
        slider = QSlider(orientation, parent)
        slider.setStyleSheet(VoxSigilStyles.get_slider_stylesheet())
        return slider
    
    @staticmethod
    def create_checkbox(text: str, parent=None) -> QCheckBox:
        """Create a styled checkbox"""
        checkbox = QCheckBox(text, parent)
        checkbox.setStyleSheet(VoxSigilStyles.get_checkbox_stylesheet())
        return checkbox
    
    @staticmethod
    def create_status_bar(parent=None) -> QStatusBar:
        """Create a styled status bar"""
        status_bar = QStatusBar(parent)
        status_bar.setStyleSheet(VoxSigilStyles.get_status_bar_stylesheet())
        return status_bar
    
    @staticmethod
    def create_scroll_area(parent=None) -> QScrollArea:
        """Create a styled scroll area"""
        scroll = QScrollArea(parent)
        scroll.setStyleSheet(VoxSigilStyles.get_scrollbar_stylesheet())
        return scroll
    
    @staticmethod
    def create_splitter(orientation=Qt.Horizontal, parent=None) -> QSplitter:
        """Create a styled splitter"""
        splitter = QSplitter(orientation, parent)
        splitter.setStyleSheet(VoxSigilStyles.get_splitter_stylesheet())
        return splitter
    
    @staticmethod
    def create_calendar(parent=None) -> QCalendarWidget:
        """Create a styled calendar widget"""
        calendar = QCalendarWidget(parent)
        calendar.setStyleSheet(VoxSigilStyles.get_calendar_stylesheet())
        return calendar


class VoxSigilThemeManager:
    """Advanced theme management with dynamic switching capabilities"""
    
    def __init__(self):
        self.current_theme = "dark"
        self.custom_themes = {}
        
    def register_custom_theme(self, name: str, colors: Dict[str, str]):
        """Register a custom color theme"""
        self.custom_themes[name] = colors
    
    def apply_theme(self, app: QApplication, theme_name: str = "dark"):
        """Apply a theme to the application"""
        if theme_name == "dark":
            VoxSigilStyles.apply_dark_theme(app)
        elif theme_name in self.custom_themes:
            # Apply custom theme
            original_colors = VoxSigilStyles.COLORS.copy()
            VoxSigilStyles.COLORS.update(self.custom_themes[theme_name])
            VoxSigilStyles.apply_dark_theme(app)
            app.setStyleSheet(VoxSigilStyles.get_complete_stylesheet())
            VoxSigilStyles.COLORS = original_colors
        
        self.current_theme = theme_name
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        return ["dark"] + list(self.custom_themes.keys())


def bind_agent_buttons(parent_widget: QWidget, registry, meta_path: str = "agents.json") -> None:
    """Enhanced agent button binding with improved styling"""
    if not parent_widget or not registry:
        return

    meta: Dict[str, Dict] = {}
    try:
        with open(meta_path, "r") as f:
            for entry in json.load(f):
                meta[entry.get("name")] = entry
    except Exception:
        meta = {}

    # Create enhanced frame for agent buttons
    frame = VoxSigilWidgetFactory.create_frame(parent_widget)
    frame.setStyleSheet(f"""
        QFrame {{
            background-color: {VoxSigilStyles.COLORS["bg_secondary"]};
            border: 1px solid {VoxSigilStyles.COLORS["border_inactive"]};
            border-radius: 6px;
            padding: 5px;
        }}
    """)
    
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(8)
    
    # Add enhanced "Agents" label
    agents_label = VoxSigilWidgetFactory.create_label("🤖 Agents", "section")
    layout.addWidget(agents_label)
    
    # Add agent buttons with enhanced styling
    for agent_name, agent in registry.get_all_agents():
        if not hasattr(agent, "on_gui_call"):
            continue

        meta_entry = meta.get(agent_name, {})
        invocations = getattr(agent, "invocations", [])
        label = invocations[0] if invocations else f"Invoke {agent_name}"

        btn = VoxSigilWidgetFactory.create_button(label, "primary")
        btn.clicked.connect(lambda checked, a=agent: a.on_gui_call())
        
        # Enhanced tooltip
        tooltip_text = f"Agent: {agent_name}\nClass: {meta_entry.get('class', 'Unknown')}\nTags: {', '.join(getattr(agent, 'tags', []))}"
        tooltip = AnimatedToolTip(btn, tooltip_text)
        
        layout.addWidget(btn)
    
    layout.addStretch()
    
    # Add frame to parent widget's layout if it has one
    if hasattr(parent_widget, 'layout') and parent_widget.layout():
        parent_widget.layout().addWidget(frame)


class VoxSigilGUIUtils:
    """Enhanced PyQt5 GUI utility functions for VoxSigil"""
    
    @staticmethod
    def create_animated_tooltip(widget: QWidget, text: str, delay: int = 1000) -> AnimatedToolTip:
        """Create an animated tooltip for a widget"""
        return AnimatedToolTip(widget, text, delay)
    
    @staticmethod
    def set_widget_tooltip(widget: QWidget, text: str):
        """Set a simple tooltip for a widget using PyQt5's built-in tooltip"""
        widget.setToolTip(text)
    
    @staticmethod
    def create_notification_frame(parent=None, message: str = "", variant: str = "info") -> QFrame:
        """Create a notification frame with different variants"""
        frame = QFrame(parent)
        
        color_map = {
            "info": VoxSigilStyles.COLORS["info"],
            "success": VoxSigilStyles.COLORS["success"],
            "warning": VoxSigilStyles.COLORS["warning"],
            "error": VoxSigilStyles.COLORS["error"]
        }
        
        bg_color = color_map.get(variant, VoxSigilStyles.COLORS["info"])
        
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                color: {VoxSigilStyles.COLORS["text_primary"]};
                border-radius: 6px;
                padding: 12px;
                font-weight: 500;
            }}
        """)
        
        layout = QHBoxLayout(frame)
        label = QLabel(message)
        layout.addWidget(label)
        
        return frame
    
    @staticmethod
    def create_card_widget(parent=None, title: str = "", content: str = "") -> QFrame:
        """Create a card-style widget"""
        card = QFrame(parent)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {VoxSigilStyles.COLORS["bg_secondary"]};
                border: 1px solid {VoxSigilStyles.COLORS["border_inactive"]};
                border-radius: 8px;
                padding: 16px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        
        if title:
            title_label = VoxSigilWidgetFactory.create_label(title, "section")
            layout.addWidget(title_label)
        
        if content:
            content_label = VoxSigilWidgetFactory.create_label(content)
            layout.addWidget(content_label)
        
        return card
    
    @staticmethod
    def create_loading_widget(parent=None, text: str = "Loading...") -> QWidget:
        """Create a loading widget with progress bar"""
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        
        label = VoxSigilWidgetFactory.create_label(text, "info")
        progress = VoxSigilWidgetFactory.create_progress_bar()
        progress.setRange(0, 0)  # Indeterminate progress
        
        layout.addWidget(label)
        layout.addWidget(progress)
        
        return widget
    
    @staticmethod
    def setup_window_properties(window: QWidget, title: str = "VoxSigil Application"):
        """Setup standard window properties"""
        window.setWindowTitle(title)
        VoxSigilStyles.apply_icon(window)
        VoxSigilStyles.apply_theme(window)
        window.setMinimumSize(800, 600)
    
    @staticmethod
    def create_action_toolbar(parent=None, actions: List[Dict[str, Any]] = None) -> QFrame:
        """Create an action toolbar with buttons"""
        toolbar = QFrame(parent)
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {VoxSigilStyles.COLORS["bg_secondary"]};
                border-bottom: 1px solid {VoxSigilStyles.COLORS["border_inactive"]};
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(toolbar)
        layout.setSpacing(8)
        
        actions = actions or []
        for action in actions:
            btn = VoxSigilWidgetFactory.create_button(
                action.get("text", "Action"),
                action.get("variant", "default")
            )
            if action.get("callback"):
                btn.clicked.connect(action["callback"])
            layout.addWidget(btn)
        
        layout.addStretch()
        return toolbar
    
    @staticmethod
    def create_info_panel(parent=None, info_items: List[Dict[str, str]] = None) -> QFrame:
        """Create an information panel with key-value pairs"""
        panel = QFrame(parent)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {VoxSigilStyles.COLORS["bg_tertiary"]};
                border: 1px solid {VoxSigilStyles.COLORS["border_inactive"]};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        
        info_items = info_items or []
        for item in info_items:
            key_label = VoxSigilWidgetFactory.create_label(f"{item.get('key', '')}:", "info")
            value_label = VoxSigilWidgetFactory.create_label(item.get('value', ''))
            
            row_layout = QHBoxLayout()
            row_layout.addWidget(key_label)
            row_layout.addWidget(value_label)
            row_layout.addStretch()
            
            layout.addLayout(row_layout)
        
        return panel
    
    @staticmethod
    def apply_hover_effects(widget: QWidget):
        """Apply hover effects to a widget"""
        original_style = widget.styleSheet()
        
        def on_enter():
            widget.setStyleSheet(original_style + f"""
                QWidget {{
                    background-color: {VoxSigilStyles.HOVER_COLOR};
                    border-color: {VoxSigilStyles.COLORS["accent_cyan"]};
                }}
            """)
        
        def on_leave():
            widget.setStyleSheet(original_style)
        
        widget.enterEvent = lambda event: on_enter()
        widget.leaveEvent = lambda event: on_leave()
    
    @staticmethod
    def create_responsive_layout(parent=None, min_column_width: int = 300) -> QWidget:
        """Create a responsive layout that adapts to window size"""
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        
        # This would need additional logic for true responsiveness
        # For now, it's a placeholder for future enhancement
        return container


# Initialize theme manager singleton
theme_manager = VoxSigilThemeManager()
