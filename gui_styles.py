#!/usr/bin/env python3
"""
VoxSigil GUI Styles Module
Centralized styling management for the Dynamic GridFormer GUI

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Provide consistent VoxSigil aesthetic across all GUI components
"""

import tkinter as tk
from tkinter import ttk


class VoxSigilStyles:
    """VoxSigil GUI styling management"""  # VoxSigil Color Palette

    COLORS = {
        "bg_primary": "#1a1a2e",  # Deep navy background
        "bg_secondary": "#16213e",  # Slightly lighter navy
        "bg_tertiary": "#0f1419",  # Dark code background
        "accent_cyan": "#00d4ff",  # Bright cyan
        "accent_coral": "#ff6b6b",  # Coral red
        "accent_mint": "#4ecdc4",  # Mint green
        "accent_gold": "#ffd93d",  # Gold highlights
        "text_primary": "#ffffff",  # White text
        "text_secondary": "#b0bec5",  # Light gray text
        "border_active": "#00d4ff",  # Active border color
        "border_inactive": "#3a4750",  # Inactive border color
    }

    # Additional color constants for backwards compatibility
    ALT_ROW_COLOR = "#2a2a3e"

    # Font configurations - direct access
    LABEL_FONT = ("Segoe UI", 10)
    LABEL_FONT_BOLD = ("Segoe UI", 10, "bold")
    HEADER_FONT = ("Segoe UI", 14, "bold")
    TEXT_FONT = ("Consolas", 10)

    @classmethod
    def heading_font(cls):
        """Heading font configuration"""
        return ("Segoe UI", 14, "bold")

    @classmethod
    def label_font(cls):
        """Label font configuration"""
        return ("Segoe UI", 10)

    @classmethod
    def text_font(cls):
        """Text widget font configuration"""
        return ("Consolas", 10)

    @classmethod
    def get_frame_config(cls):
        """Get standard frame configuration"""
        return {"bg": cls.COLORS["bg_primary"], "relief": "flat", "borderwidth": 0}

    @classmethod
    def get_label_frame_config(cls, text=""):
        """Get labeled frame configuration"""
        return {
            "bg": cls.COLORS["bg_secondary"],
            "fg": cls.COLORS["text_primary"],
            "text": text,
            "font": cls.heading_font,
            "relief": "solid",
            "borderwidth": 1,
        }

    @classmethod
    def get_text_widget_config(cls):
        """Get text widget configuration"""
        return {
            "bg": cls.COLORS["bg_tertiary"],
            "fg": cls.COLORS["text_primary"],
            "insertbackground": cls.COLORS["accent_cyan"],
            "selectbackground": cls.COLORS["accent_cyan"],
            "selectforeground": cls.COLORS["bg_primary"],
            "font": cls.text_font,
            "relief": "solid",
            "borderwidth": 1,
            "wrap": "word",
        }

    @classmethod
    def get_button_config(cls):
        """Get button configuration"""
        return {
            "bg": cls.COLORS["bg_secondary"],
            "fg": cls.COLORS["text_primary"],
            "activebackground": cls.COLORS["accent_cyan"],
            "activeforeground": cls.COLORS["bg_primary"],
            "font": cls.label_font,
            "relief": "solid",
            "borderwidth": 1,
            "cursor": "hand2",
        }

    @classmethod
    def get_entry_config(cls):
        """Get entry widget configuration"""
        return {
            "bg": cls.COLORS["bg_tertiary"],
            "fg": cls.COLORS["text_primary"],
            "insertbackground": cls.COLORS["accent_cyan"],
            "selectbackground": cls.COLORS["accent_cyan"],
            "selectforeground": cls.COLORS["bg_primary"],
            "font": cls.label_font,
            "relief": "solid",
            "borderwidth": 1,
        }

    @classmethod
    def get_listbox_config(cls):
        """Get listbox configuration"""
        return {
            "bg": cls.COLORS["bg_tertiary"],
            "fg": cls.COLORS["text_primary"],
            "selectbackground": cls.COLORS["accent_cyan"],
            "selectforeground": cls.COLORS["bg_primary"],
            "font": cls.label_font,
            "relief": "solid",
            "borderwidth": 1,
        }

    @classmethod
    def setup_styles(cls):
        """Setup all custom styles for the application"""
        style = ttk.Style()
        style.theme_use("clam")

        # Title styles
        style.configure(
            "Title.TLabel",
            foreground=cls.COLORS["accent_cyan"],
            background=cls.COLORS["bg_primary"],
            font=("Segoe UI", 16, "bold"),
        )

        # Section header styles
        style.configure(
            "Section.TLabel",
            foreground=cls.COLORS["accent_coral"],
            background=cls.COLORS["bg_primary"],
            font=("Segoe UI", 12, "bold"),
        )

        # Info text styles
        style.configure(
            "Info.TLabel",
            foreground=cls.COLORS["text_secondary"],
            background=cls.COLORS["bg_primary"],
            font=("Segoe UI", 10),
        )

        # Button styles
        style.configure(
            "TButton",
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["bg_secondary"],
            borderwidth=1,
            focusthickness=3,
            focuscolor=cls.COLORS["accent_cyan"],
            font=("Segoe UI", 10),
        )

        style.map(
            "TButton",
            background=[
                ("active", cls.COLORS["accent_cyan"]),
                ("pressed", cls.COLORS["accent_cyan"]),
            ],
            foreground=[
                ("active", cls.COLORS["bg_primary"]),
                ("pressed", cls.COLORS["bg_primary"]),
            ],
        )

        # Action button - for confirm/proceed actions
        style.configure(
            "Action.TButton",
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["accent_cyan"],
            borderwidth=1,
            focusthickness=3,
            focuscolor=cls.COLORS["accent_cyan"],
            font=("Segoe UI", 10, "bold"),
        )

        style.map(
            "Action.TButton", background=[("active", "#00a6c7"), ("pressed", "#008fb0")]
        )

        # Danger button - for destructive actions
        style.configure(
            "Danger.TButton",
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["accent_coral"],
            borderwidth=1,
            focusthickness=3,
            focuscolor=cls.COLORS["accent_coral"],
            font=("Segoe UI", 10, "bold"),
        )

        style.map(
            "Danger.TButton", background=[("active", "#f04c4c"), ("pressed", "#d64141")]
        )

        # Entry styles
        style.configure(
            "TEntry",
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["bg_tertiary"],
            fieldbackground=cls.COLORS["bg_tertiary"],
            borderwidth=1,
            bordercolor=cls.COLORS["border_inactive"],
            lightcolor=cls.COLORS["border_inactive"],
            darkcolor=cls.COLORS["border_inactive"],
            insertcolor=cls.COLORS["accent_cyan"],
        )

        style.map(
            "TEntry",
            bordercolor=[("focus", cls.COLORS["border_active"])],
            lightcolor=[("focus", cls.COLORS["border_active"])],
            darkcolor=[("focus", cls.COLORS["border_active"])],
        )

        # Combobox styles
        style.configure(
            "TCombobox",
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["bg_tertiary"],
            fieldbackground=cls.COLORS["bg_tertiary"],
            selectbackground=cls.COLORS["accent_cyan"],
            selectforeground=cls.COLORS["bg_primary"],
            borderwidth=1,
            arrowcolor=cls.COLORS["accent_cyan"],
        )

        # Frame styles
        style.configure("TFrame", background=cls.COLORS["bg_primary"])

        # Panel styles (for grouping content)
        style.configure(
            "Panel.TFrame",
            background=cls.COLORS["bg_secondary"],
            borderwidth=1,
            relief="solid",
        )

        # Code frame style for showing source code
        style.configure(
            "Code.TFrame",
            background=cls.COLORS["bg_tertiary"],
            borderwidth=1,
            relief="solid",
        )

        # Notebook styles
        style.configure(
            "TNotebook",
            background=cls.COLORS["bg_primary"],
            tabmargins=[2, 5, 2, 0],
            borderwidth=0,
        )

        style.configure(
            "TNotebook.Tab",
            background=cls.COLORS["bg_secondary"],
            foreground=cls.COLORS["text_secondary"],
            padding=[10, 5],
            font=("Segoe UI", 9),
        )

        style.map(
            "TNotebook.Tab",
            background=[("selected", cls.COLORS["accent_cyan"]), ("active", "#15a6c7")],
            foreground=[
                ("selected", cls.COLORS["bg_primary"]),
                ("active", cls.COLORS["text_primary"]),
            ],
            expand=[("selected", [1, 1, 1, 0])],
        )

    @classmethod
    def apply_dark_theme(cls, root: tk.Tk):
        """Apply dark theme to the root window and all its children"""
        root.configure(background=cls.COLORS["bg_primary"])
        cls.setup_styles()

        # Configure Text widget style
        text_style = {
            "background": cls.COLORS["bg_tertiary"],
            "foreground": cls.COLORS["text_primary"],
            "insertbackground": cls.COLORS["accent_cyan"],  # Cursor color
            "selectbackground": cls.COLORS["accent_cyan"],
            "selectforeground": cls.COLORS["bg_primary"],
            "borderwidth": 0,
            "highlightthickness": 1,
            "highlightbackground": cls.COLORS["border_inactive"],
            "highlightcolor": cls.COLORS["border_active"],
            "font": ("Consolas", 10),
        }

        # Apply the style to all Text widgets
        # This is used for child windows created later
        root.option_add("*Text.background", text_style["background"])
        root.option_add("*Text.foreground", text_style["foreground"])
        root.option_add("*Text.insertBackground", text_style["insertbackground"])
        root.option_add("*Text.selectBackground", text_style["selectbackground"])
        root.option_add("*Text.selectForeground", text_style["selectforeground"])
        root.option_add("*Text.borderWidth", text_style["borderwidth"])
        root.option_add("*Text.highlightThickness", text_style["highlightthickness"])
        root.option_add("*Text.highlightBackground", text_style["highlightbackground"])
        root.option_add("*Text.highlightColor", text_style["highlightcolor"])
        root.option_add("*Text.font", text_style["font"])

        # Listbox style
        root.option_add("*Listbox.background", cls.COLORS["bg_tertiary"])
        root.option_add("*Listbox.foreground", cls.COLORS["text_primary"])
        root.option_add("*Listbox.selectBackground", cls.COLORS["accent_cyan"])
        root.option_add("*Listbox.selectForeground", cls.COLORS["bg_primary"])

    @classmethod
    def apply_icon(cls, window):
        """Apply VoxSigil icon to window (placeholder implementation)"""
        try:
            # Try to set a default icon if available
            # This is a placeholder - in a real implementation you'd load an actual icon file
            pass
        except Exception:
            # If icon loading fails, just continue without icon
            pass

    @classmethod
    def apply_theme(cls, window):
        """Apply VoxSigil theme to window"""
        try:
            window.configure(bg=cls.COLORS["bg_primary"])
        except Exception:
            # If theme application fails, just continue
            pass
