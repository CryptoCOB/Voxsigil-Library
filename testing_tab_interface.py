#!/usr/bin/env python3
"""
VoxSigil Testing Tab Interface
Modular component for model testing functionality

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Encapsulated testing interface for Dynamic GridFormer GUI
"""

import tkinter as tk


class VoxSigilTestingInterface:
    """
    Legacy testing interface for VoxSigil Dynamic GridFormer.
    This panel is now deprecated. Testing is managed via UnifiedVantaCore and the main GUI.
    """

    def __init__(self, parent_gui, parent_frame, *args, **kwargs):
        self.parent_gui = parent_gui
        self.parent_frame = parent_frame

        # Replace the testing tab with a deprecation message
        msg = tk.Label(
            self.parent_frame,
            text=(
                "Testing is now managed via UnifiedVantaCore.\n"
                "Please use the main GUI for all testing and orchestration."
            ),
            font=("Segoe UI", 12, "bold"),
            fg="#ffaa00",
            bg="#222233",
            justify=tk.CENTER,
            wraplength=500,
        )
        msg.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
