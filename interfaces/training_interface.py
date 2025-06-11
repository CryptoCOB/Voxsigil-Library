"""
VoxSigil Training Interface Module
Encapsulated training functionality for the Dynamic GridFormer GUI
"""

import tkinter as tk


class VoxSigilTrainingInterface:
    """
    Legacy training interface for VoxSigil Dynamic GridFormer.
    This panel is now deprecated. Training is managed via UnifiedVantaCore and the main GUI.
    """

    def __init__(self, parent_gui, notebook):
        """
        Initialize the training interface

        Args:
            parent_gui: Reference to the main GUI class
            notebook: ttk.Notebook to add the training tab to
        """
        self.parent_gui = parent_gui
        self.notebook = notebook

        # Replace the training tab with a deprecation message
        training_frame = tk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🔥 Training (Legacy)")
        msg = tk.Label(
            training_frame,
            text="Training is now managed via UnifiedVantaCore.\nPlease use the main GUI for all training and orchestration.",
            font=("Segoe UI", 12, "bold"),
            fg="#ffaa00",
            bg="#222233",
            justify=tk.CENTER,
            wraplength=500,
        )
        msg.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
