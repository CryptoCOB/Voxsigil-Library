#!/usr/bin/env python3
"""
VoxSigil Model Discovery Tab Interface
Modular component for model discovery and selection functionality

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Encapsulated model discovery interface for Dynamic GridFormer GUI
"""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .gui_styles import VoxSigilStyles


class VoxSigilModelInterface:
    """Model discovery and selection interface for VoxSigil Dynamic GridFormer"""

    def __init__(self, parent_gui, notebook):
        """
        Initialize the model interface

        Args:
            parent_gui: Reference to the main GUI class
            notebook: ttk.Notebook to add the model tab to
        """
        self.parent_gui = parent_gui
        self.notebook = notebook

        # Model state
        self.discovered_models = {}
        self.current_model = None
        self.current_model_path = None

        # Create the model tab
        self.create_model_tab()

        # Discover models on initialization
        self.discover_models()

    def create_model_tab(self):
        """Create the model discovery and selection tab"""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="🔍 Model Discovery")

        # Main container
        main_container = tk.Frame(model_frame, **VoxSigilStyles.get_frame_config())
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Model discovery controls
        self._create_discovery_panel(main_container)

        # Right panel: Model information and selection
        self._create_selection_panel(main_container)

        # Bottom panel: Agent–Component Relationships Graph
        graph_frame = tk.LabelFrame(
            model_frame,
            text="Agent–Component Relationships",
            **VoxSigilStyles.get_label_frame_config("Relationships Graph"),
        )
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.relationships_canvas = tk.Canvas(graph_frame, bg="#1a1a2e", height=200)
        self.relationships_canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            graph_frame,
            text="🔄 Refresh Graph",
            command=self._draw_relationship_graph,
            style="VoxSigil.TButton",
        ).pack(pady=5)
        # Initial draw
        self._draw_relationship_graph()

    def _create_discovery_panel(self, parent):
        """Create the model discovery panel"""
        discovery_panel = tk.LabelFrame(
            parent, **VoxSigilStyles.get_label_frame_config("Model Discovery")
        )
        discovery_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Discovery controls
        controls_frame = tk.Frame(discovery_panel, **VoxSigilStyles.get_frame_config())
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Auto-discover button
        ttk.Button(
            controls_frame,
            text="🔍 Auto-Discover Models",
            command=self.discover_models,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        # Manual model loading
        ttk.Button(
            controls_frame,
            text="📂 Load Custom Model",
            command=self.load_custom_model,
            style="VoxSigil.TButton",
        ).pack(side=tk.LEFT, padx=5)

        # Model list
        list_frame = tk.Frame(discovery_panel, **VoxSigilStyles.get_frame_config())
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(list_frame, text="Discovered Models:", style="Section.TLabel").pack(
            anchor=tk.W
        )

        # Model listbox with scrollbar
        listbox_frame = tk.Frame(list_frame, **VoxSigilStyles.get_frame_config())
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.model_listbox = tk.Listbox(
            listbox_frame,
            **VoxSigilStyles.get_text_widget_config(),
            selectmode=tk.SINGLE,
            height=10,
        )

        scrollbar = ttk.Scrollbar(
            listbox_frame, orient=tk.VERTICAL, command=self.model_listbox.yview
        )
        self.model_listbox.configure(yscrollcommand=scrollbar.set)

        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        # Load model button
        ttk.Button(
            list_frame,
            text="⚡ Load Selected Model",
            command=self.load_selected_model,
            style="VoxSigil.TButton",
        ).pack(pady=10)

    def _create_selection_panel(self, parent):
        """Create the model selection and info panel"""
        selection_panel = tk.LabelFrame(
            parent, **VoxSigilStyles.get_label_frame_config("Model Information")
        )
        selection_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Current model status
        status_frame = tk.Frame(selection_panel, **VoxSigilStyles.get_frame_config())
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(status_frame, text="Current Model:", style="Section.TLabel").pack(
            anchor=tk.W
        )
        self.current_model_label = ttk.Label(
            status_frame, text="None loaded", style="Info.TLabel"
        )
        self.current_model_label.pack(anchor=tk.W)

        # Model details
        details_frame = tk.Frame(selection_panel, **VoxSigilStyles.get_frame_config())
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(details_frame, text="Model Details:", style="Section.TLabel").pack(
            anchor=tk.W
        )

        self.model_details_text = tk.Text(
            details_frame, **VoxSigilStyles.get_text_widget_config(), height=15
        )
        details_scroll = ttk.Scrollbar(
            details_frame, orient=tk.VERTICAL, command=self.model_details_text.yview
        )
        self.model_details_text.configure(yscrollcommand=details_scroll.set)

        self.model_details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize with placeholder text
        self.model_details_text.insert(1.0, "Select a model to view details...")
        self.model_details_text.config(state=tk.DISABLED)

    def _draw_relationship_graph(self):
        """Draw a simple graph of registered agents and their capabilities"""
        self.relationships_canvas.delete("all")
        # Fetch agents and their capabilities from supervisor integration
        agents = []
        if (
            hasattr(self.parent_gui, "voxsigil_integration")
            and self.parent_gui.voxsigil_integration
        ):
            try:
                registry = (
                    self.parent_gui.voxsigil_integration.unified_core.agent_registry
                )
                # Fetch agent types and statuses
                for name, agent in registry.get_all_agents():
                    meta = registry.agents.get(name, {}).get("metadata", {})
                    caps = meta.get("capabilities", [])
                    agent_type = meta.get("type", "Unknown")
                    status = meta.get("status", "Inactive")
                    agents.append((name, caps, agent_type, status))
            except Exception:
                agents = []
        # Layout agents vertically
        x0, y0, dy = 50, 20, 30
        for idx, (name, caps, agent_type, status) in enumerate(agents):
            y = y0 + idx * dy
            # Draw agent node
            self.relationships_canvas.create_oval(
                x0, y, x0 + 20, y + 20, outline="#fff", width=2
            )
            cap_str = ", ".join(caps) if caps else "(no caps)"
            self.relationships_canvas.create_text(
                x0 + 30,
                y + 10,
                anchor=tk.W,
                text=f"{name} ({agent_type}, {status}): {cap_str}",
                fill="#fff",
            )
        if not agents:
            self.relationships_canvas.create_text(
                x0, y0, anchor=tk.NW, text="No agents registered", fill="#888"
            )

    def discover_models(self):
        """Discover available models in the workspace"""
        self.discovered_models.clear()
        self.model_listbox.delete(0, tk.END)

        # Search patterns for model files
        search_paths = [
            self.parent_gui.PROJECT_ROOT / "models",
            self.parent_gui.PROJECT_ROOT / "active" / "core",
            self.parent_gui.PROJECT_ROOT / "enhanced_training_project",
            Path.cwd(),
        ]

        model_extensions = [".pt", ".pth", ".ckpt", ".model"]

        for search_path in search_paths:
            if search_path.exists():
                for ext in model_extensions:
                    for model_file in search_path.rglob(f"*{ext}"):
                        if model_file.is_file():
                            model_name = model_file.stem
                            self.discovered_models[model_name] = str(model_file)
                            self.model_listbox.insert(
                                tk.END, f"{model_name} ({model_file.name})"
                            )

        # Update model details
        self.update_model_details(
            f"Discovered {len(self.discovered_models)} models:\\n\\n"
            + "\\n".join(
                [f"• {name}: {path}" for name, path in self.discovered_models.items()]
            )
        )

        # Add status update to parent GUI
        if hasattr(self.parent_gui, "update_status"):
            self.parent_gui.update_status(
                f"Discovered {len(self.discovered_models)} models"
            )

    def load_custom_model(self):
        """Load a custom model file"""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch Models", "*.pt *.pth"),
                ("Checkpoint Files", "*.ckpt"),
                ("All Model Files", "*.pt *.pth *.ckpt *.model"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            model_name = Path(file_path).stem
            self.discovered_models[model_name] = file_path
            self.model_listbox.insert(tk.END, f"{model_name} (custom)")

            if hasattr(self.parent_gui, "update_status"):
                self.parent_gui.update_status(f"Added custom model: {model_name}")

    def on_model_select(self, event):
        """Handle model selection from listbox"""
        selection = self.model_listbox.curselection()
        if selection:
            selected_text = self.model_listbox.get(selection[0])
            # Extract model name from display text
            model_name = selected_text.split(" (")[0]

            if model_name in self.discovered_models:
                model_path = self.discovered_models[model_name]
                self.show_model_info(model_name, model_path)

    def show_model_info(self, model_name, model_path):
        """Display detailed information about selected model"""
        from pathlib import Path

        import torch

        details = f"Model: {model_name}\\n"
        details += f"Path: {model_path}\\n"
        details += (
            f"File Size: {Path(model_path).stat().st_size / (1024 * 1024):.2f} MB\\n\\n"
        )

        try:
            # Try to load model metadata
            if model_path.endswith((".pt", ".pth")):
                checkpoint = torch.load(model_path, map_location="cpu")

                if isinstance(checkpoint, dict):
                    details += "Checkpoint Contents:\\n"
                    for key in checkpoint.keys():
                        if key == "model_state_dict":
                            details += (
                                f"  • {key}: {len(checkpoint[key])} parameters\\n"
                            )
                        elif key == "optimizer_state_dict":
                            details += f"  • {key}: Optimizer state\\n"
                        else:
                            details += f"  • {key}: {type(checkpoint[key]).__name__}\\n"

                    # Show some model architecture info if available
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        details += f"\\nModel Parameters: {len(state_dict)} layers\\n"
                        details += "Layer Names:\\n"
                        for i, layer_name in enumerate(list(state_dict.keys())[:10]):
                            details += f"  • {layer_name}\\n"
                        if len(state_dict) > 10:
                            details += (
                                f"  ... and {len(state_dict) - 10} more layers\\n"
                            )
                else:
                    details += f"Direct model object: {type(checkpoint).__name__}\\n"

        except Exception as e:
            details += f"\\nError loading model info: {str(e)}\\n"

        self.update_model_details(details)

    def update_model_details(self, details):
        """Update the model details text widget"""
        self.model_details_text.config(state=tk.NORMAL)
        self.model_details_text.delete(1.0, tk.END)
        self.model_details_text.insert(1.0, details)
        self.model_details_text.config(state=tk.DISABLED)

    def load_selected_model(self):
        """Load the currently selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to load.")
            return

        selected_text = self.model_listbox.get(selection[0])
        model_name = selected_text.split(" (")[0]

        if model_name in self.discovered_models:
            model_path = self.discovered_models[model_name]

            try:
                # Use parent GUI's model loader
                success = self.parent_gui.model_loader.load_model(model_path)

                if success:
                    self.current_model = model_name
                    self.current_model_path = model_path
                    self.current_model_label.config(text=f"{model_name}")

                    # Update parent GUI state
                    self.parent_gui.current_model = self.parent_gui.model_loader.model
                    self.parent_gui.current_model_path = model_path

                    if hasattr(self.parent_gui, "update_status"):
                        self.parent_gui.update_status(f"Loaded model: {model_name}")

                    messagebox.showinfo(
                        "Success", f"Model '{model_name}' loaded successfully!"
                    )

                else:
                    messagebox.showerror(
                        "Error", f"Failed to load model '{model_name}'"
                    )

            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
        else:
            messagebox.showerror(
                "Error", "Selected model not found in discovered models."
            )

    def get_current_model_info(self):
        """Get information about the currently loaded model"""
        return {
            "name": self.current_model,
            "path": self.current_model_path,
            "model_object": getattr(self.parent_gui, "current_model", None),
        }
