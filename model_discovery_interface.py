"""
Model Discovery Interface Module
Extracted model discovery functionality for the Dynamic GridFormer GUI
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .gui_styles import VoxSigilStyles


class ModelDiscoveryInterface:
    """Model discovery and loading interface"""

    def __init__(self, parent_gui, model_loader=None, model_selected_callback=None):
        self.parent_gui = parent_gui
        self.model_loader = model_loader
        self.model_selected_callback = model_selected_callback

        # Since we're now getting the frame directly, we need to detect how we're being called
        # If parent_gui is a frame (has 'pack' method), use it directly
        # If parent_gui is the GUI instance, create our own tab
        if hasattr(parent_gui, "pack"):  # It's a frame
            self.model_frame = parent_gui
            self.notebook = None
            self.create_model_interface_in_frame()
        else:  # It's the GUI instance, get the notebook from it
            self.notebook = getattr(parent_gui, "notebook", None)
            if self.notebook is not None:
                self.create_model_tab()
            else:
                print("Warning: No notebook available for ModelDiscoveryInterface")

    def create_model_interface_in_frame(self):
        """Create the model discovery interface in the provided frame"""
        if not hasattr(self, "model_frame") or self.model_frame is None:
            print("Warning: No frame available for ModelDiscoveryInterface")
            return

        # Apply VoxSigil styling
        styles = VoxSigilStyles()

        # Header with icon
        header_frame = ttk.Frame(self.model_frame)
        header_frame.pack(pady=10, padx=20, fill="x")
        header_label = ttk.Label(
            header_frame,
            text="🧠 GridFormer Model Discovery",
            font=styles.heading_font(),
        )
        header_label.pack(side="left", padx=10)

        # Model discovery section
        discovery_frame = ttk.LabelFrame(self.model_frame, text="Model Discovery")
        discovery_frame.pack(pady=10, padx=20, fill="x")

        # Directory selection
        dir_frame = ttk.Frame(discovery_frame)
        dir_frame.pack(pady=5, padx=10, fill="x")

        ttk.Label(dir_frame, text="Model Directory:").pack(side="left")
        self.model_dir_var = tk.StringVar(value="./models")
        dir_entry = ttk.Entry(dir_frame, textvariable=self.model_dir_var, width=50)
        dir_entry.pack(side="left", padx=(10, 5), expand=True, fill="x")

        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_model_dir)
        browse_btn.pack(side="right")

        # Scan button
        scan_btn = ttk.Button(
            discovery_frame, text="Scan for Models", command=self.scan_models
        )
        scan_btn.pack(pady=10)

        # Model list
        list_frame = ttk.LabelFrame(self.model_frame, text="Available Models")
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Create treeview for model list
        columns = ("Name", "Type", "Size", "Path")
        self.model_tree = ttk.Treeview(list_frame, columns=columns, show="headings")

        # Define column headings
        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.model_tree.yview
        )
        self.model_tree.configure(yscrollcommand=scrollbar.set)

        self.model_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load button
        load_btn = ttk.Button(
            list_frame, text="Load Selected Model", command=self.load_selected_model
        )
        load_btn.pack(pady=10)

    def create_model_tab(self):
        """Create the model discovery and selection tab"""
        if self.notebook is None:
            print("Warning: No notebook available for ModelDiscoveryInterface")
            return

        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="🔍 Model Discovery")

        # Apply VoxSigil styling
        styles = VoxSigilStyles()
        # Header with icon
        header_frame = ttk.Frame(model_frame)
        header_frame.pack(pady=10, padx=20, fill="x")
        header_label = ttk.Label(
            header_frame,
            text="🧠 GridFormer Model Discovery",
            font=styles.heading_font(),
        )
        header_label.pack(side="left", padx=10)

        # Model discovery section
        discovery_frame = ttk.LabelFrame(
            model_frame, text="Discover Models", padding=15
        )
        discovery_frame.pack(padx=20, pady=10, fill="both")

        # Model directory selection
        dir_frame = ttk.Frame(discovery_frame)
        dir_frame.pack(fill="x", pady=5)

        ttk.Label(dir_frame, text="Model Directory:").pack(side="left", padx=5)

        self.model_dir_var = tk.StringVar()
        model_dir_entry = ttk.Entry(
            dir_frame, textvariable=self.model_dir_var, width=50
        )
        model_dir_entry.pack(side="left", padx=5, fill="x", expand=True)

        browse_btn = ttk.Button(
            dir_frame, text="Browse...", command=self.browse_model_dir
        )
        browse_btn.pack(side="left", padx=5)

        # Scan button
        scan_btn = ttk.Button(
            discovery_frame, text="🔍 Scan for Models", command=self.scan_models
        )
        scan_btn.pack(pady=10)

        # Model selection section
        selection_frame = ttk.LabelFrame(model_frame, text="Select Model", padding=15)
        selection_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # Model treeview
        self.model_tree = ttk.Treeview(
            selection_frame, columns=("Name", "Type", "Size", "Path"), show="headings"
        )

        self.model_tree.heading("Name", text="Model Name")
        self.model_tree.heading("Type", text="Type")
        self.model_tree.heading("Size", text="Size")
        self.model_tree.heading("Path", text="Path")

        self.model_tree.column("Name", width=150)
        self.model_tree.column("Type", width=100)
        self.model_tree.column("Size", width=80)
        self.model_tree.column("Path", width=300)

        scrollbar = ttk.Scrollbar(
            selection_frame, orient="vertical", command=self.model_tree.yview
        )
        self.model_tree.configure(yscrollcommand=scrollbar.set)

        self.model_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load button
        load_btn = ttk.Button(
            model_frame, text="Load Selected Model", command=self.load_selected_model
        )
        load_btn.pack(pady=10)

    def browse_model_dir(self):
        """Browse for model directory"""
        directory = filedialog.askdirectory(
            title="Select Model Directory", initialdir="./models"
        )
        if directory:
            self.model_dir_var.set(directory)

    def scan_models(self):
        """Scan for models in the selected directory"""
        directory = self.model_dir_var.get()
        if not directory:
            messagebox.showwarning("Warning", "Please select a model directory first")
            return

        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)

        # Store discovered models
        self.discovered_models = {}

        try:
            # Use the model loader to discover models if available
            if self.model_loader and hasattr(self.model_loader, "discover_models"):
                discovered = self.model_loader.discover_models(directory)
                for path, metadata in discovered.items():
                    model_name = Path(path).stem if isinstance(path, str) else str(path)
                    self.discovered_models[model_name] = {
                        "path": path,
                        "metadata": metadata,
                    }
            else:
                # Manual discovery - look for common model file extensions
                model_extensions = [
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".safetensors",
                    ".bin",
                    ".sigil",
                ]

                directory_path = Path(directory)
                if directory_path.exists():
                    for ext in model_extensions:
                        for model_file in directory_path.glob(f"**/*{ext}"):
                            model_name = model_file.stem
                            size = model_file.stat().st_size
                            self.discovered_models[model_name] = {
                                "path": str(model_file),
                                "metadata": {
                                    "type": "PyTorch"
                                    if ext in [".pt", ".pth"]
                                    else "Unknown",
                                    "size": size,
                                    "extension": ext,
                                },
                            }

            # Update the tree view
            for model_name, model_info in self.discovered_models.items():
                path = model_info.get("path", "N/A")
                metadata = model_info.get("metadata", {})
                model_type = metadata.get("type", "Unknown")
                size = metadata.get("size", "Unknown")

                # Format size
                if isinstance(size, (int, float)):
                    if size > 1024 * 1024:
                        size_str = f"{size / 1024 / 1024:.1f}MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                else:
                    size_str = str(size)

                self.model_tree.insert(
                    "",
                    "end",
                    values=(model_name, model_type, size_str, path),
                )

            count = len(self.discovered_models)
            messagebox.showinfo("Scan Complete", f"Found {count} compatible models")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan models: {str(e)}")

    def load_selected_model(self):
        """Load the selected model"""
        selected_items = self.model_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a model to load")
            return

        selected_item = selected_items[0]
        model_path = self.model_tree.item(selected_item)["values"][3]

        # Pass the selected model to the parent GUI
        if hasattr(self.parent_gui, "load_model") and callable(
            self.parent_gui.load_model
        ):
            self.parent_gui.load_model(model_path)

    def update_model_list(self, discovered_models):
        """Update the model list with discovered models.

        Args:
            discovered_models: Dictionary where keys are model_ids and values are dicts
                              containing 'path' and 'metadata' keys
        """
        if not hasattr(self, "model_tree"):
            print("Warning: Model tree not initialized, cannot update model list")
            return

        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)

        # Add discovered models to the tree
        for model_id, model_info in discovered_models.items():
            path = model_info.get("path", "N/A")
            metadata = model_info.get("metadata", {})

            # Extract model information from metadata
            model_type = metadata.get("type", "Unknown")
            size = metadata.get("size", "Unknown")

            # Format size if it's a number
            if isinstance(size, (int, float)):
                size = (
                    f"{size / 1024 / 1024:.1f}MB"
                    if size > 1024 * 1024
                    else f"{size / 1024:.1f}KB"
                )

            # Add to tree
            self.model_tree.insert(
                "",
                "end",
                values=(model_id, model_type, size, path or "Supervisor Model"),
            )

    def display_model_details(self, model_id, metadata, model_obj=None):
        """Display detailed information about a selected model.

        Args:
            model_id: The model identifier
            metadata: Model metadata dictionary
            model_obj: The loaded model object (optional)
        """
        # This method can be expanded to show detailed model information
        # in a separate panel or dialog
        print(f"Model selected: {model_id}")
        print(f"Metadata: {metadata}")
