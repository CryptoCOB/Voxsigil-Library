#!/usr/bin/env python3
"""Test script to check MeshMapPanel method availability."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

try:
    from gui.components.mesh_map_panel import MeshMapPanel

    # Check if the class has the method
    print("MeshMapPanel class imported successfully")
    print(f"auto_refresh_mesh method exists: {hasattr(MeshMapPanel, 'auto_refresh_mesh')}")

    # List all methods
    methods = [method for method in dir(MeshMapPanel) if not method.startswith("_")]
    print(f"Available methods: {methods}")

    # Try to create an instance (without Qt app)
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    panel = MeshMapPanel()
    print(
        f"Instance created, auto_refresh_mesh method exists: {hasattr(panel, 'auto_refresh_mesh')}"
    )

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
