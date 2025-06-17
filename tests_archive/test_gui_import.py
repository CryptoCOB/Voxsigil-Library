#!/usr/bin/env python3
"""
Quick test script for GUI imports
"""

try:
    from gui.components.pyqt_main import VoxSigilMainWindow

    print("✅ GUI imports successfully!")
    print("✅ VoxSigilMainWindow imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
