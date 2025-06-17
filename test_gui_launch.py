#!/usr/bin/env python3
"""
Simple test script to verify GUI can be imported and instantiated
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all critical modules can be imported"""
    print("🔄 Testing imports...")

    try:
        # Test PyQt5
        from PyQt5.QtWidgets import QApplication as _QApp

        print("✅ PyQt5 imported successfully")

        # Test GUI module
        from working_gui.complete_live_gui import CompleteVoxSigilGUI as _GUI

        print("✅ CompleteVoxSigilGUI imported successfully")

        # Test monitoring module
        from monitoring.vanta_registration import MonitoringModule as _Monitor

        print("✅ MonitoringModule imported successfully")

        # Test core modules
        from core.base import BaseCore as _Core

        print("✅ BaseCore imported successfully")

        # Use the imports to avoid unused warnings
        _ = _QApp, _GUI, _Monitor, _Core

        print("🎉 All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_instantiation():
    """Test that critical objects can be instantiated"""
    print("\n🔄 Testing instantiation...")

    try:
        # Import here to avoid unused import warnings
        from PyQt5.QtWidgets import QApplication
        from working_gui.complete_live_gui import CompleteVoxSigilGUI

        # Create QApplication (required for Qt widgets)
        app = QApplication(sys.argv)
        print("✅ QApplication created successfully")

        # Test GUI instantiation (without showing)
        gui = CompleteVoxSigilGUI()
        print("✅ CompleteVoxSigilGUI instantiated successfully")
        print(f"   GUI has {gui.tab_widget.count()} tabs")

        print("🎉 Instantiation successful!")
        app.quit()  # Clean up
        return True

    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 VoxSigil GUI Test Suite")
    print("=" * 50)

    # Test imports
    if not test_imports():
        sys.exit(1)

    # Test instantiation
    if not test_instantiation():
        sys.exit(1)

    print("\n🎉 All tests passed! GUI is ready to launch.")
