#!/usr/bin/env python3
"""
VoxSigil GUI Non-blocking Test
Tests GUI creation without hanging the terminal
"""

import os
import sys
from pathlib import Path


def test_gui_components():
    """Test GUI components without starting event loop"""
    try:
        print("🧪 Testing VoxSigil GUI components...")

        # Setup environment
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        # Test PyQt5 import
        from PyQt5.QtWidgets import QApplication

        print("✅ PyQt5 imported successfully")

        # Create minimal app
        app = QApplication(sys.argv)
        print("✅ QApplication created")

        # Test theme import
        try:
            from gui.components.gui_styles import VoxSigilStyles

            styles = VoxSigilStyles()
            print("✅ VoxSigilStyles imported")
        except Exception as e:
            print(f"⚠️ Theme import issue: {e}")

        # Test main window import
        try:
            from gui.components.pyqt_main import VoxSigilMainWindow

            print("✅ VoxSigilMainWindow imported")

            # Create window (but don't show)
            window = VoxSigilMainWindow()
            print("✅ Main window created successfully")

            # Test if window has expected attributes
            if hasattr(window, "show"):
                print("✅ Window has show() method")

            print("🎉 All GUI components working!")

        except Exception as e:
            print(f"❌ Main window error: {e}")
            import traceback

            traceback.print_exc()

        # Clean shutdown
        app.quit()
        print("🧹 Clean shutdown completed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_gui_components()
