#!/usr/bin/env python3
"""
Fixed VoxSigil GUI Launcher
This version fixes the import and styling issues
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """Setup the environment for VoxSigil GUI"""
    # Add project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set Qt platform
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")

    print("🔧 Environment setup complete")


def launch_gui():
    """Launch the VoxSigil GUI with proper error handling"""
    try:
        print("🚀 Launching VoxSigil GUI...")

        # Import PyQt5
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtWidgets import QApplication

        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("VoxSigil Library")
        app.setApplicationVersion("1.0.0")

        print("✅ QApplication created")

        # Apply theme
        try:
            from gui.components.gui_styles import VoxSigilStyles

            styles = VoxSigilStyles()
            app.setStyleSheet(styles.get_dark_theme())
            print("🌙 Dark theme applied")
        except Exception as theme_error:
            print(f"⚠️ Theme warning: {theme_error}")

        # Import and create main window
        from gui.components.pyqt_main import VoxSigilMainWindow

        print("📦 Creating main window...")
        main_window = VoxSigilMainWindow()

        # Show window
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        print("🎉 VoxSigil GUI launched successfully!")
        print("📱 Main window is now visible")

        # Start the event loop
        return app.exec_()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return 1
    except Exception as e:
        print(f"❌ Launch error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    setup_environment()
    exit_code = launch_gui()
    sys.exit(exit_code)
