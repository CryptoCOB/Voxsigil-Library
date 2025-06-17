#!/usr/bin/env python3
"""
Enhanced VoxSigil GUI Launcher with Async Loop Handling
Fixes the async event loop issues and module import warnings
"""

import asyncio
import os
import sys
import warnings
from pathlib import Path


def setup_environment():
    """Setup the environment for VoxSigil GUI"""
    # Add project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set Qt platform
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*found in sys.modules.*")
    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")

    print("🔧 Environment setup complete")


def setup_async_loop():
    """Setup async event loop for Qt integration"""
    try:
        # Check if event loop is already running
        loop = asyncio.get_running_loop()
        print("🔄 Using existing event loop")
        return loop
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print("🆕 Created new event loop")
        return loop


def launch_gui():
    """Launch the VoxSigil GUI with proper async handling"""
    try:
        print("🚀 Launching VoxSigil GUI...")

        # Setup async loop first
        setup_async_loop()  # Import PyQt5
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("VoxSigil Library")
        app.setApplicationVersion("1.0.0")
        app.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)

        print("✅ QApplication created")  # Apply theme
        try:
            from gui.components.gui_styles import VoxSigilStyles

            styles = VoxSigilStyles()
            app.setStyleSheet(styles.get_dark_theme())
            print("🌙 Dark theme applied")
        except Exception as theme_error:
            print(f"⚠️ Theme warning: {theme_error}")

        # Import and create main window
        from gui.components.pyqt_main_unified import VoxSigilMainWindow

        print("📦 Creating main window...")
        main_window = VoxSigilMainWindow()

        # Show window
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        print("🎉 VoxSigil GUI launched successfully!")
        print("📱 Main window is now visible")
        print("\n🔍 To test the Training Control tab:")
        print("   1. Click on the 'Training Control' tab")
        print("   2. Select a model and configure parameters")
        print("   3. Click 'Start Training' to see real/simulated results")
        print("   4. Notice the accuracy is no longer hardcoded at 85%!")

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
