#!/usr/bin/env python3
"""
VoxSigil Enhanced GUI Launcher
Simple launcher that properly sets up paths and runs the enhanced GUI
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set working directory to project root
os.chdir(project_root)


def main():
    """Launch the VoxSigil Enhanced GUI"""
    print("🚀 Launching VoxSigil Enhanced GUI...")
    print(f"📁 Project root: {project_root}")

    try:
        # Import PyQt5 first to check availability
        from PyQt5.QtWidgets import QApplication

        print("✅ PyQt5 is available")

        # Import the GUI module with proper path setup
        from gui.components.pyqt_main_unified import VoxSigilMainWindow

        print("✅ GUI module imported successfully")

        # Create QApplication
        app = QApplication(sys.argv)
        print("✅ QApplication created")

        # Create and show main window
        window = VoxSigilMainWindow()
        window.show()
        print("✅ Main window created and shown")

        print("\n🎉 VoxSigil Enhanced GUI is now running!")
        print("   - All enhanced tabs are available")
        print("   - Dev mode controls are ready")
        print("   - Neural TTS is integrated")
        print("   - Configuration system is active")

        # Start the event loop
        sys.exit(app.exec_())

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure PyQt5 is installed: uv add PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
