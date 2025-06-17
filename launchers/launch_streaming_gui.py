#!/usr/bin/env python3
"""
Simple Enhanced GUI launcher to test streaming functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Launch the Enhanced GUI with all streaming tabs."""
    print("🚀 Launching VoxSigil Enhanced GUI with Real Streaming Data...")

    try:
        from PyQt5.QtWidgets import QApplication

        from gui.components.pyqt_main_unified import VoxSigilMainWindow

        app = QApplication(sys.argv)

        # Create the enhanced main window
        window = VoxSigilMainWindow()
        window.show()

        print("✅ Enhanced GUI launched successfully!")
        print("🔥 All tabs should now show REAL streaming data, not just checkmarks!")
        print("📊 Check the Live Dashboard for unified real-time metrics")
        print("🎯 Training tab has real VantaCore integration")
        print("📈 Visualization tab shows real system metrics")
        print("🎵 Music tab has real-time audio metrics")
        print("🤖 Model tab has real-time model discovery")

        return app.exec_()

    except Exception as e:
        print(f"❌ Error launching enhanced GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
