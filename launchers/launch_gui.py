#!/usr/bin/env python3
"""
VoxSigil GUI Launch Script
Test the main GUI with all tabs and dark mode styling
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from gui.components.gui_styles import VoxSigilStyles
from gui.components.pyqt_main import VoxSigilMainWindow


def main():
    """Launch the VoxSigil GUI"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("VoxSigil Dashboard")
    app.setApplicationVersion("2.0")

    # Apply dark theme
    VoxSigilStyles.apply_dark_theme(app)
    logger.info("✅ Applied VoxSigil dark theme")

    try:
        # Create main window
        main_window = VoxSigilMainWindow()
        logger.info("✅ Created VoxSigil main window")

        # Setup window
        main_window.setWindowTitle("🌌 VoxSigil Dynamic GridFormer Dashboard v2.0")
        main_window.resize(1400, 900)
        main_window.show()
        logger.info("✅ Displayed main window")

        # Start event loop
        logger.info("🚀 Starting VoxSigil GUI...")
        return app.exec_()

    except Exception as e:
        logger.error(f"❌ Error launching GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
