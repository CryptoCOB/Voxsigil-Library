#!/usr/bin/env python3
"""
VoxSigil Streamlined Training GUI - Enhanced Version
==================================================
A clean, modern GUI that properly displays all VantaCore components
and provides intuitive training interface.
"""

import logging
import sys
import traceback

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("voxsigil_gui.log"), logging.StreamHandler()],
)
logger = logging.getLogger("VoxSigil")


def log_debug(message: str) -> None:
    """Helper function to log debug messages."""
    logger.debug(message)


# Main window class
class VoxSigilMainWindow(QMainWindow):
    """Main window for the VoxSigil GUI application."""

    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("VoxSigil Enhanced Training GUI")
        self.setMinimumSize(1024, 768)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_training_tab()
        self.create_model_management_tab()
        self.create_settings_tab()

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        logger.info("Main window initialized")

    def create_training_tab(self):
        """Create the training tab with voice training controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Add training components
        layout.addWidget(QLabel("Voice Training"))
        layout.addWidget(QPushButton("Select Training Data"))
        layout.addWidget(QPushButton("Start Training"))

        # Progress indicators
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        layout.addWidget(progress_bar)

        # Log display
        log_display = QTextEdit()
        log_display.setReadOnly(True)
        layout.addWidget(log_display)

        self.tab_widget.addTab(tab, "Training")

    def create_model_management_tab(self):
        """Create the model management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Model Management"))
        layout.addWidget(QPushButton("Import Model"))
        layout.addWidget(QPushButton("Export Model"))

        # Model list
        model_list = QTreeWidget()
        layout.addWidget(model_list)

        self.tab_widget.addTab(tab, "Models")

    def create_settings_tab(self):
        """Create the settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Settings"))
        layout.addWidget(QPushButton("Configure Paths"))
        layout.addWidget(QPushButton("Save Settings"))

        self.tab_widget.addTab(tab, "Settings")


def main():
    """Main application entry point."""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("VoxSigil Training GUI")

        # Set application style - simplified for stability
        app.setStyle("Fusion")

        # Simple style string with just basic colors
        basic_style = "QPushButton { background-color: #2E86AB; color: white; }"
        app.setStyleSheet(basic_style)

        # Create and show main window
        window = VoxSigilMainWindow()
        window.show()

        logger.info("[LAUNCH] VoxSigil GUI started successfully")

        # Run application
        return app.exec_()
    except Exception as e:
        logger.critical(f"[ERROR] CRITICAL ERROR in main(): {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
