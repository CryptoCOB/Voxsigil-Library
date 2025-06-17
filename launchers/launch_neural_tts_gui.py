"""
VoxSigil GUI with Neural TTS Integration Demo
Launches the VoxSigil GUI with integrated Neural TTS capabilities.
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VoxSigilGUIDemo")


def main():
    """Launch VoxSigil GUI with Neural TTS integration."""

    print("🎙️ VoxSigil GUI with Neural TTS Integration")
    print("=" * 50)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("VoxSigil Neural TTS GUI")
    app.setApplicationVersion("1.0.0")

    try:
        # Import and create main window
        logger.info("Loading VoxSigil GUI components...")
        from gui.components.pyqt_main_unified import VoxSigilMainWindow

        # Create main window
        main_window = VoxSigilMainWindow()
        main_window.setWindowTitle("🎙️ VoxSigil - Neural TTS Integrated GUI")

        # Show welcome message
        QMessageBox.information(
            main_window,
            "🎉 VoxSigil Neural TTS Ready!",
            """Welcome to VoxSigil with integrated Neural TTS!

🎭 Features Available:
• Neural TTS Tab: Complete voice control interface
• Agent Status Panel: Quick voice controls for all agents
• 5 Unique Agent Voices: Nova, Aria, Kai, Echo, Sage
• Real-time speech synthesis and audio file generation

🚀 Getting Started:
1. Click on the "🎙️ Neural TTS" tab for full controls
2. Use the "📈 Agent Status" tab for quick voice tests
3. Select any agent and click "Speak Status" or "Greeting"
4. Enjoy human-like AI conversations!

The system is ready for production use with free, open-source neural TTS.""",
        )

        # Show main window
        main_window.show()
        main_window.raise_()

        logger.info("✅ VoxSigil GUI launched successfully with Neural TTS integration")

        # Start event loop
        sys.exit(app.exec_())

    except ImportError as e:
        logger.error(f"❌ Failed to import GUI components: {e}")

        # Show error dialog
        error_app = (
            QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        )
        QMessageBox.critical(
            None,
            "Import Error",
            f"Failed to load VoxSigil GUI components:\n\n{str(e)}\n\nPlease ensure all dependencies are installed.",
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

        # Show error dialog
        error_app = (
            QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        )
        QMessageBox.critical(None, "Startup Error", f"Failed to start VoxSigil GUI:\n\n{str(e)}")


if __name__ == "__main__":
    main()
