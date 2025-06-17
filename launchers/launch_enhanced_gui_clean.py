#!/usr/bin/env python3
"""
Enhanced GUI Launcher - No VantaCore Mode
Launches the enhanced GUI with real-time data provider only, avoiding VantaCore warnings.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedGUI")


def launch_enhanced_gui_only():
    """Launch only the enhanced GUI without VantaCore initialization."""
    try:
        logger.info("🚀 Launching Enhanced GUI (No VantaCore Mode)")

        # Test real-time data provider
        logger.info("🔍 Testing real-time data provider...")
        from gui.components.real_time_data_provider import RealTimeDataProvider

        data_provider = RealTimeDataProvider()

        # Get sample metrics to verify it works
        all_metrics = data_provider.get_all_metrics()
        logger.info(f"✅ Real-time data provider working: {len(all_metrics)} metrics available")        # Import and launch the enhanced GUI
        logger.info("🔍 Starting Enhanced GUI...")
        
        try:
            import sys

            from PyQt5.QtWidgets import QApplication

            from gui.components.complete_enhanced_gui import CompleteEnhancedGUI

            app = QApplication(sys.argv)
            window = CompleteEnhancedGUI()
            window.show()

            logger.info("✅ Enhanced GUI launched successfully!")
            logger.info(
                "💡 GUI is using real-time streaming data without VantaCore event loop issues"
            )

            return app.exec_()

        except ImportError as e:
            logger.error(f"❌ GUI import error: {e}")
            logger.info("💡 Make sure PyQt5 is installed: pip install PyQt5")
            return 1

    except Exception as e:
        logger.error(f"❌ Enhanced GUI launch failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    logger.info("🎯 Enhanced VoxSigil GUI - Real-Time Data Mode")
    logger.info("=" * 60)
    logger.info("This mode launches the GUI with real streaming data")
    logger.info("and avoids VantaCore initialization to prevent event loop errors.")
    logger.info("=" * 60)

    exit_code = launch_enhanced_gui_only()
    sys.exit(exit_code)
