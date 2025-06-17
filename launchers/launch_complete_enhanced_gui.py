#!/usr/bin/env python3
"""
Complete Enhanced GUI Launcher - Full VoxSigil Functionality
This launches the COMPLETE enhanced GUI with all tabs, using lazy loading to prevent hangs.
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CompleteEnhancedGUI")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def launch_complete_enhanced_gui():
    """Launch the complete enhanced GUI with all functionality"""
    try:
        logger.info("🎯 Complete VoxSigil Enhanced GUI Launcher")
        logger.info("=" * 70)
        logger.info("Loading the COMPLETE enhanced GUI with ALL tabs and features")
        logger.info("Using lazy loading to prevent initialization hangs")
        logger.info("=" * 70)

        # Step 1: Test data provider
        logger.info("🔍 Initializing real-time data provider...")
        from gui.components.real_time_data_provider import RealTimeDataProvider
        
        data_provider = RealTimeDataProvider()
        all_metrics = data_provider.get_all_metrics()
        logger.info(f"✅ Real-time data provider ready: {len(all_metrics)} metrics available")

        # Step 2: Import PyQt5
        logger.info("🔍 Importing PyQt5 framework...")
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        # Step 3: Create application
        logger.info("🔍 Creating QApplication...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            # Enable high DPI scaling
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        logger.info("✅ QApplication ready")

        # Step 4: Import and create the complete enhanced GUI
        logger.info("🔍 Loading complete enhanced GUI...")
        from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
        
        # Create the main window
        window = CompleteEnhancedGUI()
        logger.info("✅ Complete enhanced GUI created")
        
        # Step 5: Show the window
        logger.info("🔍 Displaying GUI window...")
        window.show()
        logger.info("✅ GUI window displayed")
        
        # Log success
        logger.info("=" * 70)
        logger.info("🎉 COMPLETE ENHANCED GUI LAUNCHED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("✅ All enhanced tabs available:")
        logger.info("   📊 Status - System overview and metrics")
        logger.info("   📡 Live Dashboard - Real-time streaming data")
        logger.info("   🤖 Models - Enhanced model management")
        logger.info("   🎯 Training - Advanced training pipelines")
        logger.info("   📈 Visualization - Real-time data visualization")
        logger.info("   🎵 Music - Enhanced music generation")
        logger.info("   🔄 GridFormer - Grid formation systems")
        logger.info("   🧠 Novel Reasoning - Advanced reasoning capabilities")
        logger.info("   🎙️ Neural TTS - Text-to-speech systems")
        logger.info("   💓 Heartbeat Monitor - System vital signs")
        logger.info("   🔧 System Integration - Integration management")
        logger.info("   📝 Real-time Logs - Live logging interface")
        logger.info("")
        logger.info("💡 Usage Tips:")
        logger.info("• Tabs load on-demand when you click them (prevents hangs)")
        logger.info("• All VantaCore integration features are available")
        logger.info("• Real-time data streaming is active across all components")
        logger.info("• Use the Status tab to monitor system health")
        logger.info("")
        logger.info("🚀 Starting GUI event loop...")

        # Step 6: Start event loop
        return app.exec_()
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure all dependencies are installed:")
        logger.info("   pip install PyQt5")
        logger.info("   Check that all GUI components are present")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Complete enhanced GUI launch failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = launch_complete_enhanced_gui()
    if exit_code == 0:
        logger.info("👋 GUI session ended successfully")
    else:
        logger.error("❌ GUI session ended with errors")
    sys.exit(exit_code)
