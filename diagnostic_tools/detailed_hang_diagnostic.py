#!/usr/bin/env python3
"""
Detailed Hang Diagnostic - Find Exact Hang Point
This script will report exactly where the GUI hangs.
"""

import logging
import sys
import os
import time

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HangDiagnostic")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def checkpoint(message, step_num):
    """Log a checkpoint with timing"""
    logger.info(f"🔍 STEP {step_num}: {message}")
    time.sleep(0.1)  # Small delay to ensure output is visible

def main():
    """Run detailed diagnostic"""
    try:
        checkpoint("Starting GUI hang diagnostic", 1)
        
        checkpoint("Importing basic Python modules", 2)
        import traceback
        import json
        
        checkpoint("Adding path and checking file existence", 3)
        gui_path = os.path.join(os.path.dirname(__file__), "gui", "components")
        logger.info(f"GUI components path: {gui_path}")
        logger.info(f"Path exists: {os.path.exists(gui_path)}")
        
        checkpoint("Testing data provider import", 4)
        try:
            from gui.components.real_time_data_provider import RealTimeDataProvider
            logger.info("✅ Data provider import successful")
        except Exception as e:
            logger.error(f"❌ Data provider import failed: {e}")
            traceback.print_exc()
        
        checkpoint("Testing data provider initialization", 5)
        try:
            data_provider = RealTimeDataProvider()
            logger.info("✅ Data provider initialization successful")
            
            # Test getting metrics
            metrics = data_provider.get_all_metrics()
            logger.info(f"✅ Got {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"❌ Data provider initialization failed: {e}")
            traceback.print_exc()
        
        checkpoint("Testing PyQt5 import", 6)
        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
            from PyQt5.QtCore import Qt
            logger.info("✅ PyQt5 import successful")
        except Exception as e:
            logger.error(f"❌ PyQt5 import failed: {e}")
            traceback.print_exc()
            return 1
        
        checkpoint("Creating QApplication", 7)
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
                app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            logger.info("✅ QApplication created successfully")
        except Exception as e:
            logger.error(f"❌ QApplication creation failed: {e}")
            traceback.print_exc()
            return 1
        
        checkpoint("Testing basic window creation", 8)
        try:
            window = QMainWindow()
            window.setWindowTitle("Diagnostic Test Window")
            window.setGeometry(100, 100, 400, 300)
            
            central_widget = QWidget()
            layout = QVBoxLayout()
            label = QLabel("Diagnostic Test - GUI is working!")
            layout.addWidget(label)
            central_widget.setLayout(layout)
            window.setCentralWidget(central_widget)
            
            logger.info("✅ Basic window created successfully")
        except Exception as e:
            logger.error(f"❌ Basic window creation failed: {e}")
            traceback.print_exc()
            return 1
        
        checkpoint("Testing window.show()", 9)
        try:
            window.show()
            logger.info("✅ Window.show() successful")
        except Exception as e:
            logger.error(f"❌ Window.show() failed: {e}")
            traceback.print_exc()
            return 1
        
        checkpoint("Testing complete enhanced GUI import", 10)
        try:
            from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
            logger.info("✅ Complete enhanced GUI import successful")
        except Exception as e:
            logger.error(f"❌ Complete enhanced GUI import failed: {e}")
            traceback.print_exc()
            # Continue anyway to test other parts
        
        checkpoint("Testing complete enhanced GUI creation", 11)
        try:
            enhanced_gui = CompleteEnhancedGUI()
            logger.info("✅ Complete enhanced GUI creation successful")
        except Exception as e:
            logger.error(f"❌ Complete enhanced GUI creation failed: {e}")
            traceback.print_exc()
            # Don't return, let's see if the basic window works
        
        checkpoint("Testing complete enhanced GUI show", 12)
        try:
            enhanced_gui.show()
            logger.info("✅ Complete enhanced GUI show successful")
        except Exception as e:
            logger.error(f"❌ Complete enhanced GUI show failed: {e}")
            traceback.print_exc()
        
        checkpoint("Starting event loop with timeout", 13)
        try:
            # Start event loop with a timer to exit after 5 seconds
            from PyQt5.QtCore import QTimer
            
            def timeout_handler():
                logger.info("✅ Event loop running successfully for 5 seconds")
                app.quit()
            
            timer = QTimer()
            timer.timeout.connect(timeout_handler)
            timer.start(5000)  # 5 seconds
            
            logger.info("🚀 Starting GUI event loop (will exit after 5 seconds)...")
            exit_code = app.exec_()
            logger.info(f"✅ Event loop completed with code: {exit_code}")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Event loop failed: {e}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        logger.error(f"❌ Diagnostic failed at unknown step: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"🏁 Diagnostic completed with exit code: {exit_code}")
    input("Press Enter to exit...")
    sys.exit(exit_code)
