#!/usr/bin/env python3
"""
Simple Hang Diagnostic - Find Exact Hang Point
This script will report exactly where the GUI hangs (no user input required).
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
logger = logging.getLogger("SimpleHangDiagnostic")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def checkpoint(message, step_num):
    """Log a checkpoint with timing"""
    print(f"🔍 STEP {step_num}: {message}")
    logger.info(f"🔍 STEP {step_num}: {message}")
    time.sleep(0.1)  # Small delay to ensure output is visible

def main():
    """Run simple diagnostic"""
    try:
        checkpoint("Starting GUI hang diagnostic", 1)
        
        checkpoint("Testing PyQt5 import", 2)
        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
            from PyQt5.QtCore import Qt, QTimer
            print("✅ PyQt5 import successful")
        except Exception as e:
            print(f"❌ PyQt5 import failed: {e}")
            return 1
        
        checkpoint("Creating QApplication", 3)
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            print("✅ QApplication created successfully")
        except Exception as e:
            print(f"❌ QApplication creation failed: {e}")
            return 1
        
        checkpoint("Testing basic window creation", 4)
        try:
            window = QMainWindow()
            window.setWindowTitle("Simple Diagnostic Test")
            window.setGeometry(100, 100, 400, 300)
            
            central_widget = QWidget()
            layout = QVBoxLayout()
            label = QLabel("Basic GUI Test - Working!")
            layout.addWidget(label)
            central_widget.setLayout(layout)
            window.setCentralWidget(central_widget)
            print("✅ Basic window created successfully")
        except Exception as e:
            print(f"❌ Basic window creation failed: {e}")
            return 1
        
        checkpoint("Testing window.show()", 5)
        try:
            window.show()
            print("✅ Window.show() successful")
        except Exception as e:
            print(f"❌ Window.show() failed: {e}")
            return 1
        
        checkpoint("Testing real-time data provider import", 6)
        try:
            from gui.components.real_time_data_provider import RealTimeDataProvider
            print("✅ Data provider import successful")
        except Exception as e:
            print(f"❌ Data provider import failed: {e}")
            print("Continuing without data provider...")
        
        checkpoint("Testing data provider initialization", 7)
        try:
            data_provider = RealTimeDataProvider()
            metrics = data_provider.get_all_metrics()
            print(f"✅ Data provider working: {len(metrics)} metrics")
        except Exception as e:
            print(f"❌ Data provider failed: {e}")
            print("Continuing without data provider...")
        
        checkpoint("Testing complete enhanced GUI import", 8)
        try:
            from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
            print("✅ Enhanced GUI import successful")
        except Exception as e:
            print(f"❌ Enhanced GUI import failed: {e}")
            return 1
        
        checkpoint("Testing complete enhanced GUI creation (THIS IS LIKELY WHERE IT HANGS)", 9)
        try:
            enhanced_gui = CompleteEnhancedGUI()
            print("✅ Enhanced GUI creation successful")
        except Exception as e:
            print(f"❌ Enhanced GUI creation failed: {e}")
            return 1
        
        checkpoint("Testing enhanced GUI show", 10)
        try:
            enhanced_gui.show()
            print("✅ Enhanced GUI show successful")
        except Exception as e:
            print(f"❌ Enhanced GUI show failed: {e}")
            return 1
        
        checkpoint("Starting short event loop test", 11)
        try:
            # Exit after 3 seconds
            def timeout_exit():
                print("✅ Event loop test completed successfully")
                app.quit()
            
            timer = QTimer()
            timer.timeout.connect(timeout_exit)
            timer.start(3000)  # 3 seconds
            
            print("🚀 Starting GUI event loop (3 second test)...")
            exit_code = app.exec_()
            print(f"✅ Event loop completed with code: {exit_code}")
            return 0
            
        except Exception as e:
            print(f"❌ Event loop failed: {e}")
            return 1
            
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"🏁 Diagnostic completed with exit code: {exit_code}")
    sys.exit(exit_code)
