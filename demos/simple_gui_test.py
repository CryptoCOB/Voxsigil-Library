#!/usr/bin/env python3
"""
Simple GUI test without heavy components.
"""

import sys
import logging

# Simple logging without emojis
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SimpleGUITest")

def main():
    logger.info("=== Simple GUI Test ===")
    
    try:
        logger.info("1. Testing PyQt5...")
        from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        logger.info("   PyQt5 imported successfully")
        
        logger.info("2. Creating QApplication...")
        app = QApplication(sys.argv)
        logger.info("   QApplication created")
        
        logger.info("3. Creating simple window...")
        window = QMainWindow()
        window.setWindowTitle("VoxSigil GUI Test")
        window.resize(400, 300)
        
        # Create simple content
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("VoxSigil Enhanced GUI Test\n\nIf you can see this, the GUI is working!")
        label.setStyleSheet("font-size: 14px; padding: 20px;")
        layout.addWidget(label)
        
        central_widget.setLayout(layout)
        window.setCentralWidget(central_widget)
        
        logger.info("4. Showing window...")
        window.show()
        
        logger.info("5. GUI window should now be visible!")
        logger.info("   Close the window to continue...")
        
        # Run the GUI
        return app.exec_()
        
    except Exception as e:
        logger.error(f"GUI Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"GUI test completed with exit code: {exit_code}")
    sys.exit(exit_code)
