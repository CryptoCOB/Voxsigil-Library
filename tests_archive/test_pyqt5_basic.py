#!/usr/bin/env python3
"""
PyQt5 Basic Test - Check if PyQt5 works at all
"""

print("Testing PyQt5 installation...")

try:
    print("Step 1: Testing PyQt5 import...")
    import PyQt5
    print(f"✅ PyQt5 version: {PyQt5.Qt.PYQT_VERSION_STR}")
    
    print("Step 2: Testing QtWidgets import...")
    from PyQt5.QtWidgets import QApplication, QLabel, QWidget
    print("✅ QtWidgets imported")
    
    print("Step 3: Testing QApplication creation...")
    import sys
    app = QApplication(sys.argv)
    print("✅ QApplication created")
    
    print("Step 4: Testing widget creation...")
    widget = QWidget()
    label = QLabel("Hello PyQt5!")
    print("✅ Widgets created")
    
    print("Step 5: Testing widget show...")
    widget.resize(300, 200)
    widget.setWindowTitle("PyQt5 Test")
    widget.show()
    print("✅ Widget shown")
    
    print("Step 6: Testing event loop for 2 seconds...")
    from PyQt5.QtCore import QTimer
    def exit_test():
        print("✅ PyQt5 test completed successfully!")
        app.quit()
    
    timer = QTimer()
    timer.timeout.connect(exit_test)
    timer.start(2000)
    
    print("Starting event loop...")
    app.exec_()
    print("✅ Event loop completed")
    
except ImportError as e:
    print(f"❌ PyQt5 import error: {e}")
    print("💡 Solution: pip install PyQt5")

except Exception as e:
    print(f"❌ PyQt5 error: {e}")
    import traceback
    traceback.print_exc()

print("PyQt5 test finished.")
