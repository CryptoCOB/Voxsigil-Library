#!/usr/bin/env python3
"""
Minimal GUI Test - Test Just the Core Components
"""

print("Starting minimal test...")

try:
    print("Step 1: Testing PyQt5...")
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QTabWidget
    from PyQt5.QtCore import Qt
    print("✅ PyQt5 imported")
    
    print("Step 2: Creating QApplication...")
    import sys
    app = QApplication(sys.argv)
    print("✅ QApplication created")
    
    print("Step 3: Creating simple window...")
    window = QMainWindow()
    window.setWindowTitle("Minimal Test")
    print("✅ Window created")
    
    print("Step 4: Testing data provider import...")
    from gui.components.real_time_data_provider import RealTimeDataProvider
    print("✅ Data provider imported")
    
    print("Step 5: Testing data provider creation...")
    data_provider = RealTimeDataProvider()
    print("✅ Data provider created")
    
    print("Step 6: Testing enhanced GUI import...")
    from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
    print("✅ Enhanced GUI imported")
    
    print("Step 7: THIS IS THE CRITICAL TEST - Creating CompleteEnhancedGUI...")
    # This is likely where it hangs
    enhanced_gui = CompleteEnhancedGUI()
    print("✅ Enhanced GUI created successfully!")
    
    print("Step 8: Showing GUI...")
    enhanced_gui.show()
    print("✅ GUI shown")
    
    print("Step 9: Testing event loop briefly...")
    from PyQt5.QtCore import QTimer
    def exit_app():
        print("✅ Event loop test completed")
        app.quit()
    
    timer = QTimer()
    timer.timeout.connect(exit_app)
    timer.start(2000)  # 2 seconds
    
    print("Starting event loop...")
    app.exec_()
    print("✅ All tests passed!")

except Exception as e:
    print(f"❌ Error at current step: {e}")
    import traceback
    traceback.print_exc()
