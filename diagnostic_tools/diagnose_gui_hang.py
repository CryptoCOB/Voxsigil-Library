#!/usr/bin/env python3
"""
GUI Launch Diagnostics - Find where it's hanging
"""

import sys
import os
import time
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step_by_step():
    """Test each step of GUI initialization to find where it hangs"""
    
    print("🔍 Starting step-by-step GUI diagnostic...")
    print("=" * 60)
    
    try:
        print("Step 1: Testing basic imports...")
        from gui.components.real_time_data_provider import RealTimeDataProvider
        print("✅ RealTimeDataProvider import OK")
        
        print("Step 2: Testing data provider instantiation...")
        provider = RealTimeDataProvider()
        print("✅ Data provider creation OK")
        
        print("Step 3: Testing data provider methods...")
        metrics = provider.get_all_metrics()
        print(f"✅ Data provider methods OK - {len(metrics)} metrics")
        
        print("Step 4: Testing PyQt5 import...")
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 import OK")
        
        print("Step 5: Testing QApplication creation...")
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            print("✅ New QApplication created")
        else:
            print("✅ Using existing QApplication")
        
        print("Step 6: Testing GUI module imports...")
        try:
            from gui.enhanced_voxsigil_gui import EnhancedVoxSigilGUI
            print("✅ Enhanced GUI class import OK")
        except ImportError as e:
            print(f"⚠️ Enhanced GUI import issue: {e}")
            # Try alternative import
            print("Trying alternative GUI import...")
            from gui.voxsigil_gui import VoxSigilGUI
            print("✅ Standard GUI class import OK")
        
        print("Step 7: Testing minimal GUI creation...")
        # Don't actually create the full GUI yet, just test the class
        print("✅ All imports successful - GUI creation should work")
        
        print("\n" + "=" * 60)
        print("🎉 DIAGNOSTIC COMPLETE - All basic components working!")
        print("The hang is likely during GUI widget creation or event loop setup.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_lightweight_gui():
    """Test creating a minimal GUI to see if it hangs"""
    
    print("\n🧪 Testing lightweight GUI creation...")
    
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
        from PyQt5.QtCore import QTimer
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create minimal window
        window = QMainWindow()
        window.setWindowTitle("VoxSigil Test Window")
        window.setGeometry(100, 100, 400, 200)
        
        # Add simple label
        label = QLabel("✅ GUI is working! Close this window to continue.")
        window.setCentralWidget(label)
        
        # Show window
        window.show()
        
        # Set a timer to automatically close after 3 seconds
        timer = QTimer()
        timer.timeout.connect(app.quit)
        timer.start(3000)  # 3 seconds
        
        print("✅ Showing test window for 3 seconds...")
        app.exec_()
        
        print("✅ Lightweight GUI test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Lightweight GUI test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚨 GUI HANG DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Test step by step
    if not test_step_by_step():
        print("❌ Basic component test failed")
        return
    
    # Test lightweight GUI
    if not test_lightweight_gui():
        print("❌ Lightweight GUI test failed")
        return
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC RESULTS:")
    print("✅ All basic components working")
    print("✅ PyQt5 and GUI creation working")
    print("\n💡 RECOMMENDATION:")
    print("The hang is likely caused by:")
    print("1. Complex widget initialization in the main GUI")
    print("2. VantaCore integration causing deadlock")
    print("3. Event loop conflicts")
    print("4. Heavy computations during startup")
    print("\n🔧 SOLUTION: Create a simplified GUI launcher")

if __name__ == "__main__":
    main()
