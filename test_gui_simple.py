#!/usr/bin/env python3
"""
Simple GUI Test - Just run the complete GUI and capture what happens
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Test the complete GUI"""
    try:
        print("🚀 Starting VoxSigil Complete GUI Test...")
        
        # Import PyQt5
        from PyQt5.QtWidgets import QApplication
        
        # Create application
        app = QApplication(sys.argv)
        
        # Import and create the GUI
        from working_gui.complete_live_gui import CompleteVoxSigilGUI
        
        print("✅ GUI imported successfully")
        
        # Create the GUI instance
        gui = CompleteVoxSigilGUI()
        print(f"✅ GUI created with {gui.main_tabs.count()} tabs")
        
        # Show GUI
        gui.show()
        print("✅ GUI displayed")
        
        # Print tab information
        print("\n📋 TABS CREATED:")
        for i in range(gui.main_tabs.count()):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            tab_type = type(tab_widget).__name__
            print(f"   {i+1:2d}. {tab_name} ({tab_type})")
        
        print(f"\n🎯 Total tabs: {gui.main_tabs.count()}")
        print("🔍 Press Ctrl+C to exit after examining the GUI")
        
        # Run the application
        app.exec_()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
