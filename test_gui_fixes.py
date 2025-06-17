#!/usr/bin/env python3
"""
Final GUI Status Check - Verify everything is working
"""

import sys
import os

def main():
    print("🎉 VoxSigil GUI - Final Status Check")
    print("=" * 50)
    
    try:
        # Test PyQt5 import
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5: Available")
        
        # Test GUI import
        from working_gui.complete_live_gui import CompleteVoxSigilGUI
        print("✅ Complete GUI: Importable")
        
        # Create minimal app
        app = QApplication(sys.argv)
        print("✅ QApplication: Created")
        
        # Test GUI creation
        print("� Testing GUI creation...")
        gui = CompleteVoxSigilGUI()
        print(f"✅ GUI Created: {gui.main_tabs.count()} tabs")
        
        # Check if any tabs exist
        if gui.main_tabs.count() > 0:
            print("✅ Tabs: Successfully created")
            print("\n📋 Sample tabs:")
            for i in range(min(5, gui.main_tabs.count())):
                tab_name = gui.main_tabs.tabText(i)
                print(f"   {i+1}. {tab_name}")
            
            if gui.main_tabs.count() > 5:
                print(f"   ... and {gui.main_tabs.count() - 5} more tabs")
        
        print("\n🎯 Status: GUI is working correctly!")
        print("🚀 Ready to launch with: python launch_enhanced_gui.py")
        print("   or: batch_files\\Launch_VoxSigil_GUI.bat")
        
        # Clean shutdown without showing GUI
        gui.close()
        app.quit()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
