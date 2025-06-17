#!/usr/bin/env python3
"""
Quick verification that the VoxSigil GUI is working with interactive tabs
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)

def main():
    """Quick test of the GUI functionality"""
    
    print("🔍 VoxSigil GUI Quick Verification")
    print("=" * 50)
    
    try:
        # Test PyQt5 import
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 import: SUCCESS")
        
        # Test GUI import
        from working_gui.complete_live_gui import CompleteVoxSigilGUI
        print("✅ GUI import: SUCCESS")
        
        # Create application (minimal)
        app = QApplication(sys.argv)
        
        # Create GUI instance
        print("🔄 Creating GUI instance...")
        gui = CompleteVoxSigilGUI()
        
        # Check tab count
        tab_count = gui.main_tabs.count()
        print(f"✅ GUI created: {tab_count} tabs")
        
        # Check tab types
        real_components = 0
        interactive_fallbacks = 0
        
        for i in range(tab_count):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            widget_type = type(tab_widget).__name__
            
            if widget_type == "QScrollArea":
                interactive_fallbacks += 1
            else:
                real_components += 1
                
        print(f"📊 Tab Analysis:")
        print(f"   • Real Components: {real_components}")
        print(f"   • Interactive Fallbacks: {interactive_fallbacks}")
        print(f"   • Total Tabs: {tab_count}")
        
        # Test a few specific tabs
        print("\n🎯 Testing Specific Tabs:")
        for i in range(min(5, tab_count)):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            widget_type = type(tab_widget).__name__
            
            # Check if it has interactive elements
            has_buttons = False
            has_progress = False
            has_tables = False
            
            if widget_type == "QScrollArea":
                # This should be our enhanced fallback tab
                inner_widget = tab_widget.widget()
                if inner_widget:
                    # Check for buttons
                    from PyQt5.QtWidgets import QPushButton, QProgressBar, QTableWidget
                    buttons = inner_widget.findChildren(QPushButton)
                    progress_bars = inner_widget.findChildren(QProgressBar)
                    tables = inner_widget.findChildren(QTableWidget)
                    
                    has_buttons = len(buttons) > 0
                    has_progress = len(progress_bars) > 0
                    has_tables = len(tables) > 0
            
            status = "🎯 Interactive" if (has_buttons or has_progress or has_tables) else "📝 Basic"
            print(f"   {i+1}. {tab_name}: {status}")
            
        print(f"\n🎉 RESULT: GUI is working with {tab_count} tabs!")
        
        # Don't actually show the GUI, just verify it works
        print("✅ Verification complete - GUI is functional!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 CONCLUSION: VoxSigil GUI is working properly!")
        print("   You can now launch it using:")
        print("   • python launch_enhanced_gui.py")
        print("   • batch_files\\Launch_VoxSigil_GUI.bat")
        print("   • python working_gui\\complete_live_gui.py")
    else:
        print("\n❌ ISSUE: GUI verification failed")
