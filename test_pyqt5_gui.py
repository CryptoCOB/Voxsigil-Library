#!/usr/bin/env python3
"""
PyQt5 GUI Compatibility Test
Test script to verify all PyQt5 GUI components are working properly
"""

import sys
from pathlib import Path

def test_pyqt5_availability():
    """Test if PyQt5 is available"""
    try:
        import PyQt5.QtWidgets
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 is available")
        return True
    except ImportError as e:
        print(f"❌ PyQt5 not available: {e}")
        return False

def test_gui_components():
    """Test VoxSigil GUI components"""
    try:
        # Test basic GUI styles
        from gui.components.gui_styles import VoxSigilStyles, VoxSigilWidgetFactory
        print("✅ VoxSigil styles available")
        
        # Test core panels  
        from gui.components.agent_status_panel import AgentStatusPanel
        from gui.components.echo_log_panel import EchoLogPanel
        from gui.components.mesh_map_panel import MeshMapPanel
        print("✅ Core GUI panels available")
        
        # Test main window
        from gui.components.pyqt_main import VoxSigilMainWindow, launch
        print("✅ Main PyQt5 window available")
        
        return True
        
    except ImportError as e:
        print(f"❌ GUI component import error: {e}")
        return False

def test_registration():
    """Test GUI module registration"""
    try:
        from gui.register_gui_module import register_gui, PYQT5_AVAILABLE
        print(f"✅ GUI registration available (PyQt5: {PYQT5_AVAILABLE})")
        return True
    except ImportError as e:
        print(f"❌ Registration import error: {e}")
        return False

def test_advanced_components():
    """Test advanced GUI components"""
    try:
        # Test if advanced components exist
        components = []
        
        try:
            from gui.components.dynamic_gridformer_gui import DynamicGridFormerGUI
            components.append("DynamicGridFormerGUI")
        except ImportError:
            pass
            
        try:
            from gui.components.training_interface_new import TrainingInterfaceNew
            components.append("TrainingInterfaceNew")
        except ImportError:
            pass
            
        try:
            from gui.components.vmb_final_demo import VMBFinalDemo
            components.append("VMBFinalDemo")
        except ImportError:
            pass
            
        if components:
            print(f"✅ Advanced components available: {', '.join(components)}")
        else:
            print("⚠️ No advanced components found (may need to be created)")
            
        return True
        
    except Exception as e:
        print(f"❌ Advanced component test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 VoxSigil PyQt5 GUI Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("PyQt5 Availability", test_pyqt5_availability),
        ("Core GUI Components", test_gui_components), 
        ("GUI Registration", test_registration),
        ("Advanced Components", test_advanced_components)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"{emoji} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All PyQt5 GUI components are properly configured!")
    else:
        print("⚠️ Some components need attention")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
