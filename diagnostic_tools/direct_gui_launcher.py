#!/usr/bin/env python3
"""
Direct GUI Launcher - Hang Detection and Bypass
This launcher isolates each step to identify exactly where the hang occurs
"""

import sys
import os
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DirectGUI")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def step_by_step_launch():
    """Launch GUI step by step with timeout detection"""
    
    print("🔍 DIRECT GUI LAUNCHER - HANG DETECTION")
    print("=" * 50)
    
    try:
        # Step 1: Basic imports
        print("Step 1: Testing basic imports...")
        start_time = time.time()
        
        from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        from PyQt5.QtCore import Qt
        
        elapsed = time.time() - start_time
        print(f"✅ PyQt5 imports OK ({elapsed:.2f}s)")
        
        # Step 2: Test data provider import
        print("Step 2: Testing data provider import...")
        start_time = time.time()
        
        from gui.components.real_time_data_provider import RealTimeDataProvider
        
        elapsed = time.time() - start_time
        print(f"✅ Data provider import OK ({elapsed:.2f}s)")
        
        # Step 3: Test data provider instantiation
        print("Step 3: Testing data provider instantiation...")
        start_time = time.time()
        
        provider = RealTimeDataProvider()
        
        elapsed = time.time() - start_time
        print(f"✅ Data provider created OK ({elapsed:.2f}s)")
        
        # Step 4: Test QApplication creation
        print("Step 4: Testing QApplication creation...")
        start_time = time.time()
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        elapsed = time.time() - start_time
        print(f"✅ QApplication created OK ({elapsed:.2f}s)")
        
        # Step 5: Test minimal window creation
        print("Step 5: Testing minimal window creation...")
        start_time = time.time()
        
        window = QMainWindow()
        window.setWindowTitle("VoxSigil - Direct Launch Test")
        window.setGeometry(100, 100, 800, 600)
        
        # Simple content
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Get some data to display
        try:
            metrics = provider.get_all_metrics()
            metric_count = len(metrics)
            data_status = f"✅ {metric_count} metrics available"
        except Exception as e:
            data_status = f"⚠️ Data provider error: {e}"
        
        label = QLabel(f"""
🎉 DIRECT GUI LAUNCH SUCCESSFUL!

This minimal GUI confirms that:
✅ PyQt5 is working correctly
✅ RealTimeDataProvider is functional
✅ Basic GUI creation works
✅ No hanging in core components

Data Provider Status:
{data_status}

The hanging issue is likely in the complex
enhanced GUI components, not the core system.

Next steps:
1. Use this minimal GUI as baseline
2. Add components progressively  
3. Identify which specific component causes hanging

Close this window to continue testing...
        """)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 11px;
            padding: 20px;
            background-color: #1e1e1e;
            color: #00ff00;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        """)
        
        layout.addWidget(label)
        central_widget.setLayout(layout)
        window.setCentralWidget(central_widget)
        
        elapsed = time.time() - start_time
        print(f"✅ Minimal window created OK ({elapsed:.2f}s)")
        
        # Step 6: Show window
        print("Step 6: Showing window...")
        start_time = time.time()
        
        window.show()
        
        elapsed = time.time() - start_time
        print(f"✅ Window shown OK ({elapsed:.2f}s)")
        
        # Step 7: Test event loop
        print("Step 7: Starting event loop...")
        print("💡 Window should be visible - close it to continue tests")
        
        return app.exec_()
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_problematic_imports():
    """Test imports that might be causing the hang"""
    
    print("\n🔍 TESTING PROBLEMATIC IMPORTS")
    print("=" * 50)
    
    problematic_imports = [
        ("complete_enhanced_gui", "gui.components.complete_enhanced_gui"),
        ("pyqt_main_unified", "gui.components.pyqt_main_unified"),  
        ("enhanced_model_tab", "gui.components.enhanced_model_tab"),
        ("enhanced_training_tab", "gui.components.enhanced_training_tab"),
        ("enhanced_visualization_tab", "gui.components.enhanced_visualization_tab"),
        ("streaming_dashboard", "gui.components.streaming_dashboard"),
    ]
    
    for name, module_path in problematic_imports:
        try:
            print(f"Testing {name}...")
            start_time = time.time()
            
            __import__(module_path)
            
            elapsed = time.time() - start_time
            print(f"✅ {name} import OK ({elapsed:.2f}s)")
            
            if elapsed > 5.0:
                print(f"⚠️ {name} takes a long time to import ({elapsed:.2f}s)")
                
        except ImportError as e:
            print(f"⚠️ {name} not available: {e}")
        except Exception as e:
            print(f"❌ {name} import failed: {e}")

def test_enhanced_gui_creation():
    """Test creating the enhanced GUI to see where it hangs"""
    
    print("\n🔍 TESTING ENHANCED GUI CREATION")
    print("=" * 50)
    
    try:
        print("Testing CompleteEnhancedGUI import...")
        start_time = time.time()
        
        from gui.components.complete_enhanced_gui import CompleteEnhancedGUI
        
        elapsed = time.time() - start_time
        print(f"✅ CompleteEnhancedGUI import OK ({elapsed:.2f}s)")
        
        print("Testing CompleteEnhancedGUI instantiation...")
        start_time = time.time()
        
        # This is where it might hang
        gui = CompleteEnhancedGUI()
        
        elapsed = time.time() - start_time
        print(f"✅ CompleteEnhancedGUI created OK ({elapsed:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced GUI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    logger.info("🔍 Testing prerequisites...")
    
    # Test PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        logger.info("✅ PyQt5 is available")
    except ImportError as e:
        logger.error(f"❌ PyQt5 not available: {e}")
        logger.error("Please install PyQt5: pip install PyQt5")
        return False
    
    # Test real-time data provider
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider
        provider = RealTimeDataProvider()
        metrics = provider.get_all_metrics()
        logger.info(f"✅ RealTimeDataProvider works: {len(metrics)} metrics available")
    except Exception as e:
        logger.error(f"❌ RealTimeDataProvider error: {e}")
        return False
    
    return True

def launch_gui():
    """Launch the enhanced GUI directly."""
    logger.info("🚀 Launching Enhanced VoxSigil GUI")
    logger.info("Using real-time streaming data without VantaCore dependencies")
    
    try:
        # Import PyQt5
        from PyQt5.QtWidgets import QApplication
        import sys
        
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Try to import the main GUI class
        try:
            from gui.components.pyqt_main_unified import VoxSigilApp
            main_class = VoxSigilApp
            logger.info("✅ Using unified VoxSigil GUI")
        except ImportError:
            try:
                from gui.components.pyqt_main import VoxSigilApp
                main_class = VoxSigilApp
                logger.info("✅ Using main VoxSigil GUI")
            except ImportError:
                logger.error("❌ Could not import VoxSigil GUI class")
                return 1
        
        # Create and show the main window
        logger.info("🎯 Creating main window...")
        window = main_class()
        window.show()
        
        logger.info("✅ Enhanced GUI launched successfully!")
        logger.info("💡 GUI is using real-time streaming data")
        logger.info("💡 Close the GUI window to exit")
        
        # Run the application
        return app.exec_()
        
    except Exception as e:
        logger.error(f"❌ Failed to launch GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("🎯 VoxSigil Enhanced GUI - Direct Launcher")
    logger.info("=" * 60)
    
    # Setup paths
    setup_paths()
    
    # Test prerequisites
    if not test_prerequisites():
        logger.error("❌ Prerequisites check failed")
        return 1
    
    # Launch GUI
    exit_code = launch_gui()
    
    if exit_code == 0:
        logger.info("✅ GUI closed successfully")
    else:
        logger.error(f"❌ GUI exited with error code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
