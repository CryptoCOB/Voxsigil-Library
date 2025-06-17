#!/usr/bin/env python3
"""
Enhanced VoxSigil GUI Launcher with Performance Profiling
This version provides rich, interactive tabs with real controls and functionality
while also profiling the application to identify performance bottlenecks
"""

import logging
import sys
import time
import os
from datetime import datetime

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up performance logging
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)

# Create a file handler for performance logs
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
perf_log_file = f"logs/performance_{timestamp}.log"
file_handler = logging.FileHandler(perf_log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
perf_logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("⏱️ PERF: %(message)s"))
perf_logger.addHandler(console_handler)

def log_time(message, start_time=None):
    """Log elapsed time with a message"""
    if start_time is None:
        return time.time()
    
    elapsed = time.time() - start_time
    perf_logger.info(f"{message}: {elapsed:.4f} seconds")
    return time.time()

def profile_import(module_name):
    """Profile the import of a module"""
    start = time.time()
    try:
        module = __import__(module_name, fromlist=['*'])
        elapsed = time.time() - start
        perf_logger.info(f"Import {module_name}: {elapsed:.4f} seconds")
        return module
    except Exception as e:
        elapsed = time.time() - start
        perf_logger.error(f"Import {module_name} failed after {elapsed:.4f} seconds: {e}")
        raise

def main():
    """Launch the enhanced VoxSigil GUI with interactive tabs and performance profiling"""
    total_start = time.time()
    
    print("=" * 80)
    print("🚀 VoxSigil Enhanced GUI Launcher (Performance Profiling)")
    print("   Now featuring fully interactive tabs with real controls!")
    print("=" * 80)
    perf_logger.info("Starting application profiling")

    try:
        # Profile PyQt5 import
        start_time = log_time("Starting PyQt5 import")
        from PyQt5.QtWidgets import QApplication, QMessageBox
        start_time = log_time("PyQt5 import complete", start_time)
        
        print("✅ PyQt5 loaded successfully")

        # Create application
        start_time = log_time("Creating QApplication")
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Modern, dark-friendly style
        start_time = log_time("QApplication created", start_time)

        # Import and create the GUI
        start_time = log_time("Importing CompleteVoxSigilGUI")
        from working_gui.complete_live_gui import CompleteVoxSigilGUI
        start_time = log_time("CompleteVoxSigilGUI import complete", start_time)
        
        print("✅ Complete GUI system imported")

        # Create the GUI instance
        print("🔄 Creating GUI with all interactive components...")
        start_time = log_time("Creating GUI instance")
        gui = CompleteVoxSigilGUI()
        start_time = log_time("GUI instance created", start_time)

        # Show information about what was created
        tab_count = gui.main_tabs.count()
        print(f"✅ GUI created successfully with {tab_count} tabs")

        # Show the GUI
        start_time = log_time("Showing GUI")
        gui.show()
        gui.raise_()  # Bring to front
        gui.activateWindow()  # Give focus
        start_time = log_time("GUI shown and activated", start_time)

        print("\n" + "=" * 60)
        print("📋 TABS AVAILABLE:")
        print("=" * 60)

        # List all tabs
        for i in range(tab_count):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            widget_type = type(tab_widget).__name__

            # Check if it's a real component or fallback
            is_scroll_area = widget_type == "QScrollArea"
            status = (
                "🎯 Interactive Fallback" if is_scroll_area else "🔧 Real Component"
            )

            print(f"   {i + 1:2d}. {tab_name:<25} {status}")

        # Log total initialization time
        total_elapsed = time.time() - total_start
        perf_logger.info(f"Total initialization time: {total_elapsed:.4f} seconds")
        print(f"\n⏱️ Total initialization time: {total_elapsed:.4f} seconds")
        print(f"📊 Performance log written to {perf_log_file}")

        print("\n" + "=" * 60)
        print("🎉 FEATURES AVAILABLE:")
        print("=" * 60)
        print("   • 📊 Live system metrics with real-time updates")
        print("   • 🎛️ Interactive control panels with working buttons")
        print("   • ⚙️ Configuration settings that respond to changes")
        print("   • 📈 Progress bars showing system health")
        print("   • 📋 Data tables with live information")
        print("   • 📝 Activity logs with real-time event tracking")
        print("   • 🔄 Auto-refresh functionality")
        print("   • 📤 Export capabilities")
        print("   • 🎯 Start/Stop/Restart system controls")

        print("\n" + "=" * 60)
        print("🔍 INSTRUCTIONS:")
        print("=" * 60)
        print("   1. Click through each tab to explore the interfaces")
        print("   2. Try the buttons - they provide real feedback!")
        print("   3. Check the activity logs to see actions being recorded")
        print("   4. Adjust settings like verbosity and auto-refresh")
        print("   5. Watch the progress bars and metrics update")
        print("   6. Use Start/Stop/Restart to see system state changes")

        print(f"\n🎯 GUI is now ready! Total tabs: {tab_count}")
        print("   Press Ctrl+C in this terminal to exit when done")
        print("=" * 80)

        # Run the application
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("\n\n🔄 Shutting down VoxSigil GUI...")
        print("✅ Goodbye!")

    except Exception as e:
        print(f"\n❌ Error starting GUI: {e}")
        import traceback

        traceback.print_exc()
        perf_logger.error(f"Application crashed after {time.time() - total_start:.4f} seconds: {e}")

        # Show error dialog if PyQt5 is available
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox

            app = (
                QApplication(sys.argv)
                if not QApplication.instance()
                else QApplication.instance()
            )

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("VoxSigil GUI Error")
            msg.setText(f"Failed to start GUI: {str(e)}")
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
        except ImportError as qt_error:
            print(f"Failed to start GUI and Qt not available for error dialog: {e}")
            print(f"Qt import error: {qt_error}")
        except Exception as dialog_error:
            print(f"Failed to start GUI and error dialog failed: {e}")
            print(f"Dialog error: {dialog_error}")


if __name__ == "__main__":
    main()
