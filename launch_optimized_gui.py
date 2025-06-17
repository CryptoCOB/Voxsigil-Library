#!/usr/bin/env python3
"""
Launch the optimized VoxSigil GUI with lazy loading
This version loads tabs on demand to improve startup time
"""

import logging
import sys
import time

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set up performance logger
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)
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

def main():
    """Launch the optimized VoxSigil GUI with lazy loading"""
    total_start = time.time()
    
    print("=" * 80)
    print("🚀 VoxSigil Optimized GUI Launcher")
    print("   Now featuring lazy-loaded tabs for faster startup!")
    print("=" * 80)
    
    try:
        # Import PyQt5
        start_time = log_time("Starting PyQt5 import")
        from PyQt5.QtWidgets import QApplication
        start_time = log_time("PyQt5 import complete", start_time)
        
        print("✅ PyQt5 loaded successfully")

        # Create application
        start_time = log_time("Creating QApplication")
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Modern, dark-friendly style
        start_time = log_time("QApplication created", start_time)

        # Import and create the GUI
        start_time = log_time("Importing LazyLoadGUI")
        from working_gui.lazy_load_gui import CompleteVoxSigilGUI
        start_time = log_time("LazyLoadGUI import complete", start_time)
        
        print("✅ Optimized GUI system imported")

        # Create the GUI instance
        print("🔄 Creating GUI with lazy-loaded components...")
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
        
        # Log total initialization time
        total_elapsed = time.time() - total_start
        perf_logger.info(f"Total initialization time: {total_elapsed:.4f} seconds")
        print(f"\n⏱️ Total initialization time: {total_elapsed:.4f} seconds")

        print("\n" + "=" * 60)
        print("📋 TABS AVAILABLE:")
        print("=" * 60)

        # List all tabs
        for i in range(tab_count):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            widget_type = type(tab_widget).__name__
            
            # Check if it's a lazy-loaded tab
            is_lazy = widget_type == "LazyTabWidget"
            status = "🎯 Lazy-Loaded (loads on demand)" if is_lazy else "🔧 Always Loaded"
            
            print(f"   {i + 1:2d}. {tab_name:<25} {status}")

        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE IMPROVEMENTS:")
        print("=" * 60)
        print("   • ⏱️ Faster startup time with lazy tab loading")
        print("   • 🧠 Lower memory usage during startup")
        print("   • 📊 Background initialization of non-critical components")
        print("   • 🔄 Automatic loading when tab is selected")
        print("   • 📈 Improved responsiveness during startup")
        print("   • 🔍 Detailed performance logging")

        print("\n" + "=" * 60)
        print("🔍 INSTRUCTIONS:")
        print("=" * 60)
        print("   1. Critical tabs (System Status, Heartbeat) load immediately")
        print("   2. Other tabs load when you click on them or hover over them")
        print("   3. All system components initialize in the background")
        print("   4. System status updates as components come online")
        print("   5. You can start using the core functionality immediately")

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

if __name__ == "__main__":
    main()
