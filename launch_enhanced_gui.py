#!/usr/bin/env python3
"""
Enhanced VoxSigil GUI Launcher - Now with Fully Interactive Tabs
This version provides rich, interactive tabs with real controls and functionality
"""

import logging
import sys
import os

# Force UTF-8 encoding to handle special characters properly
os.environ["PYTHONIOENCODING"] = "utf-8"

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Launch the enhanced VoxSigil GUI with interactive tabs"""

    print("=" * 80)
    print("VoxSigil Enhanced GUI Launcher")
    print("   Now featuring fully interactive tabs with real controls!")
    print("=" * 80)

    try:  # Import PyQt5
        from PyQt5.QtWidgets import QApplication, QMessageBox

        print("[OK] PyQt5 loaded successfully")

        # Create application
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Modern, dark-friendly style

        # Import and create the GUI
        from working_gui.complete_live_gui import CompleteVoxSigilGUI

        print("[OK] Complete GUI system imported")

        # Create the GUI instance
        print("[...] Creating GUI with all interactive components...")
        gui = CompleteVoxSigilGUI()

        # Show information about what was created
        tab_count = gui.main_tabs.count()
        print(f"[OK] GUI created successfully with {tab_count} tabs")

        # Show the GUI
        gui.show()
        gui.raise_()  # Bring to front
        gui.activateWindow()  # Give focus

        print("\n" + "=" * 60)
        print("TABS AVAILABLE:")
        print("=" * 60)

        # List all tabs
        for i in range(tab_count):
            tab_name = gui.main_tabs.tabText(i)
            tab_widget = gui.main_tabs.widget(i)
            widget_type = type(tab_widget).__name__

            # Check if it's a real component or fallback
            is_scroll_area = widget_type == "QScrollArea"
            status = (
                "[FALLBACK] Interactive Fallback" if is_scroll_area else "[REAL] Real Component"
            )

            print(f"   {i + 1:2d}. {tab_name:<25} {status}")

        print("\n" + "=" * 60)
        print("FEATURES AVAILABLE:")
        print("=" * 60)
        print("   • Live system metrics with real-time updates")
        print("   • Interactive control panels with working buttons")
        print("   • Configuration settings that respond to changes")
        print("   • Progress bars showing system health")
        print("   • Data tables with live information")
        print("   • Activity logs with real-time event tracking")
        print("   • Auto-refresh functionality")
        print("   • Export capabilities")
        print("   • Start/Stop/Restart system controls")

        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        print("=" * 60)
        print("   1. Click through each tab to explore the interfaces")
        print("   2. Try the buttons - they provide real feedback!")
        print("   3. Check the activity logs to see actions being recorded")
        print("   4. Adjust settings like verbosity and auto-refresh")
        print("   5. Watch the progress bars and metrics update")
        print("   6. Use Start/Stop/Restart to see system state changes")

        print(f"\nGUI is now ready! Total tabs: {tab_count}")
        print("   Press Ctrl+C in this terminal to exit when done")
        print("=" * 80)

        # Run the application
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("\n\nShutting down VoxSigil GUI...")
        print("Goodbye!")

    except Exception as e:
        print(f"\n[ERROR] Error starting GUI: {e}")
        import traceback

        traceback.print_exc()

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
