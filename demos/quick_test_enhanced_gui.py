#!/usr/bin/env python3
"""
Quick Test Enhanced GUI
Test the enhanced tabs to verify they show real streaming data instead of checkmarks.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def quick_test_gui():
    """Quick test of enhanced GUI with focus on streaming data."""
    print("🚀 Quick Test: Enhanced GUI with Real Streaming Data")
    print("=" * 60)

    try:
        from PyQt5.QtWidgets import QApplication

        from gui.components.pyqt_main_unified import VoxSigilMainWindow

        print("✅ Importing GUI components...")

        # Create application
        app = QApplication(sys.argv)
        print("✅ QApplication created")

        # Create main window with enhanced tabs
        window = VoxSigilMainWindow()
        print("✅ Main window created with enhanced tabs")

        # Show window
        window.show()
        print("✅ Window shown")

        print("\n🎯 Enhanced GUI Features:")
        print("   📊 Enhanced Model Tab - Real model discovery & streaming metrics")
        print("   📈 Enhanced Visualization Tab - Live system & VantaCore metrics")
        print("   🎯 Enhanced Training Tab - Real training integration & progress")
        print("   🎵 Enhanced Music Tab - Real audio metrics & device monitoring")
        print("   📊 Live Dashboard - Unified real-time streaming data")

        print("\n💡 Look for:")
        print("   🔴 LIVE indicators showing real-time data")
        print("   📊 Charts updating with streaming metrics")
        print("   🔗 VantaCore integration status")
        print("   ⚡ Real system resource monitoring")
        print("   🤖 Agent activity and coordination metrics")

        print("\n🏃‍♂️ GUI is running! Check tabs for streaming data...")
        print("   (Press Ctrl+C to exit)")

        # Run the application
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("\n👋 GUI test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error testing GUI: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    quick_test_gui()
