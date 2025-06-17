#!/usr/bin/env python3
"""
Simple GUI launch test script
"""

import os
import sys

# Set Qt platform to offscreen to prevent display issues during testing
os.environ["QT_QPA_PLATFORM"] = "windows"

try:
    print("🚀 Starting VoxSigil GUI launch test...")

    # Test basic imports first
    print("📦 Testing imports...")
    from gui.components.pyqt_main import VoxSigilMainWindow

    print("✅ VoxSigilMainWindow imported")

    # Test Qt
    from PyQt5.QtWidgets import QApplication

    print("✅ PyQt5 imported")

    # Create QApplication
    print("🎨 Creating QApplication...")
    app = QApplication(sys.argv)

    # Apply dark theme
    print("🌙 Applying dark theme...")
    try:
        from gui.components.gui_styles import VoxSigilStyles

        styles = VoxSigilStyles()
        app.setStyleSheet(styles.get_dark_theme())
        print("✅ Dark theme applied")
    except Exception as e:
        print(f"⚠️ Theme warning: {e}")

    # Create main window
    print("🪟 Creating main window...")
    window = VoxSigilMainWindow()

    print("📱 Showing window...")
    window.show()

    print("🎉 GUI launched successfully!")
    print("👀 Window should be visible now...")

    # Run the app
    sys.exit(app.exec_())

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
