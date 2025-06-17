#!/usr/bin/env python3
"""
Minimal GUI test
"""

import sys

try:
    print("Testing minimal GUI...")

    from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("VoxSigil Test")
    window.resize(400, 300)

    label = QLabel("VoxSigil GUI Test - Working!")
    window.setCentralWidget(label)

    window.show()

    print("✅ Minimal GUI launched!")

    # Don't call exec_() to avoid hanging
    # app.exec_()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
