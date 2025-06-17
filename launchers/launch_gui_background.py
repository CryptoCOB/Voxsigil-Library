#!/usr/bin/env python3
"""
VoxSigil GUI Background Launcher
This version runs the GUI in the background
"""

import os
import subprocess
import sys
from pathlib import Path


def create_gui_script():
    """Create a standalone GUI script"""
    script_content = """
import sys
import os
from pathlib import Path

# Setup environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')

# Import and run
from PyQt5.QtWidgets import QApplication
from gui.components.pyqt_main import VoxSigilMainWindow

app = QApplication(sys.argv)
window = VoxSigilMainWindow()
window.show()
app.exec_()
"""

    script_path = Path("voxsigil_gui_standalone.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path


def launch_background():
    """Launch GUI in background process"""
    try:
        print("🚀 Creating standalone GUI launcher...")
        script_path = create_gui_script()

        print("🎬 Starting VoxSigil GUI in background...")

        # Start the GUI in a separate process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print(f"✅ VoxSigil GUI started with PID: {process.pid}")
        print("🪟 GUI window should appear shortly...")
        print("📝 Check the new console window for GUI output")

        return process

    except Exception as e:
        print(f"❌ Launch failed: {e}")
        return None


if __name__ == "__main__":
    process = launch_background()
    if process:
        print("🎉 VoxSigil GUI launch initiated!")
        print("💡 The GUI is running in a separate window")
    else:
        print("❌ Failed to launch GUI")
