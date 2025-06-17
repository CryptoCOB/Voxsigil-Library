#!/usr/bin/env python3
"""
Test script to launch the enhanced GUI in clean mode without VantaCore warnings.
"""

import os
import subprocess
import sys


def launch_clean_gui():
    """Launch the GUI in clean mode."""
    print("🚀 Launching Enhanced VoxSigil GUI in Clean Mode")
    print("=" * 60)
    print("This mode avoids VantaCore initialization to prevent event loop errors.")
    print("The GUI will use real-time streaming data from the system metrics.")
    print("=" * 60)

    # Set environment variable for clean mode
    env = os.environ.copy()
    env["VOXSIGIL_ENHANCED_CLEAN_MODE"] = "true"

    try:
        # Launch the GUI launcher in clean mode
        result = subprocess.run([sys.executable, "gui/launcher.py"], env=env, capture_output=False)

        return result.returncode

    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1


if __name__ == "__main__":
    exit_code = launch_clean_gui()
    if exit_code == 0:
        print("✅ GUI launched successfully!")
    else:
        print("❌ GUI launch failed.")
    sys.exit(exit_code)
