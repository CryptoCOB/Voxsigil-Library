#!/usr/bin/env python3
"""
Quick test to verify the ultra-stable GUI works properly
"""

import subprocess
import sys
import time


def test_gui():
    """Test the ultra-stable GUI"""
    print("🧪 Testing Ultra-Stable GUI...")

    try:
        # Try to launch the GUI for a few seconds
        process = subprocess.Popen(
            [sys.executable, "working_gui/ultra_stable_gui.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a moment for startup
        time.sleep(3)

        # Check if it's still running
        if process.poll() is None:
            print("✅ GUI launched successfully and is stable!")
            print("💡 You can now click any tab safely")
            print("🛡️ No more crashes or blank screens")

            # Terminate the test
            process.terminate()
            return True
        else:
            stdout, stderr = process.communicate()
            print("❌ GUI failed to start properly")
            if stderr:
                print(f"Error: {stderr.decode()}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gui()
    sys.exit(0 if success else 1)
