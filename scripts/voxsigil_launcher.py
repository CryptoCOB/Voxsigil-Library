#!/usr/bin/env python3
"""
Main VoxSigil Application Launcher
Launches the optimized VoxSigil GUI application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Launch the VoxSigil application."""
    try:
        from voxsigil.gui.optimized_enhanced_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
