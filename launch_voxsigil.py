#!/usr/bin/env python3
"""
VoxSigil GUI Launcher

This script launches the main VoxSigil GUI application.
It provides a clean entry point to the complete GUI with all components.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Launch the complete VoxSigil GUI"""
    try:
        # Import and run the complete GUI
        from working_gui.complete_live_gui_real_components_only import VoxSigilGUI

        app = VoxSigilGUI(sys.argv)
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
