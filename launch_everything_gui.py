#!/usr/bin/env python3
"""
VoxSigil Complete System Launcher
Launches the entire VoxSigil system through the GUI interface.
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Launch the complete VoxSigil system."""
    print(
        "🚀 Launching VoxSigil Complete System..."
    )  # Try different GUI options in order of preference
    gui_options = [
        ("working_gui/ultra_stable_gui.py", "Ultra-Stable GUI"),
        ("working_gui/crash_proof_enhanced_gui.py", "Crash-Proof Enhanced GUI"),
        ("working_gui/standalone_enhanced_gui.py", "Standalone Enhanced GUI"),
        ("working_gui/optimized_enhanced_gui.py", "Optimized Enhanced GUI"),
        ("scripts/ultra_minimal_gui.py", "Ultra Minimal GUI"),
    ]

    for gui_path, gui_name in gui_options:
        full_path = Path(gui_path)
        if full_path.exists():
            print(f"✅ Found {gui_name} at {gui_path}")
            try:
                # Import and run the GUI
                import importlib.util

                spec = importlib.util.spec_from_file_location("gui_module", full_path)
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)

                # Try to find and call the main function
                if hasattr(gui_module, "main"):
                    print(f"🎯 Launching {gui_name}...")
                    gui_module.main()
                    return
                elif hasattr(gui_module, "run"):
                    print(f"🎯 Launching {gui_name}...")
                    gui_module.run()
                    return
                else:
                    print(f"⚠️ {gui_name} doesn't have main() or run() function")

            except Exception as e:
                print(f"❌ Failed to launch {gui_name}: {e}")
                continue

    # Fallback: try to import from working_gui directly
    print("🔄 Trying direct imports...")
    try:
        sys.path.append(str(Path("working_gui")))
        from standalone_enhanced_gui import main as gui_main

        print("🎯 Launching Standalone Enhanced GUI...")
        gui_main()
    except ImportError:
        try:
            from crash_proof_enhanced_gui import main as gui_main

            print("🎯 Launching Crash-Proof Enhanced GUI...")
            gui_main()
        except ImportError:
            print("❌ Could not find any working GUI module")
            print("📝 Available options:")
            print("   1. Run: python working_gui/standalone_enhanced_gui.py")
            print("   2. Run: python working_gui/crash_proof_enhanced_gui.py")
            print("   3. Run: python scripts/ultra_minimal_gui.py")


if __name__ == "__main__":
    main()
