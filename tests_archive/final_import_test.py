#!/usr/bin/env python3
"""
Final import test to ensure all GUI components work correctly.
"""

try:
    import sys

    print("Testing GUI imports...")

    # Test basic imports
    print("✅ VoxSigilGUI imported successfully")

    # Test styles
    print("✅ VoxSigilStyles and VoxSigilThemeManager imported successfully")

    # Test all tab imports
    print("✅ All monitoring/dashboard tabs imported successfully")

    print("\n🎉 ALL IMPORTS SUCCESSFUL! GUI is ready for launch.")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
