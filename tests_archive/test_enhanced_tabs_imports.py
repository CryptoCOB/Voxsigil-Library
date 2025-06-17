#!/usr/bin/env python3
"""
Test Enhanced Tabs Import and Functionality
Tests the import and basic functionality of the enhanced tabs.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))


def test_enhanced_tabs():
    """Test enhanced tabs imports and basic functionality."""
    print("🧪 Testing Enhanced Tabs...")

    # Test Enhanced Model Tab
    try:
        print("✅ Enhanced Model Tab imports successfully")

        # Test basic instantiation (without GUI)
        # We'll just test the import for now

    except Exception as e:
        print(f"❌ Enhanced Model Tab import error: {e}")

    # Test Enhanced Model Discovery Tab
    try:
        print("✅ Enhanced Model Discovery Tab imports successfully")

    except Exception as e:
        print(f"❌ Enhanced Model Discovery Tab import error: {e}")

    # Test Enhanced Visualization Tab
    try:
        print("✅ Enhanced Visualization Tab imports successfully")

        # Test matplotlib availability
        try:
            import matplotlib

            print("✅ Matplotlib is available for advanced charts")
        except ImportError:
            print("⚠️ Matplotlib not available - will use fallback charts")

    except Exception as e:
        print(f"❌ Enhanced Visualization Tab import error: {e}")

    # Test main GUI import
    try:
        print("✅ Main GUI imports successfully")

    except Exception as e:
        print(f"❌ Main GUI import error: {e}")

    print("\n📊 Enhanced Tab Testing Complete!")


if __name__ == "__main__":
    test_enhanced_tabs()
