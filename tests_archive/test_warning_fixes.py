#!/usr/bin/env python3
"""
Test the warning fixes for VantaCore imports.
"""

import logging
import sys

# Set up logging to capture warnings
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_integration_warnings():
    """Test if the integration warnings are reduced."""
    print("🔍 Testing integration warnings...")

    # Capture warnings
    import io
    from contextlib import redirect_stderr

    warning_output = io.StringIO()

    try:
        with redirect_stderr(warning_output):
            # Test voxsigil_integration
            try:
                from integration.voxsigil_integration import _get_vanta_components

                components = _get_vanta_components()
                print(
                    f"✅ voxsigil_integration imported, VantaCore available: {components.get('available', False)}"
                )
            except Exception as e:
                print(f"❌ voxsigil_integration error: {e}")

            # Test checkin_manager_vosk
            try:
                from core.checkin_manager_vosk import HAVE_VANTA_CORE

                print(f"✅ checkin_manager_vosk imported, HAVE_VANTA_CORE: {HAVE_VANTA_CORE}")
            except Exception as e:
                print(f"❌ checkin_manager_vosk error: {e}")

        warnings = warning_output.getvalue()
        if warnings:
            print("⚠️ Warnings captured:")
            print(warnings)
            return False
        else:
            print("✅ No warnings captured!")
            return True

    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def test_real_time_data_provider():
    """Test that the real-time data provider works without warnings."""
    print("\n🔍 Testing real-time data provider...")

    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider

        provider = RealTimeDataProvider()

        # Test all metrics
        all_metrics = provider.get_all_metrics()
        vanta_metrics = provider.get_vanta_core_metrics()

        print(f"✅ RealTimeDataProvider works: {len(all_metrics)} metrics")
        print(f"✅ VantaCore connected: {vanta_metrics['vanta_core_connected']}")

        return True
    except Exception as e:
        print(f"❌ RealTimeDataProvider error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Warning Fixes for VantaCore Integration")
    print("=" * 60)

    test_results = []

    # Test integration warnings
    test_results.append(test_integration_warnings())

    # Test data provider
    test_results.append(test_real_time_data_provider())

    print("\n" + "=" * 60)
    print("📋 RESULTS:")

    if all(test_results):
        print("🎉 SUCCESS: All tests passed! Warnings should be reduced.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check output above.")
        sys.exit(1)
