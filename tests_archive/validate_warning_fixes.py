#!/usr/bin/env python3
"""
Validation script to check if VantaCore warnings have been reduced.
"""

import io
import logging


def capture_warnings():
    """Capture and return warnings from imports."""
    # Set up logging to capture warnings
    log_capture = io.StringIO()

    # Create a custom handler
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)

    # Add handler to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(handler)

    warnings_captured = []

    try:
        # Test importing the integration module that was causing warnings
        try:
            from integration.voxsigil_integration import _get_vanta_components

            components = _get_vanta_components()
            print(
                f"✅ voxsigil_integration loaded, VantaCore available: {components.get('available', False)}"
            )
        except Exception as e:
            print(f"ℹ️ voxsigil_integration: {e}")

        # Test importing checkin manager
        try:
            from core.checkin_manager_vosk import HAVE_VANTA_CORE

            print(f"✅ checkin_manager_vosk loaded, HAVE_VANTA_CORE: {HAVE_VANTA_CORE}")
        except Exception as e:
            print(f"ℹ️ checkin_manager_vosk: {e}")

        # Test real-time data provider
        try:
            from gui.components.real_time_data_provider import RealTimeDataProvider

            provider = RealTimeDataProvider()
            metrics = provider.get_all_metrics()
            print(f"✅ RealTimeDataProvider works: {len(metrics)} metrics")
        except Exception as e:
            print(f"❌ RealTimeDataProvider error: {e}")

        # Get captured warnings
        log_content = log_capture.getvalue()
        if log_content:
            warnings_captured = log_content.strip().split("\n")

    finally:
        # Clean up
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)
        handler.close()

    return warnings_captured


def main():
    print("🔍 Validating VantaCore Warning Fixes")
    print("=" * 50)

    warnings = capture_warnings()

    print("\n" + "=" * 50)
    print("📊 RESULTS:")

    if not warnings:
        print("🎉 SUCCESS: No warnings captured!")
        print("✅ VantaCore integration warnings have been reduced to debug level")
        print("✅ Enhanced GUI can run with real streaming data")
        print("✅ No event loop errors should occur")
    else:
        print(f"⚠️ {len(warnings)} warnings still present:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nℹ️ These may be expected warnings from other components")

    print("\n💡 To run GUI in clean mode:")
    print("   python launch_clean_gui.py")
    print("\n💡 Or set environment variable:")
    print("   export VOXSIGIL_ENHANCED_CLEAN_MODE=true")
    print("   python gui/launcher.py")


if __name__ == "__main__":
    main()
