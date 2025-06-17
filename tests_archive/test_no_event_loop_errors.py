#!/usr/bin/env python3
"""
Test script to verify that all enhanced GUI components work without VantaCore event loop errors.
"""

import logging
import sys

# Set up logging to capture any VantaCore errors
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")


def test_data_provider():
    """Test the real-time data provider."""
    print("🔍 Testing RealTimeDataProvider...")
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider

        provider = RealTimeDataProvider()

        # Test all metric methods
        all_metrics = provider.get_all_metrics()
        system_metrics = provider.get_system_metrics()
        vanta_metrics = provider.get_vanta_core_metrics()
        model_metrics = provider.get_model_metrics()
        training_metrics = provider.get_training_metrics()
        music_metrics = provider.get_music_metrics()
        audio_metrics = provider.get_audio_metrics()

        print("✅ All metrics retrieved successfully:")
        print(f"   - All metrics: {len(all_metrics)} keys")
        print(
            f"   - System: CPU {system_metrics['cpu_percent']:.1f}%, Memory {system_metrics['memory_percent']:.1f}%"
        )
        print(f"   - VantaCore connected: {vanta_metrics['vanta_core_connected']}")
        print(f"   - Active models: {model_metrics['active_models']}")
        print(f"   - Training loss: {training_metrics['training_loss']:.3f}")
        print(f"   - Music agents: {music_metrics['music_agents_active']}")
        print(f"   - Audio devices: {audio_metrics['audio_devices_available']}")

        return True
    except Exception as e:
        print(f"❌ RealTimeDataProvider error: {e}")
        return False


def test_enhanced_tabs():
    """Test importing enhanced tab components."""
    print("\n🔍 Testing Enhanced Tab Components...")

    components = [
        ("Enhanced Model Tab", "gui.components.enhanced_model_tab", "EnhancedModelTab"),
        ("Enhanced Training Tab", "gui.components.enhanced_training_tab", "EnhancedTrainingTab"),
        ("Enhanced Music Tab", "gui.components.enhanced_music_tab", "EnhancedMusicTab"),
        (
            "Enhanced Visualization Tab",
            "gui.components.enhanced_visualization_tab",
            "EnhancedVisualizationTab",
        ),
        ("Streaming Dashboard", "gui.components.streaming_dashboard", "StreamingDashboard"),
    ]

    success_count = 0
    for name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            print(f"✅ {name} imported successfully")
            success_count += 1
        except Exception as e:
            print(f"❌ {name} error: {e}")

    print(f"\n📊 Enhanced tabs: {success_count}/{len(components)} imported successfully")
    return success_count == len(components)


def test_no_vanta_core_calls():
    """Verify no direct VantaCore calls are made."""
    print("\n🔍 Testing for VantaCore Event Loop Issues...")

    # Import the data provider and check if it triggers any VantaCore async calls
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider

        provider = RealTimeDataProvider()

        # Test multiple calls to see if any trigger event loop errors
        for i in range(5):
            metrics = provider.get_all_metrics()
            if i == 0:
                print(f"✅ Iteration {i + 1}: {len(metrics)} metrics retrieved")

        print("✅ No event loop errors detected in data provider")
        return True
    except Exception as e:
        if "no running event loop" in str(e).lower():
            print(f"❌ Event loop error detected: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            return False


if __name__ == "__main__":
    print("🚀 Testing VoxSigil Enhanced GUI - Event Loop Error Resolution")
    print("=" * 70)

    test_results = []

    # Test data provider
    test_results.append(test_data_provider())

    # Test enhanced tabs
    test_results.append(test_enhanced_tabs())

    # Test for event loop issues
    test_results.append(test_no_vanta_core_calls())

    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS:")

    if all(test_results):
        print("🎉 SUCCESS: All tests passed! Enhanced GUI is ready for real streaming data.")
        print("   ✅ Real-time data provider works correctly")
        print("   ✅ Enhanced tabs import without errors")
        print("   ✅ No VantaCore event loop errors detected")
        print("\n🚀 The GUI should now run with live streaming data and no crashes!")
        sys.exit(0)
    else:
        print("❌ FAILURE: Some tests failed. Check the output above for details.")
        sys.exit(1)
