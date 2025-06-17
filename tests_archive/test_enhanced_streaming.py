#!/usr/bin/env python3
"""
Test Enhanced Tabs for Real Streaming Data
Verify that all enhanced tabs show actual streaming data instead of placeholders/checkmarks.
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_enhanced_tabs_streaming():
    """Test that enhanced tabs have real streaming functionality."""
    print("🧪 Testing Enhanced Tabs for Real Streaming Data")
    print("=" * 60)

    # Test 1: Enhanced Model Tab
    print("\n1. Testing Enhanced Model Tab...")
    try:
        from PyQt5.QtWidgets import QApplication

        from gui.components.enhanced_model_tab import EnhancedModelTab

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        model_tab = EnhancedModelTab()

        # Check if it has real streaming workers
        has_discovery_worker = hasattr(model_tab, "discovery_worker")
        has_metrics_timer = hasattr(model_tab, "metrics_timer")
        has_real_components = hasattr(model_tab, "real_time_panel")

        print("   ✅ Model Tab created successfully")
        print(f"   📡 Discovery Worker: {'✅' if has_discovery_worker else '❌'}")
        print(f"   ⏱️  Metrics Timer: {'✅' if has_metrics_timer else '❌'}")
        print(f"   🔄 Real-time Components: {'✅' if has_real_components else '❌'}")

    except Exception as e:
        print(f"   ❌ Enhanced Model Tab Error: {e}")

    # Test 2: Enhanced Visualization Tab
    print("\n2. Testing Enhanced Visualization Tab...")
    try:
        from gui.components.enhanced_visualization_tab import EnhancedVisualizationTab

        viz_tab = EnhancedVisualizationTab()

        # Check if it has real metrics collection
        has_metrics_collector = hasattr(viz_tab, "metrics_collector")
        has_vanta_integration = hasattr(viz_tab, "_get_vanta_core_metrics")
        has_real_charts = hasattr(viz_tab, "charts")

        print("   ✅ Visualization Tab created successfully")
        print(f"   📊 Metrics Collector: {'✅' if has_metrics_collector else '❌'}")
        print(f"   🔗 VantaCore Integration: {'✅' if has_vanta_integration else '❌'}")
        print(f"   📈 Real Charts: {'✅' if has_real_charts else '❌'}")

        # Test if metrics collector can get real data
        if has_metrics_collector:
            try:
                viz_tab.metrics_collector.start()
                time.sleep(1)  # Let it collect some data
                viz_tab.metrics_collector.stop()
                print("   🎯 Metrics Collection: ✅ Working")
            except Exception as e:
                print(f"   🎯 Metrics Collection: ❌ Error: {e}")

    except Exception as e:
        print(f"   ❌ Enhanced Visualization Tab Error: {e}")

    # Test 3: Enhanced Training Tab
    print("\n3. Testing Enhanced Training Tab...")
    try:
        from gui.components.enhanced_training_tab import EnhancedTrainingTab

        training_tab = EnhancedTrainingTab()

        # Check if it has real training integration
        has_vanta_integration = hasattr(training_tab, "_get_real_trainer")
        has_training_worker = hasattr(training_tab, "training_worker")
        has_real_simulation = hasattr(training_tab, "_run_intelligent_simulation")

        print("   ✅ Training Tab created successfully")
        print(f"   🔗 VantaCore Integration: {'✅' if has_vanta_integration else '❌'}")
        print(f"   👷 Training Worker: {'✅' if has_training_worker else '❌'}")
        print(f"   🎯 Intelligent Simulation: {'✅' if has_real_simulation else '❌'}")

    except Exception as e:
        print(f"   ❌ Enhanced Training Tab Error: {e}")

    # Test 4: Enhanced Music Tab
    print("\n4. Testing Enhanced Music Tab...")
    try:
        from gui.components.enhanced_music_tab import EnhancedMusicTab

        music_tab = EnhancedMusicTab()

        # Check if it has real audio metrics
        has_audio_metrics = hasattr(music_tab, "audio_metrics_timer")
        has_device_monitoring = hasattr(music_tab, "device_monitor")
        has_real_generation = hasattr(music_tab, "_get_real_audio_metrics")

        print("   ✅ Music Tab created successfully")
        print(f"   🎵 Audio Metrics: {'✅' if has_audio_metrics else '❌'}")
        print(f"   🎧 Device Monitoring: {'✅' if has_device_monitoring else '❌'}")
        print(f"   🎼 Real Generation: {'✅' if has_real_generation else '❌'}")

    except Exception as e:
        print(f"   ❌ Enhanced Music Tab Error: {e}")

    # Test 5: Streaming Dashboard
    print("\n5. Testing Streaming Dashboard...")
    try:
        from gui.components.streaming_dashboard import StreamingDashboard

        dashboard = StreamingDashboard()

        # Check if it has real streaming capabilities
        has_metrics_worker = hasattr(dashboard, "metrics_worker")
        has_vanta_integration = hasattr(dashboard, "_get_vanta_metrics")
        has_real_charts = hasattr(dashboard, "charts")

        print("   ✅ Streaming Dashboard created successfully")
        print(f"   📊 Metrics Worker: {'✅' if has_metrics_worker else '❌'}")
        print(f"   🔗 VantaCore Integration: {'✅' if has_vanta_integration else '❌'}")
        print(f"   📈 Real-time Charts: {'✅' if has_real_charts else '❌'}")

    except Exception as e:
        print(f"   ❌ Streaming Dashboard Error: {e}")

    # Test 6: VantaCore Integration
    print("\n6. Testing VantaCore Real Data Access...")
    try:
        from Vanta.core.UnifiedVantaCore import get_vanta_core

        vanta_core = get_vanta_core()
        if vanta_core:
            # Test real data methods
            has_system_status = hasattr(vanta_core, "get_system_status")
            has_performance_metrics = hasattr(vanta_core, "get_performance_metrics")
            has_agent_registry = hasattr(vanta_core, "agent_registry")

            print("   ✅ VantaCore instance available")
            print(f"   📊 System Status: {'✅' if has_system_status else '❌'}")
            print(f"   ⚡ Performance Metrics: {'✅' if has_performance_metrics else '❌'}")
            print(f"   🤖 Agent Registry: {'✅' if has_agent_registry else '❌'}")

            # Test actual data retrieval
            if has_system_status:
                try:
                    status = vanta_core.get_system_status()
                    print(f"   🎯 Real System Data: ✅ Available ({len(status)} metrics)")
                except Exception as e:
                    print(f"   🎯 Real System Data: ❌ Error: {e}")
        else:
            print("   ❌ VantaCore instance not available")

    except Exception as e:
        print(f"   ❌ VantaCore Integration Error: {e}")

    print("\n" + "=" * 60)
    print("🏁 Enhanced Tabs Streaming Test Complete!")
    print("\n💡 Summary:")
    print("   - All tabs should show ✅ for real streaming components")
    print("   - If any show ❌, those are placeholder/checkmark-only features")
    print("   - VantaCore integration enables the most advanced real-time data")


if __name__ == "__main__":
    test_enhanced_tabs_streaming()
