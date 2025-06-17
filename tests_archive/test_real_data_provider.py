#!/usr/bin/env python3
"""
Test script to verify the real data provider works correctly.
"""


def test_real_data_provider():
    """Test all real data provider functions."""
    try:
        from gui.components.real_time_data_provider import (
            get_all_metrics,
            get_audio_metrics,
            get_system_metrics,
            get_training_metrics,
            get_vanta_metrics,
        )

        print("✅ Real data provider imports successfully")

        # Test each function
        system_metrics = get_system_metrics()
        print(
            f"✅ System metrics: CPU {system_metrics['cpu_percent']:.1f}%, Memory {system_metrics['memory_percent']:.1f}%"
        )

        vanta_metrics = get_vanta_metrics()
        print(
            f"✅ VantaCore metrics: Connected={vanta_metrics['vanta_core_connected']}, Components={vanta_metrics['active_components']}"
        )

        training_metrics = get_training_metrics()
        print(
            f"✅ Training metrics: Loss={training_metrics['training_loss']:.3f}, Accuracy={training_metrics['validation_accuracy']:.3f}"
        )

        audio_metrics = get_audio_metrics()
        print(
            f"✅ Audio metrics: Level={audio_metrics['audio_level']:.1f}dB, Latency={audio_metrics['audio_latency']:.1f}ms"
        )

        all_metrics = get_all_metrics()
        print(f"✅ All metrics aggregated: {len(all_metrics)} total metrics available")

        print("🎉 ALL REAL DATA PROVIDER TESTS PASSED!")
        return True

    except Exception as e:
        print(f"❌ Error testing real data provider: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_real_data_provider()
