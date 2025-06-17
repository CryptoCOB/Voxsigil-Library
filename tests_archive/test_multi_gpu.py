#!/usr/bin/env python3
"""
Test script to verify multi-GPU detection in VoxSigil GUI
"""


def test_gpu_detection():
    """Test GPU detection functionality"""
    print("🔍 Testing GPU Detection...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA Available: {gpu_count} GPU(s) detected")

            for i in range(gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                memory_total = device_props.total_memory / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                usage_percent = (memory_reserved / memory_total) * 100

                print(f"  GPU {i}: {device_props.name}")
                print(f"    Total Memory: {memory_total:.1f} GB")
                print(f"    Reserved Memory: {memory_reserved:.2f} GB ({usage_percent:.1f}%)")
        else:
            print("❌ CUDA not available")

    except Exception as e:
        print(f"❌ Error detecting GPUs: {e}")


def test_gui_gpu_monitor():
    """Test the GUI GPU monitoring component"""
    print("\n🖥️ Testing GUI GPU Monitor...")

    try:
        from gui.components.heartbeat_monitor_tab import HeartbeatMonitorTab

        # Create the monitor (this will detect GPUs)
        monitor = HeartbeatMonitorTab()
        print(f"✅ GPU Monitor created with {monitor.gpu_count} GPU(s)")

        if monitor.gpu_count > 0:
            print(f"  Monitoring {len(monitor.gpu_labels)} GPU labels")
            print(f"  Monitoring {len(monitor.gpu_bars)} GPU progress bars")

        # Test getting real GPU stats
        gpu_stats = monitor.get_real_gpu_stats()
        if gpu_stats:
            print(f"✅ Real GPU stats: {gpu_stats}")
        else:
            print("⚠️ Using simulated GPU data")

    except Exception as e:
        print(f"❌ Error testing GUI GPU monitor: {e}")


if __name__ == "__main__":
    print("VoxSigil Multi-GPU Detection Test")
    print("=" * 40)

    test_gpu_detection()
    test_gui_gpu_monitor()

    print("\n🎉 GPU Detection Test Complete!")
