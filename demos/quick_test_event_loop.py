#!/usr/bin/env python3

# Quick test to verify no event loop errors
try:
    from gui.components.real_time_data_provider import RealTimeDataProvider

    provider = RealTimeDataProvider()
    metrics = provider.get_all_metrics()
    print(f"SUCCESS: RealTimeDataProvider works! Got {len(metrics)} metrics")

    # Test VantaCore metrics specifically
    vanta_metrics = provider.get_vanta_core_metrics()
    print(f"VantaCore connected: {vanta_metrics['vanta_core_connected']}")
    print("No event loop errors detected!")

except Exception as e:
    print(f"ERROR: {e}")
    if "no running event loop" in str(e):
        print("DETECTED: Event loop error still present")
    else:
        print("Different error type")
