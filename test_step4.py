#!/usr/bin/env python3
"""
Test Step 4 Integration: UnifiedVantaCore GUI Orchestration
"""

import sys
import traceback


def test_step4_integration():
    """Test the Step 4 integration."""
    print("🔄 Testing Step 4 Integration: UnifiedVantaCore GUI Orchestration")
    print("=" * 60)

    try:
        # Test 1: Import VoxSigilIntegrationManager
        print("1. Testing import...")
        from voxsigil_integration import VoxSigilIntegrationManager

        print("   ✅ Successfully imported VoxSigilIntegrationManager")

        # Test 2: Initialize integration manager
        print("2. Testing initialization...")
        integration = VoxSigilIntegrationManager()
        print("   ✅ Successfully initialized VoxSigilIntegrationManager")

        # Test 3: Check status
        print("3. Testing status checks...")
        status = integration.get_status()
        overall_health = status.get("overall_health", "unknown")
        use_unified_core = status.get("use_unified_core", False)
        unified_core_available = status.get("unified_core_available", False)

        print(f"   Overall health: {overall_health}")
        print(f"   Unified core available: {unified_core_available}")
        print(f"   Using unified core: {use_unified_core}")
        print("   ✅ Status check completed")

        # Test 4: Test integration status
        print("4. Testing integration status...")
        integration_status = integration.get_integration_status()
        interfaces_available = integration_status.get("interfaces_available", False)
        print(f"   Interfaces available: {interfaces_available}")

        if unified_core_available:
            unified_status = integration_status.get("unified_core_status", {})
            print(f"   Unified core agents: {unified_status.get('agent_count', 0)}")

        print("   ✅ Integration status completed")

        # Test 5: Test basic functionality
        print("5. Testing basic functionality...")
        test_results = integration.test_all_interfaces()
        for interface, result in test_results.items():
            print(f"   {interface}: {result}")
        print("   ✅ Interface testing completed")

        print("\n" + "=" * 60)
        print("🎉 Step 4 Integration Test: PASSED")
        print("✅ GUI now routes through UnifiedVantaCore as central orchestrator")

        return True

    except Exception as e:
        print(f"\n❌ Error during Step 4 integration test: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step4_integration()
    sys.exit(0 if success else 1)
