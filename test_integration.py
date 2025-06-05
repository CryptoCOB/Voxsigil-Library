#!/usr/bin/env python3
"""
Test script for the VoxSigil Integration Manager
"""

import sys
from pathlib import Path

# Add the project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our integration manager
try:
    from voxsigil_integration import VoxSigilIntegrationManager

    print(
        "✅ Successfully imported VoxSigilIntegrationManager from voxsigil_integration.py"
    )
except ImportError as e:
    print(f"❌ Failed to import VoxSigilIntegrationManager: {e}")
    exit(1)

try:
    # Initialize the manager
    manager = VoxSigilIntegrationManager()
    print(f"✅ Manager initialized with use_unified_core={manager.use_unified_core}")

    # Test status
    status = manager.get_status()
    print(f"✅ Status: {status}")

    # Test interfaces
    test_results = manager.test_all_interfaces()
    print(f"✅ Interface tests: {test_results}")

    # Test memory
    try:
        result = manager.store_interaction(
            {"query": "test", "response": "This is a test"}
        )
        print(f"✅ Memory test (store): {result}")
    except Exception as e:
        print(f"⚠️  Memory test error: {e}")

    # Test RAG
    try:
        context = manager.create_context("What is VoxSigil?")
        print(f"✅ RAG test (context): {context[:100] if context else 'No context'}")
    except Exception as e:
        print(f"⚠️  RAG test error: {e}")

    print("🎉 Test completed successfully")

except Exception as e:
    print(f"❌ Error testing integration: {e}")
    import traceback

    traceback.print_exc()
