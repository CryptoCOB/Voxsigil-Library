#!/usr/bin/env python3
"""
VoxSigil Integration Test Script

This script tests the fixed VoxSigil integration module, verifying that the interface compatibility issues
have been resolved and all core functionality works correctly.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import and test both the original and fixed integration modules
try:
    # Import the original integration module
    from GUI.components.voxsigil_integration import (
        VoxSigilIntegrationManager as OriginalIntegrationManager,
    )

    has_original = True
except ImportError as e:
    print(f"Could not import original integration module: {e}")
    has_original = False

try:
    # Import the fixed integration module
    from GUI.components.voxsigil_integration_fixed import (
        VoxSigilIntegrationManager as FixedIntegrationManager,
    )

    has_fixed = True
except ImportError as e:
    print(f"Could not import fixed integration module: {e}")
    has_fixed = False


def test_integration_manager(manager_class, name: str) -> Dict[str, Any]:
    """Test an integration manager class."""
    print(f"\n---- Testing {name} ----")
    results = {"name": name, "tests": {}}

    try:
        # Initialize the manager
        print(f"Initializing {name}...")
        manager = manager_class()
        results["tests"]["initialization"] = "PASS"
        print("✅ Initialization successful")
    except Exception as e:
        results["tests"]["initialization"] = f"FAIL: {str(e)}"
        print(f"❌ Initialization failed: {e}")
        return results

    # Test memory methods
    try:
        print("Testing memory interface...")
        interaction = {
            "query": "test query",
            "response": "test response",
            "timestamp": "2024-01-01",
        }
        store_result = manager.store_interaction(interaction)
        retrieve_result = manager.retrieve_interactions()
        search_result = manager.search_memory("test")

        results["tests"]["memory_interface"] = "PASS"
        print("✅ Memory interface working")
        print(f"  - Store result: {store_result}")
        print(f"  - Retrieved {len(retrieve_result)} interactions")
        print(f"  - Search returned {len(search_result)} results")
    except Exception as e:
        results["tests"]["memory_interface"] = f"FAIL: {str(e)}"
        print(f"❌ Memory interface failed: {e}")

    # Test RAG methods
    try:
        print("Testing RAG interface...")
        context = manager.create_context("test query")
        enhanced_context = manager.inject_voxsigil_context(
            context, {"additional": "data"}
        )
        search_results = manager.search_rag_context("test query")

        results["tests"]["rag_interface"] = "PASS"
        print("✅ RAG interface working")
        print(f"  - Context: {context[:50]}...")
        print(f"  - Enhanced context: {enhanced_context[:50]}...")
        print(f"  - Search returned {len(search_results)} results")
    except Exception as e:
        results["tests"]["rag_interface"] = f"FAIL: {str(e)}"
        print(f"❌ RAG interface failed: {e}")

    # Test learning methods
    try:
        print("Testing learning interface...")
        start_result = manager.start_learning_mode()
        insights = manager.get_learning_insights()
        stop_result = manager.stop_learning_mode()

        results["tests"]["learning_interface"] = "PASS"
        print("✅ Learning interface working")
        print(f"  - Start learning: {start_result}")
        print(f"  - Insights: {insights[:50]}...")
        print(f"  - Stop learning: {stop_result}")
    except Exception as e:
        results["tests"]["learning_interface"] = f"FAIL: {str(e)}"
        print(f"❌ Learning interface failed: {e}")

    # Test model methods
    try:
        print("Testing model interface...")
        embeddings = manager.generate_embeddings("test query")
        models = manager.get_available_models()

        results["tests"]["model_interface"] = "PASS"
        print("✅ Model interface working")
        print(f"  - Embeddings length: {len(embeddings)}")
        print(f"  - Available models: {len(models)}")
    except Exception as e:
        results["tests"]["model_interface"] = f"FAIL: {str(e)}"
        print(f"❌ Model interface failed: {e}")

    # Test integration status
    try:
        print("Testing integration status...")
        status = manager.get_integration_status()
        test_results = manager.test_all_interfaces()

        results["tests"]["status"] = "PASS"
        print("✅ Status methods working")
        print(f"  - Integration status: {status.get('overall_health', 'unknown')}")
        print(f"  - Test results: {len(test_results)} interfaces tested")
    except Exception as e:
        results["tests"]["status"] = f"FAIL: {str(e)}"
        print(f"❌ Status methods failed: {e}")

    # Calculate overall results
    passes = sum(1 for result in results["tests"].values() if result == "PASS")
    total = len(results["tests"])
    results["overall"] = f"{passes}/{total} tests passed"

    print(f"\nOverall results for {name}: {results['overall']}")
    return results


def main():
    """Run the integration tests."""
    results = []

    # Test original integration if available
    if has_original:
        try:
            original_results = test_integration_manager(
                OriginalIntegrationManager, "Original Integration Manager"
            )
            results.append(original_results)
        except Exception as e:
            print(f"Error testing original integration: {e}")

    # Test fixed integration
    if has_fixed:
        try:
            fixed_results = test_integration_manager(
                FixedIntegrationManager, "Fixed Integration Manager"
            )
            results.append(fixed_results)
        except Exception as e:
            print(f"Error testing fixed integration: {e}")

    # Print comparison summary
    if len(results) > 1:
        print("\n---- Comparison Summary ----")
        for result in results:
            print(f"{result['name']}: {result['overall']}")
            for test_name, test_result in result["tests"].items():
                status = "✅" if test_result == "PASS" else "❌"
                print(f"  {status} {test_name}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
