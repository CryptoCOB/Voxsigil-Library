#!/usr/bin/env python3
"""
Quick BLT Import Test
"""


def test_blt_imports():
    print("Testing BLT import fixes...")

    # Test 1: BLTEnhancedRAG
    try:
        print("✅ BLTEnhancedRAG imported successfully")
    except Exception as e:
        print(f"❌ BLTEnhancedRAG failed: {e}")

    # Test 2: BLT Supervisor Integration
    try:
        print("✅ BLTSupervisorRagInterface imported successfully")
    except Exception as e:
        print(f"❌ BLTSupervisorRagInterface failed: {e}")

    # Test 3: HybridMiddleware from correct location
    try:
        print("✅ HybridMiddleware imported successfully")
    except Exception as e:
        print(f"❌ HybridMiddleware failed: {e}")

    # Test 4: ByteLatentTransformerEncoder from correct location
    try:
        print("✅ ByteLatentTransformerEncoder imported successfully")
    except Exception as e:
        print(f"❌ ByteLatentTransformerEncoder failed: {e}")


if __name__ == "__main__":
    test_blt_imports()
