#!/usr/bin/env python3
"""
Test which BLTEnhancedRAG implementation is being imported
"""


def test_blt_enhanced_rag():
    print("Testing BLTEnhancedRAG imports...")

    # Test 1: Try importing from the real implementation
    try:
        from VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG as RealBLT

        print("✅ Real BLTEnhancedRAG imported from voxsigil_blt_rag")
        print(f"   Module: {RealBLT.__module__}")
        print(f"   Doc: {RealBLT.__doc__[:100]}...")
    except Exception as e:
        print(f"❌ Real BLTEnhancedRAG failed: {e}")

    # Test 2: Try importing from hybrid_blt (should be stub)
    try:
        from VoxSigilRag.hybrid_blt import BLTEnhancedRAG as StubBLT

        print("✅ Stub BLTEnhancedRAG imported from hybrid_blt")
        print(f"   Module: {StubBLT.__module__}")
        print(f"   Doc: {StubBLT.__doc__[:100]}...")
    except Exception as e:
        print(f"❌ Stub BLTEnhancedRAG failed: {e}")

    # Test 3: Check if they're the same class
    try:
        from VoxSigilRag.hybrid_blt import BLTEnhancedRAG as StubBLT
        from VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG as RealBLT

        if RealBLT is StubBLT:
            print("⚠️  WARNING: Real and Stub are the same class!")
        else:
            print("✅ Real and Stub are different classes (expected)")
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


if __name__ == "__main__":
    test_blt_enhanced_rag()
