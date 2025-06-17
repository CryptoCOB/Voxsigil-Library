#!/usr/bin/env python3
"""
Import Test Script - Test all major components for import issues
"""


def test_import(module_name, from_module=None, as_name=None):
    """Test importing a module and report success/failure"""
    try:
        if from_module:
            if as_name:
                exec(f"from {from_module} import {module_name} as {as_name}")
            else:
                exec(f"from {from_module} import {module_name}")
        else:
            exec(f"import {module_name}")
        print(f"✅ {module_name} - SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {module_name} - FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("VoxSigil Library Import Test")
    print("=" * 60)

    # Test core components
    print("\n📦 Core Components:")
    test_import("UnifiedVantaCore", "Vanta.core")
    test_import("VoxAgent", "agents")
    test_import("HoloMesh", "agents")

    # Test BLT components
    print("\n🔧 BLT Components:")
    test_import("BLTEnhancedRAG", "VoxSigilRag.voxsigil_blt_rag")
    test_import("BLTSupervisorRagInterface", "BLT.blt_supervisor_integration")
    test_import("hybrid_blt", "VoxSigilRag")
    test_import("HybridMiddlewareConfig", "VoxSigilRag.hybrid_blt")

    # Test VoxSigilRag components
    print("\n📚 VoxSigilRag Components:")
    test_import("VoxSigilRAG", "VoxSigilRag.voxsigil_rag")
    test_import("voxsigil_blt", "VoxSigilRag")

    # Test ART components
    print("\n🎨 ART Components:")
    test_import("ARTManager", "ART.art_manager")
    test_import("ARTAdapter", "ART.adapter")

    # Test ARC components
    print("\n🧩 ARC Components:")
    test_import("ARCReasoner", "ARC.arc_reasoner")
    test_import("HybridARCSolver", "ARC.arc_integration")

    # Test Gridformer components
    print("\n🔄 Gridformer Components:")
    test_import("GridFormerConnector", "Gridformer")
    test_import("enhanced_grid_connector", "core")

    print("\n" + "=" * 60)
    print("Import test completed!")


if __name__ == "__main__":
    main()
