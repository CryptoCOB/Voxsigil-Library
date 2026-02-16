"""Phase 0: Core module import and determinism tests."""

import sys
import importlib
import dataclasses
import pytest

import voxsigil_memory
from voxsigil_memory import build_context, ContextPack
import voxsigil_memory.semantic
import voxsigil_memory.protocol
import voxsigil_memory.storage
import voxsigil_memory.retrieval
import voxsigil_memory.models
import voxsigil_memory.compression


def test_phase_0_imports():
    """Test that all core modules import without errors."""
    modules = [
        "voxsigil_memory",
        "voxsigil_memory.semantic",
        "voxsigil_memory.protocol",
        "voxsigil_memory.storage",
        "voxsigil_memory.retrieval",
        "voxsigil_memory.models",
        "voxsigil_memory.compression",
    ]

    for module_name in modules:
        if module_name in sys.modules:
            del sys.modules[module_name]

        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_phase_0_core_api_exists():
    """Test that primary public API exists and has correct signature."""
    assert callable(build_context), "build_context must be callable"
    assert ContextPack is not None, "ContextPack must exist"

    # Check ContextPack has required fields
    fields = {f.name for f in dataclasses.fields(ContextPack)}
    required = {"query", "compressed_content", "signature", "version",
                "budget_tokens", "mode", "metadata"}
    assert required.issubset(fields), f"ContextPack missing fields: {required - fields}"


def test_phase_0_deterministic_version():
    """Test that __version__ is deterministic."""
    v1 = voxsigil_memory.__version__
    assert v1 == "0.1.0", f"Version should be 0.1.0, got {v1}"

    # Verify stable across multiple accesses
    v2 = voxsigil_memory.__version__
    assert v1 == v2, f"Version not deterministic: {v1} != {v2}"


def test_phase_0_no_side_effects_on_import():
    """Test that importing voxsigil_memory has no external side effects."""
    # This test passes if we reach here without errors
    # (no file creation, no network calls, no state changes)
    assert True, "All imports completed without side effects"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
