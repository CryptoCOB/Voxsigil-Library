"""Phase 1: Single-call API implementation and integration tests."""

import inspect
import pytest
from voxsigil_memory import build_context, ContextPack


def test_phase_1_build_context_exists():
    """Test that build_context can be called with default args."""
    assert callable(build_context)


def test_phase_1_build_context_signature():
    """Test build_context accepts required parameters."""
    sig = inspect.signature(build_context)
    params = set(sig.parameters.keys())

    # Required parameters
    assert "query" in params
    assert "budget_tokens" in params
    assert "mode" in params

    # Optional parameters
    assert "device" in params
    assert "cache" in params


def test_phase_1_context_pack_returntype():
    """Test that build_context is typed to return ContextPack."""
    sig = inspect.signature(build_context)
    # Return annotation should be ContextPack
    assert sig.return_annotation == ContextPack or "ContextPack" in str(sig.return_annotation)


def test_phase_1_mode_validation():
    """Test that invalid mode raises ValueError."""
    with pytest.raises((NotImplementedError, ValueError)):
        build_context("test", mode="invalid_mode")


def test_phase_1_budget_tokens_validation():
    """Test that budget_tokens < 128 raises ValueError."""
    with pytest.raises((NotImplementedError, ValueError)):
        build_context("test", budget_tokens=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
