"""Phase 1 Integration: End-to-end build_context flow."""

import pytest
from voxsigil_memory import build_context, ContextPack


def test_phase_1_e2e_happy_path():
    """Test happy path: build_context returns valid ContextPack."""
    result = build_context("test query", budget_tokens=1024, mode="balanced")
    
    assert isinstance(result, ContextPack)
    assert result.query == "test query"
    assert result.budget_tokens == 1024
    assert result.mode == "balanced"
    assert result.version == "0.1.0"
    assert isinstance(result.signature, str)
    assert isinstance(result.compressed_content, bytes)
    assert isinstance(result.metadata, dict)


def test_phase_1_e2e_mode_aggressive():
    """Test with aggressive mode."""
    result = build_context("test", mode="aggressive")
    assert result.mode == "aggressive"


def test_phase_1_e2e_mode_quality():
    """Test with quality mode."""
    result = build_context("test", mode="quality")
    assert result.mode == "quality"


def test_phase_1_e2e_custom_budget():
    """Test with custom budget tokens."""
    result = build_context("test", budget_tokens=512)
    assert result.budget_tokens == 512


def test_phase_1_e2e_device_selection():
    """Test device parameter (cpu/cuda)."""
    result = build_context("test", device="cpu")
    assert result.metadata["device"] == "cpu"


def test_phase_1_e2e_cache_disabled():
    """Test with cache disabled."""
    result = build_context("test", cache=False)
    assert result.metadata["cache"] is False


def test_phase_1_e2e_query_in_metadata():
    """Test that query length is stored in metadata."""
    result = build_context("test query")
    assert "query_length" in result.metadata
    assert result.metadata["query_length"] == 10


def test_phase_1_empty_query_raises_error():
    """Test that empty query raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        build_context("")


def test_phase_1_whitespace_query_raises_error():
    """Test that whitespace-only query raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        build_context("   ")


def test_phase_1_invalid_device_raises_error():
    """Test that invalid device raises ValueError."""
    with pytest.raises(ValueError, match="device"):
        build_context("test", device="gpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
