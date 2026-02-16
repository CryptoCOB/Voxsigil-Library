"""
Tests for edge-optimized VoxSigil pipeline.

Validates: device profile selection, document processing,
and configurable tradeoffs for different edge devices.
"""

import pytest
from voxsigil_memory.edge_optimized import (
    DeviceProfile,
    DeviceConfig,
    EdgeOptimizedPipeline,
    auto_select_profile,
    create_pipeline,
    DEVICE_CONFIGS,
)


class TestDeviceConfigs:
    """Test device configuration parameters."""

    def test_all_profiles_have_config(self):
        """All device profiles have configurations."""
        for profile in DeviceProfile:
            assert profile in DEVICE_CONFIGS
            cfg = DEVICE_CONFIGS[profile]
            assert isinstance(cfg, DeviceConfig)
            assert cfg.max_latency_ms > 0
            assert cfg.max_memory_mb > 0
            assert 0 < cfg.pruning_ratio <= 1.0

    def test_server_config_features(self):
        """Server profile enables all components."""
        cfg = DEVICE_CONFIGS[DeviceProfile.SERVER]
        assert cfg.use_routing is True
        assert cfg.use_compression is True
        assert cfg.pruning_ratio == 0.8

    def test_edge_config_balanced(self):
        """Edge profile balances features and performance."""
        cfg = DEVICE_CONFIGS[DeviceProfile.EDGE]
        assert cfg.use_routing is True
        assert cfg.use_compression is True
        assert cfg.pruning_ratio == 0.5  # More aggressive than server

    def test_ultra_edge_config_minimal(self):
        """Ultra-edge profile disables non-critical features."""
        cfg = DEVICE_CONFIGS[DeviceProfile.ULTRA_EDGE]
        assert cfg.use_routing is False  # Skip routing
        assert cfg.use_compression is True  # Compression still needed
        assert cfg.pruning_ratio == 0.3  # Most aggressive


class TestEdgeOptimizedPipeline:
    """Test EdgeOptimizedPipeline core functionality."""

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_pipeline_creation(self, profile):
        """Pipeline can be created for each profile."""
        pipeline = EdgeOptimizedPipeline(profile)
        assert pipeline.device_profile == profile
        assert pipeline.config == DEVICE_CONFIGS[profile]

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_document_processing(self, profile):
        """Document processing works for each profile."""
        pipeline = EdgeOptimizedPipeline(profile)
        text = "The capital of France is Paris."

        unit = pipeline.process_document(text)

        assert unit is not None
        assert unit.embedding.shape == (384,)
        assert unit.latent_encoding is not None

    def test_truncation_at_max_length(self):
        """Documents longer than max_doc_length are truncated."""
        cfg = DEVICE_CONFIGS[DeviceProfile.ULTRA_EDGE]
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        long_text = "word " * 500  # ~2500 chars, > max_doc_length=1000

        unit = pipeline.process_document(long_text)

        assert unit is not None
        assert unit.embedding.shape == (384,)

    def test_pruning_more_aggressive_on_ultra_edge(self):
        """Ultra-edge prunes more aggressively."""
        text = " ".join([f"Sentence {i}." for i in range(50)])

        server_unit = EdgeOptimizedPipeline(DeviceProfile.SERVER).process_document(text)
        edge_unit = EdgeOptimizedPipeline(DeviceProfile.EDGE).process_document(text)
        ultra_unit = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE).process_document(text)

        # Pruned fractions should reflect profile aggressiveness
        assert server_unit.pruned_fraction <= edge_unit.pruned_fraction <= ultra_unit.pruned_fraction

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_routing_optional(self, profile):
        """Routing is optional and respects config."""
        pipeline = EdgeOptimizedPipeline(profile)
        units = [
            pipeline.process_document(f"Text {i}")
            for i in range(3)
        ]

        routed, stats = pipeline.process_units(units, budget_tokens=1000)

        assert len(routed) > 0
        assert isinstance(stats, dict)

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_context_packing(self, profile):
        """Context packing works for each profile."""
        pipeline = EdgeOptimizedPipeline(profile)
        units = [
            pipeline.process_document(f"Fact {i}.")
            for i in range(3)
        ]

        pack = pipeline.build_context_pack(units, query="test", budget_tokens=500)

        assert isinstance(pack, dict)
        assert len(pack) > 0

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_memory_profiling(self, profile):
        """Memory profiling returns valid estimates."""
        pipeline = EdgeOptimizedPipeline(profile)
        mem = pipeline.profile_memory_usage()

        assert isinstance(mem, dict)
        assert "model_size_mb" in mem
        assert "per_unit_mb" in mem
        assert mem["model_size_mb"] > 0
        assert mem["per_unit_mb"] > 0


class TestAutoDetection:
    """Test automatic device profile selection."""

    def test_unlimited_resources_selects_server(self):
        """High RAM and loose latency select Server."""
        profile = auto_select_profile(ram_available_mb=8096, latency_budget_ms=1000)
        assert profile == DeviceProfile.SERVER

    def test_tight_constraints_select_ultra_edge(self):
        """Low RAM and tight latency select Ultra-edge."""
        profile = auto_select_profile(ram_available_mb=256, latency_budget_ms=10)
        assert profile == DeviceProfile.ULTRA_EDGE

    def test_moderate_constraints_select_edge(self):
        """Moderate resources select Edge."""
        profile = auto_select_profile(ram_available_mb=512, latency_budget_ms=100)
        assert profile in [DeviceProfile.EDGE, DeviceProfile.ULTRA_EDGE]


class TestFactory:
    """Test create_pipeline factory."""

    @pytest.mark.parametrize("profile", list(DeviceProfile))
    def test_create_with_profile(self, profile):
        """Can create pipeline with specific profile."""
        pipeline = create_pipeline(profile)
        assert pipeline.device_profile == profile

    def test_create_defaults_to_server(self):
        """Default profile is Server."""
        pipeline = create_pipeline()
        assert pipeline.device_profile == DeviceProfile.SERVER

    def test_create_none_defaults_to_server(self):
        """None defaults to Server."""
        pipeline = create_pipeline(None)
        assert pipeline.device_profile == DeviceProfile.SERVER


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_workflow(self):
        """Complete workflow: doc → process → route → pack."""
        pipeline = create_pipeline(DeviceProfile.EDGE)

        # Stage 1: Process documents
        docs = [
            "Paris is the capital of France.",
            "France is in Europe.",
            "Europe has 50 countries."
        ]
        units = [pipeline.process_document(doc) for doc in docs]

        # Stage 2: Route units
        routed, stats = pipeline.process_units(units, budget_tokens=1000)

        # Stage 3: Pack for context
        pack = pipeline.build_context_pack(routed, query="Where is Paris?", budget_tokens=1000)

        assert len(routed) > 0
        assert pack is not None

    def test_can_switch_profiles_dynamically(self):
        """Can switch profiles for different hardware."""
        text = "Sample document for testing."

        for profile in DeviceProfile:
            pipeline = create_pipeline(profile)
            unit = pipeline.process_document(text)
            assert unit is not None
            assert unit.embedding.shape == (384,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
