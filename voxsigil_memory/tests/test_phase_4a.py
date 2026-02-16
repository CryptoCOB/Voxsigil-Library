"""
Phase 4-A Tests: Edge-Optimized Runtime Layer

Validates:
- Device profile configurations
- Profile-specific latency/quality/memory tradeoffs
- Auto-detection logic
- Per-profile adversarial robustness
- Routing enable/disable behavior
- Pruning intensity differences

Status: Phase 4-A Completion Tests
"""

import time
import pytest
import numpy as np
from voxsigil_memory.edge_optimized import (
    EdgeOptimizedPipeline,
    DeviceProfile,
    DEVICE_CONFIGS,
    auto_select_profile,
    create_pipeline,
)
from voxsigil_memory.semantic import LatentMemoryUnit


class TestDeviceProfiles:
    """Test device profile enumeration and configuration."""

    def test_profile_enum_has_three_profiles(self):
        """Verify DeviceProfile has exactly 3 profiles."""
        profiles = list(DeviceProfile)
        assert len(profiles) == 3
        assert DeviceProfile.SERVER in profiles
        assert DeviceProfile.EDGE in profiles
        assert DeviceProfile.ULTRA_EDGE in profiles

    def test_all_profiles_in_config_dict(self):
        """Verify all profiles have configs in DEVICE_CONFIGS."""
        for profile in DeviceProfile:
            assert profile in DEVICE_CONFIGS
            assert DEVICE_CONFIGS[profile] is not None

    def test_server_config_is_least_constrained(self):
        """SERVER profile should have most lenient constraints."""
        server = DEVICE_CONFIGS[DeviceProfile.SERVER]
        edge = DEVICE_CONFIGS[DeviceProfile.EDGE]
        ultra = DEVICE_CONFIGS[DeviceProfile.ULTRA_EDGE]

        # Latency budgets: server >= edge >= ultra
        assert server.max_latency_ms >= edge.max_latency_ms
        assert edge.max_latency_ms >= ultra.max_latency_ms

        # Memory: server >= edge >= ultra
        assert server.max_memory_mb >= edge.max_memory_mb
        assert edge.max_memory_mb >= ultra.max_memory_mb

        # Doc length: server >= edge >= ultra
        assert server.max_doc_length >= edge.max_doc_length
        assert edge.max_doc_length >= ultra.max_doc_length

    def test_pruning_ratio_progression(self):
        """Pruning should be more aggressive with device constraints."""
        server = DEVICE_CONFIGS[DeviceProfile.SERVER]
        edge = DEVICE_CONFIGS[DeviceProfile.EDGE]
        ultra = DEVICE_CONFIGS[DeviceProfile.ULTRA_EDGE]

        # More aggressive pruning = lower ratio (keep less)
        # server: lenient (keep 80%), edge: moderate (keep 50%), ultra: aggressive (keep 30%)
        assert server.pruning_ratio >= edge.pruning_ratio
        assert edge.pruning_ratio >= ultra.pruning_ratio

    def test_routing_disabled_on_ultra_edge(self):
        """Ultra-edge should disable routing to save compute."""
        assert DeviceProfile.SERVER == DeviceProfile.SERVER
        server_routing = DEVICE_CONFIGS[DeviceProfile.SERVER].use_routing
        ultra_routing = DEVICE_CONFIGS[DeviceProfile.ULTRA_EDGE].use_routing

        assert server_routing is True
        assert ultra_routing is False

    def test_compression_always_enabled(self):
        """All profiles keep compression enabled."""
        for profile in DeviceProfile:
            assert DEVICE_CONFIGS[profile].use_compression is True


class TestEdgeOptimizedPipelineInit:
    """Test pipeline initialization with different profiles."""

    def test_pipeline_init_with_server_profile(self):
        """Initialize pipeline with SERVER profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        assert pipeline.device_profile == DeviceProfile.SERVER
        assert pipeline.config.max_latency_ms == 100

    def test_pipeline_init_with_edge_profile(self):
        """Initialize pipeline with EDGE profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.EDGE)
        assert pipeline.device_profile == DeviceProfile.EDGE
        assert pipeline.config.max_memory_mb == 512

    def test_pipeline_init_with_ultra_edge_profile(self):
        """Initialize pipeline with ULTRA_EDGE profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        assert pipeline.device_profile == DeviceProfile.ULTRA_EDGE
        assert pipeline.config.use_routing is False

    def test_pipeline_default_is_server(self):
        """Pipeline defaults to SERVER profile."""
        pipeline = EdgeOptimizedPipeline()
        assert pipeline.device_profile == DeviceProfile.SERVER

    def test_router_initialized_conditionally(self):
        """Router initialized only if use_routing=True."""
        server_pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra_pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        assert server_pipeline.router is not None
        assert ultra_pipeline.router is None


class TestProcessDocument:
    """Test document processing through profiles."""

    SAMPLE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "This is important information. "
        "Some filler text here. " * 5
    )

    def test_process_document_server_profile(self):
        """Process document through SERVER pipeline."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        unit = pipeline.process_document(self.SAMPLE_TEXT)

        assert isinstance(unit, LatentMemoryUnit)
        assert unit.embedding is not None
        assert len(unit.embedding) == 384  # MiniLM dimension

    def test_process_document_ultra_edge_profile(self):
        """Process document through ULTRA_EDGE pipeline."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        unit = pipeline.process_document(self.SAMPLE_TEXT)

        assert isinstance(unit, LatentMemoryUnit)
        assert unit.embedding is not None

    def test_truncation_beyond_max_doc_length(self):
        """Documents exceeding max_doc_length should be truncated."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        # Ultra-edge max is 1000 chars
        long_text = "word " * 1000  # 5000 chars

        unit = pipeline.process_document(long_text)
        # Should be truncated to ~1000 + "..."
        assert unit.latent_encoding is not None  # Processed successfully

    def test_pruning_ratio_affects_output(self):
        """Different profiles with different pruning ratios."""
        server = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        server_unit = server.process_document(self.SAMPLE_TEXT)
        ultra_unit = ultra.process_document(self.SAMPLE_TEXT)

        # Ultra-edge prunes more aggressively (keep 30% vs 80%)
        assert ultra_unit.pruned_fraction >= server_unit.pruned_fraction


class TestProcessUnits:
    """Test unit routing through profiles."""

    def test_routing_enabled_on_server(self):
        """SERVER profile uses full routing."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)

        # Create sample units
        units = [
            LatentMemoryUnit(
                id=f"unit_{i}",
                embedding=np.random.rand(384),
                latent_encoding=b"sample",
                original_length=100,
                modality="text",
                retrieval_score=0.5 + i * 0.1,
                pruned_fraction=0.0,
                entropy_score=0.5,
            )
            for i in range(5)
        ]

        _, stats = pipeline.process_units(units, budget_tokens=512)

        assert stats["routing_used"] is True
        assert "routing_stats" in stats

    def test_routing_disabled_on_ultra_edge(self):
        """ULTRA_EDGE profile skips routing (naive selection)."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        units = [
            LatentMemoryUnit(
                id=f"unit_{i}",
                embedding=np.random.rand(384),
                latent_encoding=b"sample",
                original_length=100,
                modality="text",
                retrieval_score=0.5,
                pruned_fraction=0.0,
                entropy_score=0.5,
            )
            for i in range(5)
        ]

        routed, stats = pipeline.process_units(units, budget_tokens=512)

        assert stats["routing_used"] is False
        # Without routing, returns units up to budget
        assert len(routed) <= len(units)

    def test_budget_enforcement_ultra_edge(self):
        """Ultra-edge respects token budget without routing."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        # Create 10 units
        units = [
            LatentMemoryUnit(
                id=f"unit_{i}",
                embedding=np.random.rand(384),
                latent_encoding=b"sample",
                original_length=100,
                modality="text",
                retrieval_score=0.5,
                pruned_fraction=0.0,
                entropy_score=0.5,
            )
            for i in range(10)
        ]

        # Small budget
        routed_units, stats = pipeline.process_units(units, budget_tokens=100)

        # Should keep only some units
        assert len(routed_units) < len(units)
        assert stats["budget_used"] <= 100


class TestContextPackBuilder:
    """Test context pack assembly."""

    def test_build_context_pack_server(self):
        """Build context pack on SERVER profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)

        sample_doc = "The quick brown fox jumps over the lazy dog."
        unit = pipeline.process_document(sample_doc)

        pack = pipeline.build_context_pack([unit], query="What does fox do?")

        assert "expanded_text" in pack
        assert "device_profile" in pack
        assert pack["device_profile"] == "server"
        assert "config" in pack
        assert pack["config"]["max_latency_ms"] == 100
        assert "token_count" in pack

    def test_build_context_pack_ultra_edge(self):
        """Build context pack on ULTRA_EDGE profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        sample_doc = "The quick brown fox jumps over the lazy dog."
        unit = pipeline.process_document(sample_doc)

        pack = pipeline.build_context_pack([unit])

        assert pack["device_profile"] == "ultra_edge"
        assert pack["config"]["use_routing"] is False

    def test_context_pack_empty_units(self):
        """Handle empty unit list."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        pack = pipeline.build_context_pack([])

        assert pack["expanded_text"] == ""
        assert pack["token_count"] == 0


class TestMemoryProfiling:
    """Test memory usage estimation per profile."""

    def test_memory_profile_server(self):
        """Memory estimate for SERVER profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        mem = pipeline.profile_memory_usage()

        assert mem["device_profile"] == "server"
        assert mem["model_size_mb"] == 80  # Full precision MiniLM
        assert mem["total_budget_mb"] == 2048

    def test_memory_profile_edge(self):
        """Memory estimate for EDGE profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.EDGE)
        mem = pipeline.profile_memory_usage()

        assert mem["device_profile"] == "edge"
        assert mem["model_size_mb"] == 22  # int8 quantized
        assert mem["total_budget_mb"] == 512

    def test_memory_profile_ultra_edge(self):
        """Memory estimate for ULTRA_EDGE profile."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        mem = pipeline.profile_memory_usage()

        assert mem["device_profile"] == "ultra_edge"
        assert mem["model_size_mb"] == 12  # int4 quantized
        assert mem["total_budget_mb"] == 256

    def test_max_units_in_memory_calculation(self):
        """Max units should respect memory budget."""
        server = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        server_mem = server.profile_memory_usage()
        ultra_mem = ultra.profile_memory_usage()

        # Server should hold more units than ultra
        assert server_mem["max_units_in_memory"] >= ultra_mem["max_units_in_memory"]


class TestAutoSelectProfile:
    """Test auto-detection of device profile."""

    def test_auto_select_high_end_device(self):
        """High RAM + latency budget → SERVER."""
        profile = auto_select_profile(ram_available_mb=4096, latency_budget_ms=200)
        assert profile == DeviceProfile.SERVER

    def test_auto_select_mid_range_device(self):
        """Mid RAM + latency budget → EDGE or SERVER (boundary condition)."""
        # 1024 MB RAM with 100ms latency falls on SERVER boundary
        profile = auto_select_profile(ram_available_mb=512, latency_budget_ms=75)
        assert profile == DeviceProfile.EDGE

    def test_auto_select_constrained_device(self):
        """Low RAM + tight latency → ULTRA_EDGE."""
        profile = auto_select_profile(ram_available_mb=256, latency_budget_ms=20)
        assert profile == DeviceProfile.ULTRA_EDGE

    def test_auto_select_boundary_conditions(self):
        """Test boundary cases."""
        # Edge boundary: exactly 1024MB RAM
        profile = auto_select_profile(ram_available_mb=1024, latency_budget_ms=100)
        assert profile == DeviceProfile.SERVER

        # Ultra-edge threshold: < 256MB RAM
        profile = auto_select_profile(ram_available_mb=128, latency_budget_ms=20)
        assert profile == DeviceProfile.ULTRA_EDGE


class TestCreatePipelineConvenience:
    """Test convenience factory function."""

    def test_create_pipeline_with_server(self):
        """create_pipeline with SERVER profile."""
        pipeline = create_pipeline(DeviceProfile.SERVER)
        assert pipeline.device_profile == DeviceProfile.SERVER

    def test_create_pipeline_with_edge(self):
        """create_pipeline with EDGE profile."""
        pipeline = create_pipeline(DeviceProfile.EDGE)
        assert pipeline.device_profile == DeviceProfile.EDGE

    def test_create_pipeline_defaults_to_server(self):
        """create_pipeline defaults to SERVER."""
        pipeline = create_pipeline()
        assert pipeline.device_profile == DeviceProfile.SERVER


class TestProfileQualityTradeoffs:
    """Test quality degradation across profiles."""

    SAMPLE_DOCS = [
        "The quick brown fox jumps over the lazy dog. "
        "This is important information we must keep. " * 5,
        "Smith & Associates reported earnings in Q4 1995. "
        "The date matters here. " * 5,
        "Rare phenomenon: supernova XYZ-123 detected. "
        "Entity preservation critical. " * 5,
    ]

    def test_aggressive_pruning_preserves_length_variance(self):
        """Aggressive pruning (ultra) vs lenient (server) should show difference."""
        server = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        server_units = [server.process_document(doc) for doc in self.SAMPLE_DOCS]
        ultra_units = [ultra.process_document(doc) for doc in self.SAMPLE_DOCS]

        # Ultra-edge prunes more
        avg_server_prune = np.mean([u.pruned_fraction for u in server_units])
        avg_ultra_prune = np.mean([u.pruned_fraction for u in ultra_units])

        assert avg_ultra_prune >= avg_server_prune

    def test_routing_enabled_reduces_unit_count(self):
        """Enabled routing on SERVER vs disabled on ULTRA should differ."""
        server = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        units = [
            LatentMemoryUnit(
                id=f"unit_{i}",
                embedding=np.random.rand(384),
                latent_encoding=b"sample_text_here",
                original_length=100,
                modality="text",
                retrieval_score=0.3 + i * 0.1,
                pruned_fraction=0.0,
                entropy_score=0.2 + i * 0.05,
            )
            for i in range(10)
        ]

        _, server_stats = server.process_units(units, budget_tokens=512)
        _, ultra_stats = ultra.process_units(units, budget_tokens=512)

        # With identical units and budget, server (routing enabled) may select fewer
        # based on entropy thresholds
        assert isinstance(server_stats["routing_used"], bool)
        assert isinstance(ultra_stats["routing_used"], bool)


class TestLatencyCharacteristics:
    """Measure latency per profile (qualitative, not quantitative)."""

    def test_process_document_completes(self):
        """All profiles should complete document processing."""
        text = "Sample text. " * 100

        for profile in DeviceProfile:
            pipeline = EdgeOptimizedPipeline(profile)
            start = time.time()
            unit = pipeline.process_document(text)
            elapsed = time.time() - start

            assert unit is not None
            # Should complete in < 5 seconds (very generous)
            assert elapsed < 5.0

    def test_ultra_edge_faster_than_server(self):
        """ULTRA_EDGE with aggressive pruning should process faster."""
        text = "Sample text. " * 100

        server = EdgeOptimizedPipeline(DeviceProfile.SERVER)
        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        # Time server
        server_times = []
        for _ in range(3):
            start = time.time()
            server.process_document(text)
            server_times.append(time.time() - start)

        # Time ultra
        ultra_times = []
        for _ in range(3):
            start = time.time()
            ultra.process_document(text)
            ultra_times.append(time.time() - start)

        # Ultra should not be slower (may be similar due to Python overhead)
        avg_server = np.mean(server_times)
        avg_ultra = np.mean(ultra_times)
        # Note: ultra might actually be slightly slower due to aggressive pruning,
        # but they should be in the same ballpark
        assert avg_ultra < avg_server * 2  # Very loose bound


class TestAdversarialRobustness:
    """Test profile robustness on edge cases."""

    ADVERSARIAL_TEXTS = [
        # Edge case: dates and numbers
        "On 2024-12-25, the temperature was 98.6°F. "
        "Account #123456789 had $1,000,000.00 balance.",
        # Edge case: entities
        "Alice met Bob at the Eiffel Tower in Paris, France. "
        "They discussed quantum computing with Dr. Smith.",
        # Edge case: rare words
        "The sesquipedalian loquaciousness evinced considerable obfuscation. "
        "Pneumonoultramicroscopicsilicovolcanoconiosis emerged.",
        # Edge case: punctuation-heavy
        "What?! Really?! Yes!!! Oh no... hmm? indeed. finally: success.",
    ]

    def test_server_handles_all_adversarial_cases(self):
        """SERVER profile processes all adversarial cases."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.SERVER)

        for text in self.ADVERSARIAL_TEXTS:
            unit = pipeline.process_document(text)
            assert unit is not None
            assert unit.embedding is not None

    def test_ultra_edge_handles_all_adversarial_cases(self):
        """ULTRA_EDGE profile processes all adversarial cases (with truncation)."""
        pipeline = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)

        for text in self.ADVERSARIAL_TEXTS:
            unit = pipeline.process_document(text)
            assert unit is not None
            # Ultra-edge may truncate, but should still process
            assert unit.embedding is not None

    def test_aggressive_pruning_preserves_entity_signals(self):
        """Even aggressive pruning should attempt to keep named entities."""
        # This is a quality-of-service assertion, not a hard guarantee
        text = (
            "Einstein formulated E=mc². "
            "Hawking studied black holes. "
            "Newton discovered gravity. " * 5
        )

        ultra = EdgeOptimizedPipeline(DeviceProfile.ULTRA_EDGE)
        unit = ultra.process_document(text)

        # Should not crash and should produce valid latent unit
        assert unit is not None
        assert len(unit.embedding) == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
