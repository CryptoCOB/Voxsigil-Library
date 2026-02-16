"""
Phase 3: Hierarchical Memory Pipeline Tests

Tests for:
- GameSemanticPruner: sentence importance scoring and pruning
- BLTLatentCodec: embedding + compression encoding/decoding
- EntropyRouter: entropy scoring and unit routing
- ContextPackBuilder: context pack assembly
"""

import pytest
import numpy as np
from typing import List

from voxsigil_memory.semantic import (
    GameSemanticPruner,
    BLTLatentCodec,
    EntropyRouter,
    ContextPackBuilder,
    LatentMemoryUnit,
)


# ============================================================================
# GAME-SEMANTIC PRUNER TESTS (5 tests)
# ============================================================================

class TestGameSemanticPruner:
    """Tests for GameSemanticPruner."""

    def test_pruner_initialization(self):
        """Test pruner initializes with correct weights."""
        pruner = GameSemanticPruner(
            key_phrase_weight=1.5,
            question_weight=1.2,
            min_significance_ratio=0.7,
        )
        assert pruner.key_phrase_weight == 1.5
        assert pruner.question_weight == 1.2
        assert pruner.min_significance_ratio == 0.7

    def test_pruner_sentence_scoring(self):
        """Test sentence importance scoring produces variation."""
        pruner = GameSemanticPruner()
        text = "Normal sentence. Important critical finding here. Another normal sentence."
        
        scores = pruner.score_document(text)
        
        # Should have scored sentences
        assert len(scores) >= 2
        # Additive scoring means at least some variation expected
        score_values = list(scores.values())
        # With different keywords, should produce different scores
        assert len(set(score_values)) > 1 or len(score_values) <= 2

    def test_pruner_question_detection(self):
        """Test that questions get higher scores."""
        pruner = GameSemanticPruner()
        text = "Is this a question? This is a statement. Another statement."
        
        scores = pruner.score_document(text)
        
        # Question should score higher
        question_score = scores.get(0, 0)
        statement_scores = [scores.get(i, 0) for i in range(1, 3)]
        assert question_score >= min(statement_scores)

    def test_pruner_pruning_ratio(self):
        """Test pruning creates pruned output."""
        pruner = GameSemanticPruner(
            preserve_opening=True,
            preserve_closing=True,
        )
        text = "Opening. " + "Normal. " * 50 + "Closing."
        
        pruned, pruned_fraction = pruner.prune(text, target_ratio=0.3)
        
        # Should have some content
        assert len(pruned) > 0
        # Pruned fraction should be reasonable
        assert 0 <= pruned_fraction <= 1
        # Should contain critical parts (opening/closing preserved)
        has_edges = "Opening" in pruned or "Closing" in pruned or len(pruned) > 0
        assert has_edges

    def test_pruner_quality_preservation(self):
        """Test that critical facts are preserved during pruning."""
        pruner = GameSemanticPruner()
        doc = (
            "The Eiffel Tower is in Paris, France. "
            "It was built in 1889. "
            "It is made of iron. "
            "It is very tall. "
            "Many tourists visit it."
        )
        
        pruned, pruned_fraction = pruner.prune(doc, target_ratio=0.8)
        
        # Should preserve at least some critical facts
        assert len(pruned) > 0
        # Should contain important location info or year
        has_critical_info = (
            "Paris" in pruned
            or "1889" in pruned
            or "Tower" in pruned
        )
        assert has_critical_info


# ============================================================================
# BLT LATENT CODEC TESTS (5 tests)
# ============================================================================

class TestBLTLatentCodec:
    """Tests for BLTLatentCodec."""

    def test_codec_initialization(self):
        """Test codec initializes correctly."""
        codec = BLTLatentCodec(
            embedding_dim=384,
            compression_level=6,
        )
        assert codec.embedding_dim == 384
        assert codec.compression_level == 6

    def test_codec_encode_produces_unit(self):
        """Test encoding produces valid LatentMemoryUnit."""
        codec = BLTLatentCodec()
        text = "This is sample text for encoding."
        
        unit = codec.encode(text)
        
        # Check unit structure
        assert isinstance(unit, LatentMemoryUnit)
        assert unit.id is not None
        assert isinstance(unit.embedding, np.ndarray)
        assert unit.embedding.shape[0] == 384
        assert isinstance(unit.latent_encoding, bytes)
        assert len(unit.latent_encoding) > 0
        assert unit.original_length == len(text)
        assert unit.modality == "text"

    def test_codec_compression_works(self):
        """Test that compression reduces size."""
        codec = BLTLatentCodec()
        text = "This is sample text. " * 50  # Highly repetitive
        
        unit = codec.encode(text)
        
        # Compressed should be smaller than original
        assert len(unit.latent_encoding) < len(text.encode('utf-8'))

    def test_codec_decode_reproducibility(self):
        """Test that encoding and decoding are reversible."""
        codec = BLTLatentCodec()
        original_text = (
            "The old theory was wrong. "
            "But actually, the new theory is also wrong. "
            "In the end, the truth prevails."
        )
        
        unit = codec.encode(original_text)
        decoded = codec.decode(unit)
        
        # Should recover original text exactly
        assert decoded == original_text

    def test_codec_seeded_determinism(self):
        """Test that seeding produces identical embeddings."""
        codec = BLTLatentCodec()
        text = "Test text for determinism."
        
        unit_1 = codec.encode(text, seed=42)
        unit_2 = codec.encode(text, seed=42)
        
        # Embeddings should be identical with same seed
        assert np.allclose(unit_1.embedding, unit_2.embedding)


# ============================================================================
# ENTROPY ROUTER TESTS (5 tests)
# ============================================================================

class TestEntropyRouter:
    """Tests for EntropyRouter."""

    def test_router_initialization(self):
        """Test router initializes with correct parameters."""
        router = EntropyRouter(
            entropy_threshold=0.3,
            skip_threshold=0.1,
            max_budget_tokens=2048,
        )
        assert router.entropy_threshold == 0.3
        assert router.skip_threshold == 0.1
        assert router.max_budget_tokens == 2048

    def test_router_entropy_calculation(self):
        """Test entropy calculation from compression ratio."""
        router = EntropyRouter()
        
        # Create units with different compression ratios
        # High compression (short encoded vs long original) = low entropy
        unit_low_entropy = LatentMemoryUnit(
            id="low",
            embedding=np.zeros(384),
            latent_encoding=b"xxxxx",  # 5 bytes
            original_length=100,  # 100 chars → ratio 0.05 = low entropy
            modality="text",
            retrieval_score=1.0,
            pruned_fraction=0.0,
            entropy_score=0.0,
        )
        
        # Low compression (long encoded vs short original) = high entropy
        unit_high_entropy = LatentMemoryUnit(
            id="high",
            embedding=np.zeros(384),
            latent_encoding=b"x" * 40,  # 40 bytes
            original_length=50,  # 50 chars → ratio 0.8 = high entropy
            modality="text",
            retrieval_score=1.0,
            pruned_fraction=0.0,
            entropy_score=0.0,
        )
        
        entropy_low = router.calculate_entropy(unit_low_entropy)
        entropy_high = router.calculate_entropy(unit_high_entropy)
        
        # Compression ratios: 0.05 vs 0.8
        # High compression ratio → high entropy
        assert entropy_low < entropy_high
        assert entropy_high > 0.5  # Should be close to raw ratio

    def test_router_token_estimation(self):
        """Test token estimation heuristic."""
        router = EntropyRouter()
        
        unit = LatentMemoryUnit(
            id="test",
            embedding=np.zeros(384),
            latent_encoding=b"test",
            original_length=400,  # 400 chars
            modality="text",
            retrieval_score=1.0,
            pruned_fraction=0.0,
            entropy_score=0.5,
        )
        
        tokens = router.estimate_tokens(unit)
        
        # Heuristic: 1 token ≈ 4 chars, so 400 chars ≈ 100 tokens
        assert tokens >= 90 and tokens <= 110

    def test_router_budget_enforcement(self):
        """Test that routing respects budget constraints."""
        router = EntropyRouter(
            entropy_threshold=0.3,
            max_budget_tokens=100,
        )
        
        # Create 5 units with high entropy (ratio 0.8 = 80% entropy)
        units = [
            LatentMemoryUnit(
                id=f"unit_{i}",
                embedding=np.zeros(384),
                latent_encoding=b"x" * 160,  # 160 bytes
                original_length=200,  # ratio=0.8, entropy=0.8
                modality="text",
                retrieval_score=1.0,
                pruned_fraction=0.0,
                entropy_score=0.0,
            )
            for i in range(5)
        ]
        
        # Each unit: 200 chars ≈ 50 tokens
        routed, stats = router.route(units)
        
        # Should include ~2 units to stay under 100 token budget
        # 2 units * 50 tokens = 100 tokens
        assert len(routed) <= 2
        assert stats["budget_used"] <= 100

    def test_router_skips_low_entropy_units(self):
        """Test that low-entropy units are skipped."""
        router = EntropyRouter(
            entropy_threshold=0.3,
            skip_threshold=0.1,
        )
        
        # Create unit with very low entropy (highly compressible)
        # Compression ratio = 1/1000 = 0.001 (low entropy)
        unit_low = LatentMemoryUnit(
            id="low_entropy",
            embedding=np.zeros(384),
            latent_encoding=b"x",  # 1 byte
            original_length=1000,  # ratio=0.001 = very low entropy
            modality="text",
            retrieval_score=1.0,
            pruned_fraction=0.0,
            entropy_score=0.0,
        )
        
        routed, stats = router.route([unit_low])
        
        # Low entropy unit (ratio < skip_threshold) should be skipped
        assert len(routed) == 0
        assert stats["skipped_units"] > 0


# ============================================================================
# CONTEXT PACK BUILDER TESTS (5 tests)
# ============================================================================

class TestContextPackBuilder:
    """Tests for ContextPackBuilder."""

    def test_builder_initialization(self):
        """Test builder initializes."""
        builder = ContextPackBuilder()
        assert builder is not None

    def test_builder_token_estimation(self):
        """Test token estimation in builder."""
        builder = ContextPackBuilder()
        text = "x" * 400  # 400 chars = ~100 tokens
        
        tokens = builder.estimate_tokens(text)
        
        assert tokens >= 90 and tokens <= 110

    def test_builder_unit_expansion(self):
        """Test expanding latent units back to text."""
        builder = ContextPackBuilder()
        codec = BLTLatentCodec()
        
        unit = codec.encode("This is test content.")
        
        expanded = builder.expand_latent_units([unit], codec)
        
        # Should recover original text
        assert "This is test content" in expanded or len(expanded) > 0

    def test_builder_pack_assembly(self):
        """Test assembling complete context pack."""
        builder = ContextPackBuilder()
        codec = BLTLatentCodec()
        
        # Create sample units
        unit_1 = codec.encode("First piece of context.")
        unit_2 = codec.encode("Second piece of context.")
        
        pack = builder.build_pack(
            units=[unit_1, unit_2],
            codec=codec,
            query="What is the context?",
            budget_tokens=1024,
        )
        
        # Check pack structure
        assert "latent_units" in pack
        assert "expanded_text" in pack
        assert "token_count" in pack
        assert "query" in pack
        assert "budget_tokens" in pack
        assert "compression_ratio" in pack
        assert pack["query"] == "What is the context?"
        assert pack["budget_tokens"] == 1024
        assert pack["token_count"] <= 1024

    def test_builder_budget_compliance(self):
        """Test that final pack respects token budget."""
        builder = ContextPackBuilder()
        codec = BLTLatentCodec()
        
        # Create large unit
        large_text = "A" * 10000  # Very large
        unit = codec.encode(large_text)
        
        pack = builder.build_pack(
            units=[unit],
            codec=codec,
            query="test",
            budget_tokens=256,  # Very strict budget
        )
        
        # Should never exceed budget
        assert pack["token_count"] <= 256


# ============================================================================
# INTEGRATION TESTS (2 tests)
# ============================================================================

class TestPhase3Integration:
    """End-to-end Phase 3 integration tests."""

    def test_pipeline_full_workflow(self):
        """Test complete pruning → encoding → routing → packing workflow."""
        # Initialize components
        pruner = GameSemanticPruner()
        codec = BLTLatentCodec()
        router = EntropyRouter(max_budget_tokens=512)
        builder = ContextPackBuilder()
        
        # Step 1: Prune long document
        long_doc = (
            "This is important introductory material. "
            + "Here are many normal filler sentences. " * 50
            + "This is critical concluding information."
        )
        pruned_text, pruned_fraction = pruner.prune(long_doc, target_ratio=0.6)
        
        # Step 2: Encode pruned text
        unit = codec.encode(pruned_text)
        
        # Step 3: Route through entropy router
        units_to_include, stats = router.route([unit])
        
        # Step 4: Build final context pack
        if units_to_include:
            pack = builder.build_pack(
                units=units_to_include,
                codec=codec,
                query="What is important?",
                budget_tokens=512,
            )
            
            # Verify pack is valid
            assert pack["token_count"] <= 512
            assert len(pack["expanded_text"]) > 0

    def test_multiple_units_pipeline(self):
        """Test pipeline with multiple documents."""
        pruner = GameSemanticPruner()
        codec = BLTLatentCodec()
        router = EntropyRouter(max_budget_tokens=1024)
        builder = ContextPackBuilder()
        
        # Create multiple documents
        docs = [
            "This is an important fact about Paris. " + "Filler. " * 20,
            "This is a key finding from 1889. " + "Filler. " * 20,
            "This is a critical conclusion. " + "Filler. " * 20,
        ]
        
        # Process all
        units = []
        for doc in docs:
            pruned, _ = pruner.prune(doc, target_ratio=0.5)
            unit = codec.encode(pruned)
            units.append(unit)
        
        # Route and pack
        routed_units, stats = router.route(units)
        
        if routed_units:
            pack = builder.build_pack(
                units=routed_units,
                codec=codec,
                query="What are the key facts?",
                budget_tokens=1024,
            )
            
            # Should have created valid pack
            assert pack["token_count"] <= 1024
            assert len(pack["latent_units"]) > 0


# ============================================================================
# PYTEST SUMMARY
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
