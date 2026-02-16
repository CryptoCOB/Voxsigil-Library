"""Phase 2: In-process retrieval and semantic routing tests."""

import pytest
from voxsigil_memory.retrieval import HNSWRetriever
from voxsigil_memory.semantic import (
    EmbeddingGenerator,
    SemanticRouter,
    SemanticPruner,
    SemanticEncoder,
)


# ============================================================================
# HNSW Retriever Tests
# ============================================================================


def test_phase_2_hnsw_initialization():
    """Test HNSW retriever can be initialized."""
    retriever = HNSWRetriever(dim=384, ef=200)
    assert retriever is not None
    assert retriever.get_size() == 0


def test_phase_2_hnsw_indexing():
    """Test HNSW indexing with sample vectors."""
    retriever = HNSWRetriever(dim=3)  # Small dim for testing
    
    # Create simple vectors
    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    ids = ["doc1", "doc2", "doc3"]
    
    retriever.add_vectors(vectors, ids)
    assert retriever.get_size() == 3


def test_phase_2_hnsw_search():
    """Test HNSW search functionality."""
    retriever = HNSWRetriever(dim=3)
    
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ids = ["red", "green", "blue"]
    
    retriever.add_vectors(vectors, ids)
    
    # Search for vector similar to [1, 0, 0]
    results = retriever.search([1.0, 0.0, 0.0], k=3)
    assert len(results) == 3
    assert results[0][0] == "red"  # Should find "red" first


def test_phase_2_hnsw_empty_search():
    """Test search on empty index returns empty."""
    retriever = HNSWRetriever(dim=384)
    results = retriever.search([0.0] * 384, k=10)
    assert results == []


def test_phase_2_hnsw_dimension_validation():
    """Test dimension mismatch raises error."""
    retriever = HNSWRetriever(dim=384)
    
    with pytest.raises(ValueError, match="dimension"):
        retriever.add_vectors([[0.0] * 768], ["doc1"])  # Wrong dimension


# ============================================================================
# Embedding Generator Tests
# ============================================================================


def test_phase_2_embedding_generator_initialization():
    """Test embedding generator initializes."""
    gen = EmbeddingGenerator()
    assert gen is not None
    assert gen.dim > 0


def test_phase_2_embedding_single():
    """Test single text embedding."""
    gen = EmbeddingGenerator()
    embedding = gen.encode("hello world")
    
    assert isinstance(embedding, list)
    assert len(embedding) == gen.dim
    assert all(isinstance(x, (int, float)) for x in embedding)


def test_phase_2_embedding_batch():
    """Test batch embedding."""
    gen = EmbeddingGenerator()
    texts = ["hello world", "foo bar", "test"]
    embeddings = gen.encode_batch(texts)
    
    assert len(embeddings) == 3
    assert all(len(e) == gen.dim for e in embeddings)


def test_phase_2_embedding_empty():
    """Test empty text handling."""
    gen = EmbeddingGenerator()
    embedding = gen.encode("")
    assert len(embedding) == gen.dim


# ============================================================================
# Semantic Router Tests
# ============================================================================


def test_phase_2_router_initialization():
    """Test semantic router initializes."""
    router = SemanticRouter()
    assert router is not None
    assert router.routing_rules is not None


def test_phase_2_router_default():
    """Test default routing."""
    router = SemanticRouter()
    algo = router.route("some random text")
    
    assert algo in ["quantum", "blt", "sheaf", "game_semantic", "homotopy", "meta_learning"]


def test_phase_2_router_dialogue_detection():
    """Test dialogue type detection."""
    router = SemanticRouter()
    algo = router.route('John said: "Hello world"', content_type="dialogue")
    assert algo == "game_semantic"


def test_phase_2_router_trajectory_detection():
    """Test trajectory type detection."""
    router = SemanticRouter()
    algo = router.route("coordinate x: 10, y: 20", content_type="trajectory")
    assert algo == "homotopy"


def test_phase_2_router_infer_dialogue():
    """Test automatic dialogue inference."""
    router = SemanticRouter()
    content_type = router.infer_data_type('John said: "Hello"')
    assert content_type == "dialogue"


# ============================================================================
# Semantic Pruner Tests
# ============================================================================


def test_phase_2_pruner_initialization():
    """Test pruner initializes."""
    pruner = SemanticPruner(threshold=0.5)
    assert pruner.threshold == 0.5


def test_phase_2_pruner_short_content():
    """Test short content is not pruned."""
    pruner = SemanticPruner()
    short = "hello world"
    result = pruner.prune(short, "query", budget_tokens=100)
    assert len(result) == len(short)


def test_phase_2_pruner_long_content():
    """Test long content is pruned to budget."""
    pruner = SemanticPruner()
    long_content = "a" * 10000  # Very long
    result = pruner.prune(long_content, "query", budget_tokens=100)
    
    # Result should be shorter than original
    assert len(result) < len(long_content)
    # Result should include pruning marker
    assert "pruned" in result


# ============================================================================
# Semantic Encoder Tests
# ============================================================================


def test_phase_2_encoder_initialization():
    """Test encoder initializes."""
    encoder = SemanticEncoder()
    assert encoder is not None


def test_phase_2_encoder_balanced_mode():
    """Test encoding with balanced mode."""
    encoder = SemanticEncoder()
    encoded = encoder.encode("hello world", mode="balanced")
    
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    assert encoded[0:1] == b'\x02'  # balanced mode marker


def test_phase_2_encoder_aggressive_mode():
    """Test encoding with aggressive mode."""
    encoder = SemanticEncoder()
    encoded = encoder.encode("test", mode="aggressive")
    
    assert encoded[0:1] == b'\x01'  # aggressive mode marker


def test_phase_2_encoder_quality_mode():
    """Test encoding with quality mode."""
    encoder = SemanticEncoder()
    encoded = encoder.encode("test", mode="quality")
    
    assert encoded[0:1] == b'\x03'  # quality mode marker


# ============================================================================
# Integration Tests
# ============================================================================


def test_phase_2_e2e_embedding_to_indexing():
    """Test full pipeline: generate embeddings and index."""
    gen = EmbeddingGenerator()
    retriever = HNSWRetriever(dim=gen.dim)
    
    texts = ["machine learning", "deep learning", "neural networks"]
    embeddings = gen.encode_batch(texts)
    
    retriever.add_vectors(embeddings, texts)
    assert retriever.get_size() == 3


def test_phase_2_e2e_route_and_encode():
    """Test routing and encoding together."""
    router = SemanticRouter()
    encoder = SemanticEncoder()
    
    query = "John said hello"
    algo = router.route(query, content_type="dialogue")
    encoded = encoder.encode(query, mode="balanced")
    
    assert algo == "game_semantic"
    assert len(encoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
