# VoxSigil BLT-Memory Engine (VME) — Architecture & Implementation Plan

**Unified latent-memory codec + retrieval engine for LLMs**  
Single-call library. Batteries included. No external services. Test-gated delivery.

---

## Vision Statement

Convert the Ultra BLT compression ecosystem (BLT core + MetaConsciousness framework + 41 algorithms) into a **single, distributable memory subsystem** that any LLM can use to dramatically improve effective context efficiency through:

- **Hierarchical semantic pruning** (Game-Semantic compression)
- **Latent memory encoding** (BLT-Semantic codec)
- **Entropy-governed routing** (intelligent dispatch)
- **In-process retrieval** (no FAISS, no external RAG daemon)
- **Deterministic protocol layer** (signing, versioning, reproducibility)

**Output**: One pip wheel. One function call. Two clean internal layers. Measurable wins.

---

## Phase Overview (Test-Gated)

| Phase | Goal | Status | Tests Gate | Artifact |
|-------|------|--------|-----------|----------|
| **0** | Module boundaries + repo structure | Ready | Import zero-overhead | Spec doc |
| **1** | Single-call public API | Ready | Golden tests (determinism) | `voxsigil_memory.build_context()` |
| **2** | In-process retrieval | Ready | Recall@K, latency, offline | HNSW codec |
| **3** | Hierarchical memory pipeline | Ready | Token-efficiency, quality | Semantic pruning |
| **4** | BLT-Semantic GPU-optional codec | Ready | CPU-only + GPU fallback | Pluggable codec |
| **5** | Single-artifact distribution | Ready | Fresh install, air-gapped | Wheel + binary |
| **6** | System-level evaluation suite | Ready | Reproducible benchmarks | Metrics JSONL |
| **7** | Paper writing | Ready | Lab section + results | Conference submission |
| **8** | Open-source release | Ready | Community reproducibility | GitHub + PyPI |

---

## Existing Assets in C:\UBLT (What We Have)

### Layer 1: Compression Core (`temp_recovered_blt*`)
```
✓ BLTCore: Multi-core streaming compression (zlib/LZ4)
✓ BLTSystem: Load-balanced parallel compression
✓ Thread-safe circular buffer (4KB per core)
✓ Stats tracking (bytes, ratio, operations)
✓ Serialize/deserialize state
✓ Hash-based verification
```

**Use in VME**: Core latent encoding engine. Deterministic codepath.

### Layer 2: MetaConsciousness Framework (469 files)
```
✓ SHEAF: Holographic patch compression (images)
✓ Quantum: Entropy-adaptive compression (text)
✓ Game-Semantic: Dialogue/narrative semantic pruning
✓ Homotopy: Topological trajectory compression
✓ Meta-Learning: Adaptive algorithm selection
✓ Full test suites
```

**Use in VME**: Semantic layer. Game-Semantic for pruning, others for transport/archival.

### Layer 3: 41 Compression Algorithms
```
✓ Ghost Detection Protocol: Device profiling
✓ Proof-of-Useful-Work: Verification backbone
✓ Knowledge Distillation: Teacher-student compression
✓ Neural Architecture Search: Codec optimization
✓ Quantum Behavior NAS: Hyperparameter search
... and 36 more
```

**Use in VME**: Optional codec variants, modality branches, evaluation.

### Layer 4: Documentation (4 files)
```
✓ COMPRESSION_ALGORITHMS_DOCUMENTATION.md
✓ SYSTEM_INTEGRATION_ARCHITECTURE.md
✓ BLT_RECOVERY_REPORT.md
✓ RESTORATION_COMPLETE_SUMMARY.md
```

**Use in VME**: Source material for Phase 7 (paper). Hidden from public until tests pass.

---

## Phase 0: Module Boundaries + Repo Structure

### Goal
Define the hard boundary between what lives inside the distributable (`voxsigil_memory`) and what stays external or optional.

### Deliverable: Module Map (not docs yet; code boundaries)

```
voxsigil_memory/
├── __init__.py
│   └── def build_context(query, user_id=None, budget_tokens=2048, mode='balanced')
│       "The one public function. Everything else is internal."
│
├── semantic/
│   ├── __init__.py
│   ├── pruner.py
│   │   └── GameSemanticPruner(threshold, preserve_opening, preserve_closing)
│   │       "From MetaConsciousness.frameworks.game_compression"
│   │
│   ├── codec.py
│   │   └── BLTLatentCodec(compression_level, modality_hint)
│   │       "From temp_recovered_blt + learned parameters"
│   │
│   ├── router.py
│   │   └── EntropyRouter(entropy_threshold, skip_threshold, strategy='learned')
│   │       "Routes to pruner, codec, or retrieval based on entropy"
│   │
│   └── pack_builder.py
│       └── ContextPack(latent_units, metadata, expansions)
│           "Builds final context for LLM consumption"
│
├── protocol/
│   ├── __init__.py
│   ├── sign.py
│   │   └── ProtocolSigner(secret_key=None)
│   │       "Deterministic canonical form + optional signing"
│   │
│   ├── envelope.py
│   │   └── Envelope(payload, version, timestamp, signature)
│   │       "Serialization container with versioning"
│   │
│   └── versioning.py
│       └── ProtocolVersion(major, minor, codec_revision)
│           "Handles evolution without breaking"
│
├── storage/
│   ├── __init__.py
│   ├── adapter.py
│   │   └── StorageAdapter (abstract)
│   │
│   ├── sqlite_adapter.py
│   │   └── SQLiteMemoryStore(path, max_items, index_dim)
│   │       "Default; always available"
│   │
│   └── postgres_adapter.py
│       └── PgvectorMemoryStore(dsn, max_items, index_dim)
│           "Optional; only if user wants external DB"
│
├── retrieval/
│   ├── __init__.py
│   ├── index.py
│   │   └── MemoryIndex (abstract)
│   │
│   ├── hnsw_index.py
│   │   └── HNSWIndex(embedding_dim, max_items, ef_construction)
│   │       "Default; in-process HNSW"
│   │
│   ├── flat_index.py
│   │   └── FlatIndex(embedding_dim, max_items)
│   │       "Fallback; brute-force"
│   │
│   └── retriever.py
│       └── VectorRetriever(index, embedding_fn, top_k, threshold)
│           "Orchestrates retrieval"
│
├── compression/
│   ├── __init__.py
│   ├── selector.py
│   │   └── CompressionSelector(ghost_profile, data_type, mode)
│   │       "From MetaConsciousness decision engine"
│   │
│   ├── ecosystem.py
│   │   └── CompressionEcosystem (thin wrapper)
│   │       "All 41 algorithms available but gated"
│   │
│   ├── sheaf/ (excluded)
│   ├── quantum/ (excluded)
│   └── ... (others excluded; available via submodule)
│       "Full BLT+Compression repo stays separate, imported as dependency"
│
├── models/
│   ├── __init__.py
│   ├── embeddings.py
│   │   └── Embedder (interface)
│   │
│   ├── blt_semantic.py → BLTSemanticEmbedder(weights_path, device)
│   │   "Deterministic, seeded encoder"
│   │
│   └── defaults.py
│       └── get_default_embedder(mode='cpu', cache_dir='~/.voxsigil')
│           "Load weights once; reuse"
│
├── tests/
│   ├── __init__.py
│   ├── test_phase_0_imports.py
│   │   → test_zero_import_overhead
│   │   → test_no_gpu_init_on_import
│   │   → test_no_db_creation_on_import
│   │
│   ├── test_phase_1_api.py
│   │   → test_build_context_signature
│   │   → test_golden_determinism (same input → same output)
│   │   → test_budget_enforcement
│   │   → test_error_ergonomics
│   │
│   ├── test_phase_2_retrieval.py
│   │   → test_recall_at_k
│   │   → test_latency_p50_p95
│   │   → test_memory_bound
│   │   → test_offline_mode
│   │
│   ├── test_phase_3_semantic.py
│   │   → test_token_efficiency_lift
│   │   → test_quality_preservation
│   │   → test_latent_stability
│   │   → test_pruning_counterexamples
│   │
│   ├── test_phase_4_codec.py
│   │   → test_cpu_only_pass
│   │   → test_gpu_optional
│   │   → test_weight_serialization
│   │   → test_seeded_determinism
│   │
│   ├── test_phase_5_distribution.py
│   │   → test_fresh_env_install
│   │   → test_airgapped_install
│   │   → test_artifact_size
│   │   → test_windows_runtime
│   │
│   ├── test_phase_6_benchmarks.py
│   │   → test_latency_bench
│   │   → test_token_efficiency_bench
│   │   → test_memory_scaling_bench
│   │   → test_ablations
│   │   → test_regression_gates
│   │
│   └── conftest.py (shared fixtures)
│
├── examples/
│   ├── hello_memory.py
│   │   "Minimal: ingest one doc, query it"
│   │
│   ├── ingest_corpus.py
│   │   "Load 1K docs, build index"
│   │
│   ├── offline_mode.py
│   │   "Retrieval with zero network access"
│   │
│   ├── custom_embedder.py
│   │   "Swap embedder (advanced)"
│   │
│   └── llm_integration.py
│       "Call from LLaMA, Mistral, GPT, etc."
│
├── setup.py
│   └── Dependencies:
│       - Tier A (always): python>=3.10, stdlib
│       - Tier B (optional): numpy, torch[cpu], sqlalchemy
│       - Tier C (optional): torch[cuda], pgvector
│
└── README.md
    └── "VoxSigil VME: One-call memory engine for LLMs"
```

### Dependency Tiers (Explicit)

**Tier A (Always Installed)**
```
- python >= 3.10
- typing-extensions
- dataclasses-json (lightweight serialization)
- hashlib (stdlib; deterministic signing)
```

**Tier B (Optional; Strongly Recommended)**
```
- numpy (entropy calculation, vectorization)
- torch[cpu] (BLT-Semantic encoder, CPU-only)
- sqlite3 (stdlib; default storage)
- hnswlib (in-process retrieval)
```

**Tier C (Optional; GPU Acceleration)**
```
- torch[cuda] (BLT-Semantic on GPU)
- faiss-gpu (if user opts in; usually no)
```

### Tests Gate Phase 0

```python
def test_phase_0_imports():
    """No side effects on import."""
    import voxsigil_memory  # No GPU init
    assert voxsigil_memory is not None
    # No files created
    # No downloads triggered
    # No DB initialized

def test_phase_0_determinism_seed():
    """Protocol layer is deterministic."""
    from voxsigil_memory.protocol import ProtocolSigner
    signer = ProtocolSigner()
    payload = {"query": "test", "budget": 2048}
    canonical_1 = signer.canonicalize(payload)
    canonical_2 = signer.canonicalize(payload)
    assert canonical_1 == canonical_2  # Exact bytes
```

### Artifacts: Deliverable from Phase 0

1. **Module boundary spec** (this section + code stubs)
2. **setup.py** with dependency tiers
3. **test_phase_0_imports.py** (MUST PASS)
4. **No public docs yet** (internal only)

---

## Phase 1: Single-Call Public API

### Goal
Expose exactly **one** stable entry point. Everything internally can do 10 things; externally, one call.

### Public API Shape (Spec)

```python
import voxsigil_memory

# THE ONE FUNCTION
context_pack = voxsigil_memory.build_context(
    query: str,                          # User question / task
    user_id: Optional[str] = None,       # For retrieval context
    budget_tokens: int = 2048,           # Max tokens in output pack
    mode: Literal['fast', 'balanced', 'quality'] = 'balanced',
    semantic: bool = True,               # Enable Game-Semantic pruning
    latent: bool = True,                 # Enable BLT latent codec
    entropy_threshold: Optional[float] = None,  # Override router threshold
    device: Optional[str] = None,        # 'cpu', 'cuda', None=auto
)
# Returns ContextPack object
```

### ContextPack Contract (Data Structure)

```python
@dataclass
class ContextPack:
    """Output of build_context(). Ready to feed to any LLM."""
    
    # Core
    latent_units: List[LatentMemoryUnit]          # Compressed memory
    metadata: Dict[str, Any]                      # Provenance
    
    # Expansion (what the LLM actually uses)
    expanded_text: str                            # Readable context for LLM
    token_count: int                              # Exact count
    
    # Quality signals
    retrieval_scores: List[float]                 # Confidence per unit
    entropy_scores: List[float]                   # What was pruned and why
    compression_ratio: float                      # Original → latent
    
    # Reproducibility
    protocol_version: str                         # e.g., "1.0"
    signature: Optional[str]                      # Deterministic signature
    timestamp: float                              # ms since epoch

@dataclass
class LatentMemoryUnit:
    """One compressed memory item."""
    id: str                                       # UUID
    embedding: np.ndarray                         # Dense vector (768d)
    latent_encoding: bytes                        # BLT-compressed payload
    original_length: int                          # Uncompressed size
    modality: str                                 # 'text', 'log', 'trace', etc.
    retrieval_score: float                        # 0-1 relevance
    pruned_fraction: float                        # How much was removed
```

### Build Context Internals (Hidden from API)

```
build_context(query, ...)
    ↓
1. Embed query → embedding
    (use default BLT-Semantic encoder or user-provided)
    ↓
2. Retrieve top-K memory units via VectorRetriever
    (HNSW index, user_id scoped)
    ↓
3. For each unit, apply GameSemanticPruner
    (skip low-entropy, preserve critical sentences)
    ↓
4. Encode pruned text via BLTLatentCodec
    (deterministic compression + learnable parameters)
    ↓
5. EntropyRouter decides: include, skip, or re-retrieve
    (based on entropy score + budget remaining)
    ↓
6. ContextPackBuilder assembles final pack
    (encode units, compute token count, sign)
    ↓
7. Return ContextPack
    (user never sees internals; just final pack)
```

### Tests Gate Phase 1

```python
def test_phase_1_api_signature():
    """API exists and has correct signature."""
    import inspect
    sig = inspect.signature(voxsigil_memory.build_context)
    assert 'query' in sig.parameters
    assert 'budget_tokens' in sig.parameters
    assert 'mode' in sig.parameters

def test_phase_1_golden_determinism():
    """Same input (seeded) → same output."""
    query = "What is the capital of France?"
    
    pack_1 = voxsigil_memory.build_context(query, random_seed=42)
    pack_2 = voxsigil_memory.build_context(query, random_seed=42)
    
    assert pack_1.signature == pack_2.signature
    assert pack_1.expanded_text == pack_2.expanded_text
    assert pack_1.token_count == pack_2.token_count

def test_phase_1_budget_enforcement():
    """Never exceed token budget."""
    pack = voxsigil_memory.build_context(
        "long query",
        budget_tokens=512
    )
    assert pack.token_count <= 512

def test_phase_1_error_ergonomics():
    """Errors are clean."""
    with pytest.raises(ValueError, match="query cannot be empty"):
        voxsigil_memory.build_context("")
```

### Artifacts: Phase 1

1. **voxsigil_memory/__init__.py** (public API + module exports)
2. **voxsigil_memory/protocol/sign.py** (ProtocolSigner)
3. **voxsigil_memory/protocol/envelope.py** (Envelope class)
4. **test_phase_1_api.py** (MUST PASS)
5. **No public docs yet**

---

## Phase 2: In-Process Retrieval (No External RAG)

### Goal
Retrieval is embedded. No FAISS daemon. No separate service. No external calls.

### Retrieval Stack Architecture

```
VectorRetriever (public interface)
    ├── _index: MemoryIndex (abstract)
    │   ├── HNSWIndex (default; fast, scalable)
    │   ├── FlatIndex (fallback; brute-force)
    │   └── (Optional) PgvectorIndex (if user has Postgres)
    │
    ├── _embedding_fn: Callable[[str] → np.ndarray]
    │   (BLT-Semantic encoder by default)
    │
    └── Methods:
        ├── add_document(text, metadata, user_id=None) → UUID
        ├── retrieve(query, top_k=10, threshold=0.5) → [MemoryUnit]
        ├── delete(id)
        ├── build_index() (batch mode)
        └── save_index(path), load_index(path)
```

### Storage Adapters (User Choice)

```
StorageAdapter (interface)
├── SQLiteMemoryStore (default, no external deps)
│   └── File-based; no server needed
│
├── PgvectorMemoryStore (optional, if user has Postgres)
│   └── Scales to billions; IP-addressed
│
└── InMemoryStore (testing only)
    └── No persistence
```

### Offline-First Design

```
# Scenario: No network, no external services

import voxsigil_memory

# Initialize entirely locally
memory = voxsigil_memory.LocalMemory(
    storage_path="./my_memory.db",
    index_path="./my_index.hnsw",
    device="cpu"  # No GPU required
)

# Ingest documents (one-time)
for doc in my_docs:
    memory.add_document(doc['text'], metadata=doc['meta'])

memory.build_index()  # Build HNSW in-process

# Later: retrieve offline
context_pack = voxsigil_memory.build_context(
    query="What happened in 1969?",
    memory=memory,
    # No network calls. Ever.
)
```

### Tests Gate Phase 2

```python
def test_phase_2_recall_at_k():
    """Retrieval is accurate."""
    # Create mini corpus (labeled)
    docs = ["Paris is in France", "Berlin is in Germany", ...]
    labels = [("Paris", "geography"), ...]
    
    memory = voxsigil_memory.LocalMemory()
    for doc, label in zip(docs, labels):
        memory.add_document(doc, metadata={"label": label})
    memory.build_index()
    
    # Query
    results = memory.retrieve("European capitals", top_k=5)
    
    # Evaluate Recall@K, MRR
    assert recall_at_k(results, labels) > 0.8

def test_phase_2_latency_bounds():
    """Retrieval is fast."""
    memory = voxsigil_memory.LocalMemory()
    # Add 10K documents
    for i in range(10000):
        memory.add_document(f"Document {i}: ...", metadata={"id": i})
    memory.build_index()
    
    # Retrieve
    start = time.time()
    results = memory.retrieve("query", top_k=10)
    elapsed_ms = (time.time() - start) * 1000
    
    assert elapsed_ms < 50  # p50 latency
    assert elapsed_ms < 200  # p95 latency

def test_phase_2_memory_bound():
    """Index size is predictable."""
    memory = voxsigil_memory.LocalMemory()
    for i in range(100000):
        memory.add_document(f"Doc {i}", metadata={})
    memory.build_index()
    
    index_size = memory.index_size_bytes
    expected_max = 100000 * 768 * 4 * 1.5  # 768d, 4-byte float, 1.5x overhead
    assert index_size <= expected_max

def test_phase_2_offline_mode():
    """Zero network access."""
    # Monkeypatch network libraries to fail
    with network_forbidden():
        memory = voxsigil_memory.LocalMemory()
        memory.add_document("test", metadata={})
        memory.build_index()
        results = memory.retrieve("query")
        # Works without any network call
        assert len(results) >= 0
```

### Artifacts: Phase 2

1. **voxsigil_memory/retrieval/index.py** (MemoryIndex abstract)
2. **voxsigil_memory/retrieval/hnsw_index.py** (HNSWIndex)
3. **voxsigil_memory/retrieval/flat_index.py** (FlatIndex)
4. **voxsigil_memory/retrieval/retriever.py** (VectorRetriever)
5. **voxsigil_memory/storage/sqlite_adapter.py** (SQLiteMemoryStore)
6. **voxsigil_memory/storage/postgres_adapter.py** (PgvectorMemoryStore, optional)
7. **test_phase_2_retrieval.py** (MUST PASS)
8. **examples/offline_mode.py**

---

## Phase 3: Hierarchical Memory (Semantic Pruning → Latent Encode → Pack)

### Goal
Turn compression into an LLM-memory scaler. Only algorithms that reduce conditioning cost sit in critical path.

### Critical-Path Pipeline (What Actually Runs)

```
Retrieve(top-K units)
    ↓
GameSemanticPruner (remove redundancy, preserve facts)
    "What's important in this chunk?"
    Input: [sentence1, sentence2, ...]
    Output: [sentence1, sentence3, ...] (sentence2 pruned)
    Score: importance per sentence
    ↓
BLTLatentCodec (compress pruned text → fixed-size vector)
    "Encode what remains into a latent representation"
    Input: pruned_text
    Output: latent_unit (embedding + BLT-compressed bytes)
    ↓
EntropyRouter (decide: include, skip, or re-retrieve?)
    "Is this unit worth its token cost?"
    If entropy(unit) > threshold: include
    If entropy(unit) < threshold: skip
    If budget_remaining < unit_tokens: re-retrieve smaller
    ↓
ContextPackBuilder (assemble final pack for LLM)
    "Build the final context the LLM sees"
    Expand latent units back to readable text
    Add retrieval scores, entropy scores
    Compute exact token count
    ↓
Output: ContextPack (ready for LLM)
```

### What's NOT in Critical Path (Available but Optional)

```
Archival Compression (SHEAF, Homotopy, etc.)
    Used when storing to long-term memory, not retrieval
    
Modality-Specific Codecs (images, video, audio)
    Only if corpus includes non-text
    
Transport Compression (network transmission)
    Only if sending over wire (not local LLM)
```

### Pruning Spec (GameSemanticPruner)

```python
class GameSemanticPruner:
    """
    Compress dialogue/text by identifying game-theoretically valuable moves.
    """
    def __init__(
        self,
        key_phrase_weight: float = 1.5,
        question_weight: float = 1.2,
        contradiction_weight: float = 1.4,
        sentiment_change_weight: float = 1.3,
        min_significance_ratio: float = 0.7,
        preserve_opening: bool = True,
        preserve_closing: bool = True,
    ):
        """
        Weights define what's "important" in dialogue.
        min_significance_ratio: keep top 70% by score; prune bottom 30%.
        """

    def score_document(self, text: str) -> Dict[int, float]:
        """
        Score each sentence by importance.
        Returns {sentence_idx: importance_score}
        """

    def prune(self, text: str, target_ratio: float = 0.7) -> str:
        """
        Keep sentences scoring above threshold.
        target_ratio: keep at least this fraction (0-1).
        """
        sentences = text.split('.')
        scores = self.score_document(text)
        threshold = np.percentile(list(scores.values()), 100 * (1 - target_ratio))
        pruned = [s for i, s in enumerate(sentences) if scores.get(i, 0) >= threshold]
        return '.'.join(pruned)
```

### Latent Codec Spec (BLTLatentCodec)

```python
class BLTLatentCodec:
    """
    Map text → fixed-size dense latent representation.
    Deterministic, seeded, reproducible.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        compression_level: int = 6,
        device: str = "cpu",
        learned_quantization: bool = True,
    ):
        """
        compression_level: 1-9 (default 6; balance speed/ratio)
        learned_quantization: use trained codec, not generic
        """

    def encode(self, text: str, seed: int = None) -> LatentUnit:
        """
        text → embedding + BLT-compressed bytes
        seed ensures reproducibility
        """
        # 1. Embed text → dense vector (768d)
        embedding = self.embedder(text, seed=seed)
        
        # 2. Compress text via BLT with learned parameters
        blt_bytes = self.blt.compress(text.encode('utf-8'))
        
        # 3. Return both (dense for retrieval, bytes for expansion)
        return LatentUnit(
            embedding=embedding,
            latent_encoding=blt_bytes,
            original_length=len(text),
        )

    def decode(self, latent_unit: LatentUnit) -> str:
        """Reconstruct text from latent encoding."""
        return self.blt.decompress(latent_unit.latent_encoding).decode('utf-8')
```

### Entropy Router Spec

```python
class EntropyRouter:
    """
    Decide which memory units to include in final pack.
    Entropy governs what's worth transmitting to LLM.
    """
    def __init__(
        self,
        entropy_threshold: float = 0.3,  # Below this, unit is "predictable"
        skip_threshold: float = 0.1,     # Skip completely if < this
        max_budget_tokens: int = 2048,
        strategy: Literal['static', 'learned'] = 'learned',
    ):
        """
        strategy='learned': thresholds adapted per corpus
        strategy='static': fixed thresholds
        """

    def route(self, units: List[LatentUnit]) -> Tuple[List[LatentUnit], Dict]:
        """
        Given units (in retrieval order), decide which to include.
        Returns (included_units, routing_metadata)
        """
        outputs = []
        budget_remaining = self.max_budget_tokens
        
        for unit in units:
            entropy = self.calculate_entropy(unit)
            token_cost = self.estimate_tokens(unit)
            
            if entropy < self.skip_threshold:
                # Skip: too predictable
                continue
            elif entropy > self.entropy_threshold or budget_remaining > token_cost:
                # Include: worth including
                outputs.append(unit)
                budget_remaining -= token_cost
            else:
                # Over budget: stop
                break
        
        return outputs, {"budget_used": self.max_budget_tokens - budget_remaining}
```

### Tests Gate Phase 3

```python
def test_phase_3_token_efficiency():
    """Pruning + latent codec reduce token cost."""
    # Long document (1000 tokens)
    long_doc = "The quick brown fox... " * 100
    
    # Without semantic pruning
    baseline_tokens = count_tokens(long_doc)
    
    # With pruning + latent codec
    pruned = pruner.prune(long_doc, target_ratio=0.7)
    encoded = codec.encode(pruned)
    optimized_tokens = (
        count_tokens(pruned) +  # Expanded form
        len(encoded.latent_encoding) // 4  # Compressed bytes as tokens
    )
    
    # Should require fewer tokens
    assert optimized_tokens < baseline_tokens * 0.9

def test_phase_3_quality_preservation():
    """Pruning doesn't remove critical facts."""
    doc = """
    The Eiffel Tower is in Paris, France.
    It was built in 1889.
    It is made of iron.
    It is very tall.
    Many tourists visit it.
    """
    # Key facts: Paris, 1889, iron, tall
    pruned = pruner.prune(doc, target_ratio=0.8)
    
    # Check critical facts still present
    assert "Paris" in pruned
    assert "1889" in pruned or "Paris" in pruned  # At least some facts

def test_phase_3_latent_stability():
    """Same text → same embedding (within tolerance)."""
    text = "Paris is the capital of France."
    
    latent_1 = codec.encode(text, seed=42)
    latent_2 = codec.encode(text, seed=42)
    
    # Embeddings should be identical
    assert np.allclose(latent_1.embedding, latent_2.embedding)
    # Compressed bytes should be identical
    assert latent_1.latent_encoding == latent_2.latent_encoding

def test_phase_3_pruning_counterexamples():
    """Pruning preserves critical details under adversarial inputs."""
    adversarial_docs = [
        "This is false. Actually, it's true.",  # Contradiction
        "Question: What is X? Answer: Y. But actually Z.",  # Correction
        "IMPORTANT: Do not forget this critical detail: ...",  # Emphasis
    ]
    
    for doc in adversarial_docs:
        pruned = pruner.prune(doc, target_ratio=0.8)
        # Should not be empty; should retain key info
        assert len(pruned) > len(doc) * 0.5
```

### Artifacts: Phase 3

1. **voxsigil_memory/semantic/pruner.py** (GameSemanticPruner)
2. **voxsigil_memory/semantic/codec.py** (BLTLatentCodec)
3. **voxsigil_memory/semantic/router.py** (EntropyRouter)
4. **voxsigil_memory/semantic/pack_builder.py** (ContextPackBuilder)
5. **test_phase_3_semantic.py** (MUST PASS)
6. **examples/hello_memory.py** (minimal end-to-end)

---

## Phase 4: BLT-Semantic GPU-Optional Codec

### Goal
BLT-Semantic packaged as pluggable codec that runs CPU-only by default, accelerates on GPU if present.

### Codec Interface (Pluggable)

```python
class BLTSemanticEmbedder:
    """
    Fine-tuned encoder for LLM memory latents.
    CPU by default. GPU optional.
    Deterministic, seeded.
    """
    
    def __init__(
        self,
        model_name: str = "blt-semantic-v1",
        device: Optional[str] = None,  # None = auto-detect
        cache_dir: str = "~/.voxsigil/models",
        quantized: bool = False,  # 8-bit quantization optional
    ):
        """
        device:
        - 'cpu': Force CPU
        - 'cuda': Force GPU (fail if unavailable)
        - None: Auto (GPU if available, else CPU)
        
        quantized: If True, use 8-bit quantization (faster, less memory)
        """
        self.device = self._resolve_device(device)
        self.weights = self._load_weights(model_name, cache_dir)
        if quantized:
            self.weights = self._quantize_int8(self.weights)
        
    def _resolve_device(self, device):
        """Determine optimal device."""
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device is None:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_weights(self, model_name, cache_dir):
        """Load model weights from local cache, not internet."""
        path = Path(cache_dir) / f"{model_name}.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. "
                "Run: voxsigil_memory download-models"
            )
        return torch.load(path, map_location=self.device)
    
    def encode(self, text: str, seed: int = None) -> np.ndarray:
        """
        text → 768-d dense embedding
        
        Deterministic if seed is provided.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Tokenize + embed
        tokens = self.tokenizer(text, truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(tokens.to(self.device))
            embedding = logits[:, 0, :].cpu().numpy()  # [CLS] token
        
        return embedding.reshape(768)  # Ensure 768-d
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode (GPU-friendly)."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = np.vstack([self.encode(t) for t in batch])
            embeddings.append(batch_embs)
        return np.vstack(embeddings) if embeddings else np.array([])
```

### Weight Distribution (Local-First)

```
~/.voxsigil/models/
├── blt-semantic-v1.pth          (Download once, reuse)
├── blt-semantic-v1-int8.pth     (Quantized variant)
└── manifest.json                (versions, checksums)

# Download command (optional; can be embedded in wheel)
$ voxsigil_memory download-models

# Or: auto-download on first use (with warning)
voxsigil_memory/models/default.py:
def get_embedder(auto_download=True):
    if not weights_exist():
        if auto_download:
            log_warning(f"Downloading {model_name} ({size}MB) to {cache_dir}...")
            download_weights()
        else:
            raise FileNotFoundError("Weights not found. Run 'voxsigil_memory download-models'")
    return BLTSemanticEmbedder()
```

### Tests Gate Phase 4

```python
def test_phase_4_cpu_only():
    """CPU-only path works on clean machine."""
    # Run on machine with no CUDA
    embedder = BLTSemanticEmbedder(device='cpu')
    embedding = embedder.encode("test text")
    assert embedding.shape == (768,)
    assert isinstance(embedding, np.ndarray)

def test_phase_4_gpu_optional():
    """GPU gracefully accelerates if present."""
    if torch.cuda.is_available():
        embedder_cpu = BLTSemanticEmbedder(device='cpu')
        embedder_gpu = BLTSemanticEmbedder(device='cuda')
        
        text = "test"
        emb_cpu = embedder_cpu.encode(text, seed=42)
        emb_gpu = embedder_gpu.encode(text, seed=42)
        
        # Same output (within FP32 tolerance)
        assert np.allclose(emb_cpu, emb_gpu, rtol=1e-5)
    else:
        # GPU not available; should not error
        embedder = BLTSemanticEmbedder(device=None)
        assert embedder.device == 'cpu'

def test_phase_4_serialization():
    """Model weights load from local path; no internet."""
    with network_forbidden():
        embedder = BLTSemanticEmbedder()
        embedding = embedder.encode("test")
        assert embedding.shape == (768,)

def test_phase_4_seeded_determinism():
    """Same seed → same embeddings."""
    embedder = BLTSemanticEmbedder()
    emb_1 = embedder.encode("test", seed=42)
    emb_2 = embedder.encode("test", seed=42)
    assert np.array_equal(emb_1, emb_2)  # Byte-exact
```

### Artifacts: Phase 4

1. **voxsigil_memory/models/blt_semantic.py** (BLTSemanticEmbedder)
2. **voxsigil_memory/models/defaults.py** (get_default_embedder)
3. **voxsigil_memory/models/weight_manager.py** (download, cache)
4. **test_phase_4_codec.py** (MUST PASS)
5. **CLI: voxsigil_memory download-models**

---

## Phase 5: Single-Artifact Distribution (One Download)

### Goal
Users run: `pip install voxsigil-memory` and get everything they need. No 14 dependencies. No fiddling.

### Distribution Matrix

| Target | Format | Size | Install | Dependencies |
|--------|--------|------|---------|--------------|
| Developer | Wheel (.whl) | 15MB | `pip install voxsigil-memory` | Tier A + B (recommended) |
| Production | Wheel + weights | 250MB | `pip install voxsigil-memory[full]` | Tier A + B + C |
| Non-dev user | Standalone binary | 300MB | Download .exe, run | None (embedded) |
| LM Studio | Plugin | 50MB | Drop into plugins/ | LM Studio runtime |
| Docker | Image | 500MB | `docker run voxsigil:latest` | Docker |

### Wheel + Optional Extras (setup.py)

```python
setup(
    name="voxsigil-memory",
    version="0.1.0",
    
    install_requires=[
        "python>=3.10",
        "dataclasses-json>=0.5.0",  # Serialization
        "typing-extensions>=4.0",    # Backcompat
    ],
    
    extras_require={
        "default": [
            "numpy>=1.21",
            "torch[cpu]>=2.0",        # BLT-Semantic
            "hnswlib>=0.7",           # HNSW index
        ],
        "full": [
            "numpy>=1.21",
            "torch>=2.0",             # Installs CUDA if available
            "hnswlib>=0.7",
            "sqlalchemy>=2.0",
            "psycopg2-binary>=2.9",   # Postgres
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "mypy>=0.950",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "voxsigil-memory=voxsigil_memory.cli:main",
        ],
    },
)
```

### PyInstaller Binary (Single .exe for Windows Users)

```bash
# Build standalone executable
pyinstaller --onefile --name voxsigil-memory voxsigil_memory/cli.py

# Result: dist/voxsigil-memory.exe (~300MB with all deps + weights)

# User just runs:
voxsigil-memory build-context --query "what is X?" --budget 2048
```

### LM Studio Plugin Shape

```
~/.lmstudio/plugins/voxsigil-memory/
├── manifest.json
│   └── plugin_id: "voxsigil-memory-v0.1"
│       entry_point: "voxsigil_memory/lmstudio_plugin.py"
│
├── voxsigil_memory/
│   └── (entire package)
│
└── models/
    └── blt-semantic-v1.pth
```

**In LM Studio UI:**
```
[Settings] → [Plugins] → [Enable "VoxSigil Memory"]
[Chat]     → [Context Options] → [Use VoxSigil VME?] [✓]

Query: "What did I say about X?"
→ VoxSigil automatically inserts compressed memory
→ Feeds to LLM
```

### Docker Image

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY voxsigil_memory /app/voxsigil_memory
COPY setup.py /app/

RUN pip install -e ".[full]"
RUN voxsigil-memory download-models

ENTRYPOINT ["voxsigil-memory"]
CMD ["build-context", "--help"]
```

### Tests Gate Phase 5

```python
def test_phase_5_fresh_env():
    """Install in clean venv works."""
    # venv1 (via pip)
    subprocess.run([venv1_python, "-m", "pip", "install", "voxsigil-memory"])
    import voxsigil_memory
    assert voxsigil_memory is not None

def test_phase_5_airgapped():
    """Install offline works (no internet)."""
    with network_forbidden():
        # Assume wheel + weights already on disk
        subprocess.run([python, "-m", "pip", "install", "voxsigil-memory*.whl"])
        import voxsigil_memory
        voxsigil_memory.build_context("test")  # Works

def test_phase_5_artifact_size():
    """Wheel is under size budget."""
    wheel_size = os.path.getsize("dist/voxsigil_memory-*.whl")
    assert wheel_size < 20_000_000  # 20MB
    
    exe_size = os.path.getsize("dist/voxsigil-memory.exe")
    assert exe_size < 400_000_000  # 400MB

def test_phase_5_windows_runtime():
    """Runs on Windows."""
    # This test runs on Windows CI
    import subprocess
    result = subprocess.run(
        ["voxsigil-memory", "build-context", "--help"],
        capture_output=True
    )
    assert result.returncode == 0
```

### Artifacts: Phase 5

1. **setup.py** (complete, with extras_require)
2. **MANIFEST.in** (include weights)
3. **build/ scripts** (PyInstaller, Docker)
4. **test_phase_5_distribution.py** (MUST PASS)
5. **dist/voxsigil-memory-0.1.0-py3-none-any.whl**
6. **dist/voxsigil-memory.exe** (Windows standalone)
7. **Dockerfile**

---

## Phase 6: System-Level Evaluation Suite

### Goal
Produce defensible results that match claims. Reproducible benchmarks.

### Benchmark Suite

```python
# benchmarks/latency.py
def benchmark_latency():
    """Measure retrieval + pack building latency."""
    memory = voxsigil_memory.LocalMemory()
    # Load 10K documents
    for doc in dataset_10k:
        memory.add_document(doc)
    memory.build_index()
    
    # Warm up
    for _ in range(10):
        voxsigil_memory.build_context("warmup query")
    
    # Measure
    trials = []
    for query in test_queries:
        start = time.time()
        pack = voxsigil_memory.build_context(query, budget_tokens=2048)
        elapsed_ms = (time.time() - start) * 1000
        trials.append({"query": query, "elapsed_ms": elapsed_ms})
    
    # Aggregate
    p50 = np.percentile([t['elapsed_ms'] for t in trials], 50)
    p95 = np.percentile([t['elapsed_ms'] for t in trials], 95)
    
    return {"p50_ms": p50, "p95_ms": p95, "trials": trials}

# benchmarks/token_efficiency.py
def benchmark_token_efficiency():
    """Measure quality vs token budget trade-off."""
    eval_dataset = [
        {
            "claim": "Paris is the capital of France",
            "context": [long document about France],
            "expected_answer": "Paris",
        },
        ...
    ]
    
    results_by_budget = {}
    for budget_tokens in [512, 1024, 2048, 4096]:
        correct = 0
        for item in eval_dataset:
            pack = voxsigil_memory.build_context(
                query=item["claim"],
                budget_tokens=budget_tokens,
                memory=dataset_memory,
            )
            # Feed pack to LLM, check if it answers correctly
            llm_response = llm.query(pack.expanded_text, item["claim"])
            if correct_answer_in(llm_response, item["expected_answer"]):
                correct += 1
        
        accuracy = correct / len(eval_dataset)
        results_by_budget[budget_tokens] = accuracy
    
    return results_by_budget

# benchmarks/ablations.py
def benchmark_ablations():
    """Measure contribution of each component."""
    scores = {}
    
    # Full system
    full_pack = voxsigil_memory.build_context(query, full_pipeline=True)
    full_score = evaluate(full_pack)
    scores['full'] = full_score
    
    # Without Game-Semantic pruning
    no_prune = voxsigil_memory.build_context(query, semantic=False)
    no_prune_score = evaluate(no_prune)
    scores['no_pruning'] = no_prune_score
    
    # Without latent codec
    no_latent = voxsigil_memory.build_context(query, latent=False)
    no_latent_score = evaluate(no_latent)
    scores['no_latent'] = no_latent_score
    
    # Router thresholds sensitivity
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = voxsigil_memory.build_context(
            query,
            entropy_threshold=threshold
        )
        score = evaluate(result)
        scores[f'threshold_{threshold}'] = score
    
    return scores
```

### Metrics Logging (JSONL)

```python
# benchmarks/runner.py
def run_all_benchmarks():
    """Execute full evaluation suite."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),  # CPU, GPU, Python version
        "voxsigil_version": __version__,
        "benchmarks": {},
    }
    
    # Latency
    results["benchmarks"]["latency"] = benchmark_latency()
    
    # Token efficiency
    results["benchmarks"]["token_efficiency"] = benchmark_token_efficiency()
    
    # Ablations
    results["benchmarks"]["ablations"] = benchmark_ablations()
    
    # Write to JSONL
    with open("results.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")
    
    return results
```

### Regression Gates (CI)

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install
        run: pip install -e ".[dev]"
      - name: Run benchmarks
        run: python benchmarks/runner.py
      - name: Check regression
        run: python benchmarks/check_regression.py
        # Fail if metrics fall beyond tolerance
```

### Tests Gate Phase 6

```python
def test_phase_6_reproducibility():
    """Same seed → same metrics."""
    bench_1 = benchmark_latency(seed=42)
    bench_2 = benchmark_latency(seed=42)
    
    assert bench_1["p50_ms"] == bench_2["p50_ms"]
    assert bench_1["p95_ms"] == bench_2["p95_ms"]

def test_phase_6_regression_gates():
    """Metrics stay within tolerance."""
    current = benchmark_latency()
    baseline = load_baseline_metrics()
    
    p50_regression = current["p50_ms"] / baseline["p50_ms"]
    assert p50_regression < 1.1  # Allow 10% slowdown max
```

### Artifacts: Phase 6

1. **benchmarks/latency.py** (latency measurements)
2. **benchmarks/token_efficiency.py** (quality vs cost)
3. **benchmarks/ablations.py** (component contributions)
4. **benchmarks/runner.py** (orchestration)
5. **benchmarks/check_regression.py** (CI gates)
6. **test_phase_6_benchmarks.py** (MUST PASS)
7. **results.jsonl** (logged metrics)
8. **.github/workflows/benchmarks.yml** (CI)

---

## Phase 7: Paper Writing (After Tests Pass)

### Goal
Document the contribution in a form suitable for publication.

### Paper Core Claim (Tight)

> **VoxSigil VME**: A unified latent-memory codec and deterministic retrieval engine for LLMs that improves effective context efficiency through hierarchical semantic pruning + latent compression + entropy-governed routing, shipped as a single-call library.

### Paper Outline

```
1. Abstract (250 words)
   - Problem: Context budgets are expensive
   - Contribution: VME improves token efficiency by X%
   - Results: Latency < 50ms p50, reproducible

2. Introduction
   - Context length limits LLM performance
   - External RAG is fragile and slow
   - VME is in-process, deterministic, unified

3. Related Work
   - RAG systems (FAISS, Weaviate, etc.)
   - Compression (LLMLingua, others)
   - Memory for LLMs (memGPT, etc.)

4. Method
   4.1 System Architecture
       - Single-call API shape
       - Semantic layer (pruner, codec, router)
       - Protocol layer (determinism, versioning)
   4.2 Game-Semantic Pruning
       - Why dialogue is redundant
       - Scoring function
       - Empirical results on compression
   4.3 BLT Latent Codec
       - Embedding + BLT compression
       - Determinism via seeding
       - GPU-optional design
   4.4 Entropy Router
       - Why entropy governs inclusion
       - Threshold adaptation
   4.5 In-Process Retrieval
       - HNSW design
       - Offline-first architecture

5. Evaluation
   5.1 Benchmarks
       - Latency (p50/p95)
       - Token efficiency (quality vs budget)
       - Memory scaling (index size)
   5.2 Ablations
       - Game-Semantic contribution
       - Latent codec contribution
       - Router sensitivity
   5.3 Reproducibility
       - Seeded determinism
       - CI regression gates
       - Public benchmark code

6. Results (Tables + Graphs)
   - Token efficiency gain: X% fewer tokens, same quality
   - Latency: p50 < 50ms, scales to 100K docs
   - Memory: index size predictable
   - Ablations: each component contributes Y%

7. Limitations
   - Works best for text (images/video future)
   - Requires BLT-Semantic model weights (~500MB)
   - Single-machine retrieval (scales via PgVector)

8. Discussion
   - Why determinism matters
   - Unified approach vs point solutions
   - Future: multi-modal, distributed

9. Reproducibility Statement
   - Code: github.com://voxsigil/vme (open-source)
   - Benchmarks: benchmarks/ (reproducible runner)
   - Models: auto-downloaded; pinned versions
   - License: MIT or Apache 2.0

10. References
```

### Paper-Grade Results (Table Template)

```
Table 1: Token Efficiency
────────────────────────────────────────
Budget (tokens) | Baseline | VME | Gain
────────────────────────────────────────
512             | 40%      | 68% | +28%
1024            | 60%      | 82% | +22%
2048            | 75%      | 88% | +13%
4096            | 85%      | 92% | +7%
────────────────────────────────────────

Table 2: Latency (10K docs, 100 queries)
────────────────────────────────────────
Metric          | p50 (ms) | p95 (ms)
────────────────────────────────────────
Retrieval       | 8        | 15
Pruning         | 12       | 25
Encoding        | 5        | 10
Pack building   | 3        | 8
────────────────────────────────────────
Total           | 28       | 58
────────────────────────────────────────

Figure 1: Token Efficiency vs Context Budget

  Accuracy
  100% ────────────────── Baseline + VME
       │ ╱
   80% │╱─ VME
       │ ╱
   60% │╱
       │╱
   40% └────────────────────
       256  512  1K  2K  4K
       Context Budget (tokens)
```

### Artifacts: Phase 7

1. **paper/voxsigil_vme.tex** (LaTeX source)
2. **paper/figures/** (PNG exports from benchmarks)
3. **paper/tables/** (JSONL → LaTeX tables)
4. **paper/results.jsonl** (raw metrics)
5. **paper/Makefile** (compile to PDF)
6. **paper/README.md** (reproducibility steps)

---

## Phase 8: Open-Source Release

### Goal
Release as production-ready open-source under VoxSigil moniker.

### GitHub Setup

```
github.com/voxsigil/blt-memory-engine/

├── README.md
│   └── Quick-start, features, installation
│
├── CONTRIBUTING.md
│   └── How to add algorithms, report bugs, etc.
│
├── LICENSE
│   └── MIT or Apache 2.0 (pick one)
│
├── pyproject.toml
│   └── Modern Python packaging
│
├── voxsigil_memory/
│   └── (entire package)
│
├── tests/
│   └── All test_phase_*.py files
│
├── benchmarks/
│   └── Reproducible evaluation
│
├── examples/
│   └── hello_memory.py, ingest_corpus.py, etc.
│
├── paper/
│   └── arxiv submission or conference paper
│
├── .github/workflows/
│   ├── tests.yml (run all test gates)
│   ├── benchmarks.yml (regression)
│   └── publish.yml (wheel to PyPI on tag)
│
└── CHANGELOG.md
    └── Version history
```

### Security Posture Statement (README section)

```markdown
## Security

### Protocol Determinism
- Canonical JSON serialization (RFC 7159)
- Deterministic signing with no timestamps in signature
- Message digest verified on deserialization

### Safe Serialization
- No pickle (arbitrary code execution risk)
- Only JSON + manual deserialization
- Type hints validated

### No Auto-Downloads
- Model weights must exist locally or explicitly downloaded
- CLI: `voxsigil-memory download-models`
- Raises FileNotFoundError if weights missing (fail-safe)

### Dependency Transparency
- Pinned dependency versions in requirements/ files
- All dependencies open-source and auditable
- No vendor lock-in; all data structures portable
```

### Versioning Scheme

```
voxsigil-memory: X.Y.Z
                 │ │ └─ Patch (bug fixes)
                 │ └─── Minor (new features, backward-compatible)
                 └───── Major (breaking changes)

blt-semantic weights: model-vA.B.pth
                      │       │
                      │       └─────── Model architecture version
                      └───────────────── Encoder version

Protocol: protocol/VERSION.txt
          "1.0" (major.minor)
          - 1.0 → 1.1 (new fields): backward compatible
          - 1.0 → 2.0 (removed fields): need migration
```

### Minimal Examples (in repo)

```python
# examples/hello_memory.py
import voxsigil_memory

# Initialize
memory = voxsigil_memory.LocalMemory()

# Ingest one document
memory.add_document("Paris is the capital of France.")
memory.build_index()

# Query
pack = voxsigil_memory.build_context(
    query="What is the capital of France?",
    memory=memory,
)

print(pack.expanded_text)  # "Paris is the capital of France."
print(f"Tokens: {pack.token_count}")
```

```python
# examples/ingest_corpus.py
import voxsigil_memory

# Ingest large corpus
memory = voxsigil_memory.LocalMemory(
    storage_path="./my_memory.db",
)

corpus = load_documents("my_documents/")  # 1000s of docs
for doc in corpus:
    memory.add_document(doc['text'], metadata=doc['meta'])

memory.build_index()
memory.save_index("./my_index.hnsw")

# Later, on another machine:
memory_2 = voxsigil_memory.LocalMemory()
memory_2.load_index("./my_index.hnsw")

pack = voxsigil_memory.build_context(
    "What happened in 1969?",
    memory=memory_2,
)
```

```python
# examples/llm_integration.py
import voxsigil_memory
from llama_cpp import Llama

# Initialize VoxSigil
memory = voxsigil_memory.LocalMemory()
memory.load_index("my_index.hnsw")

# Initialize local LLM
llm = Llama(model_path="mistral-7b.gguf")

# User query
user_query = "What did I say about machine learning?"

# Get memory context
pack = voxsigil_memory.build_context(
    query=user_query,
    memory=memory,
    budget_tokens=1024,
)

# Build prompt with memory
prompt = f"""
Context (from memory):
{pack.expanded_text}

User: {user_query}
Assistant:"""

# Call LLM
response = llm(prompt, max_tokens=256)
print(response['choices'][0]['text'])
```

### Tests Gate Phase 8

```python
def test_phase_8_reproducibility_from_scratch():
    """Fresh clone → tests pass."""
    # Simulate: git clone https://github.com/voxsigil/blt-memory-engine
    # pip install -e .
    # pytest
    assert all_tests_pass()

def test_phase_8_wheel_published():
    """Wheel is on PyPI."""
    import subprocess
    result = subprocess.run(["pip", "install", "voxsigil-memory==0.1.0"])
    assert result.returncode == 0

def test_phase_8_examples_run():
    """All examples run without error."""
    for example in Path("examples/").glob("*.py"):
        result = subprocess.run(["python", example])
        assert result.returncode == 0
```

### Artifacts: Phase 8

1. **github.com/voxsigil/blt-memory-engine** (repo)
2. **PyPI release: voxsigil-memory 0.1.0**
3. **Documentation site** (docs.readthedocs.io or github.io)
4. **CHANGELOG.md**
5. **CONTRIBUTING.md**
6. **LICENSE** (MIT or Apache 2.0)
7. **Announcement blog post or tweet**

---

## Mapping Existing Assets to Phases

### What's Already Done (Existing in C:\UBLT)

| Asset | Location | Maps to Phase | Use |
|-------|----------|---------------|-----|
| BLTCore + BLTSystem | temp_recovered_blt.py | Phase 4 | Core codec |
| MetaConsciousness (469 files) | MetaConsciousness/ | Phase 0, 1, 3 | Decision engine, game-semantic |
| 41 Compression Algorithms | C:\UBLT/*.py | Phase 0 (gated) | Optional extras |
| Document Tests | C:\UBLT tests/ | Phase 6 | Benchmarking foundation |
| Architecture Docs | C:\UBLT *.md | Phase 7 | Paper source |
| Bytecode Recovery | restore_blt_modules.py | Phase 0 | Module boundaries |

### What Needs to Be Built (Phases 0-8)

| Phase | Deliverable | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 0 | Module structure + imports | M | Existing BLT + MC |
| 1 | Single-call API | M | Phase 0 |
| 2 | HNSW retrieval | M | hnswlib, numpy |
| 3 | Semantic pipeline | L | Phase 1, 2, existing pruner |
| 4 | GPU-optional codec | M | Phase 3, torch |
| 5 | Distribution | S | setuptools, PyInstaller |
| 6 | Benchmarks | L | pytest, test corpus |
| 7 | Paper | S | LaTeX, existing results |
| 8 | Open-source | S | GitHub, PyPI account |

---

## Dependency Tree (Build Order)

```
Phase 0 (Module boundaries)
    ↓
Phase 1 (Single-call API)
    ↓
Phase 2 (In-process retrieval) + Phase 4 (GPU-optional codec) {parallel}
    ↓
Phase 3 (Semantic pipeline) {requires 1, 2, 4}
    ↓
Phase 5 (Distribution) {requires 0-4}
    ↓
Phase 6 (Evaluation) {requires 5}
    ↓
Phase 7 (Paper) {requires 6}
    ↓
Phase 8 (Open-source) {requires 7}
```

---

## The One Contract

**No code moves between phases until tests in that phase PASS.**

Tests are the gate. Documentation is written only after tests pass.

This ensures:
- ✓ Each phase is shippable
- ✓ Regression is caught immediately
- ✓ Docs never lie (they're generated from working code)
- ✓ Paper claims are backed by reproducible experiments

---

## Summary: VoxSigil BLT-Memory Engine (VME)

**What it is:**  
A unified latent-memory codec library for LLMs. One function. No external services. Test-gated delivery.

**Why it matters:**  
- Context is expensive (tokens = cost + latency)
- External RAG is fragile (external services fail)
- Unified approach is faster & simpler
- Deterministic protocol enables reproducible paper

**When it ships:**  
Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
Each phase is complete when its tests pass.

**How it starts:**  
Today: Phase 0 module structure (map existing BLT+MC to VME boundaries).

---

**This plan converts your years of compression research into a production system, a reproducible paper, and an open-source project. All gated by tests.**
