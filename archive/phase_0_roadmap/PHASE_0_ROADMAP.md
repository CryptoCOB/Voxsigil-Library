# VoxSigil VME: Phase 0 Implementation Roadmap

## Immediate Next Steps (This Week)

### 1. Create Repository Structure
```bash
# From C:\UBLT, create the voxsigil_memory package
mkdir -p voxsigil_memory/{semantic,protocol,storage,retrieval,compression,models}
mkdir -p voxsigil_memory/tests
mkdir -p voxsigil_memory/examples
mkdir -p benchmarks
mkdir -p paper
```

### 2. Setup.py (Dependency Tiers)
Create `setup.py` with:
- Tier A: python>=3.10, stdlib only
- Tier B: numpy, torch[cpu], hnswlib (optional but recommended)
- Tier C: torch[cuda] (optional for GPU)

### 3. Initial Test Suite (Phase 0 Gates)
```python
# voxsigil_memory/tests/test_phase_0_imports.py

def test_import_zero_overhead():
    """No side effects on import."""
    import voxsigil_memory
    # Verify:
    # - No files created
    # - No GPU initialized
    # - No DB opened
    # - No downloads triggered

def test_determinism_protocol():
    """Protocol layer produces bit-exact output."""
    from voxsigil_memory.protocol import ProtocolSigner
    signer = ProtocolSigner()
    payload = {"query": "test"}
    canonical_1 = signer.canonicalize(payload)
    canonical_2 = signer.canonicalize(payload)
    assert canonical_1 == canonical_2
```

### 4. Module Boundaries (Stubs)
Create placeholder files with docstrings:
- `voxsigil_memory/__init__.py` → exports `build_context()`
- `voxsigil_memory/protocol/sign.py` → `ProtocolSigner` class
- `voxsigil_memory/semantic/pruner.py` → `GameSemanticPruner` class
- `voxsigil_memory/retrieval/retriever.py` → `VectorRetriever` class
- `voxsigil_memory/models/blt_semantic.py` → `BLTSemanticEmbedder` class

All stubs should:
- Have correct signatures
- Raise `NotImplementedError` with clear message
- Have full docstrings with examples

### 5. Move/Reference Existing Code
```
├── voxsigil_memory/models/blt_semantic.py
│   └── Import or copy relevant parts from:
│       - temp_recovered_blt.py (core compression logic)
│       - Consciousness modules (semantic understanding)
│
├── voxsigil_memory/semantic/pruner.py
│   └── Reference Game-Semantic from MetaConsciousness/frameworks/
│
└── voxsigil_memory/compression/ecosystem.py
    └── Gate access to all 41 algorithms
        (they stay in submodule, not copied)
```

---

## Phase 0 Completion Criteria

All of these must be true:

1. **Import Test Passes**
   ```bash
   python -c "import voxsigil_memory; voxsigil_memory.build_context('test')"
   ```
   - Works without side effects
   - Raises clear NotImplementedError (not AttributeError)

2. **Module Signatures Correct**
   - `voxsigil_memory.build_context(query, user_id, budget_tokens, mode, ...)`
   - Internal modules exist and importable

3. **Determinism Test Passes**
   - ProtocolSigner produces bit-exact output for same inputs

4. **No External Dependencies on Import**
   - torch not loaded
   - numpy not required
   - CUDA not initialized

5. **Documentation**
   - README.md exists
   - Architecture diagram in markdown
   - Setup instructions clear

---

## Transition to Phase 1

Once Phase 0 passes:

1. Replace stubs with real implementations
2. Implement `build_context()` function
3. Create ContextPack dataclass
4. Write Phase 1 tests (golden determinism)

**Rule**: Phase 1 work doesn't start until all Phase 0 tests are green.

---

## Key Principles (Enforce Throughout)

1. **Test-First**: Write test before implementation
2. **One Function Externally**: Everything flows through `build_context()`
3. **Clean Boundaries**: Semantic layer separate from protocol layer
4. **Determinism**: Every computation must be reproducible with seed
5. **No External Services**: HNSW index, SQLite storage, all in-process
6. **Graceful Degradation**: CPU fallback if GPU unavailable
7. **Local-First**: All weights, models, indexes live on user's machine

---

## Success Metrics for Phase 0

- [  ] `pytest voxsigil_memory/tests/test_phase_0_imports.py` → PASS
- [  ] `python -c "import voxsigil_memory"` → no errors
- [  ] Setup.py builds wheel without errors
- [  ] Module boundaries documented (this file)
- [  ] Existing BLT + MC code referenced correctly
- [  ] No uncommitted third-party code in C:\UBLT

---

## Files to Create This Week

```
voxsigil_memory/
├── __init__.py                 (build_context stub)
├── __version__.py              (0.1.0-dev)
├── semantic/
│   ├── __init__.py
│   ├── pruner.py              (GameSemanticPruner stub)
│   ├── codec.py               (BLTLatentCodec stub)
│   ├── router.py              (EntropyRouter stub)
│   └── pack_builder.py        (ContextPackBuilder stub)
├── protocol/
│   ├── __init__.py
│   ├── sign.py                (ProtocolSigner)
│   ├── envelope.py            (Envelope stub)
│   └── versioning.py          (ProtocolVersion stub)
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py           (VectorRetriever stub)
│   ├── hnsw_index.py          (HNSWIndex stub)
│   └── flat_index.py          (FlatIndex stub)
├── storage/
│   ├── __init__.py
│   ├── adapter.py             (StorageAdapter abstract)
│   ├── sqlite_adapter.py      (SQLiteMemoryStore stub)
│   └── postgres_adapter.py    (PgvectorMemoryStore stub)
├── models/
│   ├── __init__.py
│   ├── blt_semantic.py        (BLTSemanticEmbedder stub)
│   ├── defaults.py            (get_default_embedder stub)
│   └── weight_manager.py      (download_models stub)
├── compression/
│   ├── __init__.py
│   ├── selector.py            (CompressionSelector stub)
│   └── ecosystem.py           (CompressionEcosystem gateway)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_phase_0_imports.py
└── examples/
    └── (empty for now)

setup.py                        (with dependency tiers)
MANIFEST.in                     (include README, LICENSE)
README.md                       (quick start)
LICENSE                         (MIT or Apache 2.0)
```

---

## Checkpoints

**By End of Week 1:**
- Package structure created
- All stubs in place
- Phase 0 tests written (but failing)

**By End of Week 2:**
- Phase 0 tests passing
- build_context signature correct
- ProtocolSigner deterministic
- Import zero-overhead verified

**Start Phase 1:**
- Implement actual build_context logic
- Create ContextPack dataclass
- Write golden determinism tests

---

## Notes for Implementation

### On Determinism
- Use `dataclasses` (not `attrs`) for reproducible JSON
- Use `json.dumps(sort_keys=True)` for canonical form
- Never use timestamps in signatures; only in metadata
- Seed all randomness (numpy, torch) explicitly

### On GPU Handling
- Detect CUDA availability, don't import torch until needed
- If user passes device='cuda' but CUDA unavailable, raise clear error
- If device=None (default), silently fallback to CPU

### On Storage
- Default is SQLite (no external service)
- Postgres adapter optional, imported only if requested
- All adapters implement same interface (abstract base class)

### On Compression
- The 41 algorithms stay in their original repo (or submodule)
- VME only exposes the ones in critical path (Game-Semantic, BLT, entropy router)
- Others available via `voxsigil_memory.compression.ecosystem.get_algorithm(name)`

---

## Estimated Timeline (No Pressure)

| Phase | Timeline | Status |
|-------|----------|--------|
| 0 | Week 1-2 | Starting now |
| 1 | Week 3-4 | After Phase 0 ✓ |
| 2 | Week 5-6 | After Phase 1 ✓ |
| 3 | Week 7-10 | After Phase 2 ✓ |
| 4 | Week 11-12 | Parallel with Phase 3 |
| 5 | Week 13-14 | After Phase 4 ✓ |
| 6 | Week 15-20 | Runs during Phase 5 |
| 7 | Week 21-22 | After Phase 6 ✓ |
| 8 | Week 23-24 | After Phase 7 ✓ |

**Total: ~6 months, test-gated, no crunches.**

---

**Remember: The test gate is the contract. Code doesn't advance until tests pass. This is how you prevent half-finished systems and ensure reproducibility from day one.**
