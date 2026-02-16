# VoxSigil VME - Complete Dependency Audit

**Date:** February 12, 2026  
**Status:** Pre-Phase 2 Environment Analysis  

---

## Current Installation

### Already Installed
✓ pytest 9.0.2 - testing framework
✓ numpy 2.4.2 - numerical computing
✓ hnswlib 0.8.0 - HNSW indexing (Phase 2)
✓ click 8.3.1 - CLI
✓ packaging, pluggy, etc. (test infrastructure)

### Total Packages Installed: 14

---

## Dependency Analysis by Codebase

### Tier 1: BLT Core Files (10 files)
These are the foundation compression engine.

**Minimal Dependencies:**
- temp_recovered_blt.py: time, logging (stdlib)
- temp_recovered_blt_interface.py: logging, typing, datetime, threading (stdlib)
- blt_student_interface.py: logging, typing, datetime, threading (stdlib)
- proof_of_useful_work.py: logging, typing (stdlib)

**Heavy Dependencies:**
- complete_blt_student_implementation.py: torch, asyncio, modules (custom)
- start_blt_integrated_training.py: torch, numpy, requests, psutil, yaml, bitsandbytes
- system_wide_blt_integration.py: torch, torch.nn, asyncio, logging
- test_blt_integration.py: torch, training (custom), research (custom)
- test_blt_distillation.py: torch, numpy, logging
- test_blt_integration.py: torch

**BLT Tier 1 Verdict:** Minimal core can work with just stdlib. Full training pipeline needs PyTorch.

### Tier 2: Compression Algorithms (41 files)
Core compression implementations.

**Sample File Dependencies:**
- quantum_behavioral_nas.py: nebula.utils (custom fallback available)
- convergence_training_system.py: torch, numpy, websockets, matplotlib
- ghost_detection_protocol.py: (checking...)
- neural_architecture_search.py: (checking...)
- knowledge_distillation_system.py: torch, sklearn likely

**Algorithms Verdict:** Mix of stdlib-only and torch-dependent. Can implement core versions without torch.

### Tier 3: MetaConsciousness Framework (136 files)
Intelligent algorithm selection.

**Expected Dependencies:**
- numpy (for math)
- torch (for ML components)
- sklearn (for classical ML)
- scipy (for signal processing)
- Likely: transformers, huggingface_hub (for embeddings)

**Framework Verdict:** Most likely requires full ML stack for intelligent selection.

### Tier 4: VoxSigil VME (New - in voxsigil_memory/)

**Phase 0-1 (Complete):**
- stdlib only (logging, typing, dataclasses)

**Phase 2 (In Progress):**
- hnswlib ✓ (installed)
- numpy (for embeddings) ✓ (installed)
- Need: embedding model (transformers OR sentence-transformers OR lightweight alternative)

**Phase 3-4 (Coming):**
- hashlib (stdlib) for signing
- datetime (stdlib) for versioning
- scipy (for compression)
- torch (optional for GPU acceleration)

---

## Strategic Dependency Plan for Phase 2

### Option A: Heavyweight (Full ML Stack)
**Install:** torch, tensorflow, transformers, scipy, scikit-learn, pandas

**Pros:**
- Full access to pre-trained embedding models
- Can use MetaConsciousness intelligent selection
- Future-proof for advanced compression

**Cons:**
- Large footprint (torch alone ~2GB)
- Slow installation
- Many unused dependencies for minimal MVP

**Recommendation:** NOT YET - wait until Phase 4+

### Option B: Lightweight (Minimal MVP)  ⭐ RECOMMENDED
**Install:** scipy, scikit-learn, sentence-transformers

**Pros:**
- sentence-transformers is lightweight (~500MB)
- Pre-trained embeddings without full torch overhead
- Can implement Phase 2 quickly
- Minimal unused dependencies

**Cons:**
- Limited to sentence-level embeddings
- Cannot use full BLT training pipeline yet

**Recommendation:** USE FOR PHASE 2

### Option C: Ultra-Minimal (DIY)
**Install:** Just what's needed for core compression

**Pros:**
- Smallest footprint
- Fastest installation
- Full control

**Cons:**
- Need to implement embedding generation
- More work
- Less semantic sophistication

**Recommendation:** Use as fallback if Option B fails

---

## Phase 2 Implementation Path

### What Phase 2 Actually Needs
- HNSW indexing ✓ (hnswlib installed)
- Vector embeddings (need sentence-transformers)
- Semantic encoder (we implement)
- Semantic router (we implement)
- Semantic pruner (we implement)

### Decision: Use Option B (Lightweight)

**New packages to install:**
```bash
pip install sentence-transformers scipy scikit-learn
```

**Rationale:**
1. sentence-transformers gives us embeddings without torch overhead
2. scipy + sklearn cover compression math
3. Keep footprint minimal for Phase 2 MVP
4. Can evolve to Option A (full ML stack) in Phase 5+ for distribution

---

## Import Requirements by Phase

### Phase 0-1 (Complete) ✓
- logging, typing, dataclasses (stdlib)
- pytest (installed)

### Phase 2 (In Progress)
**Minimum versions needed:**
```
hnswlib >= 0.8.0 ✓ (installed)
numpy >= 1.20.0 ✓ (installed: 2.4.2)
sentence-transformers >= 2.0.0 (INSTALL)
scipy >= 1.5.0 (INSTALL)
scikit-learn >= 0.24.0 (INSTALL)
```

### Phase 3-4 (Protocol + Semantic)
```
hashlib (stdlib) ✓
datetime (stdlib) ✓
scipy (Phase 2) ✓
numpy (Phase 2) ✓
```

### Phase 5+ (Distribution)
```
torch >= 2.0.0 (optional, for GPU)
transformers >= 4.0.0 (optional, for advanced models)
```

---

## Contamination Analysis

### Safe to Import Directly
- proof_of_useful_work.py (stdlib only)
- temp_recovered_blt.py (stdlib only)
- blt_student_interface.py (stdlib only)

### Conditional Import (With Error Handling)
- convergence_training_system.py (needs torch - skip if not installed)
- neural_architecture_search.py (needs torch - skip if not installed)
- quantum_behavioral_nas.py (has fallback functions)

### Not Ready for Phase 2
- MetaConsciousness framework (needs full ML stack)
- Advanced compression algorithms (need torch)
- BLT training pipeline (needs torch, bitsandbytes, etc.)

---

## Recommendation Summary

### For Phase 2 Implementation:
1. ✓ Keep voxsigil_memory/ stdlib-only at core
2. Install: `pip install sentence-transformers scipy scikit-learn`
3. Import safe algorithms (proof-of-work, ghost detection)
4. Use sentence-transformers for semantic embeddings
5. Implement HNSW retrieval with those embeddings
6. Test Phase 2 with minimal dependencies

### Libraries to Skip for Now:
- torch (Phase 5+)
- tensorflow (Phase 5+)
- transformers (Phase 5+)
- bitsandbytes (Phase 5+)
- websockets (Phase 5+)
- MetaConsciousness (Phase 5+)

### Deferred to Phase 5 (Distribution):
- Full ML stack installation
- MetaConsciousness framework integration
- Advanced BLT training components
- GPU acceleration

---

## Action Plan

### Step 1: Install Phase 2 Dependencies
```bash
pip install sentence-transformers scipy scikit-learn
```

### Step 2: Update build_context() for Phase 2
- Add HNSWRetriever initialization
- Add sentence-transformers embedding generation
- Implement SemanticRouter using embeddings
- Stub SemanticEncoder for Phase 3

### Step 3: Create Phase 2 Tests
- Test HNSW indexing
- Test embedding generation
- Test retrieval accuracy

### Step 4: Verify All Tests Pass
- Phase 0 tests (module boundaries)
- Phase 1 tests (API contracts)
- Phase 2 tests (retrieval)

---

**Decision:** Proceed with **Option B (Lightweight)** for Phase 2.

