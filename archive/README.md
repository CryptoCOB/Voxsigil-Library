# Archive - Historical & Experimental Work

**Purpose:** This directory contains experimental code, phase documentation, and generated outputs from the VoxSigil development journey. These files document the system's evolution and remain available for reference.

**Last Updated:** Feb 16, 2026

---

## Directory Structure

### `phase_0_roadmap/`
Original planning and roadmap documents:
- `PHASE_0_ROADMAP.md` - Initial system design and phases
- `PHASE_3_5_*.md` - Phase 3-5 evaluation and benchmark infrastructure planning

**Relevance:** Historical context for understanding system design decisions

---

### `phase_1_foundation/`
Foundation & infrastructure validation (Phase A):
- Early environmental setup
- Base system validation scripts
- Initial entropy and router testing

**Relevance:** Foundation work - preserved for completeness

---

### `phase_2_integration/`
Integration & embedding system (Phase B):
- Behavioral characteristic analysis
- Entropy calculation validation
- Initial hybrid router implementation
- BLT embedding integration tests

**Relevance:** Integration groundwork - may inform future embedding improvements

---

### `phase_3_operation/`
Continuous operation & evaluation (Phase C):
- Population management
- Continuous evaluation loops
- Performance tracking

**Relevance:** Legacy evaluation approach - now superseded by VME 2.0 phases 4-6

---

### `phase_4_cognitive/`
Cognitive optimization & benchmarking attempts (Phases 4-6 pre-VME):
- Early student embedder experiments
- Semantic space exploration
- Initial benchmarking frameworks
- Phase completion documentation

**Relevance:** Experimental work that informed final VME 2.0 architecture

**Note:** Final production implementations are in `vme/` directory (main repo)

---

### `other_experiments/`
Additional experimental algorithms and utilities:
- Behavioral NAS experiments
- Quantum-aware algorithms  
- Alternative cognitive approaches
- Training pipeline variants
- Utility scripts (cleanup, decompilation, recovery)
- Demo scripts and tests

**Included files:**
- `behavioral_nas_nextgen.py` - NAS for behavior patterns
- `quantum_behavioral_nas.py` - Quantum-informed architecture search
- `reasoning_engine.py`, `tot_engine.py` - Alternative reasoning approaches
- `precompute_*.py` - Various precomputation strategies
- `voxsigil_enhance_pipeline*.py` - Enrichment pipeline iterations
- `blt_*.py` - BLT system integration attempts
- And 50+ other experimental files

**Relevance:** These represent alternative design paths explored during development. Some may contain useful techniques for future optimization work.

**Note:** Do NOT use in production - these are experimental and may not maintain compatibility with current VME 2.0

---

### `generated_outputs/`
Generated data, benchmarks, and analysis:
- `phase4b_outputs/` - Phase 4-B student embedder outputs
- `phase5_outputs/` - Phase 5 attribution calculation results
- `phase6_outputs/` - Phase 6 benchmarking results
- `generated_sigils/` - Generated behavioral sigils for testing
- `*.json` - Analysis outputs (schema analysis, inventory, context summary)
- `*.txt` - Output logs and reports

**Relevance:** 
- Results data from historical phase runs
- Useful for comparing against new iterations
- Benchmark baselines for performance tracking

**Note:** These are snapshot outputs. For current production benchmarks, see `vme/` in main repo

---

## When to Use Files in Archive

### ✅ Use for:
- **Historical context**: Understanding design decisions
- **Technique reference**: Borrowing algorithms or approaches
- **Performance baseline**: Comparing against previous results
- **Documentation**: Explaining the development journey
- **Recovery**: If you need to revisit an experimental approach

### ❌ Do NOT use for:
- Production code (use `vme/` in main repo instead)
- Current system operation (use VME 2.0 components)
- New integrations (reference main repo examples)
- Active development (no longer maintained)

---

## Key Transitions

### Phase 3 → Phase 4-6 (VME 2.0)
**What changed:**
- ❌ Legacy evaluation (Phase C) → ✅ Comprehensive phases 4-6 system
- ❌ Manual entropy tuning → ✅ Automated cognitive optimization
- ❌ Simple benchmarking → ✅ Parallel multi-model orchestration
- ❌ Ad-hoc reward logic → ✅ Tiered attribution system

**Where to find replacements:**
- Cognitive optimization: `vme/phase4b/`
- Attribution system: `vme/phase5/`
- Benchmarking: `vme/phase6/`

### Experimental Algorithms → VME 2.0
**What happened:**
- NAS experiments → Informed student embedder training
- Quantum studies → Archived (not featured in current system)
- Alternative reasoning → Focused on pragmatic semantic routing
- Multiple enrichment pipelines → Consolidated into single pipeline

**Current system:**
- See `vme/` and main repo for production code
- Archive contains the exploration journey

---

## File Retrieval Guide

If you're looking for something specific:

**"I need to understand how [feature] was originally designed"**
→ Check `phase_0_roadmap/` or relevant phase directory

**"I want to see an alternative implementation of [component]"**
→ Check `other_experiments/` (but remember: experimental!)

**"I need historical benchmark results"**
→ Check `generated_outputs/phase*/`

**"I'm implementing [algorithm], can I reuse code from archive?"**
→ Possible, but review carefully - code may not match current architecture

**"What happened to [the old system]?"**
→ Most legacy systems were either:
1. Integrated into VME 2.0 (check `vme/`)
2. Archived as experimental (check `other_experiments/`)
3. Superseded by newer approaches (check phase documentation)

---

## Development Notes

### Why Keep the Archive?
1. **Accountability** - Shows the full development journey
2. **Learning** - Documents what worked and what didn't
3. **Recovery** - If a future iteration needs historical context
4. **Transparency** - Shows the evolution of design decisions

### Archive Maintenance
- ✅ READ-ONLY: Archive files should not be modified
- ✅ REFERENCE: Can be referenced in documentation
- ⚠️ INTEGRATION: Do not directly integrate code without review
- ✅ PRESERVATION: Keep indefinitely for historical record

---

## Quick Links

- **Production System**: [`vme/`](../vme/) in main repository
- **Core SDK**: [`src/`](../src/) in main repository
- **Documentation**: [`docs/`](../docs/) in main repository
- **Main README**: [`README.md`](../README.md)
- **Agent Integration Guide**: [`AGENT_INTEGRATION_GUIDE.md`](../AGENT_INTEGRATION_GUIDE.md)

---

## Questions?

For questions about archived code or historical decisions:
1. Check the relevant phase documentation in this archive
2. Review the main repo's `CHANGELOG.md` and `README.md`
3. Check GitHub issues/discussions for related topics

---

**Status:** Archive is read-only and historical. For current development, use the main repository structure.
