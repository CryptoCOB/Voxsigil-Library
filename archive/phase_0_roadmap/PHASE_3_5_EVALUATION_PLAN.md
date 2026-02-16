# Phase 3.5: Evaluation & Benchmarking Gates

**Objective**: Prove VoxSigil VME delivers value (token efficiency + answer quality) before Phase 4 GPU codec.

**Date**: February 12, 2026  
**Status**: Planning → Implementation

---

## Why Phase 3.5 Exists

Unit tests (Phase 3: 22/22 ✅) prove **correctness** not **value**:

❌ Unit tests DON'T tell us:
- Whether GameSemanticPruner removes critical facts on real documents
- Whether retrieval+packing actually improves downstream Q&A accuracy
- Whether entropy_score (compression ratio) correlates with usefulness
- Whether performance holds at 100K chunks

✅ Phase 3.5 will prove:
- Token efficiency gains on real/synthetic tasks
- Quality preservation at fixed budget
- Latency + throughput at scale
- Which components drive the gains (ablations)
- Where it safely fails (adversarial)

---

## The Measurement Strategy

### Tier 1: Deterministic Hardware Metrics (No LLM Required)
Fast, reproducible anywhere, proves the machine works.

**Deliverables**:
- Latency breakdown per component (p50/p95/p99)
- Throughput (docs/sec, tokens/sec)
- Memory scaling (index size, sqlite size)
- Determinism signature stability
- **Target**: Latency p50 ≤ 50ms @ 10K units (CPU)

### Tier 2: Utility Metrics (Synthetic Task Harness)
Uses local synthetic Q&A to avoid LLM dependency while proving usefulness.

**Deliverables**:
- Recall@budget (512/1K/2K tokens)
- Token efficiency ratio (useful facts / tokens)
- Pruning quality (key facts retained %)
- Entropy correlation with downstream utility
- **Target**: Token reduction 20-40% at fixed quality

### Tier 3: Ablations (Component Attribution)
Answers: "What's actually driving the wins?"

**Deliverables**:
- Latency contribution per component
- Quality contribution per component
- Token efficiency per component
- Identifies dead weight or bottlenecks
- **Target**: Full > any partial

### Tier 4: Adversarial & Stress (Safety Cliffs)
Answers: "Where does it break?"

**Deliverables**:
- Pruner adversarial suite (contradictions, negations, buried facts)
- Entropy proxy validity (check correlation strength)
- Scale stress tests (10K → 100K → 1M units)
- Graceful degradation patterns
- **Target**: Safe failure modes, no crashes, documented limitations

---

## File Structure (to create)

```
C:\UBLT\
├── benchmarks/
│   ├── __init__.py
│   ├── runner.py                 # Master orchestrator
│   ├── bench_latency.py          # p50/p95/p99 per component
│   ├── bench_scale.py            # 10K/100K/1M corpus sizes
│   ├── bench_ablations.py        # Component contribution
│   ├── bench_quality_synthqa.py  # Synthetic QA utility
│   ├── bench_adversarial.py      # Edge cases & failure modes
│   ├── baselines/
│   │   └── baseline_thresholds.json  # CI gates
│   └── results/
│       └── (output JSONL + tables)
│
├── datasets/
│   ├── synth_facts_10k.jsonl     # Synthetic knowledge base
│   ├── synth_qa_pairs.jsonl      # Query-answer pairs
│   ├── adversarial_pruning.jsonl # Edge case documents
│   └── README.md
│
├── PHASE_3_5_BENCHMARK_SPEC.md   # Reproducibility doc
└── VOXSIGIL_VME_PHASE_3_COMPLETE.md (updated)
```

---

## Tier 1: Latency Benchmarks (Easiest, Highest ROI)

**File**: `benchmarks/bench_latency.py`

```python
"""Measure latency breakdown at 10K units (realistic corpus)."""

def benchmark_component_latency():
    """
    Measure p50/p95/p99 for each component separately.
    
    Components:
    - HNSW retrieval (search in index)
    - GameSemanticPruner (prune candidate)
    - BLTLatentCodec.encode (embed + compress)
    - EntropyRouter (filter by entropy + budget)
    - ContextPackBuilder (assemble final pack)
    
    Output: JSONL with {component, p50_ms, p95_ms, p99_ms, trials_n}
    """

def benchmark_end_to_end_latency():
    """
    E2E workflow: query → retrieve → prune → encode → route → pack
    
    Measures:
    - Total latency at 10K units (corpus size)
    - Breakdown by phase
    - Impact of budget_tokens on latency
    
    Output: {budget_tokens: 256/512/1K/2K, latency_ms, breakdown}
    """

def benchmark_scale():
    """
    How latency scales with corpus size.
    
    Corpus sizes: 1K, 10K, 100K
    Metrics: p50, p95 total latency
    
    Gates:
    - p50(10K) ≤ 50ms
    - p95(10K) ≤ 200ms
    - sublinear growth (not just linear in N)
    """
```

**Expected Output**:
```
{
  "timestamp": "2026-02-12T...",
  "system": {"cpu": "...", "gpu": "no", "python": "3.13"},
  "corpus_size": 10000,
  "latency": {
    "hnsw_search": {"p50": 8, "p95": 15, "p99": 25},
    "prune": {"p50": 12, "p95": 25, "p99": 40},
    "encode": {"p50": 5, "p95": 10, "p99": 15},
    "route": {"p50": 3, "p95": 8, "p99": 12},
    "pack": {"p50": 2, "p95": 5, "p99": 10},
    "total": {"p50": 30, "p95": 63, "p99": 112}
  },
  "gates_passed": {
    "p50_under_50ms": true,
    "p95_under_200ms": true
  }
}
```

---

## Tier 2: Quality Benchmarks (Synthetic QA)

**File**: `benchmarks/bench_quality_synthqa.py`

```python
"""
Measure whether VoxSigil improves Q&A accuracy at fixed token budgets.

Strategy: Create synthetic dataset where ground truth facts live in memory.
"""

def create_synthetic_qa_dataset(num_facts=1000):
    """
    Build a knowledge base where:
    - Each fact is one sentence
    - Each query targets 1-3 facts
    - "Useful" = whether query-relevant facts made it to final pack
    
    Example:
    Fact: "Paris is the capital of France."
    Query: "What is the capital of France?"
    Answer: Should require "Paris" from memory.
    """

def benchmark_quality_by_budget():
    """
    Measure: Accuracy @ different token budgets
    
    Budgets: 256, 512, 1K, 2K, 4K tokens
    
    Metrics:
    - Accuracy (% of queries answered correctly)
    - Precision (% of retrieved facts used)
    - Recall (% of relevant facts included)
    - Token efficiency ratio = (accurate answers) / token_count
    
    Gates:
    - Accuracy >= 80% @ 512 tokens
    - Token reduction >= 20% vs baseline @ fixed accuracy
    """

def benchmark_with_vs_without(components=['prune', 'encode', 'route']):
    """
    Compare:
    - Baseline (HNSW retrieval only) 
    - +Prune
    - +Prune+Encode
    - Full (all components)
    
    Goal: Confirm full > any partial (no dead weight)
    """
```

**Expected Output**:
```
{
  "timestamp": "...",
  "dataset": "synth_qa_1k_facts",
  "results_by_budget": {
    "256_tokens": {"accuracy": 0.62, "token_efficiency": 0.0062},
    "512_tokens": {"accuracy": 0.78, "token_efficiency": 0.0078},
    "1024_tokens": {"accuracy": 0.88, "token_efficiency": 0.0088},
    "2048_tokens": {"accuracy": 0.95, "token_efficiency": 0.0095}
  },
  "ablation": {
    "baseline": {"512_toks_accuracy": 0.71},
    "plus_prune": {"512_toks_accuracy": 0.75},
    "plus_encode": {"512_toks_accuracy": 0.73},
    "full": {"512_toks_accuracy": 0.78}
  },
  "gates_passed": {
    "accuracy_at_512": true,
    "token_reduction_ge_20pct": true
  }
}
```

---

## Tier 3: Ablation Attribution

**File**: `benchmarks/bench_ablations.py`

```python
def ablation_latency():
    """
    Which component is the latency bottleneck?
    
    Variants:
    - Retrieval only (HNSW)
    - +Prune
    - +Encode
    - +Route
    - Full
    
    Metric: latency per variant
    
    Goal: Identify if any component is a latency cliff
    """

def ablation_quality():
    """
    Which component improves accuracy?
    
    Measure accuracy on synthetic QA with each variant.
    
    Expected: Full > any partial
    
    If prune doesn't help → maybe encoder alone is doing it
    If code doesn't help → remove it (dead weight)
    """

def ablation_token_efficiency():
    """
    Which component reduces tokens?
    
    Measure: average token count in final pack per variant
    
    Expected stacking: route < encode+route < prune+encode+route
    """
```

**Expected Output**:
```
{
  "variant": "full",
  "latency_p50_ms": 30,
  "accuracy": 0.78,
  "avg_tokens_per_pack": 487,
  "component_contribution": {
    "hnsw": 8.0,
    "prune": 12.0,
    "encode": 5.0,
    "route": 3.0,
    "pack": 2.0
  }
}
```

---

## Tier 4: Adversarial & Stress

**File**: `benchmarks/bench_adversarial.py`

```python
def test_pruner_adversarial():
    """
    Edge cases that break naive pruning:
    
    1. Contradiction flips
       "X is true. Actually, X is false."
       → Should preserve BOTH sentences
    
    2. Negation bombs
       "Not A, not B, but C is critical."
       → Should preserve critical sentence
    
    3. Buried key facts
       "Intro. Filler filler filler. KEY FACT. Filler."
       → Should preserve KEY FACT even mid-document
    
    4. Instruction collisions
       "Do NOT do X" (critical negation)
       → Should preserve negation (not prune as "low importance")
    
    Metric: "Key facts retained %" on adversarial set
    
    Gate: Retain >= 90% of marked key facts
    """

def test_entropy_proxy_validity():
    """
    Your entropy_score = len(compressed) / len(original)
    
    Is this actually predictive of "usefulness"?
    
    Measure: correlation(entropy_score, downstream_accuracy)
    
    If correlation < 0.5 → entropy proxy is weak
    If correlation >= 0.7 → proxy is good
    If 0.5-0.7 → borderline, consider improvement
    """

def stress_test_scale():
    """
    Does the system hold at larger scales?
    
    Corpus: 1K → 10K → 100K units
    
    Metrics per scale:
    - Index build time
    - Index size (MB)
    - Query latency
    - Memory peak
    
    Gates:
    - Index size grows sublinearly (not exploding)
    - Query latency <= 100ms @ 100K
    - No crashes or OOM
    """
```

---

## Tier 1.5: Determinism Verification

**File**: `benchmarks/bench_determinism.py`

```python
def verify_seeded_determinism():
    """
    Same query + same corpus + same seed = identical pack signature?
    
    Process:
    1. Encode corpus docs (with seed=42)
    2. Run query 3 times: route → pack
    3. Compute pack signature (hash of expanded_text + metadata)
    4. All 3 signatures should match exactly
    
    Gate: Signatures match across 3 runs
    """

def verify_cross_machine():
    """
    If we ship seed + corpus snapshot to another machine,
    do we get identical pack?
    
    (This is for future testing on CI)
    """
```

---

## CI Regression Gates (CI/CD Integration)

**File**: `benchmarks/baselines/baseline_thresholds.json`

```json
{
  "latency": {
    "p50_total_ms_10k": {"target": 50, "tolerance_pct": 10, "fail_if": "greater"},
    "p95_total_ms_10k": {"target": 200, "tolerance_pct": 10, "fail_if": "greater"}
  },
  "quality": {
    "accuracy_at_512_tokens": {"target": 0.75, "tolerance_pct": 2, "fail_if": "less"},
    "token_reduction": {"target": 0.20, "tolerance_pct": 5, "fail_if": "less"}
  },
  "determinism": {
    "signature_stability_runs": {"target": 3, "fail_if": "signatures_differ"}
  },
  "adversarial": {
    "key_facts_retained_pct": {"target": 90, "tolerance_pct": 5, "fail_if": "less"}
  }
}
```

**CI Pipeline**:
```yaml
# .github/workflows/benchmarks.yml
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
      - name: Check regression gates
        run: python benchmarks/check_gates.py
        # Fails PR if any gate violated
      - name: Upload results
        run: gsutil cp results/*.jsonl gs://voxsigil-benchmarks/
```

---

## Benchmark Spec Document

**File**: `PHASE_3_5_BENCHMARK_SPEC.md`

For **reproducibility & publication honesty**, document:

```markdown
# VoxSigil VME: Benchmark Specification

## System Configuration
- **Embedding Model**: sentence-transformers all-MiniLM-L6-v2 (384-d)
- **Index**: HNSW (M=16, ef_construction=200, ef_search=200)
- **Compression**: zlib level 6
- **Token Heuristic**: 1 token ≈ 4 characters
- **Seed**: 42 (for reproducibility)

## Corpus
- **Synthetic QA**: 1000 facts + 500 queries
- **Adversarial**: 50 edge-case documents
- **Scale test**: 1K, 10K, 100K generated units

## Hardware Assumption
- **CPU**: Intel Xeon or equivalent
- **RAM**: 8GB minimum
- **GPU**: None (CPU-only baseline)

## How to Reproduce
1. `git clone ...`
2. `pip install -e ".[bench]"`
3. `python benchmarks/runner.py`
4. Results → `results/bench_TIMESTAMP.jsonl`

## Metrics Definitions
- **p50/p95/p99**: Percentiles over N trials (N >= 100)
- **Accuracy**: % queries answered correctly (synthetic QA)
- **Token Efficiency**: (correct answers) / (tokens in pack)
- **Determinism**: Byte-identical signature across 3 seeded runs
```

---

## Implementation Sequence (What to Build First)

### Week 1 (Now)
1. ✅ Phase 3.5 Planning (this doc)
2. `benchmarks/bench_latency.py` (fast, proves the machine works)
3. `datasets/synth_facts_10k.jsonl` (small, static dataset)

### Week 2
4. `benchmarks/bench_quality_synthqa.py` (answers: "does it help?")
5. `benchmarks/bench_ablations.py` (answers: "what's doing it?")
6. CI gates in place

### Week 3
7. `benchmarks/bench_adversarial.py` (find safety issues early)
8. `PHASE_3_5_BENCHMARK_SPEC.md`
9. Final gates & publish results

### Phase 4 Gate
✅ **Launch Phase 4 (torch/GPU) ONLY if**:
- Latency gates: p50 ≤ 50ms ✅
- Quality gates: accuracy >= 75% @ 512 tokens ✅
- Ablation gates: Full > any partial ✅
- Adversarial gates: key facts >= 90% retained ✅
- Determinism gates: signatures stable ✅

---

## Why This Matters

**By the time you write Phase 4 code**, you'll know:

1. **Exactly which components to accelerate** (ablations show contribution)
2. **What the latency ceiling is without GPU** (benchmark bottleneks identified)
3. **Whether GPU acceleration is even needed** (if already fast enough)
4. **What adversarial cases to handle** (known failure modes)
5. **Reproducible baselines for publication** (benchmark spec enables peer reproduction)

This is the difference between "shipping a feature" and "shipping research."

---

## Success Targets (For Phase 4 Gate)

| Metric | Target | Status |
|---|---|---|
| Latency p50 @ 10K units | ≤ 50ms | TBD |
| Latency p95 @ 10K units | ≤ 200ms | TBD |
| Accuracy @ 512 tokens | ≥ 75% | TBD |
| Token reduction vs baseline | ≥ 20% | TBD |
| Adversarial key facts retained | ≥ 90% | TBD |
| Determinism signature stability | 3/3 runs match | TBD |
| Ablation: Full > any partial | ✅ | TBD |

**All TBD → will be filled in by running benchmarks (Phase 3.5 execution).**

If all gates go green → Phase 4 is a justified optimization, not a guess.
If any gate fails → debug Phase 3 before moving on.

---

## References

- Phase 3 Implementation: [VOXSIGIL_VME_PHASE_3_COMPLETE.md](VOXSIGIL_VME_PHASE_3_COMPLETE.md)
- Architecture Plan: [VOXSIGIL_VME_ARCHITECTURE_PLAN.md](VOXSIGIL_VME_ARCHITECTURE_PLAN.md)
- Benchmark Spec: [PHASE_3_5_BENCHMARK_SPEC.md](PHASE_3_5_BENCHMARK_SPEC.md) (to be created)
