"""
VoxSigil VME: Phase 3.5 Benchmarking Suite

Tier 1: Latency (component breakdown + scaling)
Tier 2: Quality (synthetic QA utility)
Tier 3: Ablations (component attribution)
Tier 4: Adversarial (safety & edge cases)
"""

__version__ = "0.1.0"

# CI gates will be loaded from baselines/baseline_thresholds.json
DEFAULT_GATES = {
    "latency_p50_ms_10k": 50,
    "latency_p95_ms_10k": 200,
    "accuracy_at_512_tokens": 0.75,
    "token_reduction_pct": 20,
    "adversarial_key_facts_retained_pct": 90,
}
