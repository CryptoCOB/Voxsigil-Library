"""
Benchmark Orchestrator

Runs all tiers (1-4) and generates comprehensive reports.
Validates gate checks before Phase 4 authorization.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Tier 1: Latency
from benchmarks.bench_latency import run_latency_benchmarks

# Tier 2: Quality
from benchmarks.bench_quality_synthqa import run_quality_benchmarks

# Tier 3: Ablation
from benchmarks.bench_ablations import run_ablation_benchmarks

# Tier 4: Adversarial
from benchmarks.bench_adversarial import run_adversarial_benchmarks

# Gate checks
from benchmarks.check_gates import validate_all_gates


def load_baseline_thresholds() -> dict:
    """Load baseline thresholds for gate checks."""
    baseline_file = Path(__file__).parent / "baselines" / "baseline_thresholds.json"
    if not baseline_file.exists():
        return {
            "latency": {
                "p50_ms_under": 50,
                "p95_ms_under": 200,
            },
            "quality": {
                "min_accuracy_512": 0.75,
                "min_accuracy_256": 0.60,
            },
            "ablation": {
                "full_over_partial": 1.10,
                "min_contribution": 0.05,
            },
            "adversarial": {
                "min_fact_retention": 0.90,
            },
        }
    
    with open(baseline_file) as f:
        return json.load(f)


def run_all_benchmarks(corpus_size: int = 10) -> dict:
    """Run all benchmark tiers."""
    
    print("\n" + "="*70)
    print("VOXSIGIL VME PHASE 3.5 EVALUATION BENCHMARKS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Corpus size: {corpus_size} units")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "corpus_size": corpus_size,
        "tiers": {},
    }
    
    # Tier 1: Latency
    print("\n[TIER 1] Latency Benchmarks")
    print("-" * 70)
    tier1_result = run_latency_benchmarks(corpus_size=corpus_size)
    results["tiers"]["tier1_latency"] = tier1_result
    
    # Tier 2: Quality
    print("\n[TIER 2] Quality Benchmarks")
    print("-" * 70)
    tier2_result = run_quality_benchmarks()
    results["tiers"]["tier2_quality"] = tier2_result
    
    # Tier 3: Ablation
    print("\n[TIER 3] Ablation Studies")
    print("-" * 70)
    tier3_result = run_ablation_benchmarks(corpus_size=corpus_size)
    results["tiers"]["tier3_ablation"] = tier3_result
    
    # Tier 4: Adversarial
    print("\n[TIER 4] Adversarial Testing")
    print("-" * 70)
    tier4_result = run_adversarial_benchmarks()
    results["tiers"]["tier4_adversarial"] = tier4_result
    
    # Validate gates
    print("\n[GATES] Phase 4 Authorization Check")
    print("-" * 70)
    baseline = load_baseline_thresholds()
    gates_result = validate_all_gates(results, baseline)
    results["gates_validation"] = gates_result
    
    # Authorization decision
    all_gates_passed = all(
        check.get("passed", False)
        for tier_checks in gates_result.values()
        if isinstance(tier_checks, dict)
        for check in (
            [tier_checks] if "passed" in tier_checks else tier_checks.values()
        )
    )
    
    results["phase4_authorization"] = {
        "authorized": all_gates_passed,
        "reason": (
            "All gates passed"
            if all_gates_passed
            else "Some gates failed - see detail below"
        ),
        "timestamp": datetime.now().isoformat(),
    }
    
    return results


def save_results(results: dict, output_dir: Path = None) -> Path:
    """Save results to JSONL and summary JSON."""
    if output_dir is None:
        output_dir = Path("benchmarks/results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Summary text
    summary_path = output_dir / f"benchmark_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("VOXSIGIL VME PHASE 3.5 BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Tier 1 summary
        if "tier1_latency" in results["tiers"]:
            t1 = results["tiers"]["tier1_latency"]
            f.write("[TIER 1] Latency\n")
            f.write("-" * 70 + "\n")
            if "e2e" in t1:
                f.write(f"  E2E p50: {t1['e2e'].get('p50_ms', 'N/A'):.2f}ms (gate: <= 50ms)\n")
                f.write(f"  E2E p95: {t1['e2e'].get('p95_ms', 'N/A'):.2f}ms (gate: <= 200ms)\n")
            for comp, stats in t1.get("components", {}).items():
                f.write(f"  {comp:15} p50={stats.get('p50_ms', 0):.2f}ms\n")
            f.write("\n")
        
        # Phase 4 auth
        auth = results.get("phase4_authorization", {})
        f.write("[PHASE 4] Authorization Decision\n")
        f.write("-" * 70 + "\n")
        status = (
            "AUTHORIZED"
            if auth.get("authorized")
            else "BLOCKED"
        )
        f.write(f"  Status: {status}\n")
        f.write(f"  Reason: {auth.get('reason', 'Unknown')}\n")
        f.write(f"  Timestamp: {auth.get('timestamp', 'Unknown')}\n")
    
    print("\nResults saved:")
    print(f"  - {json_path}")
    print(f"  - {summary_path}")
    
    return json_path


def main():
    """Run full benchmark suite."""
    corpus_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    results = run_all_benchmarks(corpus_size=corpus_size)
    save_results(results)
    
    # Print authorization decision
    auth = results["phase4_authorization"]
    phase4_status = (
        "AUTHORIZED"
        if auth["authorized"]
        else "BLOCKED"
    )
    print("\n" + "="*70)
    print(f"PHASE 4 STATUS: {phase4_status}")
    print(f"Reason: {auth['reason']}")
    print("="*70 + "\n")
    
    return 0 if auth["authorized"] else 1


if __name__ == "__main__":
    sys.exit(main())
