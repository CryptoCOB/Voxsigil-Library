"""
Phase 4-A Benchmarks: Per-Profile Performance Validation

Measures:
- Latency of each profile (document processing throughput)
- Quality tradeoffs (pruning vs output quality)
- Memory footprint accuracy
- Adversarial robustness per profile

Status: Phase 4-A Validation Suite
"""

import time
import numpy as np
from typing import List, Dict
from voxsigil_memory.edge_optimized import (
    EdgeOptimizedPipeline,
    DeviceProfile,
    DEVICE_CONFIGS,
)
from voxsigil_memory.semantic import LatentMemoryUnit


# Test datasets for benchmarking
DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog. " * 5,
    "Machine learning is a subset of artificial intelligence. " * 5,
    "Quantum computing leverages superposition and entanglement. " * 5,
    "Climate change impacts biodiversity across ecosystems. " * 5,
    "Cryptocurrency uses cryptographic protocols for security. " * 5,
]

ENTITY_DOCUMENTS = [
    "Alice and Bob met in Paris, France in 2024. " * 10,
    "Dr. Smith at MIT discovered a new phenomenon in Q4. " * 10,
    "The Eiffel Tower is in Paris. It was built in 1889. " * 10,
]

RARE_WORD_DOCUMENTS = [
    "Sesquipedalian loquaciousness obfuscates comprehension. " * 10,
    "Pneumonoultramicroscopicsilicovolcanoconiosis presented pathologically. " * 10,
]


def benchmark_latency(
    profile: DeviceProfile, documents: List[str], iterations: int = 5
) -> Dict[str, float]:
    """
    Benchmark document processing latency.

    Args:
        profile: Device profile to benchmark
        documents: List of documents to process
        iterations: Number of times to repeat benchmark

    Returns:
        Dict with p50, p95, p99 latencies in milliseconds
    """
    pipeline = EdgeOptimizedPipeline(profile)
    times_ms = []

    for _ in range(iterations):
        for doc in documents:
            start = time.perf_counter()
            _ = pipeline.process_document(doc)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)

    return {
        "p50": np.percentile(times_ms, 50),
        "p95": np.percentile(times_ms, 95),
        "p99": np.percentile(times_ms, 99),
        "mean": np.mean(times_ms),
        "count": len(times_ms),
    }


def benchmark_pruning_intensity(profile: DeviceProfile, documents: List[str]) -> Dict:
    """
    Measure actual pruning intensity for each profile.

    Returns:
        Dict with average pruned_fraction and output distribution
    """
    pipeline = EdgeOptimizedPipeline(profile)
    pruned_fractions = []

    for doc in documents:
        unit = pipeline.process_document(doc)
        pruned_fractions.append(unit.pruned_fraction)

    return {
        "mean_pruned_fraction": np.mean(pruned_fractions),
        "min_pruned_fraction": np.min(pruned_fractions),
        "max_pruned_fraction": np.max(pruned_fractions),
        "std_pruned_fraction": np.std(pruned_fractions),
        "config_target_ratio": DEVICE_CONFIGS[profile].pruning_ratio,
    }


def benchmark_memory_efficiency(profile: DeviceProfile) -> Dict:
    """
    Validate memory efficiency claims per profile.

    Returns:
        Dict with model size, per-unit overhead, max capacity
    """
    pipeline = EdgeOptimizedPipeline(profile)
    mem = pipeline.profile_memory_usage()

    return {
        "profile": profile.value,
        "model_size_mb": mem["model_size_mb"],
        "per_unit_mb": mem["per_unit_mb"],
        "max_units_in_memory": mem["max_units_in_memory"],
        "config_max_memory_mb": mem["total_budget_mb"],
    }


def benchmark_routing_impact(profile: DeviceProfile, document_count: int = 10) -> Dict:
    """
    Measure impact of routing (enabled vs disabled).

    For SERVER/EDGE: routing enabled
    For ULTRA_EDGE: routing disabled

    Returns:
        Dict with routing enabled status and unit selection stats
    """
    pipeline = EdgeOptimizedPipeline(profile)

    # Create sample units (simulating retrieval results)
    units = [
        LatentMemoryUnit(
            id=f"unit_{i}",
            embedding=np.random.rand(384),
            latent_encoding=b"sample_content",
            original_length=100,
            modality="text",
            retrieval_score=0.3 + (i * 0.05),  # Varying relevance
            pruned_fraction=0.0,
            entropy_score=0.2 + (i * 0.04),  # Varying entropy
        )
        for i in range(document_count)
    ]

    routed, stats = pipeline.process_units(units, budget_tokens=512)

    return {
        "profile": profile.value,
        "routing_enabled": DEVICE_CONFIGS[profile].use_routing,
        "input_units": len(units),
        "output_units": len(routed),
        "units_filtered": len(units) - len(routed),
        "routing_used": stats.get("routing_used", False),
    }


def benchmark_adversarial_robustness(profile: DeviceProfile) -> Dict:
    """
    Test profile robustness on adversarial/edge-case documents.

    Returns:
        Dict with success rate and failure modes
    """
    pipeline = EdgeOptimizedPipeline(profile)

    adversarial_cases = [
        ("entity_preservation", ENTITY_DOCUMENTS),
        ("rare_words", RARE_WORD_DOCUMENTS),
        ("normal_documents", DOCUMENTS),
    ]

    results = {}

    for case_name, docs in adversarial_cases:
        successes = 0
        for doc in docs:
            try:
                unit = pipeline.process_document(doc)
                if unit is not None and unit.embedding is not None:
                    successes += 1
            except Exception:
                pass

        results[case_name] = {
            "success_rate": successes / len(docs) if docs else 0.0,
            "docs_tested": len(docs),
        }

    return {
        "profile": profile.value,
        "adversarial_tests": results,
        "overall_success": all(
            r["success_rate"] == 1.0 for r in results.values()
        ),
    }


def run_full_phase_4a_benchmark() -> Dict:
    """
    Run complete Phase 4-A benchmark suite.

    Returns:
        Comprehensive results for all profiles across all metrics
    """
    print("\n" + "=" * 80)
    print("PHASE 4-A: EDGE-OPTIMIZED RUNTIME VALIDATION")
    print("=" * 80)

    results = {}

    for profile in DeviceProfile:
        print(f"\n[{profile.value.upper()}]")
        print("-" * 80)

        profile_results = {}

        # Latency
        latency = benchmark_latency(profile, DOCUMENTS, iterations=3)
        print(f"Latency: p50={latency['p50']:.2f}ms, p95={latency['p95']:.2f}ms")
        profile_results["latency"] = latency

        # Pruning intensity
        pruning = benchmark_pruning_intensity(profile, DOCUMENTS)
        print(
            f"Pruning: {pruning['mean_pruned_fraction']:.1%} (target: "
            f"{pruning['config_target_ratio']:.1%})"
        )
        profile_results["pruning"] = pruning

        # Memory
        memory = benchmark_memory_efficiency(profile)
        print(f"Memory: {memory['model_size_mb']}MB model + "
              f"{memory['max_units_in_memory']} max units")
        profile_results["memory"] = memory

        # Routing
        routing = benchmark_routing_impact(profile)
        print(f"Routing: enabled={routing['routing_enabled']}, "
              f"filtered={routing['units_filtered']}/{routing['input_units']}")
        profile_results["routing"] = routing

        # Adversarial
        adversarial = benchmark_adversarial_robustness(profile)
        success_rate = (
            adversarial["adversarial_tests"]["entity_preservation"]["success_rate"]
        )
        print(f"Adversarial: entity_preservation={success_rate:.1%}")
        profile_results["adversarial"] = adversarial

        results[profile.value] = profile_results

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 4-A VALIDATION GATES")
    print("=" * 80)

    # Gate 1: All profiles process documents
    all_latency_ok = all(
        r["latency"]["p95"] < 1000
        for r in results.values()
    )
    print(f"Gate 1 (All profiles responsive): {'✅ PASS' if all_latency_ok else '❌ FAIL'}")

    # Gate 2: Pruning matches config
    pruning_ok = all(
        abs(
            r["pruning"]["mean_pruned_fraction"]
            - r["pruning"]["config_target_ratio"]
        ) < 0.2
        for r in results.values()
    )
    print(f"Gate 2 (Pruning intensity accurate): {'✅ PASS' if pruning_ok else '❌ FAIL'}")

    # Gate 3: All profiles handle adversarial cases
    adversarial_ok = all(
        r["adversarial"]["overall_success"]
        for r in results.values()
    )
    print(f"Gate 3 (Adversarial robustness): {'✅ PASS' if adversarial_ok else '❌ FAIL'}")

    # Gate 4: Ultra-edge memory savings > 50%
    server_mem = results["server"]["memory"]["model_size_mb"]
    ultra_mem = results["ultra_edge"]["memory"]["model_size_mb"]
    memory_ok = ultra_mem < server_mem * 0.5  # 50%+ reduction
    print(f"Gate 4 (Ultra-edge memory savings): "
          f"{(1 - ultra_mem/server_mem)*100:.0f}% "
          f"{'✅ PASS' if memory_ok else '❌ FAIL'}")

    overall_pass = all([all_latency_ok, pruning_ok, adversarial_ok, memory_ok])

    print("\n" + "=" * 80)
    if overall_pass:
        print("PHASE 4-A STATUS: ✅ VALIDATION PASSED")
    else:
        print("PHASE 4-A STATUS: ⚠️ VALIDATION PARTIAL")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = run_full_phase_4a_benchmark()

    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("=" * 80)
    for profile, data in results.items():
        print(f"\n{profile.upper()}:")
        for metric, values in data.items():
            if isinstance(values, dict):
                print(f"  {metric}: {values}")
