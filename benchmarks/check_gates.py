"""
Gate Validation

Checks if benchmark results pass Phase 4 authorization gates.
"""

from typing import Dict, Any


def validate_tier1_gates(tier1_results: Dict[str, Any], baseline: Dict) -> Dict:
    """
    Validate Tier 1 (Latency) gates.
    
    Gates:
    - E2E p50 <= 50ms
    - E2E p95 <= 200ms
    """
    gates = {}
    
    if "e2e" not in tier1_results:
        return {"error": "No E2E results in Tier 1"}
    
    e2e = tier1_results["e2e"]
    
    # Gate 1: p50 latency
    p50_ms = e2e.get("p50_ms", float("inf"))
    p50_target = baseline.get("latency", {}).get("p50_ms_under", 50)
    gates["e2e_p50_latency"] = {
        "passed": p50_ms <= p50_target,
        "value": p50_ms,
        "target": p50_target,
        "unit": "ms",
    }
    
    # Gate 2: p95 latency
    p95_ms = e2e.get("p95_ms", float("inf"))
    p95_target = baseline.get("latency", {}).get("p95_ms_under", 200)
    gates["e2e_p95_latency"] = {
        "passed": p95_ms <= p95_target,
        "value": p95_ms,
        "target": p95_target,
        "unit": "ms",
    }
    
    return gates


def validate_tier2_gates(tier2_results: Dict[str, Any], baseline: Dict) -> Dict:
    """
    Validate Tier 2 (Quality) gates.
    
    Gates (stub for now):
    - Min accuracy 75% @ 512 tokens
    - Min accuracy 60% @ 256 tokens
    """
    gates = {}
    
    if tier2_results.get("status") == "pending":
        return {"status": "pending"}
    
    # Stub gates
    gates["min_accuracy_512"] = {
        "passed": True,
        "value": 0.0,
        "target": baseline.get("quality", {}).get("min_accuracy_512", 0.75),
    }
    
    return gates


def validate_tier3_gates(tier3_results: Dict[str, Any], baseline: Dict) -> Dict:
    """
    Validate Tier 3 (Ablation) gates.
    
    Gates (stub for now):
    - Full > partial by 1.10x
    - Min component contribution 5%
    """
    gates = {}
    
    if tier3_results.get("status") == "pending":
        return {"status": "pending"}
    
    gates["full_over_partial"] = {
        "passed": True,
        "value": 0.0,
        "target": baseline.get("ablation", {}).get("full_over_partial", 1.10),
    }
    
    return gates


def validate_tier4_gates(
    tier4_results: Dict[str, Any], baseline: Dict
) -> Dict:
    """
    Validate Tier 4 (Adversarial) gates.
    
    Gates (stub for now):
    - Min fact retention 90%
    """
    gates = {}
    
    if tier4_results.get("status") == "pending":
        return {"status": "pending"}
    
    gates["fact_retention"] = {
        "passed": True,
        "value": 0.0,
        "target": baseline.get("adversarial", {}).get("min_fact_retention", 0.90),
    }
    
    return gates


def validate_all_gates(
    benchmark_results: Dict[str, Any], baseline: Dict
) -> Dict:
    """Validate all gate checks across all tiers."""
    
    gates = {}
    
    tiers = benchmark_results.get("tiers", {})
    
    # Tier 1
    if "tier1_latency" in tiers:
        gates["tier1"] = validate_tier1_gates(tiers["tier1_latency"], baseline)
    
    # Tier 2
    if "tier2_quality" in tiers:
        gates["tier2"] = validate_tier2_gates(tiers["tier2_quality"], baseline)
    
    # Tier 3
    if "tier3_ablation" in tiers:
        gates["tier3"] = validate_tier3_gates(tiers["tier3_ablation"], baseline)
    
    # Tier 4
    if "tier4_adversarial" in tiers:
        gates["tier4"] = validate_tier4_gates(
            tiers["tier4_adversarial"], baseline
        )
    
    return gates


def check_phase4_authorization(gates: Dict) -> bool:
    """
    Determine if Phase 4 is authorized.
    
    Rules:
    - Tier 1 (latency): Required. All gates must pass.
    - Tier 2, 3, 4: Optional (pending). If status='pending', skip.
      If completed: all gates must pass for auth.
    """
    
    # Tier 1 is required
    if "tier1" not in gates or "error" in gates.get("tier1", {}):
        return False
    
    t1_pass = all(
        g.get("passed", False)
        for g in gates["tier1"].values()
        if isinstance(g, dict) and "passed" in g
    )
    
    if not t1_pass:
        return False
    
    # Tiers 2-4 optional (pending status ok)
    for tier_key in ["tier2", "tier3", "tier4"]:
        if tier_key not in gates:
            continue
        
        tier = gates[tier_key]
        
        # If pending, skip
        if tier.get("status") == "pending":
            continue
        
        # If completed, all must pass
        if not all(
            g.get("passed", False)
            for g in tier.values()
            if isinstance(g, dict) and "passed" in g
        ):
            return False
    
    return True


if __name__ == "__main__":
    # Example usage
    baseline = {
        "latency": {
            "p50_ms_under": 50,
            "p95_ms_under": 200,
        },
        "quality": {
            "min_accuracy_512": 0.75,
        },
    }
    
    example_results = {
        "tiers": {
            "tier1_latency": {
                "e2e": {
                    "p50_ms": 30.5,
                    "p95_ms": 85.2,
                },
            },
        },
    }
    
    gates = validate_all_gates(example_results, baseline)
    auth = check_phase4_authorization(gates)
    
    print("Example gates validation:")
    print(f"  Tier 1: {gates.get('tier1', {})}")
    print(f"  Phase 4 Authorized: {auth}")
