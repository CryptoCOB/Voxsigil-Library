"""
Tier 4: Adversarial Testing

Tests edge cases and safety:
- Fact retention on adversarial documents (contradictions, negations, buried facts)
- Key entity preservation
- De-duplication handling
- Circular reference handling
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from voxsigil_memory.semantic import (
    GameSemanticPruner,
    BLTLatentCodec,
    EntropyRouter,
    ContextPackBuilder,
)


class AdversarialBenchmark:
    """Test adversarial and edge cases."""

    def __init__(self, adversarial_dataset_path: Path = None):
        """Load adversarial dataset."""
        if adversarial_dataset_path is None:
            adversarial_dataset_path = (
                Path(__file__).parent.parent
                / "datasets"
                / "adversarial_pruning.jsonl"
            )
        
        self.adversarial_docs = self._load_adversarial_dataset(
            adversarial_dataset_path
        )
        
        # Initialize components
        self.pruner = GameSemanticPruner()
        self.codec = BLTLatentCodec()
        self.router = EntropyRouter(max_budget_tokens=1024)
        self.builder = ContextPackBuilder()
    
    def _load_adversarial_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load adversarial dataset from JSONL."""
        docs = []
        
        if not path.exists():
            # Return default adversarial cases
            return [
                {
                    "name": "contradiction",
                    "text": "Alice is tall. Alice is not tall.",
                    "key_fact": "Alice is tall",
                },
                {
                    "name": "negation",
                    "text": "The project is NOT cancelled. The team continues work.",
                    "key_fact": "The project is not cancelled",
                },
                {
                    "name": "buried_fact",
                    "text": (
                        "Lorem ipsum dolor sit amet. "
                        "The critical fact is hidden here. "
                        "More filler text follows."
                    ),
                    "key_fact": "critical fact",
                },
            ]
        
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip comments and empty lines
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        
        return docs
    
    def benchmark_fact_retention(self) -> Dict:
        """
        Measure if key facts are preserved after compression.
        
        Returns:
            Dict with retention statistics
        """
        retained = 0
        total = 0
        
        for doc in self.adversarial_docs:
            text = doc.get("text", "")
            key_fact = doc.get("key_fact", "")
            
            if not text or not key_fact:
                continue
            
            # Process through pipeline
            pruned, _ = self.pruner.prune(text, target_ratio=0.7)
            unit = self.codec.encode(pruned)
            
            # Check if key fact is in pruned text
            if key_fact.lower() in pruned.lower():
                retained += 1
            
            total += 1
        
        retention_rate = retained / total if total > 0 else 0.0
        
        return {
            "metric": "fact_retention",
            "retention_rate": retention_rate,
            "retained_facts": retained,
            "total_facts": total,
        }
    
    def benchmark_key_entity_preservation(self) -> Dict:
        """
        Measure preservation of named entities (simple approach).
        
        Returns:
            Dict with entity preservation stats
        """
        # Simple entity detection: capitalized words
        preserved = 0
        total = 0
        
        for doc in self.adversarial_docs:
            text = doc.get("text", "")
            
            # Extract "entities" (capitalized words at sentence start)
            entities = [
                word
                for word in text.split()
                if word and word[0].isupper()
            ]
            
            if not entities:
                continue
            
            pruned, _ = self.pruner.prune(text, target_ratio=0.7)
            
            for entity in entities:
                if entity in pruned:
                    preserved += 1
                total += 1
        
        preservation_rate = preserved / total if total > 0 else 0.0
        
        return {
            "metric": "entity_preservation",
            "preservation_rate": preservation_rate,
            "preserved_entities": preserved,
            "total_entities": total,
        }
    
    def benchmark_deduplication_resistance(self) -> Dict:
        """
        Test that duplicate content doesn't cause issues.
        """
        # Create document with heavy duplication
        dup_text = (
            "Important fact: system is operational. "
            "Important fact: system is operational. " * 5
        )
        
        original_len = len(dup_text)
        pruned, pruned_ratio = self.pruner.prune(dup_text, target_ratio=0.7)
        pruned_len = len(pruned)
        
        unit = self.codec.encode(pruned)
        compressed_len = len(unit.latent_encoding)
        
        return {
            "metric": "dedup_resistance",
            "original_len": original_len,
            "pruned_len": pruned_len,
            "pruned_ratio": pruned_ratio,
            "compressed_len": compressed_len,
            "overall_compression": compressed_len / original_len,
        }
    
    def benchmark_edge_cases(self) -> Dict:
        """
        Test edge cases: empty, very short, very long documents.
        """
        test_cases = {
            "empty": "",
            "single_word": "fact",
            "single_sentence": "This is a single sentence.",
            "very_long": "sentence. " * 1000,  # 8000+ chars
        }
        
        results = {}
        
        for case_name, text in test_cases.items():
            if not text:
                results[case_name] = {"error": "empty input"}
                continue
            
            try:
                pruned, _ = self.pruner.prune(text, target_ratio=0.7)
                unit = self.codec.encode(pruned)
                results[case_name] = {
                    "original_len": len(text),
                    "pruned_len": len(pruned),
                    "encoded_len": len(unit.latent_encoding),
                    "success": True,
                }
            except Exception as e:
                results[case_name] = {
                    "error": str(e),
                    "success": False,
                }
        
        return {
            "metric": "edge_cases",
            "cases": results,
        }


def run_adversarial_benchmarks(output_path: Path = None) -> Dict:
    """Run adversarial benchmarks."""
    
    bench = AdversarialBenchmark()
    
    results = {
        "timestamp": time.time(),
        "metric_group": "adversarial",
        "benchmarks": {},
    }
    
    print("\nRunning adversarial tests...")
    
    # Fact retention
    print("  fact_retention...", end=" ", flush=True)
    fact_ret = bench.benchmark_fact_retention()
    results["benchmarks"]["fact_retention"] = fact_ret
    print(f"retention={fact_ret.get('retention_rate', 0):.2%}")
    
    # Entity preservation
    print("  entity_preservation...", end=" ", flush=True)
    entity_pres = bench.benchmark_key_entity_preservation()
    results["benchmarks"]["entity_preservation"] = entity_pres
    print(f"preservation={entity_pres.get('preservation_rate', 0):.2%}")
    
    # Deduplication
    print("  dedup_resistance...", end=" ", flush=True)
    dedup = bench.benchmark_deduplication_resistance()
    results["benchmarks"]["dedup_resistance"] = dedup
    print(f"overall_compression={dedup.get('overall_compression', 0):.2%}")
    
    # Edge cases
    print("  edge_cases...", end=" ", flush=True)
    edge_results = bench.benchmark_edge_cases()
    results["benchmarks"]["edge_cases"] = edge_results
    success_count = sum(
        1
        for case in edge_results.get("cases", {}).values()
        if case.get("success", False)
    )
    print(f"{success_count} / {len(edge_results.get('cases', {}))} passed")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nAdversarial results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_adversarial_benchmarks()
    print("\n" + "="*60)
    print("ADVERSARIAL BENCHMARK SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))
