"""
Tier 2: Quality Benchmark - Synthetic QA

Measures information retention and accuracy after pruning/encoding.
Uses synthetic QA pairs from datasets/synth_qa_pairs.jsonl

Quality metrics:
- Answer relevance (is answer retrievable from compressed context?)
- Token efficiency (how many tokens to achieve 75% accuracy?)
- Budget vs accuracy tradeoff
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from voxsigil_memory.semantic import (
    GameSemanticPruner,
    BLTLatentCodec,
    EntropyRouter,
    ContextPackBuilder,
)
from voxsigil_memory.retrieval import HNSWRetriever


class QualityBenchmark:
    """Measure quality of retrieval and compression."""

    def __init__(self, qa_dataset_path: Path = None, facts_dataset_path: Path = None):
        """Load QA dataset for quality testing."""
        if qa_dataset_path is None:
            qa_dataset_path = Path(__file__).parent.parent / "datasets" / "synth_qa_pairs.jsonl"
        
        if facts_dataset_path is None:
            facts_dataset_path = Path(__file__).parent.parent / "datasets" / "synth_facts_10k.jsonl"
        
        self.qa_pairs = self._load_qa_dataset(qa_dataset_path)
        self.facts = self._load_facts_dataset(facts_dataset_path)
        
        # Create documents from facts
        self.documents = [fact.get("fact", "") for fact in self.facts]
        
        # Initialize components
        self.pruner = GameSemanticPruner()
        self.codec = BLTLatentCodec()
        self.router = EntropyRouter(max_budget_tokens=2048)
        self.builder = ContextPackBuilder()
        self.retriever = None
    
    def _load_qa_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL QA dataset."""
        qa_pairs = []
        if not path.exists():
            # Return dummy data if file doesn't exist
            return [
                {
                    "query": "What is topic 0?",
                    "answer": "topic 0",
                    "documents": ["Document about topic 0"] * 3,
                    "relevant_doc_indices": [0, 1, 2],
                },
            ]
        
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip comments and empty lines
                try:
                    qa_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        
        return qa_pairs
    
    def _load_facts_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL facts dataset."""
        facts = []
        if not path.exists():
            return [
                {
                    "id": "fact_0",
                    "fact": "This is a sample fact.",
                    "category": "general",
                },
            ]
        
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip comments and empty lines
                try:
                    facts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        
        return facts
    
    def _create_retriever(self, documents: List[str]) -> HNSWRetriever:
        """Create and index retriever from documents."""
        from voxsigil_memory.semantic import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        retriever = HNSWRetriever(dim=384, max_elements=len(documents))
        
        vectors = [embedder.encode(doc) for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        retriever.add_vectors(vectors, ids)
        
        return retriever
    
    def benchmark_answer_presence(self, budget_tokens: int = 512) -> Dict:
        """
        Measure if answer keywords are preserved in compressed context.
        
        Returns:
            Dict with accuracy metric (0-1)
        """
        correct = 0
        total = 0
        
        for qa_pair in self.qa_pairs:
            query = qa_pair.get("query", "")
            answer_keywords = qa_pair.get("answer_keywords", [])
            
            if not query or not answer_keywords:
                continue
            
            # Use loaded documents from facts
            if not self.documents:
                continue
            
            # Compress documents
            units = []
            for doc in self.documents:
                if not doc:
                    continue
                pruned, _ = self.pruner.prune(doc, target_ratio=0.7)
                unit = self.codec.encode(pruned)
                units.append(unit)
            
            if not units:
                continue
            
            # Route with budget
            routed, _ = self.router.route(units)
            
            # Build context pack
            pack = self.builder.build_pack(
                routed, self.codec, query, budget_tokens
            )
            
            # Check if any answer keyword is in context pack
            pack_text = pack.get("context", "")
            answer_found = any(
                keyword.lower() in pack_text.lower()
                for keyword in answer_keywords
                if isinstance(keyword, str)
            )
            
            if answer_found:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "metric": "answer_presence",
            "accuracy": accuracy,
            "budget_tokens": budget_tokens,
            "tested_pairs": total,
        }
    
    def benchmark_token_efficiency(self) -> Dict:
        """
        Measure accuracy at different token budgets.
        
        Returns:
            Dict with accuracy at 256, 512, 1K, 2K tokens
        """
        budgets = [256, 512, 1024, 2048]
        results = {"metric": "token_efficiency"}
        
        for budget in budgets:
            accuracy = self.benchmark_answer_presence(budget_tokens=budget)
            results[f"budget_{budget}_accuracy"] = accuracy["accuracy"]
        
        return results
    
    def benchmark_compression_ratio(self) -> Dict:
        """
        Measure average compression ratio of processed documents.
        
        Returns:
            Dict with compression stats
        """
        ratios = []
        
        for doc in self.documents:
            if not doc:
                continue
            
            original_size = len(doc.encode("utf-8"))
            
            pruned, _ = self.pruner.prune(doc, target_ratio=0.7)
            unit = self.codec.encode(pruned)
            
            compressed_size = len(unit.latent_encoding)
            
            if original_size > 0:
                ratio = 1.0 - (compressed_size / original_size)
                ratios.append(ratio)
        
        if ratios:
            return {
                "metric": "compression_ratio",
                "mean_ratio": float(np.mean(ratios)),
                "median_ratio": float(np.median(ratios)),
                "std_ratio": float(np.std(ratios)),
                "samples": len(ratios),
            }
        
        return {
            "metric": "compression_ratio",
            "mean_ratio": 0.0,
            "samples": 0,
        }


def run_quality_benchmarks(output_path: Path = None) -> Dict:
    """Run all quality benchmarks."""
    
    bench = QualityBenchmark()
    
    results = {
        "timestamp": time.time(),
        "metric_group": "quality",
        "benchmarks": {},
    }
    
    print("\nBenchmarking quality metrics...")
    
    # Answer presence
    print("  answer_presence...", end=" ", flush=True)
    answer_presence = bench.benchmark_answer_presence()
    results["benchmarks"]["answer_presence"] = answer_presence
    print(f"accuracy={answer_presence['accuracy']:.2%}")
    
    # Token efficiency
    print("  token_efficiency...", end=" ", flush=True)
    token_eff = bench.benchmark_token_efficiency()
    results["benchmarks"]["token_efficiency"] = token_eff
    for budget_key in ["budget_256_accuracy", "budget_512_accuracy"]:
        if budget_key in token_eff:
            print(f"\n    {budget_key}={token_eff[budget_key]:.2%}", end="")
    print()
    
    # Compression ratio
    print("  compression_ratio...", end=" ", flush=True)
    comp_ratio = bench.benchmark_compression_ratio()
    results["benchmarks"]["compression_ratio"] = comp_ratio
    print(f"mean={comp_ratio.get('mean_ratio', 0):.2%}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nQuality results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_quality_benchmarks()
    print("\n" + "="*60)
    print("QUALITY BENCHMARK SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))
