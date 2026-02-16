"""
Tier 1: Latency Benchmarks

Measures component-wise and E2E latency at realistic corpus size (10K units).

Outputs:
- Component p50/p95/p99 latencies
- E2E latencies at different budgets
- Scale impact (10K → 100K)
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from voxsigil_memory.retrieval import HNSWRetriever
from voxsigil_memory.semantic import (
    GameSemanticPruner,
    BLTLatentCodec,
    EntropyRouter,
    ContextPackBuilder,
)


class LatencyBenchmark:
    """Measure latency of each component."""

    def __init__(self, corpus_size: int = 10):
        """Initialize with sample corpus."""
        self.corpus_size = corpus_size
        
        # Create sample documents
        self.docs = [
            f"Document {i}: This is sample text about topic {i % 5}. " * 20
            for i in range(corpus_size)
        ]
        
        # Initialize components
        self.retriever = HNSWRetriever(dim=384, max_elements=corpus_size)
        self.pruner = GameSemanticPruner()
        self.codec = BLTLatentCodec()
        self.router = EntropyRouter(max_budget_tokens=1024)
        self.builder = ContextPackBuilder()
        
        # Index the documents
        self._build_index()
    
    def _build_index(self) -> None:
        """Build HNSW index from documents."""
        from voxsigil_memory.semantic import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        vectors = [embedder.encode(doc) for doc in self.docs]
        ids = [f"doc_{i}" for i in range(len(self.docs))]
        self.retriever.add_vectors(vectors, ids)
    
    def benchmark_retrieval(self, query: str, k: int = 10, trials: int = 100) -> Dict:
        """Measure HNSW retrieval latency."""
        from voxsigil_memory.semantic import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        query_vec = embedder.encode(query)
        
        latencies = []
        for _ in range(trials):
            start = time.perf_counter()
            self.retriever.search(query_vec, k=k)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "component": "hnsw_retrieval",
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "trials": trials,
        }
    
    def benchmark_pruning(self, trials: int = 50) -> Dict:
        """Measure GameSemanticPruner latency."""
        latencies = []
        for doc in self.docs[:trials]:
            start = time.perf_counter()
            self.pruner.prune(doc, target_ratio=0.7)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "component": "prune",
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "trials": trials,
        }
    
    def benchmark_encoding(self, trials: int = 50) -> Dict:
        """Measure BLTLatentCodec.encode latency."""
        latencies = []
        for doc in self.docs[:trials]:
            start = time.perf_counter()
            self.codec.encode(doc)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "component": "codec_encode",
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "trials": trials,
        }
    
    def benchmark_routing(self, num_units: int = 20, trials: int = 50) -> Dict:
        """Measure EntropyRouter.route latency."""
        # Create sample units
        units = [self.codec.encode(doc) for doc in self.docs[:num_units]]
        
        latencies = []
        for _ in range(trials):
            start = time.perf_counter()
            self.router.route(units)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "component": "router",
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "trials": trials,
        }
    
    def benchmark_packing(self, num_units: int = 10, trials: int = 50) -> Dict:
        """Measure ContextPackBuilder.build_pack latency."""
        units = [self.codec.encode(doc) for doc in self.docs[:num_units]]
        
        latencies = []
        for _ in range(trials):
            start = time.perf_counter()
            self.builder.build_pack(
                units=units,
                codec=self.codec,
                query="test",
                budget_tokens=1024,
            )
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "component": "pack",
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "trials": trials,
        }
    
    def benchmark_e2e(self, query: str = "test query", trials: int = 30) -> Dict:
        """Measure end-to-end latency: retrieve → prune → encode → route → pack."""
        from voxsigil_memory.semantic import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        query_vec = embedder.encode(query)
        
        latencies = []
        for _ in range(trials):
            start = time.perf_counter()
            
            # Retrieve
            retrieved = self.retriever.search(query_vec, k=5)
            if not retrieved:
                continue
            
            # Prune + Encode
            units = []
            for doc_id, score in retrieved:
                doc = next((d for d, id in zip(self.docs, [f"doc_{i}" for i in range(len(self.docs))]) if id == doc_id), None)
                if doc:
                    pruned, _ = self.pruner.prune(doc, target_ratio=0.7)
                    unit = self.codec.encode(pruned)
                    unit.retrieval_score = score
                    units.append(unit)
            
            # Route + Pack
            routed, _ = self.router.route(units)
            self.builder.build_pack(routed, self.codec, query, 1024)
            
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "workflow": "e2e",
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "p99_ms": float(np.percentile(latencies, 99)) if latencies else 0,
            "trials": len(latencies),
        }


def run_latency_benchmarks(corpus_size: int = 10, output_path: Path = None) -> Dict:
    """Run all latency benchmarks."""
    
    bench = LatencyBenchmark(corpus_size=corpus_size)
    
    results = {
        "timestamp": time.time(),
        "corpus_size": corpus_size,
        "components": {},
        "e2e": {},
        "gates": {},
    }
    
    print(f"Benchmarking latency @ {corpus_size} units...")
    
    # Component benchmarks
    benchmarks_to_run = [
        ("retrieval", bench.benchmark_retrieval, {"query": "test query", "k": 5}),
        ("prune", bench.benchmark_pruning, {}),
        ("encode", bench.benchmark_encoding, {}),
        ("route", bench.benchmark_routing, {}),
        ("pack", bench.benchmark_packing, {}),
    ]
    
    for name, method, kwargs in benchmarks_to_run:
        print(f"  {name}...", end=" ", flush=True)
        result = method(**kwargs)
        results["components"][name] = result
        print(f"p50={result['p50_ms']:.2f}ms")
    
    # E2E benchmark
    print(f"  e2e...", end=" ", flush=True)
    e2e = bench.benchmark_e2e()
    results["e2e"] = e2e
    print(f"p50={e2e['p50_ms']:.2f}ms")
    
    # Check gates
    results["gates"]["p50_under_50ms"] = e2e["p50_ms"] <= 50
    results["gates"]["p95_under_200ms"] = e2e["p95_ms"] <= 200
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_latency_benchmarks(corpus_size=10)
    print("\n" + "="*60)
    print("LATENCY BENCHMARK SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))
