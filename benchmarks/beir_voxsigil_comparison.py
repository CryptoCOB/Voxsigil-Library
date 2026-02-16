"""
BEIR Benchmark for VoxSigil Student Embedder

Compares VoxSigil 128D student embedder against MiniLM-L6 384D baseline
on 15 BEIR retrieval tasks.

Installation:
    pip install beir sentence-transformers

Usage:
    python benchmarks/beir_voxsigil_comparison.py --tasks all
    python benchmarks/beir_voxsigil_comparison.py --tasks nfcorpus,scifact
    
Output:
    - benchmarks/results/beir_minilm_baseline.json
    - benchmarks/results/beir_voxsigil_student.json
    - benchmarks/results/beir_comparison_report.md
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# BEIR imports
try:
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval import models
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("ERROR: BEIR not installed. Run: pip install beir")

# VoxSigil imports
try:
    from phase_4b1_student_embedder import StudentEmbedder
    VOXSIGIL_AVAILABLE = True
except ImportError:
    VOXSIGIL_AVAILABLE = False
    print("WARNING: VoxSigil student embedder not found")

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# BEIR TASK DEFINITIONS
# ============================================================================

BEIR_TASKS = {
    # Small datasets (fast evaluation)
    "nfcorpus": "NFCorpus (Medical)",
    "scifact": "SciFact (Scientific claims)",
    "arguana": "ArguAna (Argument retrieval)",
    "scidocs": "SCIDOCS (Citation recommendation)",
    
    # Medium datasets
    "fiqa": "FiQA (Financial QA)",
    "trec-covid": "TREC-COVID (COVID-19 research)",
    "touche-2020": "Touche-2020 (Argument retrieval)",
    "quora": "Quora (Duplicate questions)",
    
    # Large datasets (slower evaluation)
    "nq": "Natural Questions",
    "hotpotqa": "HotpotQA (Multi-hop)",
    "msmarco": "MS MARCO (Passage ranking)",
    "fever": "FEVER (Fact verification)",
    "climate-fever": "Climate-FEVER",
    "dbpedia-entity": "DBpedia Entity",
    "webis-touche2020": "Webis-Touche-2020",
}

FAST_TASKS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]
ALL_TASKS = list(BEIR_TASKS.keys())


# ============================================================================
# VOXSIGIL EMBEDDER WRAPPER (Adapts to BEIR API)
# ============================================================================

class VoxSigilBEIRWrapper:
    """Wrap VoxSigil student embedder to conform to BEIR API."""
    
    def __init__(self, model_path: str = "phase4b_outputs/student_embedder_128d.pkl"):
        """Initialize VoxSigil embedder."""
        if not VOXSIGIL_AVAILABLE:
            raise ImportError("VoxSigil student embedder not available")
        
        logger.info(f"Loading VoxSigil student embedder from {model_path}")
        self.embedder = StudentEmbedder.load(model_path)
        self.dimension = 128
    
    def encode_queries(
        self, 
        queries: List[str], 
        batch_size: int = 16, 
        **kwargs
    ) -> np.ndarray:
        """Encode queries to embeddings."""
        embeddings = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_emb = [self.embedder.embed(q) for q in batch]
            embeddings.extend(batch_emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_corpus(
        self, 
        corpus: List[Dict[str, str]], 
        batch_size: int = 8,
        **kwargs
    ) -> np.ndarray:
        """Encode corpus documents to embeddings."""
        # Combine title + text
        texts = [
            (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc in corpus
        ]
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = [self.embedder.embed(t) for t in batch]
            embeddings.extend(batch_emb)
            
            if i % 1000 == 0:
                logger.info(f"Encoded {i}/{len(texts)} documents")
        
        return np.array(embeddings, dtype=np.float32)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results for one BEIR task."""
    task_name: str
    model_name: str
    ndcg_at_10: float
    map_at_10: float
    recall_at_10: float
    precision_at_10: float
    encoding_time_seconds: float
    retrieval_time_seconds: float
    total_queries: int
    total_documents: int
    model_params: int
    embedding_dim: int


class BEIRBenchmarkRunner:
    """Run BEIR benchmark for multiple models."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def download_dataset(self, task_name: str, data_path: str = "datasets/beir"):
        """Download BEIR dataset if not exists."""
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        dataset_path = data_path / task_name
        if not dataset_path.exists():
            logger.info(f"Downloading {task_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{task_name}.zip"
            dataset_path = util.download_and_unzip(url, str(data_path))
        
        return str(dataset_path)
    
    def evaluate_model(
        self,
        task_name: str,
        model,
        model_name: str,
        model_params: int,
        embedding_dim: int
    ) -> BenchmarkResult:
        """Evaluate one model on one task."""
        logger.info(f"Evaluating {model_name} on {task_name}")
        
        # Load dataset
        dataset_path = self.download_dataset(task_name)
        corpus, queries, qrels = GenericDataLoader(
            data_folder=dataset_path
        ).load(split="test")
        
        # Encode corpus and queries
        logger.info(f"Encoding {len(corpus)} documents...")
        encode_start = time.time()
        
        retriever = EvaluateRetrieval(
            model,
            score_function="dot"  # Use dot product for normalized embeddings
        )
        
        # Retrieve
        logger.info(f"Retrieving for {len(queries)} queries...")
        retrieval_start = time.time()
        results = retriever.retrieve(corpus, queries)
        retrieval_time = time.time() - retrieval_start
        
        # Evaluate
        logger.info("Computing metrics...")
        ndcg, _map, recall, precision = retriever.evaluate(
            qrels, results, retriever.k_values
        )
        
        encoding_time = retrieval_start - encode_start
        
        return BenchmarkResult(
            task_name=task_name,
            model_name=model_name,
            ndcg_at_10=ndcg["NDCG@10"],
            map_at_10=_map["MAP@10"],
            recall_at_10=recall["Recall@10"],
            precision_at_10=precision["P@10"],
            encoding_time_seconds=round(encoding_time, 2),
            retrieval_time_seconds=round(retrieval_time, 2),
            total_queries=len(queries),
            total_documents=len(corpus),
            model_params=model_params,
            embedding_dim=embedding_dim
        )
    
    def run_benchmark_suite(
        self,
        tasks: List[str],
        run_baseline: bool = True,
        run_voxsigil: bool = True
    ):
        """Run benchmark suite on multiple tasks."""
        
        # Run MiniLM baseline
        if run_baseline:
            logger.info("="*80)
            logger.info("BASELINE: MiniLM-L6-v2")
            logger.info("="*80)
            
            baseline_model = models.SentenceBERT("sentence-transformers/all-MiniLM-L6-v2")
            baseline_wrapper = DRES(baseline_model, batch_size=16)
            
            for task in tasks:
                try:
                    result = self.evaluate_model(
                        task_name=task,
                        model=baseline_wrapper,
                        model_name="MiniLM-L6-v2",
                        model_params=22_000_000,  # 22M parameters
                        embedding_dim=384
                    )
                    self.results.append(result)
                    logger.info(f"  NDCG@10: {result.ndcg_at_10:.4f}")
                except Exception as e:
                    logger.error(f"Failed on {task}: {e}")
        
        # Run VoxSigil student
        if run_voxsigil and VOXSIGIL_AVAILABLE:
            logger.info("="*80)
            logger.info("VOXSIGIL: Student Embedder 128D")
            logger.info("="*80)
            
            voxsigil_model = VoxSigilBEIRWrapper()
            voxsigil_wrapper = DRES(voxsigil_model, batch_size=8)
            
            for task in tasks:
                try:
                    result = self.evaluate_model(
                        task_name=task,
                        model=voxsigil_wrapper,
                        model_name="VoxSigil-Student-128D",
                        model_params=3_700_000,  # 3.7M parameters
                        embedding_dim=128
                    )
                    self.results.append(result)
                    logger.info(f"  NDCG@10: {result.ndcg_at_10:.4f}")
                except Exception as e:
                    logger.error(f"Failed on {task}: {e}")
    
    def save_results(self):
        """Save results to JSON."""
        output_file = self.output_dir / "beir_comparison_results.json"
        
        results_dict = {
            "results": [asdict(r) for r in self.results],
            "summary": self.compute_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save markdown report
        self.generate_markdown_report()
    
    def compute_summary(self) -> Dict:
        """Compute summary statistics."""
        baseline_results = [r for r in self.results if "MiniLM" in r.model_name]
        voxsigil_results = [r for r in self.results if "VoxSigil" in r.model_name]
        
        def avg_metric(results, metric):
            if not results:
                return 0.0
            return sum(getattr(r, metric) for r in results) / len(results)
        
        return {
            "baseline": {
                "model": "MiniLM-L6-v2",
                "params": "22M",
                "dim": 384,
                "avg_ndcg_at_10": round(avg_metric(baseline_results, "ndcg_at_10"), 4),
                "avg_map_at_10": round(avg_metric(baseline_results, "map_at_10"), 4),
                "avg_recall_at_10": round(avg_metric(baseline_results, "recall_at_10"), 4),
            },
            "voxsigil": {
                "model": "VoxSigil-Student-128D",
                "params": "3.7M",
                "dim": 128,
                "avg_ndcg_at_10": round(avg_metric(voxsigil_results, "ndcg_at_10"), 4),
                "avg_map_at_10": round(avg_metric(voxsigil_results, "map_at_10"), 4),
                "avg_recall_at_10": round(avg_metric(voxsigil_results, "recall_at_10"), 4),
            },
            "compression": {
                "params_ratio": "6.0x smaller",
                "dim_ratio": "3.0x smaller",
                "accuracy_retention": "TBD"
            }
        }
    
    def generate_markdown_report(self):
        """Generate markdown comparison report."""
        output_file = self.output_dir / "beir_comparison_report.md"
        
        baseline_results = [r for r in self.results if "MiniLM" in r.model_name]
        voxsigil_results = [r for r in self.results if "VoxSigil" in r.model_name]
        
        with open(output_file, 'w') as f:
            f.write("# BEIR Benchmark: VoxSigil vs MiniLM\n\n")
            f.write("## Summary\n\n")
            f.write("| Model | Params | Dim | Avg NDCG@10 | Avg MAP@10 | Avg Recall@10 |\n")
            f.write("|-------|--------|-----|-------------|------------|---------------|\n")
            
            summary = self.compute_summary()
            f.write(f"| {summary['baseline']['model']} | {summary['baseline']['params']} | {summary['baseline']['dim']} | ")
            f.write(f"{summary['baseline']['avg_ndcg_at_10']:.4f} | {summary['baseline']['avg_map_at_10']:.4f} | ")
            f.write(f"{summary['baseline']['avg_recall_at_10']:.4f} |\n")
            
            f.write(f"| {summary['voxsigil']['model']} | {summary['voxsigil']['params']} | {summary['voxsigil']['dim']} | ")
            f.write(f"{summary['voxsigil']['avg_ndcg_at_10']:.4f} | {summary['voxsigil']['avg_map_at_10']:.4f} | ")
            f.write(f"{summary['voxsigil']['avg_recall_at_10']:.4f} |\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Task | Model | NDCG@10 | MAP@10 | Recall@10 | Encoding Time (s) |\n")
            f.write("|------|-------|---------|--------|-----------|-------------------|\n")
            
            # Group by task
            tasks = sorted(set(r.task_name for r in self.results))
            for task in tasks:
                task_results = [r for r in self.results if r.task_name == task]
                for r in task_results:
                    f.write(f"| {r.task_name} | {r.model_name} | {r.ndcg_at_10:.4f} | ")
                    f.write(f"{r.map_at_10:.4f} | {r.recall_at_10:.4f} | {r.encoding_time_seconds} |\n")
        
        logger.info(f"Report saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark for VoxSigil")
    parser.add_argument(
        "--tasks",
        type=str,
        default="fast",
        choices=["fast", "all"] + ALL_TASKS,
        help="Tasks to run (fast=5 quick tasks, all=15 tasks, or specific task)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip MiniLM baseline (only run VoxSigil)"
    )
    parser.add_argument(
        "--skip-voxsigil",
        action="store_true",
        help="Skip VoxSigil (only run baseline)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not BEIR_AVAILABLE:
        logger.error("BEIR not installed. Run: pip install beir")
        sys.exit(1)
    
    # Determine tasks
    if args.tasks == "fast":
        tasks = FAST_TASKS
    elif args.tasks == "all":
        tasks = ALL_TASKS
    else:
        tasks = [args.tasks]
    
    logger.info(f"Running BEIR benchmark on {len(tasks)} tasks: {tasks}")
    
    # Run benchmark
    runner = BEIRBenchmarkRunner(output_dir=args.output_dir)
    runner.run_benchmark_suite(
        tasks=tasks,
        run_baseline=not args.skip_baseline,
        run_voxsigil=not args.skip_voxsigil
    )
    
    # Save results
    runner.save_results()
    
    logger.info("="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
