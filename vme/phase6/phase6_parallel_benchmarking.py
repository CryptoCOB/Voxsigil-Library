"""
Phase 6: Parallel Multi-Model Benchmarking Orchestrator
Tests 8+ Ollama models simultaneously for VME robustness
Generates comparative report for investor proof-of-concept
"""

import json
import time
import requests
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Priority models for Phase 6 (diverse architectures)
PRIORITY_MODELS = [
    "llama3.2:latest",      # Primary reference (tested in Phase 5)
    "mistral:latest",       # Alternate base model
    "phi3:mini",            # Lightweight/efficient tier
    "deepseek-coder:6.7b",  # Specialized (coding)
    "qwen2:7b",             # Alternative architecture
    "neural-chat:7b",       # Conversational optimization
    "openchat:latest",      # Community model
    "starling-lm:7b-alpha", # Alignment-focused
]

OLLAMA_BASE_URL = "http://localhost:11434"

# BLT scenario prompts (same as Phase 5)
BLT_PROMPTS = {
    "analytical_engineer": """Generate a behavioral sigil for an analytical engineer with:
- Strong pattern recognition and problem-solving
- Data-driven decision making
- Low-risk, methodical approach
- Continuous learning mindset

Create specific behavioral metrics (0-1 scales) for: focus, adaptability, collaboration, innovation, communication.""",
    
    "creative_designer": """Generate a behavioral sigil for a creative designer with:
- High innovation and intuition
- Comfort with ambiguity
- Collaborative and expressive
- Vision-oriented approach

Create specific behavioral metrics (0-1 scales) for: creativity, execution, teamwork, attention-to-detail, responsiveness.""",
    
    "strategic_leader": """Generate a behavioral sigil for a strategic leader with:
- Systems thinking and long-term vision
- Decision-making under uncertainty  
- People development focus
- Cross-functional influence

Create specific behavioral metrics (0-1 scales) for: vision, decisiveness, empathy, delegation, resilience.""",
}

@dataclass
class ModelResult:
    """Aggregated result for a single model"""
    model: str
    timestamp: str
    total_prompts_tested: int
    successful_responses: int
    avg_blt_compatibility: float
    avg_richness: float
    avg_tokens_per_second: float
    overall_score: float
    total_tokens_generated: int
    total_time_ms: float
    per_prompt_results: List[Dict[str, Any]]
    status: str  # "success", "partial", "failed"
    error_message: str = None

def call_ollama_api(model: str, prompt: str, timeout: int = 180) -> Dict[str, Any]:
    """Call Ollama API with timeout"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        elapsed_ms = (time.time() - start_time) * 1000
        
        eval_count = data.get("eval_count", 0)
        tokens_per_sec = eval_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        return {
            "success": True,
            "response": data.get("response", ""),
            "tokens": eval_count,
            "time_ms": elapsed_ms,
            "tokens_per_sec": tokens_per_sec
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "tokens": 0,
            "time_ms": elapsed_ms,
            "tokens_per_sec": 0
        }

def score_blt_compatibility(response: str) -> float:
    """Score BLT compatibility"""
    if not response or len(response) < 150:
        return 0.2
    
    score = 0.0
    blt_patterns = {
        "metrics": ["metric", "measure", "score", "value"],
        "ranges": ["0-1", "0.0-1.0", "range", "scale"],
        "dimensions": ["dimension", "attribute", "behavioral", "trait"],
        "numerical": ["0.", "1.", "0.5", "0.8"]
    }
    
    found_patterns = 0
    for pattern_group, keywords in blt_patterns.items():
        for kw in keywords:
            if kw.lower() in response.lower():
                found_patterns += 1
                break
    
    score = min(found_patterns / len(blt_patterns), 1.0)
    
    if any(c.isdigit() for c in response):
        score = min(score * 1.2, 1.0)
    
    return score

def score_richness(response: str) -> float:
    """Score behavioral richness"""
    if not response:
        return 0.0
    
    behavior_terms = [
        "motivation", "pattern", "response", "context", 
        "trajectory", "adaptation", "trigger", "outcome",
        "communication", "collaboration", "learning", "decision"
    ]
    
    term_count = sum(1 for term in behavior_terms if term.lower() in response.lower())
    richness = min(term_count / 8, 1.0)
    length_factor = min(len(response) / 1000, 1.0) * 0.3
    richness = min(richness + length_factor, 1.0)
    
    return richness

def benchmark_single_model(model: str, return_dict: Dict) -> None:
    """Benchmark a single model (worker function for parallel execution)"""
    logger.info(f"[WORKER] Starting: {model}")
    
    per_prompt_results = []
    total_tokens = 0
    total_time = 0
    valid_responses = 0
    error_msg = None
    
    try:
        for prompt_type, prompt in BLT_PROMPTS.items():
            api_result = call_ollama_api(model, prompt)
            
            if not api_result["success"]:
                logger.warning(f"[{model}] {prompt_type}: {api_result['error']}")
                continue
            
            response = api_result["response"]
            tokens = api_result["tokens"]
            time_ms = api_result["time_ms"]
            tps = api_result["tokens_per_sec"]
            
            blt_score = score_blt_compatibility(response)
            richness = score_richness(response)
            
            is_valid = blt_score >= 0.5 and richness >= 0.4
            if is_valid:
                valid_responses += 1
            
            per_prompt_results.append({
                "prompt_type": prompt_type,
                "blt_score": blt_score,
                "richness": richness,
                "tokens_per_second": tps,
                "tokens_generated": tokens,
                "time_ms": time_ms,
                "is_valid": is_valid,
                "response_preview": response[:150] + "..." if len(response) > 150 else response
            })
            
            total_tokens += tokens
            total_time += time_ms
        
        # Aggregate scores
        if per_prompt_results:
            avg_blt = sum(r["blt_score"] for r in per_prompt_results) / len(per_prompt_results)
            avg_richness = sum(r["richness"] for r in per_prompt_results) / len(per_prompt_results)
            avg_tps = sum(r["tokens_per_second"] for r in per_prompt_results) / len(per_prompt_results)
            overall = (avg_blt * 0.4) + (avg_richness * 0.4) + (min(avg_tps / 50, 1.0) * 0.2)
            status = "success"
        else:
            avg_blt = avg_richness = overall = 0
            avg_tps = 0
            status = "failed"
            error_msg = "No valid responses"
        
        result = ModelResult(
            model=model,
            timestamp=datetime.now().isoformat(),
            total_prompts_tested=len(BLT_PROMPTS),
            successful_responses=valid_responses,
            avg_blt_compatibility=avg_blt,
            avg_richness=avg_richness,
            avg_tokens_per_second=avg_tps,
            overall_score=overall,
            total_tokens_generated=total_tokens,
            total_time_ms=total_time,
            per_prompt_results=per_prompt_results,
            status=status,
            error_message=error_msg
        )
        
        return_dict[model] = asdict(result)
        logger.info(f"[SUCCESS] {model}: score={overall:.3f}, blt={avg_blt:.2f}, richness={avg_richness:.2f}")
    
    except Exception as e:
        logger.error(f"[FATAL] {model}: {str(e)}")
        return_dict[model] = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error_message": str(e),
            "overall_score": 0,
            "per_prompt_results": [],
            "total_prompts_tested": 0,
            "successful_responses": 0
        }

def run_parallel_benchmark(models: List[str], num_workers: int = None) -> Dict[str, Any]:
    """Run benchmark on multiple models in parallel"""
    
    if num_workers is None:
        num_workers = min(len(models), mp.cpu_count() - 1)
    
    logger.info(f"Phase 6: Parallel Benchmarking Starting")
    logger.info(f"Models: {len(models)} | Workers: {num_workers} | Prompts per model: {len(BLT_PROMPTS)}")
    
    start_time = time.time()
    
    with mp.Manager() as manager:
        return_dict = manager.dict()
        
        # Create and start worker processes
        processes = []
        for model in models:
            p = mp.Process(target=benchmark_single_model, args=(model, return_dict))
            p.start()
            processes.append((model, p))
            logger.info(f"Spawned worker for: {model}")
        
        # Wait for all processes to complete
        for model, p in processes:
            p.join()
            logger.info(f"Worker completed: {model}")
        
        # Convert dict to regular dict
        all_results = dict(return_dict)
    
    elapsed = time.time() - start_time
    logger.info(f"Parallel benchmark completed in {elapsed:.1f} seconds")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 6: Multi-Model Orchestration",
        "models_tested": len(models),
        "prompts_per_model": len(BLT_PROMPTS),
        "parallel_workers": num_workers,
        "total_execution_time_seconds": elapsed,
        "results": all_results
    }

def generate_comparative_report(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparative rankings and insights"""
    
    all_results = benchmark_results["results"]
    
    # Extract models with valid scores
    valid_models = [
        (name, data) for name, data in all_results.items()
        if data.get("status") == "success" and data.get("overall_score", 0) > 0
    ]
    
    # Sort by overall score
    ranked = sorted(valid_models, key=lambda x: x[1]["overall_score"], reverse=True)
    
    # Generate rankings
    rankings = []
    for rank, (model_name, data) in enumerate(ranked, 1):
        rankings.append({
            "rank": rank,
            "model": model_name,
            "overall_score": round(data["overall_score"], 3),
            "blt_compatibility": round(data["avg_blt_compatibility"], 3),
            "richness": round(data["avg_richness"], 3),
            "tokens_per_second": round(data["avg_tokens_per_second"], 1),
            "valid_responses_pct": round((data["successful_responses"] / data["total_prompts_tested"] * 100) if data["total_prompts_tested"] > 0 else 0, 1),
            "status": data.get("status", "unknown")
        })
    
    # Statistical summary
    scores = [r["overall_score"] for r in rankings]
    blt_scores = [r["blt_compatibility"] for r in rankings]
    richness_scores = [r["richness"] for r in rankings]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 6: Multi-Model Comparative Analysis",
        "total_models_tested": benchmark_results["models_tested"],
        "successful_models": len(rankings),
        "total_execution_time_seconds": benchmark_results["total_execution_time_seconds"],
        "rankings": rankings,
        "statistics": {
            "overall_score": {
                "mean": round(sum(scores) / len(scores), 3) if scores else 0,
                "min": round(min(scores), 3) if scores else 0,
                "max": round(max(scores), 3) if scores else 0,
                "range": round(max(scores) - min(scores), 3) if scores else 0
            },
            "blt_compatibility": {
                "mean": round(sum(blt_scores) / len(blt_scores), 3) if blt_scores else 0,
                "min": round(min(blt_scores), 3) if blt_scores else 0,
                "max": round(max(blt_scores), 3) if blt_scores else 0
            },
            "richness": {
                "mean": round(sum(richness_scores) / len(richness_scores), 3) if richness_scores else 0,
                "min": round(min(richness_scores), 3) if richness_scores else 0,
                "max": round(max(richness_scores), 3) if richness_scores else 0
            }
        },
        "recommendations": {
            "best_overall": rankings[0]["model"] if rankings else None,
            "best_blt_compatible": max(rankings, key=lambda x: x["blt_compatibility"])["model"] if rankings else None,
            "best_richness": max(rankings, key=lambda x: x["richness"])["model"] if rankings else None,
            "best_speed": max(rankings, key=lambda x: x["tokens_per_second"])["model"] if rankings else None,
            "recommended_for_production": rankings[0]["model"] if rankings and rankings[0]["overall_score"] >= 0.5 else "None (improve model selection)",
            "multi_model_ensemble_candidates": [r["model"] for r in rankings[:3]] if len(rankings) >= 3 else [r["model"] for r in rankings]
        },
        "investor_metrics": {
            "system_robustness_across_models": round(len([r for r in rankings if r["overall_score"] >= 0.5]) / len(rankings) * 100, 1) if rankings else 0,
            "average_capability": round(sum(scores) / len(scores), 3) if scores else 0,
            "consistency_index": round(1 - (max(scores) - min(scores)) / max(scores), 3) if scores and max(scores) > 0 else 0,
            "vme_proven_across_architectures": len(rankings) >= 3
        }
    }
    
    return report

def main():
    """Execute Phase 6 parallel benchmarking"""
    
    print("\n" + "="*80)
    print("🚀 PHASE 6: MULTI-MODEL ORCHESTRATION & PARALLEL BENCHMARKING")
    print("="*80)
    print(f"Starting at: {datetime.now().isoformat()}")
    print(f"Models: {len(PRIORITY_MODELS)}")
    print(f"Prompts per model: {len(BLT_PROMPTS)}")
    print(f"Parallel workers: {min(len(PRIORITY_MODELS), mp.cpu_count() - 1)}")
    print("="*80)
    
    # Run parallel benchmark
    benchmark_results = run_parallel_benchmark(PRIORITY_MODELS)
    
    # Generate comparative report
    comparative_report = generate_comparative_report(benchmark_results)
    
    # Save results
    output_dir = Path("c:\\UBLT\\phase6_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full benchmark data
    benchmark_file = output_dir / f"phase6_parallel_benchmark_{timestamp}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Saved benchmark data: {benchmark_file}")
    
    # Comparative report
    report_file = output_dir / f"phase6_comparative_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(comparative_report, f, indent=2)
    logger.info(f"Saved comparative report: {report_file}")
    
    # Print summary
    print("\n\n" + "="*80)
    print("📊 PHASE 6 COMPARATIVE RANKINGS")
    print("="*80)
    
    rankings = comparative_report["rankings"]
    print(f"\n{'Rank':<5} {'Model':<30} {'Score':<8} {'BLT':<8} {'Richness':<10} {'Tok/s':<8} {'Valid%':<8}")
    print("-"*80)
    
    for r in rankings:
        print(f"{r['rank']:<5} {r['model']:<30} {r['overall_score']:<8.3f} "
              f"{r['blt_compatibility']:<8.3f} {r['richness']:<10.3f} "
              f"{r['tokens_per_second']:<8.1f} {r['valid_responses_pct']:<8.1f}")
    
    print("\n" + "="*80)
    print("🏆 PHASE 6 RECOMMENDATIONS")
    print("="*80)
    recs = comparative_report["recommendations"]
    print(f"Best Overall Model: {recs['best_overall']}")
    print(f"Best BLT Compatible: {recs['best_blt_compatible']}")
    print(f"Best Richness: {recs['best_richness']}")
    print(f"Best Speed: {recs['best_speed']}")
    print(f"Recommended for Production: {recs['recommended_for_production']}")
    print(f"Multi-Model Ensemble: {', '.join(recs['multi_model_ensemble_candidates'])}")
    
    print("\n" + "="*80)
    print("📈 INVESTOR METRICS (VME ROBUSTNESS)")
    print("="*80)
    metrics = comparative_report["investor_metrics"]
    print(f"System Robustness (% models scoring ≥0.5): {metrics['system_robustness_across_models']:.1f}%")
    print(f"Average Capability Across Models: {metrics['average_capability']:.3f}")
    print(f"Consistency Index (0-1, higher=better): {metrics['consistency_index']:.3f}")
    print(f"VME Proven Across Architectures: {metrics['vme_proven_across_architectures']}")
    
    print("\n" + "="*80)
    print(f"Execution time: {comparative_report['total_execution_time_seconds']:.1f}s")
    print(f"Reports saved to: {output_dir}")
    print("="*80 + "\n")
    
    return {
        "benchmark_results": benchmark_results,
        "comparative_report": comparative_report,
        "output_dir": str(output_dir)
    }

if __name__ == "__main__":
    results = main()
