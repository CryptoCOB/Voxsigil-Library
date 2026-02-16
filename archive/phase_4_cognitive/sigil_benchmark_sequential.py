"""
Sequential Sigil Generation Benchmark for BLT
Tests each model sequentially with diverse prompts
Measures: token speed, response variation, BLT compatibility
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import requests

# Models to test
MODELS = [
    "kimi-k2.5:cloud",
    "mistral:latest",
    "mxbai-embed-large:latest",
    "wizard-math:latest",
    "mathstral:latest",
    "phi3:mini",
    "deepseek-coder:6.7b",
    "qwen2:7b",
    "llama3:8b",
    "gpt-oss:20b",
    "llava-phi3:latest",
]

OLLAMA_BASE_URL = "http://localhost:11434"

# Different BLT scenario prompts to test model variation
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
class SigilGenerationResult:
    """Result from a single sigil generation"""
    model: str
    prompt_type: str
    response: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    blt_score: float
    richness_score: float
    length_score: float
    
def call_ollama_api(model: str, prompt: str) -> Dict[str, Any]:
    """Call Ollama API and return response with metrics"""
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
            timeout=180
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
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "tokens": 0,
            "time_ms": (time.time() - start_time) * 1000,
            "tokens_per_sec": 0
        }

def score_blt_compatibility(response: str) -> float:
    """Score BLT compatibility - realistic scoring"""
    if not response or len(response) < 150:
        return 0.2
    
    score = 0.0
    
    # Check for BLT-specific structures
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
    
    # Boost if has actual numerical values
    if any(c.isdigit() for c in response):
        score = min(score * 1.2, 1.0)
    
    return score

def score_richness(response: str) -> float:
    """Score behavioral richness - how detailed the sigil is"""
    if not response:
        return 0.0
    
    # Check for behavioral keywords
    behavior_terms = [
        "motivation", "pattern", "response", "context", 
        "trajectory", "adaptation", "trigger", "outcome",
        "communication", "collaboration", "learning", "decision"
    ]
    
    term_count = sum(1 for term in behavior_terms if term.lower() in response.lower())
    richness = min(term_count / 8, 1.0)
    
    # Length bonus (richer responses tend to be longer)
    length_factor = min(len(response) / 1000, 1.0) * 0.3
    richness = min(richness + length_factor, 1.0)
    
    return richness

def score_length(response: str, target_tokens: int = 300) -> float:
    """Score response length - optimal is around target tokens"""
    if not response:
        return 0.0
    
    # Rough estimate: ~4 chars per token
    estimated_tokens = len(response) / 4
    
    # Penalize if too short or too long
    if estimated_tokens < 100:
        return 0.2
    elif estimated_tokens < target_tokens:
        return min(estimated_tokens / target_tokens, 1.0)
    elif estimated_tokens > target_tokens * 1.5:
        return 1.0 - min((estimated_tokens - target_tokens * 1.5) / 500, 0.3)
    else:
        return 1.0

def benchmark_model_sequential(model: str) -> Dict[str, Any]:
    """Benchmark a single model with all BLT prompts"""
    print(f"\n{'='*70}")
    print(f"Testing: {model}")
    print(f"{'='*70}")
    
    results = []
    total_tokens = 0
    total_time = 0
    valid_responses = 0
    
    for prompt_type, prompt in BLT_PROMPTS.items():
        print(f"  ▶️  {prompt_type.replace('_', ' ').title()}...", end=" ", flush=True)
        
        api_result = call_ollama_api(model, prompt)
        
        if not api_result["success"]:
            print(f"❌ Error: {api_result['error']}")
            continue
        
        response = api_result["response"]
        tokens = api_result["tokens"]
        time_ms = api_result["time_ms"]
        tps = api_result["tokens_per_sec"]
        
        blt_score = score_blt_compatibility(response)
        richness = score_richness(response)
        length_score = score_length(response)
        
        is_valid = blt_score >= 0.5 and richness >= 0.4
        if is_valid:
            valid_responses += 1
        
        result = SigilGenerationResult(
            model=model,
            prompt_type=prompt_type,
            response=response,
            tokens_generated=tokens,
            generation_time_ms=time_ms,
            tokens_per_second=tps,
            blt_score=blt_score,
            richness_score=richness,
            length_score=length_score
        )
        
        results.append(result)
        total_tokens += tokens
        total_time += time_ms
        
        status = "✅" if is_valid else "⚠️"
        print(f"{status} BLT={blt_score:.2f} | Richness={richness:.2f} | {tps:.1f} tok/s")
    
    avg_blt = sum(r.blt_score for r in results) / len(results) if results else 0
    avg_richness = sum(r.richness_score for r in results) / len(results) if results else 0
    avg_length = sum(r.length_score for r in results) / len(results) if results else 0
    avg_tokens_per_sec = sum(r.tokens_per_second for r in results) / len(results) if results else 0
    overall_score = (avg_blt * 0.4) + (avg_richness * 0.4) + (avg_tokens_per_sec / 50 * 0.2)
    
    return {
        "model": model,
        "prompts_tested": len(BLT_PROMPTS),
        "results": [
            {
                "prompt_type": r.prompt_type,
                "blt_score": r.blt_score,
                "richness": r.richness_score,
                "length_score": r.length_score,
                "tokens_per_second": r.tokens_per_second,
                "tokens_generated": r.tokens_generated,
                "time_ms": r.generation_time_ms,
                "response_preview": r.response[:200] + "..." if len(r.response) > 200 else r.response
            }
            for r in results
        ],
        "summary": {
            "avg_blt_compatibility": avg_blt,
            "avg_behavioral_richness": avg_richness,
            "avg_length_adequacy": avg_length,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "valid_responses": valid_responses,
            "overall_score": overall_score
        },
        "full_results": results
    }

def main():
    """Run sequential benchmark on all models"""
    print("\n" + "="*70)
    print("🧬 SEQUENTIAL SIGIL GENERATION BENCHMARK FOR BLT")
    print("="*70)
    print(f"Models: {len(MODELS)} | Prompts per model: {len(BLT_PROMPTS)}")
    print(f"Metrics: BLT Compatibility | Behavioral Richness | Token Speed")
    
    all_results = {}
    
    for model in MODELS:
        try:
            result = benchmark_model_sequential(model)
            all_results[model] = result
        except Exception as e:
            print(f"\n❌ {model}: Fatal error - {str(e)}")
    
    # Generate report
    report_file = Path("c:\\UBLT\\sigil_benchmarks") / f"sequential_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by overall score
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]["summary"]["overall_score"],
        reverse=True
    )
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": "Sequential Sigil Generation for BLT",
        "models_tested": len(all_results),
        "prompts_per_model": len(BLT_PROMPTS),
        "results": {name: data for name, data in sorted_models},
        "rankings": [
            {
                "rank": i + 1,
                "model": name,
                "overall_score": data["summary"]["overall_score"],
                "blt_compatibility": data["summary"]["avg_blt_compatibility"],
                "richness": data["summary"]["avg_behavioral_richness"],
                "tokens_per_second": data["summary"]["avg_tokens_per_second"],
                "valid_responses": data["summary"]["valid_responses"]
            }
            for i, (name, data) in enumerate(sorted_models)
        ]
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print final summary
    print("\n\n" + "="*70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Rank':<5} {'Model':<30} {'Score':<8} {'BLT':<8} {'Richness':<10} {'Tok/s':<8}")
    print("-"*70)
    
    for i, (model, data) in enumerate(sorted_models, 1):
        s = data["summary"]
        print(f"{i:<5} {model:<30} {s['overall_score']:<8.3f} "
              f"{s['avg_blt_compatibility']:<8.2f} {s['avg_behavioral_richness']:<10.2f} "
              f"{s['avg_tokens_per_second']:<8.1f}")
    
    print("\n" + "="*70)
    print(f"🏆 RECOMMENDED FOR BLT TRAINING: {sorted_models[0][0]}")
    best = sorted_models[0][1]["summary"]
    print(f"   Overall Score: {best['overall_score']:.3f}")
    print(f"   BLT Compatibility: {best['avg_blt_compatibility']:.3f}")
    print(f"   Behavioral Richness: {best['avg_behavioral_richness']:.3f}")
    print(f"   Token Speed: {best['avg_tokens_per_second']:.1f} tokens/sec")
    print("="*70)
    print(f"\n📄 Full report: {report_file}\n")

if __name__ == "__main__":
    main()
