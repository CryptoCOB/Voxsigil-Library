"""
Model Benchmark Orchestrator - Test all 11 Ollama models for sigil enrichment quality
Runs benchmarks in parallel and generates comparative quality reports
"""

import os
import json
import time
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import hashlib

# Models to benchmark
MODELS_TO_TEST = [
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

@dataclass
class BenchmarkResult:
    """Individual model benchmark result"""
    model_name: str
    test_prompt: str
    response: str
    response_length: int
    coherence_score: float  # 0-1: semantic coherence
    richness_score: float   # 0-1: detail/enrichment
    specificity_score: float  # 0-1: domain specificity
    blt_compatibility_score: float  # 0-1: BLT structure match
    execution_time_ms: float
    tokens_per_second: float
    timestamp: str

class ModelBenchmark:
    """Orchestrates benchmarking of all models"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, output_dir: str = "c:\\UBLT\\benchmarks"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def generate_test_prompt(self) -> str:
        """Generate a sigil enrichment test prompt"""
        return """Generate a rich, detailed behavioral sigil that describes a professional with these traits:
- High analytical capability
- Strong communication skills  
- Collaborative mindset
- Adaptive and learning-oriented

Create a comprehensive sigil that captures these enrichments with specific behavioral patterns, motivations, and contextual responses. Format as structured metadata."""

    async def test_model(self, model_name: str) -> BenchmarkResult:
        """Test a single model"""
        try:
            prompt = self.generate_test_prompt()
            start_time = time.time()
            
            # Call Ollama API
            result = subprocess.run(
                ["curl", "-X", "POST", f"{self.base_url}/api/generate",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({
                     "model": model_name,
                     "prompt": prompt,
                     "stream": False
                 })],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if result.returncode != 0:
                print(f"❌ {model_name}: API Error - {result.stderr}")
                return None
                
            response_data = json.loads(result.stdout)
            response = response_data.get("response", "")
            
            # Scoring metrics
            coherence = self._score_coherence(response)
            richness = self._score_richness(response)
            specificity = self._score_specificity(response)
            blt_compat = self._score_blt_compatibility(response)
            
            tokens_per_sec = response_data.get("eval_count", 0) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            
            benchmark = BenchmarkResult(
                model_name=model_name,
                test_prompt=prompt,
                response=response,
                response_length=len(response),
                coherence_score=coherence,
                richness_score=richness,
                specificity_score=specificity,
                blt_compatibility_score=blt_compat,
                execution_time_ms=elapsed_ms,
                tokens_per_second=tokens_per_sec,
                timestamp=datetime.now().isoformat()
            )
            
            print(f"✅ {model_name}: Coherence={coherence:.2f} | Richness={richness:.2f} | BLT={blt_compat:.2f} | {elapsed_ms:.0f}ms")
            return benchmark
            
        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")
            return None
    
    def _score_coherence(self, response: str) -> float:
        """Score semantic coherence (0-1)"""
        if not response:
            return 0.0
        
        # Check for structural markers
        markers = ["behavioral", "pattern", "context", "trait", "motivation", "response"]
        marker_count = sum(1 for m in markers if m.lower() in response.lower())
        coherence = min(marker_count / len(markers), 1.0)
        
        # Penalize if too short
        if len(response) < 100:
            coherence *= 0.5
            
        return coherence
    
    def _score_richness(self, response: str) -> float:
        """Score enrichment detail level (0-1)"""
        if not response:
            return 0.0
        
        # Length as proxy for richness
        richness = min(len(response) / 1000, 1.0)
        
        # Bonus for structured content
        if "{" in response or "-" in response or ":" in response:
            richness = min(richness * 1.2, 1.0)
            
        return richness
    
    def _score_specificity(self, response: str) -> float:
        """Score domain-specific terminology (0-1)"""
        if not response:
            return 0.0
        
        sigil_terms = [
            "behavioral", "cognitive", "emotional", "contextual",
            "motivation", "adaptation", "pattern", "framework",
            "attribute", "characteristic", "trajectory"
        ]
        
        term_count = sum(1 for t in sigil_terms if t.lower() in response.lower())
        specificity = min(term_count / 5, 1.0)  # 5 terms = maximum
        
        return specificity
    
    def _score_blt_compatibility(self, response: str) -> float:
        """Score BLT (Behavioral Learning Template) compatibility (0-1)"""
        if not response:
            return 0.0
        
        # Check for BLT-compatible structure
        blt_indicators = [
            "metric",
            "score",
            "range",
            "threshold",
            "value",
            "dimension"
        ]
        
        indicator_count = sum(1 for ind in blt_indicators if ind.lower() in response.lower())
        compatibility = min(indicator_count / 3, 1.0)  # 3 indicators = good compatibility
        
        return compatibility
    
    async def run_benchmarks(self, models: List[str] = None) -> None:
        """Run benchmarks for all models"""
        if models is None:
            models = MODELS_TO_TEST
        
        print(f"\n{'='*80}")
        print(f"🔬 MODEL BENCHMARK ORCHESTRATOR")
        print(f"{'='*80}")
        print(f"Testing {len(models)} models for sigil enrichment quality\n")
        
        # Test each model
        tasks = [self.test_model(model) for model in models]
        results = await asyncio.gather(*tasks)
        self.results = [r for r in results if r is not None]
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate benchmark comparison report"""
        if not self.results:
            print("❌ No successful benchmarks to report")
            return
        
        report_file = self.output_dir / f"model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Sort by overall quality (composite score)
        self.results.sort(
            key=lambda r: (r.coherence_score + r.richness_score + r.specificity_score + r.blt_compatibility_score) / 4,
            reverse=True
        )
        
        # Create report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_tested": len(self.results),
                "purpose": "Sigil enrichment quality and BLT compatibility evaluation"
            },
            "summary": {
                "top_model": {
                    "name": self.results[0].model_name,
                    "overall_score": (self.results[0].coherence_score + self.results[0].richness_score + self.results[0].specificity_score + self.results[0].blt_compatibility_score) / 4
                }
            },
            "detailed_results": [asdict(r) for r in self.results],
            "score_breakdown": self._calculate_score_breakdown()
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary()
        print(f"\n📊 Full report saved: {report_file}")
    
    def _calculate_score_breakdown(self) -> Dict[str, Any]:
        """Calculate average scores by dimension"""
        if not self.results:
            return {}
        
        return {
            "avg_coherence": sum(r.coherence_score for r in self.results) / len(self.results),
            "avg_richness": sum(r.richness_score for r in self.results) / len(self.results),
            "avg_specificity": sum(r.specificity_score for r in self.results) / len(self.results),
            "avg_blt_compatibility": sum(r.blt_compatibility_score for r in self.results) / len(self.results),
            "avg_execution_time_ms": sum(r.execution_time_ms for r in self.results) / len(self.results),
            "avg_tokens_per_sec": sum(r.tokens_per_second for r in self.results) / len(self.results)
        }
    
    def _print_summary(self) -> None:
        """Print benchmark summary to console"""
        print(f"\n{'='*80}")
        print(f"📈 BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Model Name':<30} {'Coherence':<12} {'Richness':<12} {'Specificity':<12} {'BLT Compat':<12} {'Time (ms)':<10}")
        print(f"{'-'*98}")
        
        for result in self.results:
            print(f"{result.model_name:<30} "
                  f"{result.coherence_score:<12.2f} "
                  f"{result.richness_score:<12.2f} "
                  f"{result.specificity_score:<12.2f} "
                  f"{result.blt_compatibility_score:<12.2f} "
                  f"{result.execution_time_ms:<10.0f}")
        
        scores = self._calculate_score_breakdown()
        print(f"\n{'-'*98}")
        print(f"{'AVERAGES':<30} "
              f"{scores.get('avg_coherence', 0):<12.2f} "
              f"{scores.get('avg_richness', 0):<12.2f} "
              f"{scores.get('avg_specificity', 0):<12.2f} "
              f"{scores.get('avg_blt_compatibility', 0):<12.2f} "
              f"{scores.get('avg_execution_time_ms', 0):<10.0f}")
        
        print(f"\n🏆 Top Model: {self.results[0].model_name}")
        print(f"   Overall Score: {(self.results[0].coherence_score + self.results[0].richness_score + self.results[0].specificity_score + self.results[0].blt_compatibility_score) / 4:.2f}/1.0")


async def main():
    """Main execution"""
    benchmark = ModelBenchmark()
    await benchmark.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
