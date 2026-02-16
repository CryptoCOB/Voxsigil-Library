"""
Sigil Generation Benchmarking System
Tests 11 Ollama models to find best sigil generator for BLT (Behavioral Learning Transformer)
Generates actual sigils and evaluates BLT compatibility
"""

import os
import json
import time
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

import requests

# Models to benchmark
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

class SigilQualityMetric(Enum):
    """Metrics for evaluating generated sigils"""
    BLT_COMPATIBILITY = "blt_compatibility"      # Can BLT parse/use this sigil?
    BEHAVIORAL_RICHNESS = "behavioral_richness"  # How detailed is the behavioral data?
    STRUCTURAL_VALIDITY = "structural_validity"  # Is it well-formed?
    SEMANTIC_COHERENCE = "semantic_coherence"    # Does it make sense?

@dataclass
class GeneratedSigil:
    """A generated sigil from a model"""
    model_name: str
    sigil_id: str
    content: str
    generated_at: str
    metrics: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None

@dataclass
class ModelBenchmarkResult:
    """Benchmark result for a single model"""
    model_name: str
    sigils_generated: int
    avg_blt_compatibility: float
    avg_behavioral_richness: float
    avg_structural_validity: float
    avg_semantic_coherence: float
    overall_score: float
    generation_time_ms: float
    tokens_per_second: float
    valid_sigils_count: int
    failed_sigils_count: int
    timestamp: str

class SigilGenerator:
    """Generates sigils using Ollama models"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, output_dir: str = "c:\\UBLT\\generated_sigils"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sigil_prompt(self, profile_type: str = "default") -> str:
        """Create prompt for sigil generation"""
        prompts = {
            "analytical": """Generate a comprehensive behavioral sigil for a high-analytical professional:
- Strengths: Problem-solving, data interpretation, logical reasoning
- Patterns: Methodical approach, preference for evidence-based decisions
- Context: Works well in structured environments with clear metrics
- Behavioral Attributes: Detail-oriented, systematic learner, low risk tolerance
- Behavioral Trajectory: Progresses from task-focused to strategic thinking
- Emotional Regulation: Steady, minimal fluctuation
Include specific metrics (0-1 scales) for each behavioral dimension.""",
            
            "creative": """Generate a behavioral sigil for a creative professional:
- Strengths: Innovation, pattern synthesis, lateral thinking  
- Patterns: Flexible approach, comfort with ambiguity
- Context: Thrives in open-ended environments with exploration time
- Behavioral Attributes: Intuitive, adaptive, high curiosity
- Behavioral Trajectory: Ideas → Execution → Refinement
- Emotional Regulation: Variable, responsive to feedback
Include specific metrics (0-1 scales) for each behavioral dimension.""",
            
            "collaborative": """Generate a behavioral sigil for a team-oriented professional:
- Strengths: Communication, consensus-building, cross-functional collaboration
- Patterns: Seeks input, values diverse perspectives, facilitates alignment
- Context: Excels in group settings, distributed teams
- Behavioral Attributes: Empathetic, relationship-focused, support-oriented
- Behavioral Trajectory: Individual → Team → Organization
- Emotional Regulation: Responsive to group dynamics
Include specific metrics (0-1 scales) for each behavioral dimension.""",
        }
        return prompts.get(profile_type, prompts["default"] if "default" in prompts else list(prompts.values())[0])
    
    async def generate_sigil(self, model_name: str, profile_type: str = "analytical") -> GeneratedSigil:
        """Generate a single sigil using specified model"""
        try:
            prompt = self.create_sigil_prompt(profile_type)
            sigil_id = hashlib.md5(
                f"{model_name}_{profile_type}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]
            
            start_time = time.time()
            
            # Call Ollama API via requests with thoughtful parameters
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,  # Allow thinking/variation
                            "top_p": 0.9,        # Nucleus sampling
                            "top_k": 40,         # Diversity
                            "num_predict": 1024  # Allow longer responses
                        }
                    },
                    timeout=180
                )
                response.raise_for_status()
                response_data = response.json()
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                return GeneratedSigil(
                    model_name=model_name,
                    sigil_id=sigil_id,
                    content="",
                    generated_at=datetime.now().isoformat(),
                    metrics={},
                    is_valid=False,
                    error_message=f"API Error: {str(e)}"
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            content = response_data.get("response", "")
            
            # Evaluate sigil quality
            metrics = self._evaluate_sigil(content)
            # Valid if BLT compatibility >= 0.65 and behavioral richness >= 0.50
            is_valid = (metrics.get(SigilQualityMetric.BLT_COMPATIBILITY.value, 0) >= 0.65 and
                       metrics.get(SigilQualityMetric.BEHAVIORAL_RICHNESS.value, 0) >= 0.50)
            
            return GeneratedSigil(
                model_name=model_name,
                sigil_id=sigil_id,
                content=content,
                generated_at=datetime.now().isoformat(),
                metrics=metrics,
                is_valid=is_valid
            )
            
        except Exception as e:
            return GeneratedSigil(
                model_name=model_name,
                sigil_id="unknown",
                content="",
                generated_at=datetime.now().isoformat(),
                metrics={},
                is_valid=False,
                error_message=str(e)
            )
    
    def _evaluate_sigil(self, content: str) -> Dict[str, float]:
        """Evaluate generated sigil quality with realistic metrics"""
        metrics = {}
        
        # BLT Compatibility: More nuanced - check for actual BLT-ready structure
        blt_score = 0.0
        blt_keywords = [
            "metric", "behavioral_dimension", "attribute", "measure",
            "score", "value", "range", "0.0", "1.0", "0-1", "0 to 1"
        ]
        blt_keyword_count = sum(1 for kw in blt_keywords if kw.lower() in content.lower())
        
        # Check for numerical ranges/values
        has_numerics = any(c.isdigit() for c in content)
        
        # Check for structure (JSON-like, YAML-like, or clear hierarchy)
        has_structure = (
            (":" in content and "\n" in content) or  # YAML-like
            ("{" in content and "}" in content) or    # JSON-like
            ("-" in content and ":" in content)       # Structured list
        )
        
        # Combine for BLT compatibility (0.0 - 1.0)
        blt_score = (blt_keyword_count / 8) * 0.5 + (0.5 if has_numerics else 0) + (0.3 if has_structure else 0)
        metrics[SigilQualityMetric.BLT_COMPATIBILITY.value] = min(blt_score, 1.0)
        
        # Behavioral Richness: Depth and detail of behavioral insights
        behavior_keywords = [
            "motivation", "response", "context", "trajectory", "adaptation",
            "pattern", "trigger", "outcome", "decision", "learning",
            "emotional", "cognitive", "behavioral", "preference"
        ]
        behavior_count = sum(1 for kw in behavior_keywords if kw.lower() in content.lower())
        # Richness = keyword coverage + length bonus
        richness_score = (behavior_count / 7) * 0.7 + min(len(content) / 2000, 0.3)
        metrics[SigilQualityMetric.BEHAVIORAL_RICHNESS.value] = min(richness_score, 1.0)
        
        # Structural Validity: Well-formed, organized presentation
        validity_score = 0.0
        if has_structure:
            validity_score += 0.4
        if len(content) > 300:  # Sufficient depth
            validity_score += 0.3
        if content.count(".") > 5:  # Multiple coherent sentences
            validity_score += 0.3
        metrics[SigilQualityMetric.STRUCTURAL_VALIDITY.value] = min(validity_score, 1.0)
        
        # Semantic Coherence: Makes logical sense for behavioral modeling
        coherence_keywords = [
            "professional", "behavioral", "strengths", "patterns", "context",
            "attributes", "trajectory", "framework", "dimension",
            "individual", "team", "organization"
        ]
        coherence_count = sum(1 for kw in coherence_keywords if kw.lower() in content.lower())
        coherence_score = min(coherence_count / 6, 1.0)
        metrics[SigilQualityMetric.SEMANTIC_COHERENCE.value] = coherence_score
        
        return metrics
    
    def save_sigil(self, sigil: GeneratedSigil) -> Path:
        """Save generated sigil to file"""
        filename = self.output_dir / f"{sigil.sigil_id}_{sigil.model_name.replace(':', '_')}.json"
        
        data = {
            "sigil_id": sigil.sigil_id,
            "model": sigil.model_name,
            "generated_at": sigil.generated_at,
            "content": sigil.content,
            "metrics": sigil.metrics,
            "is_valid": sigil.is_valid,
            "error": sigil.error_message
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename

class SigilBenchmarkOrchestrator:
    """Orchestrates benchmarking all models for sigil generation"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.generator = SigilGenerator(base_url)
        self.results: Dict[str, ModelBenchmarkResult] = {}
        self.all_sigils: List[GeneratedSigil] = []
        
    async def benchmark_model(self, model_name: str, num_sigils: int = 3) -> ModelBenchmarkResult:
        """Benchmark a single model by generating multiple sigils"""
        print(f"\n🧬 Benchmarking {model_name}...")
        
        sigils = []
        total_time = 0
        profile_types = ["analytical", "creative", "collaborative"]
        
        for i in range(num_sigils):
            profile_type = profile_types[i % len(profile_types)]
            sigil = await self.generator.generate_sigil(model_name, profile_type)
            sigils.append(sigil)
            self.all_sigils.append(sigil)
            total_time += sigil.generated_at.__sizeof__()  # Placeholder
            
            if sigil.is_valid:
                self.generator.save_sigil(sigil)
                status = "✅"
            else:
                status = "❌"
            
            print(f"  {status} Sigil {i+1}: BLT={sigil.metrics.get('blt_compatibility', 0):.2f} | Richness={sigil.metrics.get('behavioral_richness', 0):.2f}")
        
        # Calculate averages
        valid_sigils = [s for s in sigils if s.is_valid]
        failed_sigils = [s for s in sigils if not s.is_valid]
        
        metrics = {}
        for metric_key in SigilQualityMetric:
            metric_name = metric_key.value
            values = [s.metrics.get(metric_name, 0) for s in valid_sigils]
            metrics[metric_name] = sum(values) / len(values) if values else 0.0
        
        overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        result = ModelBenchmarkResult(
            model_name=model_name,
            sigils_generated=num_sigils,
            avg_blt_compatibility=metrics.get('blt_compatibility', 0),
            avg_behavioral_richness=metrics.get('behavioral_richness', 0),
            avg_structural_validity=metrics.get('structural_validity', 0),
            avg_semantic_coherence=metrics.get('semantic_coherence', 0),
            overall_score=overall_score,
            generation_time_ms=total_time,
            tokens_per_second=0,
            valid_sigils_count=len(valid_sigils),
            failed_sigils_count=len(failed_sigils),
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    async def run_benchmark(self, models: List[str] = None, sigils_per_model: int = 3) -> None:
        """Run full benchmark suite"""
        if models is None:
            models = MODELS
        
        print("\n" + "="*80)
        print("🔬 SIGIL GENERATION BENCHMARK - BLT TRAINING DATA QUALITY TEST")
        print("="*80)
        print(f"Testing {len(models)} models | {sigils_per_model} sigils per model\n")
        
        # Benchmark each model
        for model_name in models:
            try:
                result = await self.benchmark_model(model_name, sigils_per_model)
                self.results[model_name] = result
            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate comprehensive benchmark report"""
        if not self.results:
            print("❌ No benchmark results")
            return
        
        # Sort by overall score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        report_file = Path("c:\\UBLT\\sigil_benchmarks") / f"sigil_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "purpose": "Find best model for generating training data for BLT (Behavioral Learning Transformer)",
                "models_tested": len(self.results),
                "sigils_generated": len(self.all_sigils),
                "valid_sigils": len([s for s in self.all_sigils if s.is_valid])
            },
            "top_models": [
                {
                    "rank": i + 1,
                    "model": name,
                    "overall_score": result.overall_score,
                    "blt_compatibility": result.avg_blt_compatibility,
                    "behavioral_richness": result.avg_behavioral_richness,
                    "valid_sigils": result.valid_sigils_count
                }
                for i, (name, result) in enumerate(sorted_results[:5])
            ],
            "detailed_results": {
                name: asdict(result) 
                for name, result in sorted_results
            },
            "recommendation": {
                "best_model": sorted_results[0][0],
                "best_score": sorted_results[0][1].overall_score,
                "reason": f"Highest BLT compatibility ({sorted_results[0][1].avg_blt_compatibility:.2f}) and behavioral richness ({sorted_results[0][1].avg_behavioral_richness:.2f})"
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(sorted_results)
        print(f"\n📊 Full report: {report_file}")
    
    def _print_summary(self, sorted_results) -> None:
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("📈 SIGIL GENERATION BENCHMARK RESULTS")
        print("="*80)
        print(f"\n{'Rank':<5} {'Model':<30} {'Score':<8} {'BLT':<8} {'Richness':<10} {'Valid':<6}")
        print("-" * 80)
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i:<5} {model_name:<30} {result.overall_score:<8.2f} "
                  f"{result.avg_blt_compatibility:<8.2f} {result.avg_behavioral_richness:<10.2f} "
                  f"{result.valid_sigils_count:<6}")
        
        best_model = sorted_results[0]
        print("\n" + "="*80)
        print(f"🏆 RECOMMENDED FOR BLT TRAINING: {best_model[0]}")
        print(f"   Overall Score: {best_model[1].overall_score:.3f}/1.0")
        print(f"   BLT Compatibility: {best_model[1].avg_blt_compatibility:.3f}/1.0")
        print(f"   Behavioral Richness: {best_model[1].avg_behavioral_richness:.3f}/1.0")
        print("="*80)

async def main():
    """Main execution"""
    orchestrator = SigilBenchmarkOrchestrator()
    await orchestrator.run_benchmark(sigils_per_model=3)

if __name__ == "__main__":
    asyncio.run(main())
