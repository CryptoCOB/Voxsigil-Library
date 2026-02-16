"""
BLT Training Data Generator
Uses fastest model (wizard-math:latest) with VoxSigil context
Generates high-quality training sigils for Behavioral Learning Transformer
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
from voxsigil_middleware import VoxSigilMiddleware

# Winner from speed test
FASTEST_MODEL = "wizard-math:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

class BLTTrainingDataGenerator:
    """Generate training data for BLT using contextualized VoxSigils"""
    
    def __init__(self, model: str = FASTEST_MODEL):
        self.model = model
        self.middleware = VoxSigilMiddleware()
        self.middleware.load_all_sigils()
        self.middleware.build_context()
        self.generated_sigils = []
        
    def generate_training_sigil(self, persona_description: str) -> dict:
        """Generate a single training sigil with full VoxSigil context"""
        base_prompt = f"Create a VoxSigil for: {persona_description}"
        contextualized_prompt = self.middleware.create_contextualized_prompt(base_prompt)
        
        print(f"🧬 Generating sigil for: {persona_description[:50]}...", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": contextualized_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            data = response.json()
            content = data.get("response", "")
            tokens = data.get("eval_count", 0)
            tps = tokens / elapsed if elapsed > 0 else 0
            
            result = {
                "persona": persona_description,
                "sigil_content": content,
                "tokens_generated": tokens,
                "generation_time_sec": elapsed,
                "tokens_per_second": tps,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model
            }
            
            self.generated_sigils.append(result)
            print(f"✅ {tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def generate_training_batch(self, count: int = 10) -> None:
        """Generate a batch of training sigils for BLT"""
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING {count} TRAINING SIGILS FOR BLT")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Context: {self.middleware.context.total_sigils} reference sigils loaded\n")
        
        # Diverse persona types for training data
        personas = [
            "Analytical engineer with strong pattern recognition and data-driven decision making",
            "Creative designer with high innovation and comfort with ambiguity",
            "Strategic leader with systems thinking and long-term vision",
            "Collaborative facilitator focused on consensus-building",
            "Technical specialist with deep domain expertise and methodical approach",
            "Adaptive learner with high curiosity and growth mindset",
            "Risk-aware operator with safety-first priorities",
            "Visionary innovator pushing boundaries and exploring novel solutions",
            "Pragmatic executor balancing quality with timely delivery",
            "Empathetic communicator bridging diverse perspectives"
        ]
        
        for i, persona in enumerate(personas[:count], 1):
            print(f"\n[{i}/{count}] ", end="")
            self.generate_training_sigil(persona)
        
        # Save results
        self.save_training_data()
    
    def save_training_data(self) -> None:
        """Save generated training data"""
        output_dir = Path("c:\\UBLT\\blt_training_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"training_sigils_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        training_data = {
            "metadata": {
                "model_used": self.model,
                "reference_sigils": self.middleware.context.total_sigils,
                "behavioral_dimensions": len(self.middleware.context.common_patterns["behavioral_dimensions"]),
                "generation_timestamp": datetime.now().isoformat(),
                "total_sigils_generated": len(self.generated_sigils)
            },
            "sigils": self.generated_sigils,
            "summary": {
                "avg_tokens": sum(s["tokens_generated"] for s in self.generated_sigils) / len(self.generated_sigils),
                "avg_time_sec": sum(s["generation_time_sec"] for s in self.generated_sigils) / len(self.generated_sigils),
                "avg_speed_tps": sum(s["tokens_per_second"] for s in self.generated_sigils) / len(self.generated_sigils)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Training data saved: {filename}")
        print(f"📊 Summary:")
        print(f"   Sigils generated: {training_data['metadata']['total_sigils_generated']}")
        print(f"   Avg tokens: {training_data['summary']['avg_tokens']:.0f}")
        print(f"   Avg speed: {training_data['summary']['avg_speed_tps']:.1f} tok/s")
        print(f"{'='*70}\n")

def main():
    """Generate BLT training data with fastest model and VoxSigil context"""
    generator = BLTTrainingDataGenerator()
    generator.generate_training_batch(count=10)
    
    print("✅ BLT training data generation complete!")
    print("   Ready to train Behavioral Learning Transformer")

if __name__ == "__main__":
    main()
