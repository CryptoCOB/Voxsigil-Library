"""
VoxSigil Psychology-Focused Generator
Generates behavioral/cognitive variations for BLT training
Focuses on psychological dimensions: personality traits, cognitive patterns, behavioral archetypes
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from voxsigil_complete_middleware import VoxSigilCompleteMiddleware
from datetime import datetime

FASTEST_MODEL = "wizard-math:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

class PsychologyDimension:
    """Psychological dimensions for behavioral variation"""
    
    COGNITIVE_PATTERNS = [
        "analytical_reasoning", "pattern_recognition", "systems_thinking",
        "abstract_conceptualization", "holistic_perception", "sequential_processing",
        "parallel_processing", "intuitive_inference", "deductive_logic", 
        "inductive_reasoning", "analogical_thinking", "meta_cognitive_awareness"
    ]
    
    PERSONALITY_TRAITS = [
        "curiosity", "openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism", "risk_tolerance", "ambiguity_tolerance",
        "novelty_seeking", "persistence", "adaptability", "empathy"
    ]
    
    BEHAVIORAL_ARCHETYPES = [
        "explorer", "analyzer", "creator", "optimizer", "mediator",
        "visionary", "executor", "guardian", "innovator", "synthesizer",
        "catalyst", "architect", "navigator", "harmonizer", "disruptor"
    ]
    
    COGNITIVE_STYLES = [
        "reflective", "impulsive", "convergent", "divergent",
        "field_dependent", "field_independent", "systematic", "intuitive",
        "verbal", "visual", "kinesthetic", "abstract"
    ]
    
    DECISION_MAKING = [
        "rational", "emotional", "collaborative", "independent",
        "cautious", "bold", "data_driven", "values_driven",
        "incremental", "transformational", "pragmatic", "idealistic"
    ]

class VoxSigilPsychologyGenerator:
    """Generate psychology-focused VoxSigil variations for BLT training"""
    
    def __init__(self, model: str = FASTEST_MODEL):
        self.model = model
        self.middleware = VoxSigilCompleteMiddleware()
        self.inventory = None
        self.generated_variations = []
        
    def initialize(self):
        """Load all sigils and analyze psychological patterns"""
        print(f"\n{'='*70}")
        print(f"🧠 PSYCHOLOGY-FOCUSED VOXSIGIL GENERATOR")
        print(f"{'='*70}")
        self.inventory = self.middleware.load_all_sigils()
        
    def create_psychology_variation_prompt(self, 
                                           base_sigil: Dict[str, Any],
                                           psychological_focus: Dict[str, Any]) -> str:
        """Create prompt focusing on specific psychological dimensions"""
        
        name = base_sigil["name"]
        category = base_sigil["category"]
        reference = base_sigil["raw_content"][:1200]
        
        prompt = f"""You are creating a NEW VoxSigil with a specific PSYCHOLOGICAL PROFILE.

BASE REFERENCE SIGIL:
Name: {name}
Category: {category}

{reference}

PSYCHOLOGICAL VARIATION PARAMETERS:
Cognitive Pattern: {psychological_focus['cognitive_pattern']}
Personality Archetype: {psychological_focus['archetype']}
Key Traits: {', '.join(psychological_focus['traits'])}
Cognitive Style: {psychological_focus['cognitive_style']}
Decision Making: {psychological_focus['decision_style']}

TASK: Create a NEW VoxSigil that embodies these psychological characteristics.

REQUIREMENTS:
1. Follow VoxSigil YAML format (schema_version, meta, cognitive, holo_mesh, implementation)
2. DIFFERENT name than '{name}' (suggest name reflecting psychological profile)
3. Emphasize the specified psychological dimensions in the 'cognitive' section
4. Adjust behavioral parameters to match the psychological profile
5. Include relevant tags reflecting cognitive patterns
6. Make it psychologically DISTINCT and behaviorally COHERENT

Focus on making this a realistic, nuanced behavioral profile that would help train a Behavioral Learning Transformer.

Generate complete VoxSigil YAML:"""
        
        return prompt
    
    def generate_psychology_variation(self, 
                                    base_sigil: Dict[str, Any],
                                    psychological_focus: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single psychological variation"""
        
        archetype = psychological_focus['archetype']
        cognitive = psychological_focus['cognitive_pattern']
        
        print(f"\n🧬 Generating: {archetype} + {cognitive}", end=" ... ", flush=True)
        
        prompt = self.create_psychology_variation_prompt(base_sigil, psychological_focus)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1024,
                        "temperature": 0.8,  # Higher for more creative psychological profiles
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
            
            # Check for duplicates
            is_dup = self.middleware.is_duplicate(content)
            
            result = {
                "base_sigil": base_sigil["name"],
                "base_category": base_sigil["category"],
                "psychological_profile": psychological_focus,
                "generated_content": content,
                "is_duplicate": is_dup,
                "is_valid": self._quick_validate(content),
                "tokens": tokens,
                "time_sec": elapsed,
                "tokens_per_sec": tps,
                "timestamp": datetime.now().isoformat()
            }
            
            self.generated_variations.append(result)
            
            status = "✅" if result["is_valid"] and not is_dup else ("⚠️ DUP" if is_dup else "❌")
            print(f"{status} {tokens}tok {elapsed:.1f}s ({tps:.1f}tok/s)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _quick_validate(self, content: str) -> bool:
        """Quick validation of generated sigil"""
        # Check for YAML structure markers
        required_fields = ["schema_version", "meta", "cognitive"]
        return all(field in content for field in required_fields)
    
    def generate_psychology_training_set(self, count: int = 20) -> None:
        """Generate diverse psychological training variations"""
        import random
        
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING {count} PSYCHOLOGY-FOCUSED TRAINING SIGILS")
        print(f"{'='*70}\n")
        
        for i in range(count):
            print(f"[{i+1}/{count}] ", end="")
            
            # Pick random base sigil
            base_sigil = self.middleware.get_random_sigil_for_variation()
            
            # Create unique psychological profile
            psychological_focus = {
                "cognitive_pattern": random.choice(PsychologyDimension.COGNITIVE_PATTERNS),
                "archetype": random.choice(PsychologyDimension.BEHAVIORAL_ARCHETYPES),
                "traits": random.sample(PsychologyDimension.PERSONALITY_TRAITS, 3),
                "cognitive_style": random.choice(PsychologyDimension.COGNITIVE_STYLES),
                "decision_style": random.choice(PsychologyDimension.DECISION_MAKING)
            }
            
            self.generate_psychology_variation(base_sigil, psychological_focus)
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)
        
        self.save_training_set()
    
    def save_training_set(self) -> None:
        """Save generated psychological training variations"""
        output_dir = Path("c:\\UBLT\\blt_psychology_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"psychology_variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate statistics
        valid_count = sum(1 for v in self.generated_variations if v and v["is_valid"])
        duplicate_count = sum(1 for v in self.generated_variations if v and v["is_duplicate"])
        avg_tokens = sum(v["tokens"] for v in self.generated_variations if v) / len(self.generated_variations)
        avg_tps = sum(v["tokens_per_sec"] for v in self.generated_variations if v) / len(self.generated_variations)
        
        training_set = {
            "metadata": {
                "model_used": self.model,
                "base_sigils_count": self.inventory.total_sigils,
                "generated_count": len(self.generated_variations),
                "valid_count": valid_count,
                "duplicate_count": duplicate_count,
                "timestamp": datetime.now().isoformat()
            },
            "psychological_dimensions": {
                "cognitive_patterns": PsychologyDimension.COGNITIVE_PATTERNS,
                "personality_traits": PsychologyDimension.PERSONALITY_TRAITS,
                "behavioral_archetypes": PsychologyDimension.BEHAVIORAL_ARCHETYPES,
                "cognitive_styles": PsychologyDimension.COGNITIVE_STYLES,
                "decision_making_styles": PsychologyDimension.DECISION_MAKING
            },
            "variations": self.generated_variations,
            "statistics": {
                "avg_tokens": avg_tokens,
                "avg_tokens_per_sec": avg_tps,
                "valid_percentage": (valid_count / len(self.generated_variations)) * 100,
                "duplicate_percentage": (duplicate_count / len(self.generated_variations)) * 100
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(training_set, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Psychology training set saved: {filename}")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Total generated: {len(self.generated_variations)}")
        print(f"   Valid: {valid_count} ({valid_count/len(self.generated_variations)*100:.1f}%)")
        print(f"   Duplicates: {duplicate_count}")
        print(f"   Avg tokens: {avg_tokens:.0f}")
        print(f"   Avg speed: {avg_tps:.1f} tok/s")
        print(f"{'='*70}\n")

def main():
    """Generate psychology-focused BLT training data"""
    generator = VoxSigilPsychologyGenerator()
    generator.initialize()
    generator.generate_psychology_training_set(count=20)
    
    print("✅ Psychology-focused training data generation complete!")
    print("   Ready for BLT validation and training")

if __name__ == "__main__":
    main()
