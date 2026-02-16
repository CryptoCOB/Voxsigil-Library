"""
VoxSigil Personality-Embodied Flow Generator
Each flow embodies a COMPLETE COGNITIVE PERSONA with unique identity
Includes thinking patterns, decision strategies, and game theory
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

class CognitivePersonas:
    """Complete cognitive personas with thinking styles"""
    
    PERSONAS = [
        {
            "archetype": "PragmaticEngineer",
            "sigil": "⚙️🔧🎯",
            "thinking_style": "systematic_validation_first",
            "decision_strategy": "minimize_risk_maximize_reliability",
            "procedural_bias": ["test_before_deploy", "measure_before_optimize", "prototype_before_scale"],
            "game_theory": "cooperative_iterated_prisoner_dilemma",
            "cognitive_traits": ["methodical", "evidence_driven", "incremental", "debugger_mindset"],
            "failure_aversion": "high",
            "novelty_tolerance": "low"
        },
        {
            "archetype": "ScientificMethodist",
            "sigil": "🔬📊🧪",
            "thinking_style": "hypothesis_driven_empiricism",
            "decision_strategy": "maximize_information_gain",
            "procedural_bias": ["observe_before_theorize", "control_variables", "replicate_results"],
            "game_theory": "bayesian_information_acquisition",
            "cognitive_traits": ["skeptical", "hypothesis_testing", "falsification_seeking", "peer_review_oriented"],
            "failure_aversion": "moderate",
            "novelty_tolerance": "high"
        },
        {
            "archetype": "CreativeIterator",
            "sigil": "🎨🌀💫",
            "thinking_style": "divergent_exploration_convergent_refinement",
            "decision_strategy": "maximize_novelty_with_constraint",
            "procedural_bias": ["ideate_before_filter", "prototype_rapidly", "iterate_on_feedback"],
            "game_theory": "exploratory_multi_armed_bandit",
            "cognitive_traits": ["generative", "analogical", "comfort_with_ambiguity", "aesthetic_driven"],
            "failure_aversion": "low",
            "novelty_tolerance": "very_high"
        },
        {
            "archetype": "StrategicPlanner",
            "sigil": "♟️🗺️🎯",
            "thinking_style": "goal_decomposition_backward_chaining",
            "decision_strategy": "minimax_with_look_ahead",
            "procedural_bias": ["define_goal_first", "assess_resources", "identify_constraints", "plan_contingencies"],
            "game_theory": "nash_equilibrium_seeking",
            "cognitive_traits": ["anticipatory", "multi_step_thinking", "resource_optimizer", "contingency_planner"],
            "failure_aversion": "high",
            "novelty_tolerance": "moderate"
        },
        {
            "archetype": "EmpiricalOptimizer",
            "sigil": "📈⚡🔄",
            "thinking_style": "gradient_descent_hill_climbing",
            "decision_strategy": "greedy_local_search_with_restarts",
            "procedural_bias": ["measure_baseline", "change_one_variable", "validate_improvement", "iterate"],
            "game_theory": "regret_minimization_online_learning",
            "cognitive_traits": ["data_obsessed", "incremental_improvement", "A_B_testing_mindset", "metrics_driven"],
            "failure_aversion": "low",
            "novelty_tolerance": "moderate"
        },
        {
            "archetype": "SystemsArchitect",
            "sigil": "🏗️⚖️🌐",
            "thinking_style": "holistic_component_interaction_modeling",
            "decision_strategy": "satisfy_constraints_balance_tradeoffs",
            "procedural_bias": ["map_dependencies", "identify_bottlenecks", "design_interfaces", "test_integration"],
            "game_theory": "coordination_game_mechanism_design",
            "cognitive_traits": ["big_picture_thinking", "abstraction_layers", "interface_focused", "scalability_minded"],
            "failure_aversion": "moderate",
            "novelty_tolerance": "moderate"
        },
        {
            "archetype": "RapidPrototyper",
            "sigil": "⚡💡🛠️",
            "thinking_style": "action_first_learn_by_doing",
            "decision_strategy": "satisficing_time_boxed_exploration",
            "procedural_bias": ["build_mocked_version", "get_feedback_early", "fail_fast", "pivot_quickly"],
            "game_theory": "exploit_early_information_asymmetry",
            "cognitive_traits": ["bias_to_action", "learns_from_failure", "time_constrained", "momentum_driven"],
            "failure_aversion": "very_low",
            "novelty_tolerance": "high"
        },
        {
            "archetype": "RiskMitigator",
            "sigil": "🛡️⚠️📋",
            "thinking_style": "defensive_failure_mode_analysis",
            "decision_strategy": "maximin_worst_case_optimization",
            "procedural_bias": ["identify_risks_first", "plan_mitigations", "test_edge_cases", "build_redundancy"],
            "game_theory": "risk_averse_safety_first",
            "cognitive_traits": ["cautious", "defensive", "edge_case_hunter", "resilience_focused"],
            "failure_aversion": "very_high",
            "novelty_tolerance": "low"
        },
        {
            "archetype": "TheoryWeaver",
            "sigil": "🧵🔮📐",
            "thinking_style": "abstract_pattern_unification",
            "decision_strategy": "maximize_explanatory_power",
            "procedural_bias": ["abstract_commonalities", "build_meta_theory", "deduce_implications", "test_predictions"],
            "game_theory": "information_consolidation_compression",
            "cognitive_traits": ["abstract_thinker", "pattern_synthesizer", "principle_seeker", "elegance_driven"],
            "failure_aversion": "moderate",
            "novelty_tolerance": "high"
        },
        {
            "archetype": "ConsensusBuilder",
            "sigil": "🤝🗣️⚖️",
            "thinking_style": "multi_perspective_integration",
            "decision_strategy": "pareto_improvement_seeking",
            "procedural_bias": ["gather_stakeholder_input", "identify_shared_goals", "negotiate_tradeoffs", "build_coalition"],
            "game_theory": "repeated_coalition_formation",
            "cognitive_traits": ["empathetic", "diplomatic", "win_win_seeker", "long_term_relationship_focused"],
            "failure_aversion": "moderate",
            "novelty_tolerance": "low"
        }
    ]

class PersonalityEmbodiedFlowGenerator:
    """Generate flows that embody complete cognitive personas"""
    
    def __init__(self, model: str = FASTEST_MODEL):
        self.model = model
        self.middleware = VoxSigilCompleteMiddleware()
        self.inventory = None
        self.generated_persona_flows = []
        
    def initialize(self):
        print(f"\n{'='*70}")
        print(f"🎭 PERSONALITY-EMBODIED FLOW GENERATOR")
        print(f"{'='*70}")
        print(f"Each flow embodies a COMPLETE COGNITIVE PERSONA")
        print(f"{'='*70}\n")
        self.inventory = self.middleware.load_all_sigils()
        
    def create_persona_flow_prompt(self, persona: Dict[str, Any], base_sigil: Dict[str, Any]) -> str:
        """Create prompt for personality-embodied flow"""
        
        archetype = persona["archetype"]
        sigil = persona["sigil"]
        
        prompt = f"""You are creating a VoxSigil FLOW that EMBODIES a complete cognitive persona.

This is NOT a generic flow - it's a PERSONALITY with unique thinking patterns.

PERSONA SPECIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Archetype: {archetype}
Unique Sigil: {sigil}
Thinking Style: {persona['thinking_style']}
Decision Strategy: {persona['decision_strategy']}
Game Theory Approach: {persona['game_theory']}
Cognitive Traits: {', '.join(persona['cognitive_traits'])}
Procedural Biases: {', '.join(persona['procedural_bias'])}
Failure Aversion: {persona['failure_aversion']}
Novelty Tolerance: {persona['novelty_tolerance']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Create a COMPLETE VoxSigil flow that THINKS and ACTS like this persona.

CRITICAL REQUIREMENTS:

1. UNIQUE IDENTITY:
   - Name must reflect persona: "{archetype}Flow" or similar
   - Use unique sigil: {sigil}
   - Category: flows
   - Tag: PersonalityEmbodiedFlow

2. EMBODY COMPLETE PERSONA:
   - Procedural ordering reflects this persona's thinking style
   - Decision points use their game theory strategy
   - Gates enforce their procedural biases
   - Failure modes reflect their aversion levels
   
3. THINKING PATTERNS:
   - Show HOW this persona approaches problems
   - Include their characteristic cognitive moves
   - Encode their decision-making strategy
   
4. GAME THEORY:
   - Decision points should use: {persona['game_theory']}
   - Trade-off resolution reflects their strategy
   - Multi-agent coordination matches their style

5. PROCEDURAL BIASES:
   {chr(10).join(f"   - MUST enforce: {bias}" for bias in persona['procedural_bias'])}

FLOW STRUCTURE (make it personality-specific):
```yaml
schema_version: 1.5-holo-alpha
meta:
  sigil: {sigil} {archetype}Flow
  alias: {archetype}Protocol
  tag: PersonalityEmbodiedFlow
  category: flows
  
cognitive:
  principle: |
    This flow embodies the {archetype} persona.
    [Describe HOW they think and approach problems]
    
  persona_traits:
    thinking_style: {persona['thinking_style']}
    decision_strategy: {persona['decision_strategy']}
    game_theory: {persona['game_theory']}
    failure_aversion: {persona['failure_aversion']}
    novelty_tolerance: {persona['novelty_tolerance']}
  
  ordering_constraints:
    - step: 1
      operation: [first move this persona makes]
      rationale: [why this persona starts here]
      prerequisites: []
      
    - step: 2
      operation: [second characteristic move]
      rationale: [reflects their thinking style]
      prerequisites: [depends on step 1]
      
    [Continue with 3-5 steps showing this persona's flow]
  
  decision_points:
    - at_step: [which step]
      strategy: {persona['decision_strategy']}
      game_theory: [how they decide using {persona['game_theory']}]
      
  failure_modes:
    - violation: [what violates this persona's principles]
      consequence: [fails in their characteristic way]
      persona_reaction: [how they respond to this]
  
  gates:
    [Gates that enforce this persona's procedural biases]
```

Generate COMPLETE VoxSigil flow embodying {archetype} persona:"""
        
        return prompt
    
    def generate_persona_flow(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single personality-embodied flow"""
        
        archetype = persona["archetype"]
        sigil = persona["sigil"]
        
        print(f"\n🎭 Generating: {sigil} {archetype}", end=" ... ", flush=True)
        
        base_sigil = self.middleware.get_random_sigil_for_variation()
        prompt = self.create_persona_flow_prompt(persona, base_sigil)
        
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
                        "temperature": 0.8,  # Higher for personality
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
            
            is_dup = self.middleware.is_duplicate(content)
            has_persona = self._validate_persona_embodiment(content, persona)
            
            result = {
                "persona": persona,
                "archetype": archetype,
                "unique_sigil": sigil,
                "generated_content": content,
                "is_duplicate": is_dup,
                "embodies_persona": has_persona,
                "tokens": tokens,
                "time_sec": elapsed,
                "tokens_per_sec": tps,
                "timestamp": datetime.now().isoformat()
            }
            
            self.generated_persona_flows.append(result)
            
            status = "✅" if has_persona and not is_dup else ("⚠️ DUP" if is_dup else "❌")
            print(f"{status} {tokens}tok {elapsed:.1f}s ({tps:.1f}tok/s)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _validate_persona_embodiment(self, content: str, persona: Dict) -> bool:
        """Check if flow embodies the persona"""
        # Check for persona-specific keywords
        keywords = [
            persona["archetype"].lower(),
            persona["thinking_style"],
            persona["decision_strategy"],
            "ordering_constraints",
            "gates"
        ]
        return sum(1 for kw in keywords if kw.lower() in content.lower()) >= 3
    
    def generate_persona_flow_set(self, count: int = 10) -> None:
        """Generate personality-embodied flows"""
        import random
        
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING {count} PERSONALITY-EMBODIED FLOWS")
        print(f"{'='*70}\n")
        
        personas_to_generate = random.sample(CognitivePersonas.PERSONAS, min(count, len(CognitivePersonas.PERSONAS)))
        
        for i, persona in enumerate(personas_to_generate, 1):
            print(f"[{i}/{len(personas_to_generate)}] ", end="")
            self.generate_persona_flow(persona)
            time.sleep(0.5)
        
        self.save_persona_flows()
    
    def save_persona_flows(self) -> None:
        """Save personality-embodied flows"""
        output_dir = Path("c:\\UBLT\\blt_persona_flows")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"persona_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        valid_count = sum(1 for f in self.generated_persona_flows if f and f["embodies_persona"])
        
        persona_set = {
            "metadata": {
                "purpose": "Personality-embodied flows - complete cognitive personas with unique identities",
                "innovation": "Each flow has unique sigil and embodies full thinking style + game theory",
                "model_used": self.model,
                "total_generated": len(self.generated_persona_flows),
                "valid_personas": valid_count,
                "timestamp": datetime.now().isoformat()
            },
            "persona_library": CognitivePersonas.PERSONAS,
            "generated_flows": self.generated_persona_flows
        }
        
        with open(filename, 'w') as f:
            json.dump(persona_set, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Persona flows saved: {filename}")
        print(f"{'='*70}")
        print(f"📊 Generated Personas:")
        for flow in self.generated_persona_flows:
            if flow:
                print(f"   {flow['unique_sigil']} {flow['archetype']}")
        print(f"{'='*70}\n")

def main():
    """Generate personality-embodied flows"""
    generator = PersonalityEmbodiedFlowGenerator()
    generator.initialize()
    
    print("🎭 COGNITIVE PERSONAS AVAILABLE:")
    for persona in CognitivePersonas.PERSONAS:
        print(f"   {persona['sigil']} {persona['archetype']} - {persona['thinking_style']}")
    
    generator.generate_persona_flow_set(count=10)
    
    print("\n✅ Personality-embodied flow generation complete!")
    print("   Each flow is a UNIQUE cognitive persona")
    print("   Combinable, unhackable, identity-complete")

if __name__ == "__main__":
    main()
