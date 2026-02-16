"""
VoxSigil Meta-Cognitive Architecture Generator
Focuses on COGNITIVE STRUCTURES and THINKING ORGANIZATION PATTERNS
VoxSigil = Meta-language describing how minds organize and process thought
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

class CognitiveArchitectureDimensions:
    """Meta-cognitive structures - HOW thoughts are organized"""
    
    REASONING_STRUCTURES = [
        "hierarchical_decomposition",      # Top-down breakdown
        "associative_network",             # Web of connections
        "sequential_chain",                # Linear logical flow
        "parallel_synthesis",              # Multiple streams merging
        "recursive_refinement",            # Self-improving loops
        "dialectic_opposition",            # Thesis-antithesis-synthesis
        "analogical_mapping",              # Pattern transfer across domains
        "categorical_classification",      # Taxonomic organization
        "probabilistic_inference",         # Bayesian reasoning
        "narrative_construction",          # Story-based understanding
        "symbolic_abstraction",            # High-level concept manipulation
        "embodied_simulation"              # Grounded sensorimotor thinking
    ]
    
    CONCEPTUAL_ORGANIZATION = [
        "radial_clustering",               # Hub-and-spoke concepts
        "lattice_hierarchy",               # Multi-path taxonomies
        "semantic_web",                    # Meaning-based links
        "temporal_sequence",               # Time-ordered knowledge
        "spatial_topology",                # Location-based mental maps
        "causal_network",                  # Cause-effect chains
        "functional_grouping",             # Purpose-based categories
        "component_assembly",              # Part-whole structures
        "dimensional_scaling",             # Spectrum-based organization
        "fractal_nesting"                  # Self-similar recursive patterns
    ]
    
    INFORMATION_PROCESSING = [
        "bottom_up_aggregation",           # Data-driven synthesis
        "top_down_prediction",             # Model-driven expectations
        "bidirectional_resonance",         # Feedback loops
        "attentional_gating",              # Selective focus filtering
        "working_memory_buffer",           # Transient active state
        "episodic_encoding",               # Experience recording
        "semantic_integration",            # Meaning consolidation
        "procedural_automation",           # Skill compilation
        "pattern_completion",              # Filling gaps from partial input
        "anomaly_detection",               # Novelty flagging
        "compression_abstraction",         # Lossy summarization
        "entropy_minimization"             # Predictive reduction
    ]
    
    COGNITIVE_MESH_PATTERNS = [
        "centralized_hub",                 # Single integration point
        "distributed_consensus",           # Parallel agreement
        "layered_hierarchy",               # Stacked processing levels
        "lateral_inhibition",              # Competitive mutual suppression
        "resonant_amplification",          # Mutually reinforcing
        "modular_encapsulation",           # Isolated subsystems
        "holographic_redundancy",          # Information distributed everywhere
        "sparse_activation",               # Minimal efficient patterns
        "dense_connectivity",              # Highly interconnected
        "dynamic_routing"                  # Flexible path selection
    ]
    
    METACOGNITIVE_CONTROL = [
        "reflective_monitoring",           # Watching own thinking
        "strategic_planning",              # Multi-step goal setting
        "error_correction",                # Self-debugging
        "uncertainty_quantification",      # Confidence tracking
        "resource_allocation",             # Cognitive budget management
        "context_switching",               # Task transition handling
        "hypothesis_generation",           # Possibility exploration
        "validation_checking",             # Truth verification
        "abstraction_climbing",            # Moving up conceptual levels
        "instantiation_descent"            # Moving to concrete examples
    ]
    
    COGNITIVE_TOOLS = [
        # Analysis Tools
        "pattern_extraction",              # Find regularities in data
        "anomaly_detection",               # Identify outliers
        "causal_inference",                # Determine cause-effect
        "statistical_aggregation",         # Summarize distributions
        "dimensional_reduction",           # Compress complexity
        
        # Synthesis Tools
        "conceptual_blending",             # Merge disparate ideas
        "analogy_construction",            # Build cross-domain mappings
        "narrative_generation",            # Create explanatory stories
        "hypothesis_formation",            # Generate testable predictions
        "solution_composition",            # Combine partial solutions
        
        # Transformation Tools
        "symbolic_manipulation",           # Rewrite expressions
        "perspective_shifting",            # View from different angles
        "abstraction_extraction",          # Pull out essence
        "concrete_instantiation",          # Generate specific examples
        "reformulation",                   # Restate in different terms
        
        # Evaluation Tools
        "consistency_checking",            # Verify logical coherence
        "confidence_estimation",           # Quantify uncertainty
        "utility_assessment",              # Evaluate value/cost
        "feasibility_analysis",            # Check if doable
        "impact_prediction",               # Forecast consequences
        
        # Interface Tools
        "query_parsing",                   # Understand requests
        "result_formatting",               # Structure outputs
        "context_retrieval",               # Pull relevant background
        "memory_encoding",                 # Store experiences
        "explanation_generation"           # Justify reasoning
    ]
    
    COGNITIVE_SKILLS = [
        # Learned Procedures
        "systematic_debugging",            # Step-by-step error finding
        "incremental_refinement",          # Iterative improvement
        "constraint_satisfaction",         # Meet multiple requirements
        "resource_optimization",           # Maximize efficiency
        "conflict_resolution",             # Reconcile contradictions
        
        # Domain Expertise
        "mathematical_reasoning",          # Quantitative analysis
        "linguistic_processing",           # Language understanding
        "spatial_reasoning",               # Geometric manipulation
        "temporal_reasoning",              # Time-based inference
        "social_modeling",                 # Theory of mind
        
        # Meta-Skills
        "learning_from_examples",          # Few-shot adaptation
        "skill_composition",               # Combine techniques
        "transfer_application",            # Apply to new domains
        "strategy_selection",              # Pick appropriate method
        "performance_monitoring"           # Track success metrics
    ]
    
    AFFORDANCES = [
        # What can be invoked/activated
        "read_access",                     # Can retrieve information
        "write_access",                    # Can modify state
        "compute_access",                  # Can run calculations
        "memory_access",                   # Can store/recall
        "communication_access",            # Can send/receive messages
        "planning_access",                 # Can simulate futures
        "reasoning_access",                # Can draw inferences
        "learning_access",                 # Can update from experience
        "introspection_access",            # Can examine self
        "control_access"                   # Can direct processes
    ]
    
    PROCEDURAL_ORDERING = [
        # Logical sequence constraints - WHAT COMES BEFORE WHAT
        
        # Development Flow
        "design_before_implementation",    # Plan before coding
        "test_before_documentation",       # Verify before documenting
        "prototype_before_optimization",   # Make it work before making it fast
        "validation_before_deployment",    # Test before release
        "requirements_before_architecture", # Understand need before design
        
        # Analysis Flow  
        "observe_before_hypothesize",      # Data before theory
        "measure_before_conclude",         # Evidence before claims
        "understand_before_critique",      # Comprehend before judging
        "explore_before_exploit",          # Learn before executing
        "question_before_answer",          # Clarify before solving
        
        # Execution Flow
        "prepare_before_execute",          # Setup before action
        "verify_prerequisites_first",      # Check dependencies
        "atomic_before_composite",         # Simple before complex
        "sequential_before_parallel",      # Single-threaded before concurrent
        "local_before_distributed",        # Work locally before deploying
        
        # Learning Flow
        "experience_before_generalization", # Specific cases before rules
        "practice_before_teaching",        # Master before instructing
        "feedback_before_adaptation",      # Observe results before changing
        "error_detection_before_correction", # Find bug before fixing
        "foundation_before_specialization", # Basics before advanced
        
        # Communication Flow
        "listen_before_respond",           # Understand input before output
        "clarify_before_commit",           # Confirm before promising
        "example_before_abstraction",      # Concrete before abstract
        "context_before_content",          # Frame before details
        "acknowledge_before_redirect",     # Show understanding before changing topic
        
        # Decision Flow
        "gather_before_decide",            # Collect info before choosing
        "evaluate_before_commit",          # Assess options before locking in
        "simulate_before_execute",         # Model consequences before acting
        "reversible_before_irreversible",  # Try safe options first
        "consensus_before_controversial"   # Easy agreements before hard ones
    ]
    
    DEPENDENCY_CONSTRAINTS = [
        # Hard prerequisites - X REQUIRES Y to exist first
        "input_requires_source",           # Can't process without data
        "output_requires_computation",     # Can't return without calculating
        "validation_requires_specification", # Can't test without requirements
        "optimization_requires_baseline",  # Can't improve without measuring
        "comparison_requires_alternatives", # Can't choose with one option
        "integration_requires_components", # Can't combine without parts
        "iteration_requires_initial",      # Can't refine nothing
        "correction_requires_error",       # Can't fix what's not broken
        "explanation_requires_understanding", # Can't teach what you don't know
        "generalization_requires_examples" # Can't abstract without instances
    ]
    
    COGNITIVE_HYGIENE = [
        # Practices for maintaining logical coherence
        "validate_assumptions_early",      # Check premises before proceeding
        "fail_fast_on_blockers",          # Stop if prerequisites missing
        "checkpoint_before_branching",     # Save state before exploring
        "backtrack_on_contradiction",      # Undo when inconsistent
        "defer_premature_optimization",    # Don't optimize too early
        "isolate_before_integrate",        # Test parts separately first
        "version_control_changes",         # Track what changed when
        "document_after_stability",        # Only describe what's settled
        "refactor_incrementally",          # Small changes with validation
        "maintain_invariants"              # Preserve essential properties
    ]

class VoxSigilMetaCognitiveGenerator:
    """Generate VoxSigils as META-LANGUAGE for cognitive structures"""
    
    def __init__(self, model: str = FASTEST_MODEL):
        self.model = model
        self.middleware = VoxSigilCompleteMiddleware()
        self.inventory = None
        self.generated_structures = []
        
    def initialize(self):
        """Load all sigils and prepare for cognitive structure generation"""
        print(f"\n{'='*70}")
        print(f"🧠 VOXSIGIL META-COGNITIVE ARCHITECTURE GENERATOR")
        print(f"{'='*70}")
        print(f"Purpose: Generate meta-language descriptions of THINKING STRUCTURES")
        print(f"Focus: HOW thoughts are organized, not behavioral outputs")
        print(f"{'='*70}\n")
        self.inventory = self.middleware.load_all_sigils()
        
    def create_cognitive_architecture_prompt(self,
                                            base_sigil: Dict[str, Any],
                                            cognitive_structure: Dict[str, Any]) -> str:
        """Create prompt for generating cognitive architecture variation"""
        
        name = base_sigil["name"]
        category = base_sigil["category"]
        reference = base_sigil["raw_content"][:1200]
        
        prompt = f"""You are creating a VoxSigil as a META-LANGUAGE description of a COGNITIVE ARCHITECTURE.

VoxSigil encodes HOW a mind ORGANIZES and PROCESSES thoughts - the STRUCTURE of thinking itself.

BASE REFERENCE SIGIL:
Name: {name}
Category: {category}

{reference}

COGNITIVE ARCHITECTURE SPECIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reasoning Structure: {cognitive_structure['reasoning_structure']}
  → How logical inferences flow and connect

Conceptual Organization: {cognitive_structure['conceptual_organization']}
  → How knowledge is categorized and linked

Information Processing: {cognitive_structure['information_processing']}
  → How inputs are transformed and integrated

Cognitive Mesh: {cognitive_structure['mesh_pattern']}
  → How cognitive modules interconnect

Metacognitive Control: {cognitive_structure['metacognitive_control']}
  → How the system monitors and directs itself

COGNITIVE TOOLS & CAPABILITIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Available Tools: {', '.join(cognitive_structure['cognitive_tools'])}
  → What operations this cognitive architecture can perform

Skills: {', '.join(cognitive_structure['cognitive_skills'])}
  → Learned procedures and domain expertise

Affordances: {', '.join(cognitive_structure['affordances'])}
  → What interfaces/actions are available

PROCEDURAL ORDERING & DEPENDENCIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ordering Rules: {', '.join(cognitive_structure['procedural_ordering'])}
  → What must come BEFORE what (logical sequencing)

Dependencies: {', '.join(cognitive_structure['dependencies'])}
  → Hard prerequisites - what REQUIRES what

Cognitive Hygiene: {', '.join(cognitive_structure['cognitive_hygiene'])}
  → Practices for maintaining logical coherence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Create a NEW VoxSigil encoding this cognitive architecture WITH tools AND ordering constraints.

CRITICAL REQUIREMENTS:
1. VoxSigil is a META-LANGUAGE - describe STRUCTURE + FUNCTIONAL API + SEQUENCING RULES
2. Include 'implementation' section with tools/skills AND their ordering constraints
3. Specify WHAT operations are available AND in what LOGICAL ORDER they execute
4. The 'cognitive' section must encode PROCEDURAL ORDERING - what comes before what
5. Include dependency constraints - prerequisites that must be satisfied
6. Include cognitive hygiene practices - how to maintain logical flow
7. Use DIFFERENT name than '{name}' - suggest name reflecting cognitive structure + ordering
8. Make it OPERATIONALLY DISTINCT - different sequence of operations

Think of this as the OPERATING SYSTEM + API + WORKFLOW of a mind:
- What TOOLS/FUNCTIONS are available?
- What MUST happen BEFORE other things can happen?
- What are the DEPENDENCY CHAINS?
- What ORDERING RULES prevent illogical sequences?
- How does it maintain LOGICAL HYGIENE (e.g., test before documenting)?

Example BAD ordering: Write documentation before testing code
Example GOOD ordering: Test → Verify → THEN document

This VoxSigil should enforce LOGICAL SEQUENCING to prevent cognitive mistakes.

Generate complete VoxSigil YAML encoding this cognitive architecture:"""
        
        return prompt
    
    def generate_cognitive_structure(self,
                                    base_sigil: Dict[str, Any],
                                    cognitive_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate single cognitive architecture variation"""
        
        reasoning = cognitive_structure['reasoning_structure'].replace('_', ' ').title()
        mesh = cognitive_structure['mesh_pattern'].replace('_', ' ').title()
        
        print(f"\n🏗️  Generating: {reasoning} + {mesh}", end=" ... ", flush=True)
        
        prompt = self.create_cognitive_architecture_prompt(base_sigil, cognitive_structure)
        
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
                        "temperature": 0.75,
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
            
            # Validation
            is_dup = self.middleware.is_duplicate(content)
            is_valid = self._validate_cognitive_structure(content)
            
            result = {
                "base_sigil": base_sigil["name"],
                "base_category": base_sigil["category"],
                "cognitive_architecture": cognitive_structure,
                "generated_content": content,
                "is_duplicate": is_dup,
                "is_valid_structure": is_valid,
                "tokens": tokens,
                "time_sec": elapsed,
                "tokens_per_sec": tps,
                "timestamp": datetime.now().isoformat()
            }
            
            self.generated_structures.append(result)
            
            status = "✅" if is_valid and not is_dup else ("⚠️ DUP" if is_dup else "❌")
            print(f"{status} {tokens}tok {elapsed:.1f}s ({tps:.1f}tok/s)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _validate_cognitive_structure(self, content: str) -> bool:
        """Validate that sigil describes cognitive structure"""
        # Check for cognitive architecture markers
        structure_markers = [
            "schema_version", "meta", "cognitive", "holo_mesh",
            "process", "flow", "structure", "pattern"
        ]
        return sum(1 for marker in structure_markers if marker in content.lower()) >= 4
    
    def generate_cognitive_training_set(self, count: int = 15) -> None:
        """Generate diverse cognitive architecture training set"""
        import random
        
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING {count} COGNITIVE ARCHITECTURE VARIATIONS")
        print(f"{'='*70}")
        print(f"Each VoxSigil encodes a unique way of ORGANIZING THOUGHT\n")
        
        for i in range(count):
            print(f"[{i+1}/{count}] ", end="")
            
            # Pick random base sigil
            base_sigil = self.middleware.get_random_sigil_for_variation()
            
            # Define unique cognitive architecture with ordering constraints
            cognitive_structure = {
                "reasoning_structure": random.choice(CognitiveArchitectureDimensions.REASONING_STRUCTURES),
                "conceptual_organization": random.choice(CognitiveArchitectureDimensions.CONCEPTUAL_ORGANIZATION),
                "information_processing": random.choice(CognitiveArchitectureDimensions.INFORMATION_PROCESSING),
                "mesh_pattern": random.choice(CognitiveArchitectureDimensions.COGNITIVE_MESH_PATTERNS),
                "metacognitive_control": random.choice(CognitiveArchitectureDimensions.METACOGNITIVE_CONTROL),
                "cognitive_tools": random.sample(CognitiveArchitectureDimensions.COGNITIVE_TOOLS, 3),
                "cognitive_skills": random.sample(CognitiveArchitectureDimensions.COGNITIVE_SKILLS, 2),
                "affordances": random.sample(CognitiveArchitectureDimensions.AFFORDANCES, 3),
                "procedural_ordering": random.sample(CognitiveArchitectureDimensions.PROCEDURAL_ORDERING, 3),
                "dependencies": random.sample(CognitiveArchitectureDimensions.DEPENDENCY_CONSTRAINTS, 2),
                "cognitive_hygiene": random.sample(CognitiveArchitectureDimensions.COGNITIVE_HYGIENE, 2)
            }
            
            self.generate_cognitive_structure(base_sigil, cognitive_structure)
            
            # Small delay for Ollama
            time.sleep(0.5)
        
        self.save_training_set()
    
    def save_training_set(self) -> None:
        """Save cognitive architecture training set"""
        output_dir = Path("c:\\UBLT\\blt_cognitive_architecture_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"cognitive_architectures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Statistics
        valid_count = sum(1 for s in self.generated_structures if s and s["is_valid_structure"])
        duplicate_count = sum(1 for s in self.generated_structures if s and s["is_duplicate"])
        avg_tokens = sum(s["tokens"] for s in self.generated_structures if s) / len(self.generated_structures)
        avg_tps = sum(s["tokens_per_sec"] for s in self.generated_structures if s) / len(self.generated_structures)
        
        training_set = {
            "metadata": {
                "purpose": "Meta-language encoding cognitive architectures WITH tools AND logical ordering constraints for BLT training",
                "key_innovation": "Encodes PROCEDURAL ORDERING - what must come before what (e.g., test before document)",
                "model_used": self.model,
                "base_sigils_count": self.inventory.total_sigils,
                "generated_count": len(self.generated_structures),
                "valid_count": valid_count,
                "duplicate_count": duplicate_count,
                "timestamp": datetime.now().isoformat()
            },
            "cognitive_dimensions": {
                "reasoning_structures": CognitiveArchitectureDimensions.REASONING_STRUCTURES,
                "conceptual_organization": CognitiveArchitectureDimensions.CONCEPTUAL_ORGANIZATION,
                "information_processing": CognitiveArchitectureDimensions.INFORMATION_PROCESSING,
                "mesh_patterns": CognitiveArchitectureDimensions.COGNITIVE_MESH_PATTERNS,
                "metacognitive_control": CognitiveArchitectureDimensions.METACOGNITIVE_CONTROL,
                "cognitive_tools": CognitiveArchitectureDimensions.COGNITIVE_TOOLS,
                "cognitive_skills": CognitiveArchitectureDimensions.COGNITIVE_SKILLS,
                "affordances": CognitiveArchitectureDimensions.AFFORDANCES,
                "procedural_ordering": CognitiveArchitectureDimensions.PROCEDURAL_ORDERING,
                "dependency_constraints": CognitiveArchitectureDimensions.DEPENDENCY_CONSTRAINTS,
                "cognitive_hygiene": CognitiveArchitectureDimensions.COGNITIVE_HYGIENE
            },
            "generated_structures": self.generated_structures,
            "statistics": {
                "avg_tokens": avg_tokens,
                "avg_tokens_per_sec": avg_tps,
                "valid_percentage": (valid_count / len(self.generated_structures)) * 100,
                "duplicate_percentage": (duplicate_count / len(self.generated_structures)) * 100
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(training_set, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Cognitive architecture training set saved:")
        print(f"   {filename}")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Total generated: {len(self.generated_structures)}")
        print(f"   Valid structures: {valid_count} ({valid_count/len(self.generated_structures)*100:.1f}%)")
        print(f"   Duplicates: {duplicate_count}")
        print(f"   Avg tokens: {avg_tokens:.0f}")
        print(f"   Avg speed: {avg_tps:.1f} tok/s")
        print(f"{'='*70}\n")
        print(f"💡 These VoxSigils encode COGNITIVE STRUCTURES - meta-language")
        print(f"   describing how different minds organize and process thought.")

def main():
    """Generate meta-cognitive architecture training data"""
    generator = VoxSigilMetaCognitiveGenerator()
    generator.initialize()
    
    print("\n" + "="*70)
    print("📋 VOXSIGIL CATEGORIES:")
    print("="*70)
    print("  1. pglyph     - System core identity (recursive orchestration)")
    print("  2. tags       - Atomic cognitive primitives (focus, attention)")
    print("  3. scaffolds  - Composite frameworks (curriculum, learning)")
    print("  4. sigils     - Operational components (anomaly detection)")
    print("  5. flows      - Procedural sequences (test→document order)")
    print("="*70)
    print("\n🎯 GENERATING: Flow-based procedural ordering sigils")
    print("   Focus: Logical sequences, dependencies, gates\n")
    
    generator.generate_cognitive_training_set(count=15)
    
    print("\n✅ Meta-cognitive architecture generation complete!")
    print("   VoxSigils generated as meta-language for thought organization")
    print("   INCLUDING procedural flows with ordering constraints")
    print("   Ready for BLT training on cognitive diversity")

if __name__ == "__main__":
    main()
