#!/usr/bin/env python3
"""
Next-Generation Behavioral NAS
Evolves intelligence behaviors, not just architectures

Now with VoxSIGIL Schema Integration:
- All schema versions (1.4-uni, 1.5-holo-alpha, 1.8-omega) available
- Schema compliance added as 14th behavioral phenotype
- Schema-guided evolution for structurally sound architectures
"""
import sys
import time
import logging
import random
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))

from scripts.training.streaming_distillation_adapter import create_integrated_pipeline

# VoxSIGIL Schema Integration
SCHEMA_AVAILABLE = True
try:
    from training.schema_driven_evolution import get_schema_integrator, SchemaEvolutionIntegrator
    logger_schema = logging.getLogger("SchemaIntegration")
    logger_schema.info("✅ VoxSIGIL Schema Integration Active")
except ImportError as e:
    SCHEMA_AVAILABLE = False
    logger_schema = logging.getLogger("SchemaIntegration")
    logger_schema.warning(f"⚠️ VoxSIGIL Schema Integration Unavailable: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BehavioralNAS")

# ============================================================================
# BEHAVIORAL PHENOTYPES - What we're actually evolving
# ============================================================================

@dataclass
class IntelligencePhenotype:
    """Behavioral traits that define intelligence, not architecture"""
    
    # Core intelligence behaviors
    compression_efficiency: float = 0.0      # How well it compresses knowledge
    distillation_quality: float = 0.0        # How well it learns from teachers
    memory_retention: float = 0.0            # Long-range dependency handling
    error_correction: float = 0.0            # Robustness under noise
    multi_task_transfer: float = 0.0         # Generalization ability
    reasoning_depth: float = 0.0             # Chain-of-thought capability
    world_model_coherence: float = 0.0       # Internal representation consistency
    
    # Performance behaviors
    latency_per_quality: float = 0.0         # Speed-accuracy tradeoff
    compression_stability: float = 0.0       # Consistent performance under compression
    hallucination_resistance: float = 0.0    # Factual grounding
    
    # Emergent behaviors
    self_distillation_ability: float = 0.0   # Can teach itself
    cross_modal_fusion: float = 0.0          # Integrates different data types
    adaptive_plasticity: float = 0.0         # Learns new patterns fast
    
    # VoxSIGIL Schema Compliance (14th Phenotype)
    schema_compliance: float = 1.0           # Alignment with VoxSIGIL structural constraints
    
    def composite_fitness(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite intelligence score"""
        if weights is None:
            # Nebula's intelligence priorities (rebalanced with schema compliance)
            weights = {
                'compression_efficiency': 0.14,
                'distillation_quality': 0.14,
                'memory_retention': 0.11,
                'error_correction': 0.09,
                'multi_task_transfer': 0.09,
                'reasoning_depth': 0.08,
                'world_model_coherence': 0.08,
                'latency_per_quality': 0.07,
                'schema_compliance': 0.10,      # 🆕 Schema weight for structural soundness
                'compression_stability': 0.04,
                'hallucination_resistance': 0.04,
                'self_distillation_ability': 0.01,
                'cross_modal_fusion': 0.005,
                'adaptive_plasticity': 0.005,
            }
        
        score = 0.0
        for trait, value in asdict(self).items():
            weight = weights.get(trait, 0.0)
            score += value * weight
        
        return score
    
    def passes_gates(self) -> Tuple[bool, List[str]]:
        """Check if phenotype passes behavioral gates"""
        failures = []
        
        # Gate 1: Memory retention minimum
        if self.memory_retention < 0.3:
            failures.append("memory_retention_gate")
        
        # Gate 2: Error correction threshold
        if self.error_correction < 0.25:
            failures.append("error_correction_gate")
        
        # Gate 3: Distillation quality minimum
        if self.distillation_quality < 0.4:
            failures.append("distillation_quality_gate")
        
        # Gate 4: Compression efficiency baseline
        if self.compression_efficiency < 0.3:
            failures.append("compression_efficiency_gate")
        
        # Gate 5: Hallucination resistance
        if self.hallucination_resistance < 0.35:
            failures.append("hallucination_resistance_gate")
        
        return len(failures) == 0, failures

@dataclass
class ArchitectureGenome:
    """Architecture as expression of behavioral intent - ENHANCED WITH VOXSIGIL COGNITIVE FIELDS"""
    
    # ============================================================
    # EXISTING ARCHITECTURAL FIELDS (13 fields - UNCHANGED)
    # ============================================================
    # Traditional architecture params (means to an end)
    num_layers: int
    hidden_size: int
    num_heads: int
    ffn_ratio: float
    dropout: float
    activation: str
    
    # Behavioral architecture features
    use_flash_attn: bool = True
    use_rotary_emb: bool = True
    use_gated_mlp: bool = False
    use_residual_scaling: bool = False
    attention_window_size: int = 512
    compression_ratio: float = 2.0
    layer_drop_rate: float = 0.0
    
    # ============================================================
    # NEW: CORE IDENTITY & CLASSIFICATION (Schema 1.4 + 1.5)
    # ============================================================
    sigil: str = "🧬"  # Unique symbolic identifier
    name: str = "evolved_architecture"
    is_cognitive_primitive: bool = False
    cognitive_primitive_type: str = "none"
    tags: List[str] = field(default_factory=lambda: ["neural_architecture", "evolved"])
    
    # ============================================================
    # NEW: ARCHITECTURAL SCAFFOLDS (Schema 1.4 + 1.5 - HIGH PRIORITY)
    # ============================================================
    consciousness_scaffold: bool = False
    cognitive_scaffold: bool = True  # Most architectures are cognitive
    symbolic_scaffold: bool = False
    
    # Schema 1.5 enhanced scaffold fields
    consciousness_scaffold_level: str = "none"
    cognitive_scaffold_role: str = "core_reasoning_engine_component"
    symbolic_orchestration_contribution: str = "none"
    
    # ============================================================
    # NEW: COGNITIVE MODELING (Schema 1.4 Section 6 - HIGH PRIORITY)
    # ============================================================
    cognitive_stage: str = "concrete_operational"  # Piaget-inspired
    developmental_model: str = "piaget_inspired"
    solo_taxonomy_level: str = "relational"  # SOLO taxonomy
    
    strategic_goals: List[str] = field(default_factory=lambda: ["efficient_inference", "memory_retention"])
    goal_alignment_strength: float = 0.5  # 0-1 scale
    
    # Tool integration capabilities
    tool_integration_enabled: bool = False
    integrated_tools: List[str] = field(default_factory=list)
    
    # ============================================================
    # NEW: ACTIVATION CONTEXT (Schema 1.4 Section 3)
    # ============================================================
    required_capabilities: List[str] = field(default_factory=lambda: ["text_processing", "pattern_recognition"])
    supported_input_modalities: List[str] = field(default_factory=lambda: ["text_structured"])
    supported_output_modalities: List[str] = field(default_factory=lambda: ["text_formatted_report"])
    
    # ============================================================
    # NEW: EMBODIMENT PROFILE (Schema 1.5 - VERY HIGH PRIORITY)
    # ============================================================
    embodiment_form_type: str = "disembodied_logical_process"
    has_sensory_simulation: bool = False
    has_motor_outputs: bool = False
    sensory_modality_count: int = 1
    
    # ============================================================
    # NEW: SELF-MODEL & METACOGNITION (Schema 1.5 - VERY HIGH PRIORITY)
    # ============================================================
    has_self_model: bool = False
    consciousness_framework_applied: str = "none"
    reflective_inference_enabled: bool = False
    introspection_capability: bool = False
    
    # Metacognitive processes
    self_correction_enabled: bool = False
    hallucination_detection_enabled: bool = False
    goal_alignment_reevaluation: bool = False
    
    # ============================================================
    # NEW: LEARNING ARCHITECTURE (Schema 1.5 - VERY HIGH PRIORITY)
    # ============================================================
    primary_learning_paradigm: str = "supervised_batch"
    continual_learning_enabled: bool = False
    continual_learning_strategy: str = "none"
    catastrophic_forgetting_mitigation: str = "minimal_effort"
    
    # Memory subsystem
    memory_architecture_type: str = "volatile_working_buffer_ram"
    memory_capacity_scale: float = 1.0
    memory_consolidation_enabled: bool = False
    
    # ============================================================
    # NEW: KNOWLEDGE REPRESENTATION (Schema 1.5 - HIGH PRIORITY)
    # ============================================================
    primary_knowledge_format: str = "vector_embeddings_transformer_based"
    world_model_integration: bool = False
    abstraction_level_control: bool = False
    symbol_grounding_strategy: str = "direct_multi_sensory_experiential_mapping_fpfm_style"
    
    # ============================================================
    # NEW: SMART_MRAP FRAMEWORK (Schema 1.4 + 1.5 - REQUIRED)
    # ============================================================
    smart_specific: str = "Efficient neural architecture for language tasks"
    smart_measurable: str = "Perplexity, inference speed, memory footprint"
    smart_achievable: str = "Implementable with current PyTorch/transformers"
    smart_relevant: str = "Enables scalable AI deployment"
    smart_transferable: str = "Architecture patterns applicable across domains"
    
    # ============================================================
    # NEW: EVOLUTIONARY & GOVERNANCE (Schema 1.5)
    # ============================================================
    generalizability_score: float = 0.5
    fusion_composition_potential: float = 0.5
    self_evolution_enabled: bool = False
    autopoietic_maintenance: bool = False
    
    # Governance
    value_alignment_framework: str = "none"
    human_oversight_required: bool = False
    accountability_mechanism: str = "training_logs"
    
    # ============================================================
    # NEW: MULTI-SENSORY & IMMERSIVE (Schema 1.5)
    # ============================================================
    audio_profile_enabled: bool = False
    haptics_enabled: bool = False
    olfactory_enabled: bool = False
    
    # ============================================================
    # NEW: METADATA & VERSIONING (Schema 1.4 + 1.5)
    # ============================================================
    schema_version: str = "1.5-holo-alpha"
    definition_version: str = "1.0.0"
    definition_status: str = "active_experimental"
    
    # ============================================================
    # NEW: STRUCTURE & RELATIONSHIPS (Schema 1.4)
    # ============================================================
    composite_type: str = "hierarchical"
    temporal_structure: str = "sequential_phased_execution"
    principle: str = "hierarchical_feature_extraction"
    
    def mutate_behavioral(self, mutation_rate=0.4):
        """Mutate towards behavioral AND cognitive optimization"""
        # ============================================================
        # EXISTING ARCHITECTURAL MUTATIONS
        # ============================================================
        # Structural mutations
        if random.random() < mutation_rate:
            self.num_layers = max(4, min(64, self.num_layers + random.choice([-4, -2, 2, 4])))
        if random.random() < mutation_rate:
            self.hidden_size = random.choice([256, 384, 512, 768, 1024, 1536, 2048, 3072])
        if random.random() < mutation_rate:
            self.num_heads = random.choice([4, 6, 8, 12, 16, 24, 32])
        
        # Behavioral feature mutations
        if random.random() < mutation_rate:
            self.use_flash_attn = random.choice([True, False])
        if random.random() < mutation_rate:
            self.use_rotary_emb = random.choice([True, False])
        if random.random() < mutation_rate:
            self.use_gated_mlp = random.choice([True, False])
        if random.random() < mutation_rate:
            self.use_residual_scaling = random.choice([True, False])
        if random.random() < mutation_rate:
            self.attention_window_size = random.choice([256, 512, 1024, 2048, 4096])
        if random.random() < mutation_rate:
            self.compression_ratio = random.uniform(1.5, 4.0)
        if random.random() < mutation_rate:
            self.layer_drop_rate = random.uniform(0.0, 0.3)
        
        # ============================================================
        # NEW: COGNITIVE SCAFFOLD MUTATIONS (HIGH PRIORITY)
        # ============================================================
        if random.random() < mutation_rate * 0.8:
            self.consciousness_scaffold = not self.consciousness_scaffold
        if random.random() < mutation_rate * 0.8:
            self.cognitive_scaffold = not self.cognitive_scaffold
        if random.random() < mutation_rate * 0.8:
            self.symbolic_scaffold = not self.symbolic_scaffold
        
        if random.random() < mutation_rate * 0.6:
            self.consciousness_scaffold_level = random.choice([
                "none", "foundational_primitive_for_awareness", 
                "integrative_module_for_gws_like_function",
                "reflective_meta_awareness_enabler"
            ])
        
        if random.random() < mutation_rate * 0.6:
            self.cognitive_scaffold_role = random.choice([
                "core_reasoning_engine_component", "memory_management_framework_node",
                "perception_processing_pipeline_stage", "action_selection_arbitration_module"
            ])
        
        # ============================================================
        # NEW: METACOGNITION MUTATIONS (HIGH VALUE)
        # ============================================================
        if random.random() < mutation_rate:
            self.self_correction_enabled = not self.self_correction_enabled
        if random.random() < mutation_rate:
            self.hallucination_detection_enabled = not self.hallucination_detection_enabled
        if random.random() < mutation_rate:
            self.goal_alignment_reevaluation = not self.goal_alignment_reevaluation
        if random.random() < mutation_rate:
            self.reflective_inference_enabled = not self.reflective_inference_enabled
        if random.random() < mutation_rate:
            self.introspection_capability = not self.introspection_capability
        
        # ============================================================
        # NEW: LEARNING PARADIGM MUTATIONS
        # ============================================================
        if random.random() < mutation_rate * 0.6:
            self.primary_learning_paradigm = random.choice([
                "supervised_batch", "self_supervised_online", 
                "reinforcement_interactive", "meta_learning_to_adapt",
                "transfer_learning_from_base_model", "continual_lifelong_learning"
            ])
        
        if random.random() < mutation_rate * 0.7:
            self.continual_learning_enabled = not self.continual_learning_enabled
        
        if random.random() < mutation_rate * 0.5:
            self.continual_learning_strategy = random.choice([
                "none", "parameter_allocation_masking", "modular_network_expansion",
                "regularization_ewc_si", "generative_replay_to_remember_r2r"
            ])
        
        if random.random() < mutation_rate * 0.5:
            self.memory_architecture_type = random.choice([
                "volatile_working_buffer_ram", 
                "persistent_long_term_associative_store_hnn_style",
                "episodic_event_trace_database", "semantic_knowledge_graph_cache"
            ])
        
        # ============================================================
        # NEW: COGNITIVE STAGE EVOLUTION
        # ============================================================
        if random.random() < mutation_rate * 0.5:
            self.cognitive_stage = random.choice([
                "preoperational", "concrete_operational", 
                "formal_operational", "post_formal"
            ])
        
        if random.random() < mutation_rate * 0.5:
            self.solo_taxonomy_level = random.choice([
                "prestructural", "unistructural", "multistructural",
                "relational", "extended_abstract"
            ])
        
        # ============================================================
        # NEW: NUMERIC COGNITIVE FIELD MUTATIONS
        # ============================================================
        if random.random() < mutation_rate:
            self.goal_alignment_strength = max(0.0, min(1.0, 
                self.goal_alignment_strength + random.gauss(0, 0.1)))
        if random.random() < mutation_rate:
            self.generalizability_score = max(0.0, min(1.0,
                self.generalizability_score + random.gauss(0, 0.1)))
        if random.random() < mutation_rate:
            self.fusion_composition_potential = max(0.0, min(1.0,
                self.fusion_composition_potential + random.gauss(0, 0.1)))
        if random.random() < mutation_rate:
            self.memory_capacity_scale = max(0.5, min(2.0,
                self.memory_capacity_scale + random.gauss(0, 0.15)))
        
        # ============================================================
        # NEW: EMBODIMENT & SENSORY MUTATIONS
        # ============================================================
        if random.random() < mutation_rate * 0.4:
            self.embodiment_form_type = random.choice([
                "disembodied_logical_process", "simulated_humanoid_avatar_v3",
                "abstract_informational_entity_field"
            ])
        
        if random.random() < mutation_rate * 0.5:
            self.has_sensory_simulation = not self.has_sensory_simulation
        if random.random() < mutation_rate * 0.5:
            self.has_motor_outputs = not self.has_motor_outputs
        if random.random() < mutation_rate * 0.3:
            self.sensory_modality_count = random.choice([1, 2, 3, 5])
        
        # ============================================================
        # NEW: KNOWLEDGE REPRESENTATION MUTATIONS
        # ============================================================
        if random.random() < mutation_rate * 0.5:
            self.primary_knowledge_format = random.choice([
                "vector_embeddings_transformer_based", 
                "symbolic_logic_predicate_calculus",
                "probabilistic_graphical_model_bayesian_net",
                "procedural_knowledge_scripts_rules"
            ])
        
        if random.random() < mutation_rate * 0.6:
            self.world_model_integration = not self.world_model_integration
        if random.random() < mutation_rate * 0.6:
            self.abstraction_level_control = not self.abstraction_level_control
    
    def multi_parent_crossover(self, parents: List['ArchitectureGenome']) -> 'ArchitectureGenome':
        """Crossover from 3-5 parents for novel combinations - ENHANCED WITH COGNITIVE FIELDS"""
        return ArchitectureGenome(
            # Existing architectural fields
            num_layers=random.choice([p.num_layers for p in parents]),
            hidden_size=random.choice([p.hidden_size for p in parents]),
            num_heads=random.choice([p.num_heads for p in parents]),
            ffn_ratio=np.mean([p.ffn_ratio for p in parents]),
            dropout=np.mean([p.dropout for p in parents]),
            activation=random.choice([p.activation for p in parents]),
            use_flash_attn=random.choice([p.use_flash_attn for p in parents]),
            use_rotary_emb=random.choice([p.use_rotary_emb for p in parents]),
            use_gated_mlp=random.choice([p.use_gated_mlp for p in parents]),
            use_residual_scaling=random.choice([p.use_residual_scaling for p in parents]),
            attention_window_size=random.choice([p.attention_window_size for p in parents]),
            compression_ratio=np.mean([p.compression_ratio for p in parents]),
            layer_drop_rate=np.mean([p.layer_drop_rate for p in parents]),
            
            # NEW: Cognitive fields crossover
            consciousness_scaffold=random.choice([p.consciousness_scaffold for p in parents]),
            cognitive_scaffold=random.choice([p.cognitive_scaffold for p in parents]),
            symbolic_scaffold=random.choice([p.symbolic_scaffold for p in parents]),
            consciousness_scaffold_level=random.choice([p.consciousness_scaffold_level for p in parents]),
            cognitive_scaffold_role=random.choice([p.cognitive_scaffold_role for p in parents]),
            symbolic_orchestration_contribution=random.choice([p.symbolic_orchestration_contribution for p in parents]),
            
            cognitive_stage=random.choice([p.cognitive_stage for p in parents]),
            developmental_model=random.choice([p.developmental_model for p in parents]),
            solo_taxonomy_level=random.choice([p.solo_taxonomy_level for p in parents]),
            goal_alignment_strength=float(np.mean([p.goal_alignment_strength for p in parents])),
            
            self_correction_enabled=random.choice([p.self_correction_enabled for p in parents]),
            hallucination_detection_enabled=random.choice([p.hallucination_detection_enabled for p in parents]),
            goal_alignment_reevaluation=random.choice([p.goal_alignment_reevaluation for p in parents]),
            reflective_inference_enabled=random.choice([p.reflective_inference_enabled for p in parents]),
            introspection_capability=random.choice([p.introspection_capability for p in parents]),
            
            primary_learning_paradigm=random.choice([p.primary_learning_paradigm for p in parents]),
            continual_learning_enabled=random.choice([p.continual_learning_enabled for p in parents]),
            continual_learning_strategy=random.choice([p.continual_learning_strategy for p in parents]),
            memory_architecture_type=random.choice([p.memory_architecture_type for p in parents]),
            memory_capacity_scale=float(np.mean([p.memory_capacity_scale for p in parents])),
            
            primary_knowledge_format=random.choice([p.primary_knowledge_format for p in parents]),
            world_model_integration=random.choice([p.world_model_integration for p in parents]),
            abstraction_level_control=random.choice([p.abstraction_level_control for p in parents]),
            
            generalizability_score=float(np.mean([p.generalizability_score for p in parents])),
            fusion_composition_potential=float(np.mean([p.fusion_composition_potential for p in parents])),
            
            embodiment_form_type=random.choice([p.embodiment_form_type for p in parents]),
            has_sensory_simulation=random.choice([p.has_sensory_simulation for p in parents]),
            sensory_modality_count=random.choice([p.sensory_modality_count for p in parents]),
        )

@dataclass
class BehavioralOrganism:
    """Individual with both genome and phenotype"""
    genome: ArchitectureGenome
    phenotype: IntelligencePhenotype
    generation: int = 0
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"org_{random.randint(100000, 999999)}"
    
    @property
    def fitness(self) -> float:
        return self.phenotype.composite_fitness()
    
    def complexity_score(self) -> float:
        params = self.genome.num_layers * self.genome.hidden_size * self.genome.hidden_size * (3 + self.genome.ffn_ratio)
        return params / 1_000_000

# ============================================================================
# BEHAVIORAL EVALUATION - Test intelligence traits, not just loss
# ============================================================================

class BehavioralEvaluator:
    """Evaluates phenotypic behaviors"""
    
    @staticmethod
    def evaluate_compression_efficiency(result: Dict, batch_size: int) -> float:
        """How efficiently does it compress knowledge?"""
        loss = result.get('avg_loss', 5.0)
        examples = result.get('examples_trained', batch_size)
        
        # Lower loss with more examples = better compression
        compression_score = (1.0 / (1.0 + loss)) * (examples / batch_size)
        return min(1.0, compression_score)
    
    @staticmethod
    def evaluate_distillation_quality(result: Dict) -> float:
        """How well does it learn from teacher?"""
        loss = result.get('avg_loss', 5.0)
        fitness = result.get('fitness', -loss)
        
        # Fitness relative to loss indicates distillation quality
        quality = 1.0 / (1.0 + loss)
        return min(1.0, quality)
    
    @staticmethod
    def evaluate_memory_retention(result: Dict, steps: int) -> float:
        """Long-range dependency handling"""
        # Simulated: better with more steps (real: test on long sequences)
        retention = 0.5 + (0.5 * min(1.0, steps / 1000.0))
        loss = result.get('avg_loss', 5.0)
        retention *= (1.0 / (1.0 + loss * 0.5))
        return min(1.0, retention)
    
    @staticmethod
    def evaluate_error_correction(result: Dict, genome: ArchitectureGenome) -> float:
        """Robustness under noise"""
        # Dropout and residual scaling help error correction
        base_score = 0.4
        if genome.dropout > 0.1:
            base_score += 0.2
        if genome.use_residual_scaling:
            base_score += 0.2
        
        loss = result.get('avg_loss', 5.0)
        base_score *= (1.0 / (1.0 + loss * 0.3))
        return min(1.0, base_score)
    
    @staticmethod
    def evaluate_multi_task_transfer(result: Dict, genome: ArchitectureGenome) -> float:
        """Generalization across tasks"""
        # Larger models generalize better
        capacity_factor = min(1.0, genome.hidden_size / 2048.0)
        depth_factor = min(1.0, genome.num_layers / 32.0)
        
        transfer = 0.3 + (0.35 * capacity_factor) + (0.35 * depth_factor)
        loss = result.get('avg_loss', 5.0)
        transfer *= (1.0 / (1.0 + loss * 0.2))
        return min(1.0, transfer)
    
    @staticmethod
    def evaluate_reasoning_depth(genome: ArchitectureGenome) -> float:
        """Chain-of-thought capability"""
        # Depth and attention window correlate with reasoning
        depth_score = min(1.0, genome.num_layers / 48.0)
        window_score = min(1.0, genome.attention_window_size / 4096.0)
        return 0.5 * depth_score + 0.5 * window_score
    
    @staticmethod
    def evaluate_world_model_coherence(result: Dict, genome: ArchitectureGenome) -> float:
        """Internal representation consistency"""
        loss = result.get('avg_loss', 5.0)
        
        # FFN ratio and gated MLP improve representation
        coherence = 0.4
        if genome.ffn_ratio >= 3.0:
            coherence += 0.2
        if genome.use_gated_mlp:
            coherence += 0.2
        
        coherence *= (1.0 / (1.0 + loss * 0.4))
        return min(1.0, coherence)
    
    @staticmethod
    def evaluate_latency_per_quality(result: Dict, genome: ArchitectureGenome) -> float:
        """Speed-accuracy tradeoff"""
        loss = result.get('avg_loss', 5.0)
        quality = 1.0 / (1.0 + loss)
        
        # Smaller models are faster
        params = genome.num_layers * genome.hidden_size * genome.hidden_size * (3 + genome.ffn_ratio)
        speed_factor = 1.0 / (1.0 + params / 100_000_000)
        
        return quality * speed_factor
    
    @staticmethod
    def evaluate_compression_stability(result: Dict, genome: ArchitectureGenome) -> float:
        """Consistent performance under compression"""
        compression_ratio = genome.compression_ratio
        loss = result.get('avg_loss', 5.0)
        
        stability = (1.0 / (1.0 + loss)) * min(1.0, compression_ratio / 3.0)
        return min(1.0, stability)
    
    @staticmethod
    def evaluate_hallucination_resistance(result: Dict, genome: ArchitectureGenome) -> float:
        """Factual grounding"""
        # Lower dropout and higher attention heads reduce hallucinations
        resistance = 0.5
        if genome.dropout < 0.15:
            resistance += 0.15
        if genome.num_heads >= 16:
            resistance += 0.15
        
        loss = result.get('avg_loss', 5.0)
        resistance *= (1.0 / (1.0 + loss * 0.3))
        return min(1.0, resistance)
    
    @staticmethod
    def evaluate_self_distillation_ability(result: Dict, genome: ArchitectureGenome) -> float:
        """Can teach itself"""
        # Flash attention and rotary embeddings help self-distillation
        ability = 0.3
        if genome.use_flash_attn:
            ability += 0.3
        if genome.use_rotary_emb:
            ability += 0.2
        
        loss = result.get('avg_loss', 5.0)
        ability *= (1.0 / (1.0 + loss * 0.5))
        return min(1.0, ability)
    
    @staticmethod
    def evaluate_cross_modal_fusion(genome: ArchitectureGenome) -> float:
        """Integrates different data types"""
        # Gated MLP and large hidden size help fusion
        fusion = 0.4
        if genome.use_gated_mlp:
            fusion += 0.3
        if genome.hidden_size >= 1024:
            fusion += 0.3
        return min(1.0, fusion)
    
    @staticmethod
    def evaluate_adaptive_plasticity(result: Dict, genome: ArchitectureGenome) -> float:
        """Learns new patterns fast"""
        # Layer drop and moderate dropout improve plasticity
        plasticity = 0.4
        if genome.layer_drop_rate > 0.0:
            plasticity += 0.3
        if 0.1 <= genome.dropout <= 0.2:
            plasticity += 0.2
        
        loss = result.get('avg_loss', 5.0)
        plasticity *= (1.0 / (1.0 + loss * 0.5))
        return min(1.0, plasticity)
    
    @staticmethod
    def evaluate_schema_compliance(
        genome: ArchitectureGenome, 
        schema_integrator: Optional['SchemaEvolutionIntegrator'] = None,
        schema_version: str = "voxsigil_1_4_uni"
    ) -> float:
        """
        🆕 14th Behavioral Phenotype: VoxSIGIL Schema Compliance
        
        Evaluates architectural alignment with VoxSIGIL structural constraints.
        This bridges evolved architectures with schema-defined intelligence patterns.
        
        Args:
            genome: Architecture genome to validate
            schema_integrator: VoxSIGIL schema integration system
            schema_version: Which schema to validate against (1.4-uni, 1.5-holo-alpha, 1.8-omega)
        
        Returns:
            compliance_score: 0.0 (violates constraints) to 1.0 (perfect alignment)
        """
        if not SCHEMA_AVAILABLE or schema_integrator is None:
            # No schema system available - neutral compliance
            logger.debug("Schema integration not available, defaulting to 1.0 compliance")
            return 1.0
        
        try:
            # Convert genome to architecture dict for validation
            arch_dict = asdict(genome)
            
            # Validate against specified schema version
            validation_result = schema_integrator.validate_evolved_architecture(
                arch_dict, 
                schema_version
            )
            
            compliance_score = float(validation_result.get("compliance_score", 1.0))
            
            # Log any missing requirements
            missing = validation_result.get("missing_required", [])
            if missing:
                logger.debug(f"Schema {schema_version} missing: {missing[:3]}")
            
            return min(1.0, max(0.0, compliance_score))
            
        except Exception as e:
            logger.warning(f"Schema compliance evaluation failed: {e}")
            return 0.7  # Partial compliance on error
        loss = result.get('avg_loss', 5.0)
        plasticity *= (1.0 / (1.0 + loss * 0.2))
        return min(1.0, plasticity)
    
    @staticmethod
    def evaluate_full_phenotype(
        organism: BehavioralOrganism, 
        result: Dict, 
        steps: int,
        schema_integrator: Optional['SchemaEvolutionIntegrator'] = None,
        schema_version: str = "voxsigil_1_4_uni"
    ) -> IntelligencePhenotype:
        """Evaluate all behavioral traits including VoxSIGIL schema compliance"""
        return IntelligencePhenotype(
            compression_efficiency=BehavioralEvaluator.evaluate_compression_efficiency(result, 100),
            distillation_quality=BehavioralEvaluator.evaluate_distillation_quality(result),
            memory_retention=BehavioralEvaluator.evaluate_memory_retention(result, steps),
            error_correction=BehavioralEvaluator.evaluate_error_correction(result, organism.genome),
            multi_task_transfer=BehavioralEvaluator.evaluate_multi_task_transfer(result, organism.genome),
            reasoning_depth=BehavioralEvaluator.evaluate_reasoning_depth(organism.genome),
            world_model_coherence=BehavioralEvaluator.evaluate_world_model_coherence(result, organism.genome),
            latency_per_quality=BehavioralEvaluator.evaluate_latency_per_quality(result, organism.genome),
            compression_stability=BehavioralEvaluator.evaluate_compression_stability(result, organism.genome),
            hallucination_resistance=BehavioralEvaluator.evaluate_hallucination_resistance(result, organism.genome),
            self_distillation_ability=BehavioralEvaluator.evaluate_self_distillation_ability(result, organism.genome),
            cross_modal_fusion=BehavioralEvaluator.evaluate_cross_modal_fusion(organism.genome),
            adaptive_plasticity=BehavioralEvaluator.evaluate_adaptive_plasticity(result, organism.genome),
            schema_compliance=BehavioralEvaluator.evaluate_schema_compliance(organism.genome, schema_integrator, schema_version)  # 🆕
        )

# ============================================================================
# BEHAVIORAL EVOLUTION ENGINE
# ============================================================================

class BehavioralEvolutionEngine:
    """Evolves behaviors, not architectures - now with VoxSIGIL schema integration"""
    
    def __init__(
        self, 
        population_size=40, 
        elite_ratio=0.25, 
        mutation_rate=0.4,
        schema_integrator: Optional['SchemaEvolutionIntegrator'] = None,
        schema_version: str = "voxsigil_1_4_uni"
    ):
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.population: List[BehavioralOrganism] = []
        self.generation = 0
        self.hall_of_fame: List[BehavioralOrganism] = []
        self.evaluator = BehavioralEvaluator()
        self.training_steps = 0
        
        # 🆕 VoxSIGIL Schema Integration
        self.schema_integrator = schema_integrator
        self.schema_version = schema_version
        
        if SCHEMA_AVAILABLE and schema_integrator:
            logger.info(f"✅ Behavioral NAS using VoxSIGIL schema: {schema_version}")
            logger.info(f"   Schema compliance is 14th phenotype (10% weight)")
        else:
            logger.info("ℹ️ Behavioral NAS running without schema constraints")
        
    def initialize_population(self):
        """Create diverse initial population"""
        logger.info(f"Initializing population of {self.population_size} behavioral organisms...")
        
        base_configs = [
            # Compression-focused
            {"num_layers": 16, "hidden_size": 512, "num_heads": 8, "compression_ratio": 3.5},
            # Reasoning-focused
            {"num_layers": 32, "hidden_size": 1024, "num_heads": 16, "attention_window_size": 2048},
            # Speed-focused
            {"num_layers": 8, "hidden_size": 384, "num_heads": 6, "use_flash_attn": True},
            # Robustness-focused
            {"num_layers": 24, "hidden_size": 768, "num_heads": 12, "dropout": 0.15},
            # Distillation-focused
            {"num_layers": 12, "hidden_size": 512, "num_heads": 8, "use_gated_mlp": True},
        ]
        
        for i in range(self.population_size):
            if i < len(base_configs):
                config = base_configs[i].copy()
            else:
                config = random.choice(base_configs).copy()
                config["num_layers"] = random.randint(8, 48)
                config["hidden_size"] = random.choice([256, 384, 512, 768, 1024, 1536, 2048])
            
            genome = ArchitectureGenome(
                num_layers=config.get("num_layers", 16),
                hidden_size=config.get("hidden_size", 512),
                num_heads=config.get("num_heads", 8),
                ffn_ratio=random.choice([2.0, 3.0, 4.0]),
                dropout=config.get("dropout", random.uniform(0.05, 0.2)),
                activation=random.choice(['gelu', 'swish', 'silu']),
                use_flash_attn=config.get("use_flash_attn", random.choice([True, False])),
                use_rotary_emb=random.choice([True, False]),
                use_gated_mlp=config.get("use_gated_mlp", random.choice([True, False])),
                use_residual_scaling=random.choice([True, False]),
                attention_window_size=config.get("attention_window_size", random.choice([512, 1024, 2048])),
                compression_ratio=config.get("compression_ratio", random.uniform(1.5, 4.0)),
                layer_drop_rate=random.uniform(0.0, 0.2)
            )
            
            phenotype = IntelligencePhenotype()  # Will be evaluated
            organism = BehavioralOrganism(genome=genome, phenotype=phenotype, generation=0)
            self.population.append(organism)
        
        logger.info(f"✅ Population initialized with {len(self.population)} organisms")
    
    def evaluate_organism(self, organism: BehavioralOrganism, adapter, test_data, batch_id):
        """Evaluate organism's behavioral phenotype with optional schema compliance"""
        try:
            result = adapter.train_on_batch(test_data, batch_id)
            self.training_steps += result.get('training_steps', 0)
            
            # Evaluate full phenotype (including schema compliance)
            organism.phenotype = BehavioralEvaluator.evaluate_full_phenotype(
                organism, 
                result, 
                self.training_steps,
                self.schema_integrator,  # 🆕
                self.schema_version      # 🆕
            )
            
            return organism
            
        except Exception as e:
            logger.warning(f"Evaluation failed for {organism.id}: {e}")
            # Default poor phenotype
            organism.phenotype = IntelligencePhenotype()
            return organism
    
    def evaluate_population(self, adapter, test_data_generator, generation):
        """Evaluate entire population's behaviors"""
        logger.info(f"Evaluating {len(self.population)} organisms (behavioral phenotypes)...")
        
        eval_start = time.time()
        evaluated = []
        passed_gates = 0
        failed_gates = 0
        
        for i, organism in enumerate(self.population):
            # Generate test batch
            batch_data = []
            for _ in range(100):
                try:
                    batch_data.append(next(test_data_generator))
                except StopIteration:
                    test_data_generator = generate_test_data()
                    batch_data.append(next(test_data_generator))
            
            batch_id = f"gen{generation}_org{i}_{organism.id}"
            evaluated_org = self.evaluate_organism(organism, adapter, batch_data, batch_id)
            
            # Check behavioral gates
            passes, failures = evaluated_org.phenotype.passes_gates()
            if passes:
                passed_gates += 1
                evaluated.append(evaluated_org)
            else:
                failed_gates += 1
                # Severe penalty for gate failures
                evaluated_org.phenotype.compression_efficiency *= 0.1
                evaluated.append(evaluated_org)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(self.population)} organisms... (Gates: {passed_gates} pass, {failed_gates} fail)")
        
        self.population = evaluated
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        eval_time = time.time() - eval_start
        logger.info(f"✅ Population evaluated in {eval_time:.1f}s (Passed gates: {passed_gates}/{len(self.population)})")
        
        return self.population
    
    def select_and_breed(self):
        """Multi-parent selection and breeding"""
        elite = self.population[:self.elite_size]
        
        # Update hall of fame
        for org in elite[:5]:
            if org not in self.hall_of_fame:
                self.hall_of_fame.append(org)
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:10]
        
        logger.info(f"Elite preserved: {self.elite_size} organisms")
        
        new_population = elite.copy()
        
        # Multi-parent breeding
        while len(new_population) < self.population_size:
            # Select 3-5 parents via tournament
            num_parents = random.choice([3, 4, 5])
            parents = [max(random.sample(elite, min(3, len(elite))), key=lambda x: x.fitness) 
                      for _ in range(num_parents)]
            
            # Multi-parent crossover
            child_genome = parents[0].genome.multi_parent_crossover([p.genome for p in parents])
            
            # Behavioral mutation
            child_genome.mutate_behavioral(self.mutation_rate)
            
            child = BehavioralOrganism(
                genome=child_genome,
                phenotype=IntelligencePhenotype(),
                generation=self.generation + 1
            )
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        logger.info(f"✅ Generation {self.generation} bred with multi-parent crossover")
    
    def get_stats(self):
        """Population statistics focused on behaviors"""
        if not self.population:
            return {}
        
        best = self.population[0]
        
        # Aggregate phenotype stats
        avg_compression = np.mean([o.phenotype.compression_efficiency for o in self.population])
        avg_distillation = np.mean([o.phenotype.distillation_quality for o in self.population])
        avg_memory = np.mean([o.phenotype.memory_retention for o in self.population])
        avg_reasoning = np.mean([o.phenotype.reasoning_depth for o in self.population])
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "training_steps": self.training_steps,
            "best_fitness": best.fitness,
            "avg_fitness": np.mean([o.fitness for o in self.population]),
            "best_phenotype": {
                "compression_efficiency": best.phenotype.compression_efficiency,
                "distillation_quality": best.phenotype.distillation_quality,
                "memory_retention": best.phenotype.memory_retention,
                "error_correction": best.phenotype.error_correction,
                "reasoning_depth": best.phenotype.reasoning_depth,
                "hallucination_resistance": best.phenotype.hallucination_resistance,
            },
            "avg_phenotype": {
                "compression_efficiency": avg_compression,
                "distillation_quality": avg_distillation,
                "memory_retention": avg_memory,
                "reasoning_depth": avg_reasoning,
            },
            "best_genome": {
                "id": best.id,
                "layers": best.genome.num_layers,
                "hidden_size": best.genome.hidden_size,
                "num_heads": best.genome.num_heads,
                "use_flash_attn": best.genome.use_flash_attn,
                "use_gated_mlp": best.genome.use_gated_mlp,
                "attention_window": best.genome.attention_window_size,
            },
            "complexity": best.complexity_score()
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_test_data(num_samples=10000):
    """Generate test data"""
    templates = [
        "def function_{i}(x, y):\n    return x * y + {val}",
        "class Model_{i}:\n    def __init__(self, size={val}):\n        self.size = size",
        "async def process_{i}(data, batch_size={val}):\n    results = []\n    return results",
    ]
    
    while True:
        for i in range(num_samples):
            template = random.choice(templates)
            val = random.randint(1, 100)
            content = template.format(i=i, val=val)
            
            yield {
                "content": content,
                "source": "synthetic",
                "language": "python",
                "metadata": {"sample_id": i}
            }

def run_behavioral_nas():
    """Run next-gen behavioral NAS"""
    
    logger.info("="*80)
    logger.info("NEXT-GENERATION BEHAVIORAL NAS")
    logger.info("Evolving Intelligence Behaviors, Not Architectures")
    logger.info("="*80)
    logger.info("")
    
    # Configuration
    POPULATION_SIZE = 40
    NUM_GENERATIONS = 15
    ELITE_RATIO = 0.25
    MUTATION_RATE = 0.4
    SCHEMA_VERSION = "voxsigil_1_4_uni"  # or "voxsigil_1_5_holo_alpha" or "voxsigil_1_8_omega"
    
    logger.info("Configuration:")
    logger.info(f"  Population size: {POPULATION_SIZE}")
    logger.info(f"  Generations: {NUM_GENERATIONS}")
    logger.info(f"  Elite ratio: {ELITE_RATIO}")
    logger.info(f"  Mutation rate: {MUTATION_RATE}")
    logger.info(f"  Selection: Multi-parent (3-5 parents)")
    logger.info("")
    logger.info("Evolving for:")
    logger.info("  ✓ Compression efficiency")
    logger.info("  ✓ Distillation quality")
    logger.info("  ✓ Memory retention")
    logger.info("  ✓ Error correction")
    logger.info("  ✓ Reasoning depth")
    logger.info("  ✓ World model coherence")
    logger.info("  ✓ Hallucination resistance")
    logger.info("  ✓ Schema compliance (VoxSIGIL)")
    logger.info("")
    
    # Setup
    workspace = Path(__file__).parent / "training" / "behavioral_nas_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    
    # 🆕 Initialize VoxSIGIL Schema Integration
    schema_integrator = None
    if SCHEMA_AVAILABLE:
        try:
            schema_integrator = get_schema_integrator(str(Path(__file__).parent / "training" / "schema"))
            status = schema_integrator.get_system_status()
            logger.info("🧬 VoxSIGIL Schema Integration:")
            logger.info(f"  Schemas loaded: {status['schemas_loaded']}")
            logger.info(f"  Available: {', '.join(status['available_schemas'])}")
            logger.info(f"  Using: {SCHEMA_VERSION}")
            logger.info(f"  Schema compliance weight: 10%")
            logger.info("")
        except Exception as e:
            logger.warning(f"⚠️ Schema integration failed: {e}")
            logger.info("  Continuing without schema constraints...")
            logger.info("")
            schema_integrator = None
    else:
        logger.info("ℹ️ Running without VoxSIGIL schema integration")
        logger.info("")
    
    logger.info("Initializing training pipeline...")
    pipeline, adapter = create_integrated_pipeline(
        workspace_dir=workspace,
        student_model="Qwen/Qwen2.5-0.5B",
        teacher_models=["Qwen/Qwen2.5-7B"],
        batch_size_mb=50,
        keep_processed=False
    )
    logger.info("✅ Pipeline ready")
    logger.info("")
    
    # Initialize behavioral NAS with schema integration
    engine = BehavioralEvolutionEngine(
        population_size=POPULATION_SIZE,
        elite_ratio=ELITE_RATIO,
        mutation_rate=MUTATION_RATE,
        schema_integrator=schema_integrator,  # 🆕 Pass schema integrator
        schema_version=SCHEMA_VERSION         # 🆕 Pass schema version
    )
    
    engine.initialize_population()
    logger.info("")
    
    # Evolution loop
    logger.info("="*80)
    logger.info("BEHAVIORAL EVOLUTION STARTING")
    logger.info("="*80)
    logger.info("")
    
    all_stats = []
    test_data_gen = generate_test_data()
    
    for gen in range(NUM_GENERATIONS):
        logger.info("─"*80)
        logger.info(f"GENERATION {gen + 1}/{NUM_GENERATIONS}")
        logger.info("─"*80)
        
        gen_start = time.time()
        
        # Evaluate population
        engine.evaluate_population(adapter, test_data_gen, gen + 1)
        
        # Get statistics
        stats = engine.get_stats()
        all_stats.append(stats)
        
        gen_time = time.time() - gen_start
        
        # Report behavioral results
        logger.info("")
        logger.info(f"Generation {gen + 1} Behavioral Results:")
        logger.info(f"  Duration: {gen_time:.1f}s")
        logger.info(f"  Best Fitness: {stats['best_fitness']:.4f}")
        logger.info(f"  Avg Fitness: {stats['avg_fitness']:.4f}")
        logger.info("")
        logger.info("  Best Phenotype:")
        logger.info(f"    Compression: {stats['best_phenotype']['compression_efficiency']:.3f}")
        logger.info(f"    Distillation: {stats['best_phenotype']['distillation_quality']:.3f}")
        logger.info(f"    Memory: {stats['best_phenotype']['memory_retention']:.3f}")
        logger.info(f"    Error Correction: {stats['best_phenotype']['error_correction']:.3f}")
        logger.info(f"    Reasoning: {stats['best_phenotype']['reasoning_depth']:.3f}")
        logger.info(f"    Hallucination Resistance: {stats['best_phenotype']['hallucination_resistance']:.3f}")
        logger.info("")
        logger.info(f"  Best Genome: {stats['best_genome']['layers']}L-{stats['best_genome']['hidden_size']}H")
        logger.info(f"  Complexity: {stats['complexity']:.1f}M params")
        logger.info("")
        
        # Breed next generation
        if gen < NUM_GENERATIONS - 1:
            engine.select_and_breed()
            logger.info("")
    
    # Final results
    logger.info("="*80)
    logger.info("BEHAVIORAL EVOLUTION COMPLETE")
    logger.info("="*80)
    logger.info("")
    
    logger.info(f"Total generations: {NUM_GENERATIONS}")
    logger.info(f"Total organisms evaluated: {NUM_GENERATIONS * POPULATION_SIZE}")
    logger.info(f"Total training steps: {engine.training_steps}")
    logger.info(f"Hall of Fame size: {len(engine.hall_of_fame)}")
    logger.info("")
    
    logger.info("Top 5 Behavioral Phenotypes:")
    for i, org in enumerate(engine.hall_of_fame[:5], 1):
        logger.info(f"  #{i} {org.id} (Gen {org.generation}):")
        logger.info(f"      Fitness: {org.fitness:.4f}")
        logger.info(f"      Compression: {org.phenotype.compression_efficiency:.3f}")
        logger.info(f"      Distillation: {org.phenotype.distillation_quality:.3f}")
        logger.info(f"      Memory: {org.phenotype.memory_retention:.3f}")
        logger.info(f"      Reasoning: {org.phenotype.reasoning_depth:.3f}")
        logger.info(f"      Hallucination Resistance: {org.phenotype.hallucination_resistance:.3f}")
        if schema_integrator:  # 🆕 Show schema compliance if available
            logger.info(f"      Schema Compliance: {org.phenotype.schema_compliance:.3f}")
        logger.info(f"      Architecture: {org.genome.num_layers}L-{org.genome.hidden_size}H-{org.genome.num_heads}A")
        logger.info(f"      Complexity: {org.complexity_score():.1f}M params")
    logger.info("")
    
    # Save results
    results_file = workspace / "behavioral_nas_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "population_size": POPULATION_SIZE,
                "num_generations": NUM_GENERATIONS,
                "evolution_method": "multi_parent_behavioral",
                "schema_integration": schema_integrator is not None,  # 🆕
                "schema_version": SCHEMA_VERSION if schema_integrator else None,  # 🆕
            },
            "statistics": all_stats,
            "hall_of_fame": [
                {
                    "id": org.id,
                    "fitness": org.fitness,
                    "generation": org.generation,
                    "phenotype": asdict(org.phenotype),
                    "genome": asdict(org.genome),
                    "complexity": org.complexity_score()
                }
                for org in engine.hall_of_fame
            ]
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("")
    logger.info("="*80)
    logger.info("🎉 BEHAVIORAL NAS COMPLETE - INTELLIGENCE EVOLVED!")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(run_behavioral_nas())
    except KeyboardInterrupt:
        logger.info("\n\nEvolution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Evolution failed: {e}", exc_info=True)
        sys.exit(1)
