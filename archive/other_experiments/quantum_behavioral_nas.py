#!/usr/bin/env python3
"""
Quantum-Enhanced Behavioral NAS
Integrates quantum-inspired weight initialization into evolution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import quantum initialization
try:
    from nebula.utils.quantum_init import (
        quantum_initialize_weights,
        quantum_field_initialization,
        apply_quantum_perturbation,
        get_quantum_weights_distribution
    )
    QUANTUM_AVAILABLE = True
    print("✅ Quantum initialization loaded successfully")
except ImportError as e:
    QUANTUM_AVAILABLE = False
    print(f"⚠️  Quantum initialization unavailable: {e}")
    # Fallback functions
    def quantum_initialize_weights(model, **kwargs):
        return model
    def quantum_field_initialization(genome):
        return genome
    def apply_quantum_perturbation(value, **kwargs):
        return value
    def get_quantum_weights_distribution(n, **kwargs):
        return {f"model_{i}": 1.0/n for i in range(n)}

# Import the original behavioral NAS
from behavioral_nas_nextgen import *
import logging

logger = logging.getLogger("QuantumBehavioralNAS")


class QuantumEnhancedEvolutionEngine(BehavioralEvolutionEngine):
    """
    Enhanced evolution engine with quantum initialization integration.
    
    Quantum integration points:
    1. Population initialization - Better starting points
    2. Mutation - Quantum-guided perturbations
    3. Crossover - Quantum-initialized offspring
    4. Model building - Quantum weight initialization
    """
    
    def __init__(self, 
                 population_size=40, 
                 elite_ratio=0.25, 
                 mutation_rate=0.4,
                 schema_integrator: Optional['SchemaEvolutionIntegrator'] = None,
                 schema_version: str = "voxsigil_1_4_uni",
                 use_quantum_init: bool = True,
                 quantum_sparsity: float = 0.7):
        
        super().__init__(
            population_size=population_size,
            elite_ratio=elite_ratio,
            mutation_rate=mutation_rate,
            schema_integrator=schema_integrator,
            schema_version=schema_version
        )
        
        self.use_quantum_init = use_quantum_init and QUANTUM_AVAILABLE
        self.quantum_sparsity = quantum_sparsity
        
        if self.use_quantum_init:
            logger.info("🌀 QUANTUM-ENHANCED MODE ENABLED")
            logger.info(f"   Sparsity: {quantum_sparsity:.1%}")
        else:
            logger.info("⚙️  Standard initialization mode")
    
    def initialize_population(self):
        """Initialize population with quantum-enhanced genomes"""
        logger.info(f"Initializing population of {self.population_size} organisms...")
        
        if self.use_quantum_init:
            logger.info("🌀 Applying quantum initialization to genomes...")
        
        base_configs = [
            {"num_layers": 16, "hidden_size": 512, "num_heads": 8, "compression_ratio": 3.5},
            {"num_layers": 32, "hidden_size": 1024, "num_heads": 16, "attention_window_size": 2048},
            {"num_layers": 8, "hidden_size": 384, "num_heads": 6, "use_flash_attn": True},
            {"num_layers": 24, "hidden_size": 768, "num_heads": 12, "dropout": 0.15},
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
            
            # 🆕 QUANTUM ENHANCEMENT: Apply quantum field initialization
            if self.use_quantum_init:
                genome_dict = asdict(genome)
                quantum_genome_dict = quantum_field_initialization(genome_dict)
                
                # Update genome with quantum-initialized values
                for key, value in quantum_genome_dict.items():
                    if hasattr(genome, key):
                        setattr(genome, key, value)
            
            phenotype = IntelligencePhenotype()
            organism = BehavioralOrganism(genome=genome, phenotype=phenotype, generation=0)
            self.population.append(organism)
        
        logger.info(f"✅ Population initialized with {len(self.population)} organisms")
        if self.use_quantum_init:
            logger.info("   🌀 Quantum initialization applied to all genomes")
    
    def select_and_breed(self):
        """Multi-parent selection and breeding with quantum-enhanced offspring"""
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
            num_parents = random.choice([3, 4, 5])
            parents = [max(random.sample(elite, min(3, len(elite))), key=lambda x: x.fitness) 
                      for _ in range(num_parents)]
            
            # Multi-parent crossover
            child_genome = parents[0].genome.multi_parent_crossover([p.genome for p in parents])
            
            # 🆕 QUANTUM ENHANCEMENT: Apply quantum perturbation to crossover result
            if self.use_quantum_init and random.random() < 0.3:  # 30% of offspring
                genome_dict = asdict(child_genome)
                for key, value in genome_dict.items():
                    if isinstance(value, float):
                        genome_dict[key] = apply_quantum_perturbation(value, strength=0.05)
                
                # Update child genome
                for key, value in genome_dict.items():
                    if hasattr(child_genome, key):
                        setattr(child_genome, key, value)
            
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
        
        mode_str = "quantum-enhanced" if self.use_quantum_init else "standard"
        logger.info(f"✅ Generation {self.generation} bred with {mode_str} crossover")


def run_quantum_behavioral_nas(use_quantum: bool = True):
    """Run quantum-enhanced behavioral NAS"""
    
    logger.info("="*80)
    logger.info("QUANTUM-ENHANCED BEHAVIORAL NAS" if use_quantum else "STANDARD BEHAVIORAL NAS")
    logger.info("Evolving Intelligence with Quantum Initialization" if use_quantum else "Evolving Intelligence")
    logger.info("="*80)
    logger.info("")
    
    # Configuration
    POPULATION_SIZE = 40
    NUM_GENERATIONS = 15
    ELITE_RATIO = 0.25
    MUTATION_RATE = 0.4
    SCHEMA_VERSION = "voxsigil_1_4_uni"
    
    logger.info("Configuration:")
    logger.info(f"  Population size: {POPULATION_SIZE}")
    logger.info(f"  Generations: {NUM_GENERATIONS}")
    logger.info(f"  Elite ratio: {ELITE_RATIO}")
    logger.info(f"  Mutation rate: {MUTATION_RATE}")
    logger.info(f"  Quantum init: {'ENABLED ✅' if use_quantum else 'DISABLED'}")
    logger.info(f"  Schema version: {SCHEMA_VERSION}")
    logger.info(f"  Selection: Multi-parent (3-5 parents)")
    logger.info("")
    
    # Setup workspace
    workspace = Path(__file__).parent / "training" / "quantum_behavioral_nas_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Initialize schema integrator
    schema_integrator = None
    if SCHEMA_AVAILABLE:
        try:
            schema_integrator = get_schema_integrator(workspace)
            schema_integrator.load_constraints(SCHEMA_VERSION)
            logger.info(f"✅ Schema integrator loaded: {SCHEMA_VERSION}")
        except Exception as e:
            logger.warning(f"⚠️ Schema integrator unavailable: {e}")
            schema_integrator = None
    
    # Initialize distillation pipeline
    logger.info("\nInitializing distillation pipeline...")
    adapter = create_integrated_pipeline(
        student_model="Qwen/Qwen2.5-0.5B",
        teacher_models=["Qwen/Qwen2.5-7B"],
        batch_size_mb=50,
        keep_processed=False
    )
    logger.info("✅ Pipeline ready")
    logger.info("")
    
    # Initialize quantum-enhanced engine
    engine = QuantumEnhancedEvolutionEngine(
        population_size=POPULATION_SIZE,
        elite_ratio=ELITE_RATIO,
        mutation_rate=MUTATION_RATE,
        schema_integrator=schema_integrator,
        schema_version=SCHEMA_VERSION,
        use_quantum_init=use_quantum,
        quantum_sparsity=0.7
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
        
        # Report results
        logger.info("")
        logger.info(f"Generation {gen + 1} Results:")
        logger.info(f"  Duration: {gen_time:.1f}s")
        logger.info(f"  Best Fitness: {stats['best_fitness']:.4f}")
        logger.info(f"  Avg Fitness: {stats['avg_fitness']:.4f}")
        logger.info("")
        logger.info("  Best Phenotype:")
        logger.info(f"    Compression: {stats['best_phenotype']['compression_efficiency']:.3f}")
        logger.info(f"    Distillation: {stats['best_phenotype']['distillation_quality']:.3f}")
        logger.info(f"    Memory: {stats['best_phenotype']['memory_retention']:.3f}")
        logger.info(f"    Schema Compliance: {stats['best_phenotype'].get('schema_compliance', 1.0):.3f}")
        logger.info("")
        
        # Breed next generation
        if gen < NUM_GENERATIONS - 1:
            engine.select_and_breed()
            logger.info("")
    
    # Final results
    logger.info("="*80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("="*80)
    logger.info("")
    
    logger.info(f"Total generations: {NUM_GENERATIONS}")
    logger.info(f"Hall of Fame size: {len(engine.hall_of_fame)}")
    logger.info("")
    
    logger.info("Top 5 Organisms:")
    for i, org in enumerate(engine.hall_of_fame[:5], 1):
        logger.info(f"  #{i} {org.id} (Gen {org.generation}):")
        logger.info(f"      Fitness: {org.fitness:.4f}")
        logger.info(f"      Compression: {org.phenotype.compression_efficiency:.3f}")
        logger.info(f"      Distillation: {org.phenotype.distillation_quality:.3f}")
    
    # Save results
    results_file = workspace / f"quantum_nas_results_{'enabled' if use_quantum else 'disabled'}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'population_size': POPULATION_SIZE,
                'num_generations': NUM_GENERATIONS,
                'quantum_enabled': use_quantum
            },
            'stats': all_stats,
            'hall_of_fame': [
                {
                    'id': org.id,
                    'generation': org.generation,
                    'fitness': org.fitness,
                    'genome': asdict(org.genome)
                }
                for org in engine.hall_of_fame
            ]
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    
    return engine, all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum-Enhanced Behavioral NAS")
    parser.add_argument("--no-quantum", action="store_true", help="Disable quantum initialization")
    parser.add_argument("--compare", action="store_true", help="Run both modes and compare")
    args = parser.parse_args()
    
    if args.compare:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON MODE: Running both standard and quantum-enhanced")
        logger.info("="*80 + "\n")
        
        # Run standard
        logger.info("\n### RUNNING STANDARD MODE ###\n")
        engine_std, stats_std = run_quantum_behavioral_nas(use_quantum=False)
        
        # Run quantum
        logger.info("\n### RUNNING QUANTUM MODE ###\n")
        engine_q, stats_q = run_quantum_behavioral_nas(use_quantum=True)
        
        # Compare
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        
        final_std = stats_std[-1]
        final_q = stats_q[-1]
        
        improvement = ((final_q['best_fitness'] - final_std['best_fitness']) 
                      / final_std['best_fitness'] * 100)
        
        logger.info(f"\nBest Fitness:")
        logger.info(f"  Standard: {final_std['best_fitness']:.4f}")
        logger.info(f"  Quantum:  {final_q['best_fitness']:.4f}")
        logger.info(f"  Improvement: {improvement:+.1f}%")
        
        logger.info(f"\nTop Organism Comparison:")
        logger.info(f"  Standard - Genome: {final_std['best_genome']['layers']}L-{final_std['best_genome']['hidden_size']}H")
        logger.info(f"  Quantum  - Genome: {final_q['best_genome']['layers']}L-{final_q['best_genome']['hidden_size']}H")
        
    else:
        use_quantum = not args.no_quantum
        run_quantum_behavioral_nas(use_quantum=use_quantum)
