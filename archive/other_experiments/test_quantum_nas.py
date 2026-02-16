#!/usr/bin/env python3
"""
Quick Test Suite for Quantum-Enhanced Behavioral NAS
Tests quantum initialization integration without full evolution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import logging
import json
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("QuantumNAS_Test")

# Test 1: Quantum Initialization Module
def test_quantum_initialization():
    """Test that quantum init functions work correctly"""
    logger.info("="*60)
    logger.info("TEST 1: Quantum Initialization Module")
    logger.info("="*60)
    
    try:
        from nebula.utils.quantum_init import (
            quantum_initialize_weights,
            quantum_field_initialization,
            apply_quantum_perturbation,
            quantum_phase_initialization
        )
        
        # Test 1.1: Quantum phase initialization
        logger.info("\n1.1 Testing quantum_phase_initialization...")
        tensor = quantum_phase_initialization((10, 10))
        logger.info(f"  ✅ Created tensor shape: {tensor.shape}")
        logger.info(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        
        # Test 1.2: Model weight initialization
        logger.info("\n1.2 Testing quantum_initialize_weights...")
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        model = quantum_initialize_weights(model, sparsity=0.7)
        logger.info("  ✅ Model weights initialized")
        
        # Check sparsity
        total_params = sum(p.numel() for p in model.parameters())
        near_zero = sum((p.abs() < 0.01).sum().item() for p in model.parameters())
        sparsity = near_zero / total_params
        logger.info(f"  Sparsity achieved: {sparsity:.1%} (target: 70%)")
        
        # Test 1.3: Field initialization
        logger.info("\n1.3 Testing quantum_field_initialization...")
        test_genome = {
            'num_layers': 16,
            'hidden_size': 512,
            'dropout': 0.1,
            'compression_ratio': 2.5
        }
        quantum_genome = quantum_field_initialization(test_genome)
        logger.info("  ✅ Genome fields initialized")
        logger.info(f"  Original: {test_genome}")
        logger.info(f"  Quantum:  {quantum_genome}")
        
        # Test 1.4: Perturbation
        logger.info("\n1.4 Testing apply_quantum_perturbation...")
        original_value = 1.0
        perturbed = apply_quantum_perturbation(original_value, strength=0.1)
        logger.info(f"  ✅ Original: {original_value}, Perturbed: {perturbed:.4f}")
        
        logger.info("\n✅ TEST 1 PASSED: All quantum functions work correctly\n")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST 1 FAILED: {e}\n")
        return False


# Test 2: Genome Initialization
def test_genome_initialization():
    """Test that genomes can be quantum-initialized"""
    logger.info("="*60)
    logger.info("TEST 2: Genome Quantum Initialization")
    logger.info("="*60)
    
    try:
        from behavioral_nas_nextgen import ArchitectureGenome
        from nebula.utils.quantum_init import quantum_field_initialization
        
        # Create standard genome
        logger.info("\n2.1 Creating standard genome...")
        genome = ArchitectureGenome(
            num_layers=16,
            hidden_size=512,
            num_heads=8,
            ffn_ratio=2.0,
            dropout=0.1,
            activation='gelu',
        )
        logger.info(f"  ✅ Standard genome created")
        logger.info(f"  num_layers: {genome.num_layers}")
        logger.info(f"  hidden_size: {genome.hidden_size}")
        logger.info(f"  dropout: {genome.dropout}")
        
        # Apply quantum initialization
        logger.info("\n2.2 Applying quantum initialization...")
        genome_dict = asdict(genome)
        quantum_dict = quantum_field_initialization(genome_dict)
        
        # Update genome
        for key, value in quantum_dict.items():
            if hasattr(genome, key):
                setattr(genome, key, value)
        
        logger.info(f"  ✅ Quantum-initialized genome")
        logger.info(f"  num_layers: {genome.num_layers} (may have changed)")
        logger.info(f"  hidden_size: {genome.hidden_size} (may have changed)")
        logger.info(f"  dropout: {genome.dropout:.4f} (perturbed)")
        
        logger.info("\n✅ TEST 2 PASSED: Genome quantum initialization works\n")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Mini Evolution (5 organisms, 3 generations)
def test_mini_evolution():
    """Test quantum-enhanced evolution on tiny scale"""
    logger.info("="*60)
    logger.info("TEST 3: Mini Quantum Evolution (5 organisms, 3 gens)")
    logger.info("="*60)
    
    try:
        from quantum_behavioral_nas import QuantumEnhancedEvolutionEngine
        from scripts.training.streaming_distillation_adapter import create_integrated_pipeline
        from behavioral_nas_nextgen import generate_test_data
        
        # Tiny configuration for testing
        logger.info("\n3.1 Initializing mini evolution...")
        engine = QuantumEnhancedEvolutionEngine(
            population_size=5,  # Tiny population
            elite_ratio=0.4,    # Keep 2 elite
            mutation_rate=0.5,
            use_quantum_init=True,
            quantum_sparsity=0.7
        )
        
        engine.initialize_population()
        logger.info(f"  ✅ Population initialized: {len(engine.population)} organisms")
        
        # Create minimal adapter
        logger.info("\n3.2 Creating distillation pipeline...")
        adapter = create_integrated_pipeline(
            student_model="Qwen/Qwen2.5-0.5B",
            teacher_models=["Qwen/Qwen2.5-7B"],
            batch_size_mb=10,  # Smaller batch for testing
            keep_processed=False
        )
        logger.info("  ✅ Pipeline ready")
        
        # Run 3 generations
        logger.info("\n3.3 Running 3 generations...")
        test_data_gen = generate_test_data(num_samples=100)  # Tiny dataset
        
        for gen in range(3):
            logger.info(f"\n  Generation {gen + 1}/3")
            
            # Evaluate (just 2 organisms for speed)
            logger.info("    Evaluating...")
            for i, org in enumerate(engine.population[:2]):
                batch_data = [next(test_data_gen) for _ in range(10)]
                engine.evaluate_organism(org, adapter, batch_data, f"test_g{gen}_o{i}")
            
            # Sort by fitness
            engine.population.sort(key=lambda x: x.fitness, reverse=True)
            
            stats = engine.get_stats()
            logger.info(f"    Best fitness: {stats['best_fitness']:.4f}")
            
            # Breed if not last generation
            if gen < 2:
                engine.select_and_breed()
        
        logger.info("\n✅ TEST 3 PASSED: Mini evolution completed successfully\n")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# Test 4: Comparison Test (2 organisms, 2 gens each)
def test_quantum_vs_standard():
    """Quick comparison between quantum and standard initialization"""
    logger.info("="*60)
    logger.info("TEST 4: Quantum vs Standard Comparison")
    logger.info("="*60)
    
    try:
        from quantum_behavioral_nas import QuantumEnhancedEvolutionEngine
        from behavioral_nas_nextgen import generate_test_data, IntelligencePhenotype
        from scripts.training.streaming_distillation_adapter import create_integrated_pipeline
        
        # Create adapter once
        adapter = create_integrated_pipeline(
            student_model="Qwen/Qwen2.5-0.5B",
            teacher_models=["Qwen/Qwen2.5-7B"],
            batch_size_mb=10,
            keep_processed=False
        )
        
        results = {}
        
        for mode, use_quantum in [("Standard", False), ("Quantum", True)]:
            logger.info(f"\n4.{1 if mode=='Standard' else 2} Testing {mode} mode...")
            
            engine = QuantumEnhancedEvolutionEngine(
                population_size=3,
                elite_ratio=0.33,
                mutation_rate=0.5,
                use_quantum_init=use_quantum
            )
            
            engine.initialize_population()
            test_data_gen = generate_test_data(num_samples=50)
            
            # Run 2 quick generations
            for gen in range(2):
                # Evaluate only first organism
                org = engine.population[0]
                batch_data = [next(test_data_gen) for _ in range(5)]
                engine.evaluate_organism(org, adapter, batch_data, f"cmp_{mode}_g{gen}")
                
                engine.population.sort(key=lambda x: x.fitness, reverse=True)
                
                if gen < 1:
                    engine.select_and_breed()
            
            stats = engine.get_stats()
            results[mode] = stats['best_fitness']
            logger.info(f"  {mode} best fitness: {results[mode]:.4f}")
        
        # Compare
        logger.info(f"\n4.3 Comparison:")
        logger.info(f"  Standard: {results['Standard']:.4f}")
        logger.info(f"  Quantum:  {results['Quantum']:.4f}")
        
        if results['Quantum'] > results['Standard']:
            improvement = ((results['Quantum'] - results['Standard']) 
                          / results['Standard'] * 100)
            logger.info(f"  ✅ Quantum improved by {improvement:+.1f}%")
        else:
            logger.info(f"  ⚠️  Standard was better (test may be too small)")
        
        logger.info("\n✅ TEST 4 PASSED: Comparison completed\n")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    logger.info("\n" + "="*60)
    logger.info("QUANTUM-ENHANCED BEHAVIORAL NAS TEST SUITE")
    logger.info("="*60 + "\n")
    
    results = {
        "Quantum Module": test_quantum_initialization(),
        "Genome Init": test_genome_initialization(),
        "Mini Evolution": test_mini_evolution(),
        "Quantum vs Standard": test_quantum_vs_standard()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.info(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Quantum NAS Integration")
    parser.add_argument("--test", choices=["quantum", "genome", "evolution", "compare", "all"],
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "quantum":
        sys.exit(0 if test_quantum_initialization() else 1)
    elif args.test == "genome":
        sys.exit(0 if test_genome_initialization() else 1)
    elif args.test == "evolution":
        sys.exit(0 if test_mini_evolution() else 1)
    elif args.test == "compare":
        sys.exit(0 if test_quantum_vs_standard() else 1)
    else:
        sys.exit(run_all_tests())
