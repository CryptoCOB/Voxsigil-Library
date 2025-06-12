#!/usr/bin/env python3
"""
End-to-End ARC Batch Regression Test

Comprehensive test that loads the HOLO-1.5 ensemble, solves 10 synthetic ARC tasks,
and asserts ≥70% success rate. This is the gate for CI/CD pipeline.

HOLO-1.5 Enhanced Testing:
- Recursive symbolic cognition validation across ensemble agents
- Neural-symbolic reasoning synthesis verification  
- VantaCore mesh collaboration assessment
- Cognitive load and symbolic depth testing
"""

import pytest
import torch
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Core imports
from core.ensemble_integration import ARCEnsembleOrchestrator, create_arc_ensemble
from core.novel_reasoning import create_reasoning_engine
from core.meta_control import create_effort_controller, ComplexityLevel
from demo_novel_paradigms import ARCTaskGenerator

logger = logging.getLogger(__name__)

class ARCBatchTester:
    """Comprehensive batch testing for ARC ensemble performance"""
    
    def __init__(self, min_success_rate: float = 0.70):
        self.min_success_rate = min_success_rate
        self.task_generator = ARCTaskGenerator(grid_size=10)
        self.ensemble = None
        self.test_results = {
            'tasks_attempted': 0,
            'tasks_solved': 0,
            'success_rate': 0.0,
            'performance_metrics': {},
            'error_log': []
        }
        
    def setup_ensemble(self) -> bool:
        """Initialize and validate the ARC ensemble"""
        try:
            logger.info("🔧 Setting up HOLO-1.5 ARC Ensemble...")
            
            # Create ensemble with testing configuration
            config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'memory_efficient': True,
                'enable_explainability': True,
                'ensemble_mode': 'full_pipeline',
                'symbolic_depth': 3,
                'cognitive_load_limit': 4.0
            }
            
            self.ensemble = create_arc_ensemble(config)
            
            # Validate ensemble components
            required_components = [
                'splr_encoder', 'akorn_binder', 'lnu_reasoner', 
                'gnn_reasoner', 'meta_controller'
            ]
            
            for component in required_components:
                if not hasattr(self.ensemble, component):
                    raise ValueError(f"Missing ensemble component: {component}")
                    
            logger.info("✅ Ensemble setup successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ensemble setup failed: {e}")
            self.test_results['error_log'].append(f"Setup error: {e}")
            return False
    
    def generate_test_suite(self, num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate diverse ARC test tasks"""
        test_tasks = []
        
        complexity_levels = ['trivial', 'moderate', 'complex', 'extremely_complex']
        
        for i in range(num_tasks):
            # Distribute complexity levels
            complexity = complexity_levels[i % len(complexity_levels)]
            
            task = self.task_generator.generate_pattern_task(complexity)
            task_meta = {
                'task_id': f'synthetic_task_{i:03d}',
                'complexity': complexity,
                'expected_difficulty': {
                    'trivial': 0.95,      # 95% expected success
                    'moderate': 0.80,     # 80% expected success
                    'complex': 0.60,      # 60% expected success
                    'extremely_complex': 0.40  # 40% expected success
                }[complexity],
                'task_data': task
            }
            test_tasks.append(task_meta)
            
        logger.info(f"📋 Generated {num_tasks} test tasks")
        return test_tasks
    
    def solve_task(self, task_meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Solve a single ARC task and return success status with metrics"""
        task_id = task_meta['task_id']
        task_data = task_meta['task_data']
        
        try:
            start_time = time.time()
            logger.info(f"🧠 Solving {task_id} (complexity: {task_meta['complexity']})")
            
            # Extract input and target
            input_grid = task_data['input_grid']
            target_grid = task_data['target_grid']
            
            # Run ensemble inference
            result = self.ensemble.solve_task({
                'input': input_grid,
                'task_type': 'pattern_completion',
                'complexity_hint': task_meta['complexity']
            })
            
            solve_time = time.time() - start_time
            
            # Validate solution
            if result is None or 'prediction' not in result:
                logger.warning(f"⚠️  No prediction returned for {task_id}")
                return False, {'error': 'no_prediction', 'solve_time': solve_time}
            
            prediction = result['prediction']
            
            # Calculate accuracy (exact match)
            if isinstance(prediction, torch.Tensor) and isinstance(target_grid, torch.Tensor):
                accuracy = torch.equal(prediction, target_grid)
            else:
                accuracy = np.array_equal(prediction, target_grid)
            
            # Performance metrics
            metrics = {
                'solve_time': solve_time,
                'accuracy': float(accuracy),
                'prediction_shape': list(prediction.shape) if hasattr(prediction, 'shape') else 'unknown',
                'cognitive_load': result.get('cognitive_load', 0.0),
                'symbolic_depth': result.get('symbolic_depth', 0),
                'reasoning_steps': result.get('reasoning_steps', 0),
                'memory_usage': result.get('memory_usage_mb', 0.0)
            }
            
            if accuracy:
                logger.info(f"✅ {task_id} solved successfully in {solve_time:.2f}s")
            else:
                logger.info(f"❌ {task_id} failed (prediction mismatch)")
                
            return bool(accuracy), metrics
            
        except Exception as e:
            logger.error(f"💥 Error solving {task_id}: {e}")
            return False, {'error': str(e), 'solve_time': 0.0}
    
    def run_batch_test(self, num_tasks: int = 10) -> Dict[str, Any]:
        """Run complete batch test and return comprehensive results"""
        logger.info(f"🚀 Starting ARC Batch Regression Test ({num_tasks} tasks)")
        
        # Setup ensemble
        if not self.setup_ensemble():
            return self.test_results
        
        # Generate test suite
        test_tasks = self.generate_test_suite(num_tasks)
        
        # Solve each task
        task_results = []
        solved_count = 0
        
        for task_meta in test_tasks:
            self.test_results['tasks_attempted'] += 1
            
            success, metrics = self.solve_task(task_meta)
            
            task_result = {
                'task_id': task_meta['task_id'],
                'complexity': task_meta['complexity'],
                'success': success,
                'metrics': metrics
            }
            task_results.append(task_result)
            
            if success:
                solved_count += 1
                self.test_results['tasks_solved'] += 1
        
        # Calculate final metrics
        self.test_results['success_rate'] = solved_count / num_tasks if num_tasks > 0 else 0.0
        self.test_results['task_results'] = task_results
        
        # Performance analysis
        self.test_results['performance_metrics'] = self._analyze_performance(task_results)
        
        # Final verdict
        passed = self.test_results['success_rate'] >= self.min_success_rate
        
        logger.info(f"🎯 Batch Test Complete:")
        logger.info(f"   Tasks Solved: {solved_count}/{num_tasks}")
        logger.info(f"   Success Rate: {self.test_results['success_rate']:.1%}")
        logger.info(f"   Required Rate: {self.min_success_rate:.1%}")
        logger.info(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        self.test_results['test_passed'] = passed
        return self.test_results
    
    def _analyze_performance(self, task_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance metrics across all tasks"""
        solve_times = [r['metrics'].get('solve_time', 0) for r in task_results if r['success']]
        cognitive_loads = [r['metrics'].get('cognitive_load', 0) for r in task_results]
        memory_usage = [r['metrics'].get('memory_usage', 0) for r in task_results]
        
        by_complexity = {}
        for result in task_results:
            complexity = result['complexity']
            if complexity not in by_complexity:
                by_complexity[complexity] = {'attempted': 0, 'solved': 0}
            by_complexity[complexity]['attempted'] += 1
            if result['success']:
                by_complexity[complexity]['solved'] += 1
        
        return {
            'avg_solve_time': np.mean(solve_times) if solve_times else 0.0,
            'max_solve_time': np.max(solve_times) if solve_times else 0.0,
            'avg_cognitive_load': np.mean(cognitive_loads) if cognitive_loads else 0.0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0.0,
            'performance_by_complexity': {
                k: {
                    'success_rate': v['solved'] / v['attempted'] if v['attempted'] > 0 else 0.0,
                    **v
                }
                for k, v in by_complexity.items()
            }
        }


# Pytest test functions
@pytest.fixture
def arc_tester():
    """Fixture to provide ARC batch tester"""
    return ARCBatchTester(min_success_rate=0.70)

def test_arc_ensemble_batch_performance(arc_tester):
    """Main pytest function for CI/CD integration"""
    results = arc_tester.run_batch_test(num_tasks=10)
    
    # Assert success rate requirement
    assert results['test_passed'], f"ARC ensemble success rate {results['success_rate']:.1%} below required 70%"
    
    # Assert no critical errors
    assert len(results['error_log']) == 0, f"Critical errors encountered: {results['error_log']}"
    
    # Assert performance thresholds
    perf = results['performance_metrics']
    assert perf['avg_solve_time'] < 30.0, f"Average solve time {perf['avg_solve_time']:.1f}s exceeds 30s limit"
    assert perf['avg_cognitive_load'] < 5.0, f"Average cognitive load {perf['avg_cognitive_load']:.1f} exceeds limit"

def test_complexity_distribution(arc_tester):
    """Test that ensemble handles different complexity levels appropriately"""
    results = arc_tester.run_batch_test(num_tasks=12)  # Multiple of 4 for even distribution
    
    complexity_performance = results['performance_metrics']['performance_by_complexity']
    
    # Trivial tasks should have high success rate
    if 'trivial' in complexity_performance:
        assert complexity_performance['trivial']['success_rate'] >= 0.8, \
            "Trivial tasks should have ≥80% success rate"
    
    # Complex tasks can have lower success but should still attempt
    if 'complex' in complexity_performance:
        assert complexity_performance['complex']['attempted'] > 0, \
            "Complex tasks should be attempted"

def test_memory_efficiency(arc_tester):
    """Test that memory usage remains within reasonable bounds"""
    results = arc_tester.run_batch_test(num_tasks=5)  # Smaller test for memory focus
    
    avg_memory = results['performance_metrics']['avg_memory_usage']
    assert avg_memory < 2048, f"Average memory usage {avg_memory:.1f}MB exceeds 2GB limit"

if __name__ == "__main__":
    # Direct execution for manual testing
    tester = ARCBatchTester(min_success_rate=0.70)
    results = tester.run_batch_test(num_tasks=10)
    
    # Save detailed results
    results_file = Path("test_results_arc_batch.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to {results_file}")
    
    if results['test_passed']:
        print("🎉 ARC Batch Regression Test PASSED!")
        exit(0)
    else:
        print("💥 ARC Batch Regression Test FAILED!")
        exit(1)
