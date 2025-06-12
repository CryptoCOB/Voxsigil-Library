#!/usr/bin/env python3
"""
Canary Grid Validation System for Sleep Training Cycle (STC) Safety

This module implements a safety system that validates model performance
on fixed "canary" ARC patterns after each Sleep Training Cycle to detect
silent regression and prevent model degradation.

HOLO-1.5 Enhanced Safety:
- Recursive symbolic validation patterns
- Neural-symbolic safety assessment
- VantaCore mesh integrity checking
- Cognitive load drift detection
"""

import json
import torch
import numpy as np
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from core.ensemble_integration import ARCEnsembleOrchestrator, create_arc_ensemble
from core.meta_control import ComplexityMonitor

logger = logging.getLogger(__name__)

class CanaryPattern:
    """Individual canary test pattern for regression detection"""
    
    def __init__(self, pattern_id: str, description: str, 
                 input_grid: List[List[int]], expected_output: List[List[int]],
                 baseline_accuracy: float = 1.0):
        self.pattern_id = pattern_id
        self.description = description
        self.input_grid = np.array(input_grid)
        self.expected_output = np.array(expected_output)
        self.baseline_accuracy = baseline_accuracy
        self.current_accuracy = baseline_accuracy
        self.test_history = []
    
    def test_pattern(self, ensemble: ARCEnsembleOrchestrator) -> Tuple[float, Dict[str, Any]]:
        """Test this pattern against the current ensemble"""
        try:
            # Convert to tensor format
            input_tensor = torch.tensor(self.input_grid, dtype=torch.float32).unsqueeze(0)
            
            # Run inference
            result = ensemble.solve_task({
                'input': input_tensor,
                'task_type': 'canary_validation',
                'pattern_id': self.pattern_id
            })
            
            if result is None or 'prediction' not in result:
                logger.warning(f"No prediction for canary pattern {self.pattern_id}")
                return 0.0, {'error': 'no_prediction'}
            
            prediction = result['prediction']
            
            # Calculate accuracy
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            
            # Remove batch dimension if present
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]
            
            accuracy = float(np.array_equal(prediction, self.expected_output))
            
            # Update current accuracy with exponential moving average
            alpha = 0.3  # Smoothing factor
            self.current_accuracy = alpha * accuracy + (1 - alpha) * self.current_accuracy
            
            # Record test history
            test_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'accuracy': accuracy,
                'smoothed_accuracy': self.current_accuracy,
                'cognitive_load': result.get('cognitive_load', 0.0),
                'solve_time': result.get('solve_time', 0.0)
            }
            self.test_history.append(test_record)
            
            # Keep only last 100 records
            if len(self.test_history) > 100:
                self.test_history = self.test_history[-100:]
            
            return accuracy, test_record
            
        except Exception as e:
            logger.error(f"Error testing canary pattern {self.pattern_id}: {e}")
            return 0.0, {'error': str(e)}

class CanaryGridValidator:
    """Main validator for canary grid safety system"""
    
    def __init__(self, config_path: str = "logs/sleep_metrics.json"):
        self.config_path = Path(config_path)
        self.ensemble = None
        self.canary_patterns = []
        self.degradation_threshold = 0.05
        self.accuracy_threshold = 0.85
        self.load_configuration()
    
    def load_configuration(self):
        """Load existing canary configuration or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self._load_from_config(config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Creating default.")
                self._create_default_patterns()
        else:
            self._create_default_patterns()
    
    def _load_from_config(self, config: Dict[str, Any]):
        """Load canary patterns from configuration"""
        canary_config = config.get('canary_grid', {})
        self.degradation_threshold = canary_config.get('degradation_threshold', 0.05)
        self.accuracy_threshold = canary_config.get('accuracy_threshold', 0.85)
        
        patterns_data = canary_config.get('patterns', [])
        for pattern_data in patterns_data:
            pattern = CanaryPattern(
                pattern_id=pattern_data['pattern_id'],
                description=pattern_data['description'],
                input_grid=pattern_data['input'],
                expected_output=pattern_data['expected_output'],
                baseline_accuracy=pattern_data.get('baseline_accuracy', 1.0)
            )
            pattern.current_accuracy = pattern_data.get('current_accuracy', pattern.baseline_accuracy)
            self.canary_patterns.append(pattern)
    
    def _create_default_patterns(self):
        """Create default canary patterns for testing"""
        default_patterns = [
            {
                'pattern_id': 'identity_test',
                'description': 'Simple identity transformation - input equals output',
                'input': [[1, 2], [3, 4]],
                'expected_output': [[1, 2], [3, 4]]
            },
            {
                'pattern_id': 'mirror_horizontal',
                'description': 'Horizontal mirroring pattern',
                'input': [[1, 2, 3], [4, 5, 6]],
                'expected_output': [[3, 2, 1], [6, 5, 4]]
            },
            {
                'pattern_id': 'color_swap',
                'description': 'Simple color swapping rule',
                'input': [[1, 0, 1], [0, 1, 0]],
                'expected_output': [[0, 1, 0], [1, 0, 1]]
            },
            {
                'pattern_id': 'pattern_completion',
                'description': 'Basic pattern completion task',
                'input': [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                'expected_output': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            }
        ]
        
        for pattern_data in default_patterns:
            pattern = CanaryPattern(**pattern_data)
            self.canary_patterns.append(pattern)
    
    def initialize_ensemble(self) -> bool:
        """Initialize the ARC ensemble for testing"""
        try:
            config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'memory_efficient': True,
                'canary_mode': True  # Special mode for validation
            }
            self.ensemble = create_arc_ensemble(config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {e}")
            return False
    
    def run_canary_validation(self) -> Dict[str, Any]:
        """Run complete canary validation and return results"""
        logger.info("🐤 Running Canary Grid Validation...")
        
        if not self.initialize_ensemble():
            return {'status': 'ERROR', 'message': 'Failed to initialize ensemble'}
        
        validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'UNKNOWN',
            'pattern_results': [],
            'overall_accuracy': 0.0,
            'degraded_patterns': [],
            'recommendations': []
        }
        
        total_accuracy = 0.0
        degraded_count = 0
        
        for pattern in self.canary_patterns:
            logger.info(f"Testing canary pattern: {pattern.pattern_id}")
            
            accuracy, test_record = pattern.test_pattern(self.ensemble)
            
            # Check for degradation
            degradation = pattern.baseline_accuracy - pattern.current_accuracy
            is_degraded = degradation > self.degradation_threshold
            
            pattern_result = {
                'pattern_id': pattern.pattern_id,
                'description': pattern.description,
                'baseline_accuracy': pattern.baseline_accuracy,
                'current_accuracy': pattern.current_accuracy,
                'latest_accuracy': accuracy,
                'degradation': degradation,
                'is_degraded': is_degraded,
                'test_record': test_record
            }
            
            validation_results['pattern_results'].append(pattern_result)
            total_accuracy += pattern.current_accuracy
            
            if is_degraded:
                degraded_count += 1
                validation_results['degraded_patterns'].append(pattern.pattern_id)
                logger.warning(f"⚠️  Pattern {pattern.pattern_id} degraded by {degradation:.3f}")
        
        # Calculate overall metrics
        validation_results['overall_accuracy'] = total_accuracy / len(self.canary_patterns)
        
        # Determine overall status
        if degraded_count == 0 and validation_results['overall_accuracy'] >= self.accuracy_threshold:
            validation_results['overall_status'] = 'HEALTHY'
        elif degraded_count <= 1 and validation_results['overall_accuracy'] >= self.accuracy_threshold * 0.95:
            validation_results['overall_status'] = 'WARNING'
            validation_results['recommendations'].append('Monitor degraded patterns closely')
        else:
            validation_results['overall_status'] = 'CRITICAL'
            validation_results['recommendations'].append('Abort STC promotion - significant degradation detected')
        
        # Additional recommendations
        if validation_results['overall_accuracy'] < self.accuracy_threshold:
            validation_results['recommendations'].append(
                f'Overall accuracy {validation_results["overall_accuracy"]:.3f} below threshold {self.accuracy_threshold}'
            )
        
        logger.info(f"🎯 Canary validation complete: {validation_results['overall_status']}")
        logger.info(f"   Overall accuracy: {validation_results['overall_accuracy']:.3f}")
        logger.info(f"   Degraded patterns: {degraded_count}/{len(self.canary_patterns)}")
        
        return validation_results
    
    def update_sleep_metrics(self, stc_cycle_id: str, validation_results: Dict[str, Any]) -> bool:
        """Update sleep_metrics.json with validation results"""
        try:
            # Load existing metrics
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    '_metadata': {
                        'description': 'Sleep Training Cycle (STC) metrics and canary validation results',
                        'version': '1.0',
                        'last_updated': datetime.now(timezone.utc).isoformat(),
                        'holo_version': '1.5.0'
                    },
                    'canary_grid': {},
                    'stc_cycles': []
                }
            
            # Update canary grid status
            metrics['canary_grid'] = {
                'description': 'Fixed test patterns that must maintain performance after each STC cycle',
                'accuracy_threshold': self.accuracy_threshold,
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'description': p.description,
                        'input': p.input_grid.tolist(),
                        'expected_output': p.expected_output.tolist(),
                        'baseline_accuracy': p.baseline_accuracy,
                        'current_accuracy': p.current_accuracy
                    }
                    for p in self.canary_patterns
                ],
                'overall_canary_accuracy': validation_results['overall_accuracy'],
                'degradation_threshold': self.degradation_threshold,
                'status': validation_results['overall_status']
            }
            
            # Add STC cycle record
            cycle_record = {
                'cycle_id': stc_cycle_id,
                'timestamp': validation_results['timestamp'],
                'canary_validation': validation_results,
                'promotion_status': 'APPROVED' if validation_results['overall_status'] in ['HEALTHY', 'WARNING'] else 'REJECTED'
            }
            
            metrics['stc_cycles'].insert(0, cycle_record)  # Most recent first
            
            # Keep only last 50 cycles
            if len(metrics['stc_cycles']) > 50:
                metrics['stc_cycles'] = metrics['stc_cycles'][:50]
            
            # Update metadata
            metrics['_metadata']['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            # Save updated metrics
            with open(self.config_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"📊 Updated sleep metrics: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update sleep metrics: {e}")
            return False
    
    def should_abort_promotion(self, validation_results: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if STC promotion should be aborted based on canary results"""
        status = validation_results['overall_status']
        
        if status == 'CRITICAL':
            return True, f"Critical degradation detected - overall status: {status}"
        
        if len(validation_results['degraded_patterns']) > len(self.canary_patterns) // 2:
            return True, f"More than half of canary patterns degraded: {validation_results['degraded_patterns']}"
        
        if validation_results['overall_accuracy'] < self.accuracy_threshold * 0.9:
            return True, f"Overall accuracy {validation_results['overall_accuracy']:.3f} critically low"
        
        return False, "Validation passed - promotion approved"


def main():
    """CLI entry point for canary validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run canary grid validation for STC safety")
    parser.add_argument('--stc-cycle-id', required=True, help='Sleep Training Cycle ID')
    parser.add_argument('--config-path', default='logs/sleep_metrics.json', help='Path to sleep metrics config')
    parser.add_argument('--abort-on-failure', action='store_true', help='Exit with error code if validation fails')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run validation
    validator = CanaryGridValidator(args.config_path)
    results = validator.run_canary_validation()
    
    # Update metrics
    validator.update_sleep_metrics(args.stc_cycle_id, results)
    
    # Check if promotion should be aborted
    should_abort, reason = validator.should_abort_promotion(results)
    
    print(f"\n🐤 Canary Grid Validation Results:")
    print(f"   Status: {results['overall_status']}")
    print(f"   Overall Accuracy: {results['overall_accuracy']:.3f}")
    print(f"   Degraded Patterns: {len(results['degraded_patterns'])}")
    print(f"   Promotion Decision: {'❌ ABORT' if should_abort else '✅ APPROVE'}")
    print(f"   Reason: {reason}")
    
    if results['recommendations']:
        print(f"\n📋 Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    # Exit with appropriate code
    if args.abort_on_failure and should_abort:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
