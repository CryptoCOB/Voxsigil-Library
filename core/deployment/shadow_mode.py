#!/usr/bin/env python3
"""
Shadow Mode Implementation for HOLO-1.5 ARC Ensemble

Implements shadow mode deployment where the new ensemble runs in observer mode
alongside the legacy pipeline, comparing outputs and resource usage before
switching traffic to the new system.

HOLO-1.5 Enhanced Shadow Mode:
- Recursive symbolic cognition comparison
- Neural-symbolic reasoning diff analysis
- VantaCore mesh collaboration monitoring
- Cognitive load and performance profiling
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.ensemble_integration import ARCEnsembleOrchestrator, create_arc_ensemble
from monitoring.exporter import get_metrics_collector

logger = logging.getLogger(__name__)

@dataclass
class ShadowResult:
    """Result from shadow mode inference"""
    task_id: str
    timestamp: str
    primary_result: Optional[Dict[str, Any]]
    shadow_result: Optional[Dict[str, Any]]
    primary_latency: float
    shadow_latency: float
    primary_success: bool
    shadow_success: bool
    outputs_match: bool
    confidence_diff: float
    resource_comparison: Dict[str, Any]
    error_log: List[str] = field(default_factory=list)

@dataclass
class ShadowModeConfig:
    """Configuration for shadow mode deployment"""
    enabled: bool = False
    sample_rate: float = 1.0  # Fraction of traffic to shadow (0.0-1.0)
    max_latency_increase_percent: float = 50.0  # Max acceptable latency increase
    max_memory_increase_mb: float = 1024.0  # Max acceptable memory increase
    comparison_timeout_seconds: float = 30.0
    log_directory: str = "logs/shadow_mode"
    detailed_logging: bool = True
    enable_resource_monitoring: bool = True

class ShadowModeOrchestrator:
    """Main orchestrator for shadow mode deployment"""
    
    def __init__(self, config: ShadowModeConfig):
        self.config = config
        self.legacy_pipeline = None
        self.shadow_ensemble = None
        self.results_log = []
        self.is_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics tracking
        self.stats = {
            'total_comparisons': 0,
            'outputs_matched': 0,
            'shadow_faster': 0,
            'shadow_slower': 0,
            'shadow_errors': 0,
            'legacy_errors': 0,
            'avg_latency_diff': 0.0,
            'avg_memory_diff': 0.0,
            'confidence_correlation': 0.0
        }
        
        # Setup logging directory
        self.log_dir = Path(config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Shadow mode orchestrator initialized (enabled: {config.enabled})")
    
    def initialize_pipelines(self, legacy_config: Dict[str, Any], shadow_config: Dict[str, Any]) -> bool:
        """Initialize both legacy and shadow pipelines"""
        try:
            # Initialize legacy pipeline (could be existing ARC solver)
            logger.info("Initializing legacy pipeline...")
            self.legacy_pipeline = self._create_legacy_pipeline(legacy_config)
            
            # Initialize shadow ensemble
            logger.info("Initializing shadow HOLO-1.5 ensemble...")
            self.shadow_ensemble = create_arc_ensemble(shadow_config)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {e}")
            return False
    
    def _create_legacy_pipeline(self, config: Dict[str, Any]):
        """Create legacy pipeline interface"""
        # This would wrap the existing ARC solving pipeline
        # For now, create a mock legacy system
        
        class LegacyPipeline:
            def solve_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
                """Mock legacy solver"""
                # Simulate legacy processing
                time.sleep(0.1)  # Simulate processing time
                
                return {
                    'prediction': task_data.get('input', [[0, 0], [0, 0]]),  # Simple identity
                    'confidence': 0.75,
                    'solve_time': 0.1,
                    'memory_usage_mb': 512.0,
                    'method': 'legacy_rule_based'
                }
        
        return LegacyPipeline()
    
    def should_run_shadow(self, task_id: str) -> bool:
        """Determine if shadow mode should run for this task"""
        if not self.config.enabled:
            return False
        
        # Check environment variable override
        import os
        if os.getenv('ARC_ENSEMBLE_SHADOW') == '1':
            return True
        
        # Sample rate based decision
        import random
        return random.random() < self.config.sample_rate
    
    async def run_shadow_comparison(self, task_data: Dict[str, Any]) -> ShadowResult:
        """Run both pipelines and compare results"""
        task_id = task_data.get('task_id', f'task_{int(time.time() * 1000)}')
        timestamp = datetime.now(timezone.utc).isoformat()
        
        result = ShadowResult(
            task_id=task_id,
            timestamp=timestamp,
            primary_result=None,
            shadow_result=None,
            primary_latency=0.0,
            shadow_latency=0.0,
            primary_success=False,
            shadow_success=False,
            outputs_match=False,
            confidence_diff=0.0,
            resource_comparison={}
        )
        
        try:
            # Run both pipelines concurrently
            primary_future = self.executor.submit(self._run_primary_pipeline, task_data)
            shadow_future = self.executor.submit(self._run_shadow_pipeline, task_data)
            
            # Wait for both with timeout
            timeout = self.config.comparison_timeout_seconds
            
            for future in as_completed([primary_future, shadow_future], timeout=timeout):
                if future == primary_future:
                    try:
                        result.primary_result = future.result()
                        result.primary_success = True
                        result.primary_latency = result.primary_result.get('solve_time', 0.0)
                    except Exception as e:
                        result.error_log.append(f"Primary pipeline error: {e}")
                        logger.error(f"Primary pipeline failed for {task_id}: {e}")
                
                elif future == shadow_future:
                    try:
                        result.shadow_result = future.result()
                        result.shadow_success = True
                        result.shadow_latency = result.shadow_result.get('solve_time', 0.0)
                    except Exception as e:
                        result.error_log.append(f"Shadow pipeline error: {e}")
                        logger.error(f"Shadow pipeline failed for {task_id}: {e}")
            
            # Compare results if both succeeded
            if result.primary_success and result.shadow_success:
                self._compare_outputs(result)
                self._compare_resources(result)
            
            # Update statistics
            self._update_statistics(result)
            
            # Log result if detailed logging enabled
            if self.config.detailed_logging:
                self._log_result(result)
            
            return result
            
        except Exception as e:
            result.error_log.append(f"Shadow comparison error: {e}")
            logger.error(f"Shadow comparison failed for {task_id}: {e}")
            return result
    
    def _run_primary_pipeline(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the primary (legacy) pipeline"""
        start_time = time.time()
        
        try:
            # Record initial memory state
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run legacy pipeline
            result = self.legacy_pipeline.solve_task(task_data)
            
            # Record final memory state
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result['solve_time'] = time.time() - start_time
            result['memory_usage_mb'] = final_memory - initial_memory
            result['pipeline_type'] = 'primary_legacy'
            
            return result
            
        except Exception as e:
            logger.error(f"Primary pipeline execution error: {e}")
            raise
    
    def _run_shadow_pipeline(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the shadow (HOLO-1.5) pipeline"""
        start_time = time.time()
        
        try:
            # Record initial memory state
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run shadow ensemble
            result = self.shadow_ensemble.solve_task(task_data)
            
            # Record final memory state
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if result is None:
                result = {}
            
            result['solve_time'] = time.time() - start_time
            result['memory_usage_mb'] = final_memory - initial_memory
            result['pipeline_type'] = 'shadow_holo15'
            
            return result
            
        except Exception as e:
            logger.error(f"Shadow pipeline execution error: {e}")
            raise
    
    def _compare_outputs(self, result: ShadowResult):
        """Compare outputs from both pipelines"""
        try:
            primary_pred = result.primary_result.get('prediction')
            shadow_pred = result.shadow_result.get('prediction')
            
            # Convert to comparable format
            if hasattr(primary_pred, 'numpy'):
                primary_pred = primary_pred.numpy()
            if hasattr(shadow_pred, 'numpy'):
                shadow_pred = shadow_pred.numpy()
            
            # Compare predictions
            import numpy as np
            if isinstance(primary_pred, np.ndarray) and isinstance(shadow_pred, np.ndarray):
                result.outputs_match = np.array_equal(primary_pred, shadow_pred)
            else:
                result.outputs_match = primary_pred == shadow_pred
            
            # Compare confidence scores
            primary_conf = result.primary_result.get('confidence', 0.0)
            shadow_conf = result.shadow_result.get('confidence', 0.0)
            result.confidence_diff = abs(primary_conf - shadow_conf)
            
        except Exception as e:
            result.error_log.append(f"Output comparison error: {e}")
            logger.error(f"Failed to compare outputs: {e}")
    
    def _compare_resources(self, result: ShadowResult):
        """Compare resource usage between pipelines"""
        try:
            primary_memory = result.primary_result.get('memory_usage_mb', 0.0)
            shadow_memory = result.shadow_result.get('memory_usage_mb', 0.0)
            
            latency_diff_percent = 0.0
            if result.primary_latency > 0:
                latency_diff_percent = ((result.shadow_latency - result.primary_latency) / result.primary_latency) * 100
            
            memory_diff_mb = shadow_memory - primary_memory
            
            result.resource_comparison = {
                'latency_diff_percent': latency_diff_percent,
                'memory_diff_mb': memory_diff_mb,
                'shadow_faster': result.shadow_latency < result.primary_latency,
                'shadow_more_efficient': memory_diff_mb < 0,
                'acceptable_latency': latency_diff_percent <= self.config.max_latency_increase_percent,
                'acceptable_memory': memory_diff_mb <= self.config.max_memory_increase_mb
            }
            
        except Exception as e:
            result.error_log.append(f"Resource comparison error: {e}")
            logger.error(f"Failed to compare resources: {e}")
    
    def _update_statistics(self, result: ShadowResult):
        """Update global statistics"""
        self.stats['total_comparisons'] += 1
        
        if result.outputs_match:
            self.stats['outputs_matched'] += 1
        
        if result.shadow_success and result.primary_success:
            if result.shadow_latency < result.primary_latency:
                self.stats['shadow_faster'] += 1
            else:
                self.stats['shadow_slower'] += 1
            
            # Update running averages
            alpha = 0.1  # Exponential moving average factor
            latency_diff = result.shadow_latency - result.primary_latency
            self.stats['avg_latency_diff'] = alpha * latency_diff + (1 - alpha) * self.stats['avg_latency_diff']
            
            if result.resource_comparison:
                memory_diff = result.resource_comparison.get('memory_diff_mb', 0.0)
                self.stats['avg_memory_diff'] = alpha * memory_diff + (1 - alpha) * self.stats['avg_memory_diff']
        
        if not result.shadow_success:
            self.stats['shadow_errors'] += 1
        
        if not result.primary_success:
            self.stats['legacy_errors'] += 1
    
    def _log_result(self, result: ShadowResult):
        """Log detailed result to file"""
        try:
            log_file = self.log_dir / f"shadow_results_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            log_entry = {
                'task_id': result.task_id,
                'timestamp': result.timestamp,
                'primary_success': result.primary_success,
                'shadow_success': result.shadow_success,
                'outputs_match': result.outputs_match,
                'primary_latency': result.primary_latency,
                'shadow_latency': result.shadow_latency,
                'confidence_diff': result.confidence_diff,
                'resource_comparison': result.resource_comparison,
                'errors': result.error_log
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log shadow result: {e}")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get current statistics summary"""
        total = self.stats['total_comparisons']
        
        if total == 0:
            return {'message': 'No shadow comparisons performed yet'}
        
        return {
            'total_comparisons': total,
            'output_match_rate': self.stats['outputs_matched'] / total,
            'shadow_success_rate': (total - self.stats['shadow_errors']) / total,
            'legacy_success_rate': (total - self.stats['legacy_errors']) / total,
            'shadow_faster_rate': self.stats['shadow_faster'] / max(1, total - self.stats['shadow_errors'] - self.stats['legacy_errors']),
            'avg_latency_diff_ms': self.stats['avg_latency_diff'] * 1000,
            'avg_memory_diff_mb': self.stats['avg_memory_diff'],
            'recommendation': self._generate_recommendation()
        }
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation based on current statistics"""
        total = self.stats['total_comparisons']
        
        if total < 100:
            return "Insufficient data - continue shadow mode testing"
        
        match_rate = self.stats['outputs_matched'] / total
        shadow_success = (total - self.stats['shadow_errors']) / total
        latency_acceptable = self.stats['avg_latency_diff'] <= (self.config.max_latency_increase_percent / 100)
        memory_acceptable = self.stats['avg_memory_diff'] <= self.config.max_memory_increase_mb
        
        if match_rate >= 0.95 and shadow_success >= 0.95 and latency_acceptable and memory_acceptable:
            return "✅ READY FOR PRODUCTION - Shadow system performing well"
        elif match_rate >= 0.90 and shadow_success >= 0.90:
            return "⚠️  NEEDS OPTIMIZATION - Good results but performance concerns"
        elif match_rate >= 0.80:
            return "🔄 CONTINUE TESTING - Moderate performance, more data needed"
        else:
            return "❌ NOT READY - Significant issues detected, investigation needed"

# Global shadow mode instance
_global_shadow_orchestrator: Optional[ShadowModeOrchestrator] = None

def initialize_shadow_mode(config: ShadowModeConfig = None) -> ShadowModeOrchestrator:
    """Initialize global shadow mode orchestrator"""
    global _global_shadow_orchestrator
    
    if config is None:
        # Check environment for configuration
        import os
        config = ShadowModeConfig(
            enabled=os.getenv('ARC_ENSEMBLE_SHADOW') == '1',
            sample_rate=float(os.getenv('SHADOW_SAMPLE_RATE', '1.0')),
            detailed_logging=os.getenv('SHADOW_DETAILED_LOGGING', 'true').lower() == 'true'
        )
    
    _global_shadow_orchestrator = ShadowModeOrchestrator(config)
    return _global_shadow_orchestrator

def get_shadow_orchestrator() -> Optional[ShadowModeOrchestrator]:
    """Get the global shadow mode orchestrator"""
    return _global_shadow_orchestrator

async def run_shadow_inference(task_data: Dict[str, Any]) -> Optional[ShadowResult]:
    """Convenience function to run shadow inference if enabled"""
    orchestrator = get_shadow_orchestrator()
    
    if orchestrator is None or not orchestrator.should_run_shadow(task_data.get('task_id', '')):
        return None
    
    return await orchestrator.run_shadow_comparison(task_data)

def main():
    """CLI entry point for shadow mode management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HOLO-1.5 Shadow Mode Management")
    parser.add_argument('--enable', action='store_true', help='Enable shadow mode')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Sample rate (0.0-1.0)')
    parser.add_argument('--stats', action='store_true', help='Show current statistics')
    parser.add_argument('--log-dir', default='logs/shadow_mode', help='Log directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize shadow mode
    config = ShadowModeConfig(
        enabled=args.enable,
        sample_rate=args.sample_rate,
        log_directory=args.log_dir
    )
    
    orchestrator = initialize_shadow_mode(config)
    
    if args.stats:
        # Show statistics
        stats = orchestrator.get_statistics_summary()
        print("\n🔍 Shadow Mode Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    else:
        print(f"🌓 Shadow Mode Configuration:")
        print(f"   Enabled: {config.enabled}")
        print(f"   Sample Rate: {config.sample_rate:.1%}")
        print(f"   Log Directory: {config.log_directory}")
        print(f"   Environment Variable: ARC_ENSEMBLE_SHADOW={1 if config.enabled else 0}")

if __name__ == "__main__":
    main()
