#!/usr/bin/env python3
"""
HOLO-1.5 Production Readiness Integration Script

This script integrates all production readiness components and validates
the complete system for deployment to wider teams and external users.

Components Integrated:
1. End-to-end regression testing
2. Continuous training safety with canary validation
3. Metrics/telemetry with Prometheus exporters
4. Shadow mode deployment capabilities
5. User-facing configuration management
6. Explainability and reasoning traces
7. Model card and documentation
8. Community onboarding quick-start

HOLO-1.5 Enhanced Integration:
- Recursive symbolic cognition validation
- Neural-symbolic synthesis verification
- VantaCore mesh collaboration testing
- Cognitive load and safety assessment
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Import all production readiness components
try:
    from tests.regression.test_arc_batch import ARCBatchTester
    from core.safety.canary_validator import CanaryGridValidator
    from monitoring.exporter import initialize_metrics, start_metrics_server
    from core.deployment.shadow_mode import initialize_shadow_mode, ShadowModeConfig
    from core.explainability.reasoning_traces import initialize_trace_capture
    from vanta_cli import VantaCLI
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all HOLO-1.5 components are properly installed")
    sys.exit(1)

logger = logging.getLogger(__name__)

class ProductionReadinessValidator:
    """Comprehensive production readiness validation"""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'UNKNOWN',
            'components': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("🚀 Starting HOLO-1.5 Production Readiness Validation")
    
    def validate_component(self, name: str, validator_func, critical: bool = True) -> bool:
        """Validate a single component"""
        logger.info(f"🔍 Validating {name}...")
        
        try:
            start_time = time.time()
            result = validator_func()
            validation_time = time.time() - start_time
            
            success = result.get('success', False) if isinstance(result, dict) else bool(result)
            
            self.validation_results['components'][name] = {
                'status': 'PASS' if success else 'FAIL',
                'critical': critical,
                'validation_time_seconds': validation_time,
                'details': result if isinstance(result, dict) else {'result': result}
            }
            
            if success:
                logger.info(f"✅ {name} validation PASSED ({validation_time:.1f}s)")
            else:
                logger.error(f"❌ {name} validation FAILED ({validation_time:.1f}s)")
                if critical:
                    self.validation_results['critical_issues'].append(f"{name} validation failed")
                else:
                    self.validation_results['warnings'].append(f"{name} validation failed (non-critical)")
            
            return success
            
        except Exception as e:
            logger.error(f"💥 {name} validation crashed: {e}")
            self.validation_results['components'][name] = {
                'status': 'ERROR',
                'critical': critical,
                'error': str(e),
                'validation_time_seconds': 0.0
            }
            
            if critical:
                self.validation_results['critical_issues'].append(f"{name} validation error: {e}")
            
            return False
    
    def validate_end_to_end_testing(self) -> Dict[str, Any]:
        """Validate end-to-end regression testing"""
        try:
            tester = ARCBatchTester(min_success_rate=0.70)
            results = tester.run_batch_test(num_tasks=10)
            
            return {
                'success': results.get('test_passed', False),
                'success_rate': results.get('success_rate', 0.0),
                'tasks_solved': results.get('tasks_solved', 0),
                'tasks_attempted': results.get('tasks_attempted', 0),
                'performance_metrics': results.get('performance_metrics', {})
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_canary_system(self) -> Dict[str, Any]:
        """Validate canary grid safety system"""
        try:
            validator = CanaryGridValidator()
            results = validator.run_canary_validation()
            
            return {
                'success': results['overall_status'] in ['HEALTHY', 'WARNING'],
                'status': results['overall_status'],
                'overall_accuracy': results['overall_accuracy'],
                'degraded_patterns': len(results['degraded_patterns']),
                'recommendations': results.get('recommendations', [])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_metrics_system(self) -> Dict[str, Any]:
        """Validate metrics and telemetry system"""
        try:
            # Initialize metrics (don't start server for validation)
            collector = initialize_metrics()
            
            # Test metric recording
            collector.record_task_processed('test_agent', 'test_task', 'success')
            collector.update_cognitive_load('test_agent', 'test_subsystem', 2.5)
            
            # Check if metrics are being recorded
            stats = collector.get_statistics()
            
            return {
                'success': True,
                'collector_initialized': collector is not None,
                'metrics_available': len(collector.metrics) > 0,
                'stats': stats
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_shadow_mode(self) -> Dict[str, Any]:
        """Validate shadow mode deployment system"""
        try:
            config = ShadowModeConfig(enabled=False)  # Don't actually enable
            orchestrator = initialize_shadow_mode(config)
            
            # Test configuration and statistics
            stats = orchestrator.get_statistics_summary()
            
            return {
                'success': True,
                'orchestrator_initialized': orchestrator is not None,
                'configuration_valid': config.enabled is False,
                'stats': stats
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_explainability(self) -> Dict[str, Any]:
        """Validate explainability and reasoning traces"""
        try:
            # Initialize trace capture
            capture = initialize_trace_capture(
                output_directory="logs/test_traces",
                max_trace_files=100,
                detailed_mode=True
            )
            
            # Test trace creation and finalization
            trace_id = capture.start_trace(
                task_id="test_task",
                task_type="validation",
                complexity_level="trivial",
                input_summary="Test input for validation"
            )
            
            # Add test step
            from core.explainability.reasoning_traces import ReasoningStepType
            step_id = capture.add_step(
                step_type=ReasoningStepType.VERIFICATION,
                agent_name="test_agent",
                subsystem="validation",
                description="Test reasoning step",
                input_data={'test': 'input'},
                output_data={'test': 'output'},
                confidence=0.9
            )
            
            # Finalize trace
            trace_file = capture.finalize_trace()
            
            return {
                'success': trace_file is not None,
                'trace_id': trace_id,
                'step_id': step_id,
                'trace_file': trace_file,
                'stats': capture.get_statistics()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_configuration_system(self) -> Dict[str, Any]:
        """Validate user-facing configuration system"""
        try:
            # Test CLI initialization
            cli = VantaCLI()
            
            # Test configuration loading
            config = cli.load_config()
            
            # Validate config structure
            has_ensemble = 'ensemble' in config
            has_logging = 'logging' in config
            
            return {
                'success': has_ensemble and has_logging,
                'cli_initialized': cli is not None,
                'config_loaded': len(config) > 0,
                'has_ensemble_config': has_ensemble,
                'has_logging_config': has_logging,
                'config_keys': list(config.keys())
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness"""
        try:
            required_files = [
                'README.md',
                'MODEL_CARD.md',
                'LICENSES_THIRD_PARTY.md',
                'config/default.yaml'
            ]
            
            file_status = {}
            all_present = True
            
            for file_path in required_files:
                path = Path(file_path)
                exists = path.exists()
                file_status[file_path] = {
                    'exists': exists,
                    'size_bytes': path.stat().st_size if exists else 0
                }
                
                if not exists:
                    all_present = False
            
            # Check README quick-start
            readme_path = Path('README.md')
            has_quick_start = False
            if readme_path.exists():
                content = readme_path.read_text()
                has_quick_start = 'pip install vox-sigil' in content and 'vanta demo arc' in content
            
            return {
                'success': all_present and has_quick_start,
                'all_files_present': all_present,
                'has_quick_start': has_quick_start,
                'file_status': file_status
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production readiness validation"""
        logger.info("🎯 Running Complete Production Readiness Validation")
        
        # Define validation components
        validations = [
            ('End-to-End Testing', self.validate_end_to_end_testing, True),
            ('Canary Safety System', self.validate_canary_system, True),
            ('Metrics & Telemetry', self.validate_metrics_system, True),
            ('Shadow Mode Deployment', self.validate_shadow_mode, False),
            ('Explainability System', self.validate_explainability, True),
            ('Configuration System', self.validate_configuration_system, True),
            ('Documentation', self.validate_documentation, False)
        ]
        
        # Run all validations
        critical_failures = 0
        total_validations = len(validations)
        
        for name, validator, critical in validations:
            success = self.validate_component(name, validator, critical)
            if not success and critical:
                critical_failures += 1
        
        # Generate overall status
        if critical_failures == 0:
            if len(self.validation_results['warnings']) == 0:
                self.validation_results['overall_status'] = 'READY_FOR_PRODUCTION'
                self.validation_results['recommendations'].append(
                    "🎉 All systems validated - ready for production deployment!"
                )
            else:
                self.validation_results['overall_status'] = 'READY_WITH_WARNINGS'
                self.validation_results['recommendations'].append(
                    "✅ Core systems validated - address warnings before production"
                )
        else:
            self.validation_results['overall_status'] = 'NOT_READY'
            self.validation_results['recommendations'].append(
                f"❌ {critical_failures} critical issues must be resolved before deployment"
            )
        
        # Add general recommendations
        self._add_general_recommendations()
        
        return self.validation_results
    
    def _add_general_recommendations(self):
        """Add general recommendations for production deployment"""
        recommendations = [
            "📊 Enable monitoring with 'vanta monitor --port 8000'",
            "🐤 Set up canary validation with hourly checks",
            "🌓 Use shadow mode for initial production rollout",
            "📝 Enable reasoning traces for debugging support",
            "🔧 Configure resource limits based on hardware",
            "📚 Review MODEL_CARD.md for deployment guidelines",
            "🛡️ Implement backup and recovery procedures",
            "👥 Train operations team on monitoring and alerts"
        ]
        
        self.validation_results['recommendations'].extend(recommendations)
    
    def save_validation_report(self, output_file: str = None) -> str:
        """Save validation report to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"production_readiness_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"📄 Validation report saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            return ""
    
    def print_summary(self):
        """Print validation summary"""
        status = self.validation_results['overall_status']
        components = self.validation_results['components']
        
        print(f"\n{'='*60}")
        print(f"🎯 HOLO-1.5 Production Readiness Validation Summary")
        print(f"{'='*60}")
        
        # Overall status
        status_emoji = {
            'READY_FOR_PRODUCTION': '🎉',
            'READY_WITH_WARNINGS': '⚠️',
            'NOT_READY': '❌',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 Overall Status: {status_emoji.get(status, '❓')} {status}")
        
        # Component status
        print(f"\n🔍 Component Validation Results:")
        for name, result in components.items():
            status_icon = '✅' if result['status'] == 'PASS' else '❌' if result['status'] == 'FAIL' else '💥'
            critical_marker = ' (CRITICAL)' if result.get('critical', False) else ''
            print(f"  {status_icon} {name}{critical_marker}: {result['status']}")
        
        # Critical issues
        if self.validation_results['critical_issues']:
            print(f"\n🚨 Critical Issues:")
            for issue in self.validation_results['critical_issues']:
                print(f"  • {issue}")
        
        # Warnings
        if self.validation_results['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in self.validation_results['warnings']:
                print(f"  • {warning}")
        
        # Recommendations
        if self.validation_results['recommendations']:
            print(f"\n📋 Recommendations:")
            for rec in self.validation_results['recommendations']:
                print(f"  • {rec}")
        
        print(f"\n{'='*60}")

def main():
    """Main entry point for production readiness validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HOLO-1.5 Production Readiness Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script validates all production readiness components for HOLO-1.5:

1. End-to-end regression testing (≥70% success rate)
2. Continuous training safety with canary validation
3. Metrics/telemetry with Prometheus exporters
4. Shadow mode deployment capabilities
5. User-facing configuration management
6. Explainability and reasoning traces
7. Documentation and model card
8. Community onboarding features

Exit codes:
  0: Ready for production
  1: Ready with warnings
  2: Not ready - critical issues
  3: Validation error
        """
    )
    
    parser.add_argument('--output', type=str, help='Output file for validation report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--component', type=str, help='Validate specific component only')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = ProductionReadinessValidator()
    
    if args.component:
        # Validate specific component only
        component_validators = {
            'testing': validator.validate_end_to_end_testing,
            'canary': validator.validate_canary_system,
            'metrics': validator.validate_metrics_system,
            'shadow': validator.validate_shadow_mode,
            'explainability': validator.validate_explainability,
            'config': validator.validate_configuration_system,
            'docs': validator.validate_documentation
        }
        
        if args.component in component_validators:
            result = validator.validate_component(
                args.component.title(), 
                component_validators[args.component]
            )
            exit_code = 0 if result else 2
        else:
            print(f"❌ Unknown component: {args.component}")
            print(f"Available components: {', '.join(component_validators.keys())}")
            exit_code = 3
    else:
        # Run full validation
        results = validator.run_full_validation()
        
        # Save report
        if args.output:
            validator.save_validation_report(args.output)
        else:
            validator.save_validation_report()
        
        # Print summary
        validator.print_summary()
        
        # Determine exit code
        status = results['overall_status']
        if status == 'READY_FOR_PRODUCTION':
            exit_code = 0
        elif status == 'READY_WITH_WARNINGS':
            exit_code = 1
        else:
            exit_code = 2
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
