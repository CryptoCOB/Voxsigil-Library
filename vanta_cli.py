#!/usr/bin/env python3
"""
HOLO-1.5 VoxSigil Library CLI

Unified command-line interface for the HOLO-1.5 ensemble system.
Provides easy access to all major functionality with configuration management.

Usage:
  vanta --config path.yaml demo arc
  vanta train --epochs 10 --config custom.yaml
  vanta infer --task-file tasks.json
  vanta validate --canary-only
  vanta monitor --port 8080
"""

import argparse
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import core modules
from demo_novel_paradigms import main as demo_main
from core.safety.canary_validator import main as canary_main
from monitoring.exporter import main as monitor_main
from core.deployment.shadow_mode import main as shadow_main

logger = logging.getLogger(__name__)

class VantaCLI:
    """Main CLI class for HOLO-1.5 VoxSigil Library"""
    
    def __init__(self):
        self.config = {}
        self.config_file = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Try default locations
            default_paths = [
                "config/default.yaml",
                "default.yaml",
                Path.home() / ".voxsigil" / "config.yaml"
            ]
            
            for path in default_paths:
                if Path(path).exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self.config_file = config_path
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                self.config = {}
        else:
            logger.warning("No configuration file found, using defaults")
            self.config = self._get_default_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'ensemble': {
                'device': 'auto',
                'memory_efficient': True,
                'mode': 'production'
            },
            'logging': {
                'level': 'INFO',
                'file_logging': {'enabled': False}
            },
            'monitoring': {
                'metrics': {'enabled': False}
            }
        }
    
    def setup_logging(self, level: str = None):
        """Setup logging based on configuration"""
        if level is None:
            level = self.config.get('logging', {}).get('level', 'INFO')
        
        log_format = self.config.get('logging', {}).get('format', 
                                                       '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            stream=sys.stdout
        )
        
        # File logging if enabled
        file_config = self.config.get('logging', {}).get('file_logging', {})
        if file_config.get('enabled', False):
            log_dir = Path(file_config.get('log_directory', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'vanta.log')
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
    
    def cmd_demo(self, args):
        """Run demonstration of novel paradigms"""
        print("🚀 Running HOLO-1.5 Novel Paradigms Demo...")
        
        if args.task_type == 'arc':
            # Set up demo configuration
            import os
            os.environ['DEMO_MODE'] = '1'
            os.environ['DEMO_COMPLEXITY'] = args.complexity
            
            # Run the demo
            try:
                demo_main()
                print("✅ Demo completed successfully!")
            except Exception as e:
                print(f"❌ Demo failed: {e}")
                return 1
        else:
            print(f"Unknown demo type: {args.task_type}")
            return 1
        
        return 0
    
    def cmd_train(self, args):
        """Run training pipeline"""
        print("🎓 Training HOLO-1.5 Ensemble...")
        
        # TODO: Implement training pipeline
        print("Training functionality not yet implemented in this CLI")
        print("Use: python -m training.train_ensemble")
        return 0
    
    def cmd_infer(self, args):
        """Run inference on tasks"""
        print("🧠 Running Inference...")
        
        if args.task_file:
            try:
                with open(args.task_file, 'r') as f:
                    if args.task_file.endswith('.json'):
                        tasks = json.load(f)
                    else:
                        tasks = yaml.safe_load(f)
                
                print(f"Loaded {len(tasks)} tasks from {args.task_file}")
                
                # TODO: Implement inference pipeline
                print("Inference functionality not yet implemented in this CLI")
                print("Use: python -m inference.run_inference")
                
            except Exception as e:
                print(f"❌ Failed to load tasks: {e}")
                return 1
        else:
            print("No task file specified. Use --task-file option.")
            return 1
        
        return 0
    
    def cmd_validate(self, args):
        """Run validation tests"""
        print("🔍 Running Validation...")
        
        if args.canary_only:
            # Run canary validation
            stc_id = args.stc_cycle_id or f"manual_{int(time.time())}"
            
            # Prepare canary args
            canary_args = [
                '--stc-cycle-id', stc_id
            ]
            
            if args.config:
                canary_args.extend(['--config-path', str(Path(args.config).parent / 'sleep_metrics.json')])
            
            # Run canary validation
            sys.argv = ['canary_validator.py'] + canary_args
            return canary_main()
        else:
            # Run full validation suite
            print("Full validation functionality not yet implemented in this CLI")
            print("Use: python -m tests.regression.test_arc_batch")
            return 0
    
    def cmd_monitor(self, args):
        """Start monitoring dashboard"""
        print("📊 Starting Monitoring Dashboard...")
        
        # Prepare monitor args
        monitor_args = [
            '--port', str(args.port),
            '--host', args.host,
            '--interval', str(args.interval)
        ]
        
        if args.disable_gpu:
            monitor_args.append('--disable-gpu')
        
        # Run monitoring
        sys.argv = ['exporter.py'] + monitor_args
        return monitor_main()
    
    def cmd_shadow(self, args):
        """Manage shadow mode deployment"""
        print("🌓 Managing Shadow Mode...")
        
        # Prepare shadow args
        shadow_args = []
        
        if args.enable:
            shadow_args.append('--enable')
        
        if args.sample_rate:
            shadow_args.extend(['--sample-rate', str(args.sample_rate)])
        
        if args.stats:
            shadow_args.append('--stats')
        
        # Run shadow mode management
        sys.argv = ['shadow_mode.py'] + shadow_args
        return shadow_main()
    
    def cmd_config(self, args):
        """Configuration management"""
        if args.show:
            # Show current configuration
            print("📋 Current Configuration:")
            print(yaml.dump(self.config, default_flow_style=False, indent=2))
        
        elif args.validate_config:
            # Validate configuration file
            try:
                with open(args.validate_config, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ Configuration file {args.validate_config} is valid")
            except Exception as e:
                print(f"❌ Configuration file {args.validate_config} is invalid: {e}")
                return 1
        
        elif args.generate:
            # Generate default configuration
            default_config_path = Path(args.generate)
            default_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read the default config content
            default_yaml_path = Path(__file__).parent / "config" / "default.yaml"
            if default_yaml_path.exists():
                import shutil
                shutil.copy(default_yaml_path, default_config_path)
                print(f"✅ Generated default configuration at {default_config_path}")
            else:
                print(f"❌ Could not find default configuration template")
                return 1
        
        return 0

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog='vanta',
        description='HOLO-1.5 VoxSigil Library - Unified CLI for ARC reasoning ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vanta demo arc                           # Run ARC demonstration
  vanta --config custom.yaml demo arc      # Run demo with custom config
  vanta validate --canary-only             # Run canary validation only
  vanta monitor --port 8080                # Start monitoring on port 8080
  vanta shadow --enable --sample-rate 0.5  # Enable shadow mode with 50% sampling
  vanta config --show                      # Show current configuration
        """
    )
    
    # Global options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Override log level')
    parser.add_argument('--device', type=str, help='Override device (cpu, cuda, auto)')
    parser.add_argument('--version', action='version', version='HOLO-1.5 v1.5.0')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstrations')
    demo_parser.add_argument('task_type', choices=['arc'], help='Type of demo to run')
    demo_parser.add_argument('--complexity', choices=['trivial', 'moderate', 'complex', 'extremely_complex'],
                           default='moderate', help='Task complexity level')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the ensemble')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--task-file', type=str, required=True, help='JSON/YAML file with tasks')
    infer_parser.add_argument('--output', type=str, help='Output file for results')
    infer_parser.add_argument('--strategy', choices=['rule_based', 'neural', 'ensemble'],
                            default='ensemble', help='Inference strategy')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Run validation tests')
    validate_parser.add_argument('--canary-only', action='store_true', help='Run only canary validation')
    validate_parser.add_argument('--stc-cycle-id', type=str, help='Sleep training cycle ID')
    validate_parser.add_argument('--full-suite', action='store_true', help='Run full test suite')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring')
    monitor_parser.add_argument('--port', type=int, default=8000, help='HTTP server port')
    monitor_parser.add_argument('--host', type=str, default='0.0.0.0', help='HTTP server host')
    monitor_parser.add_argument('--interval', type=float, default=10.0, help='Collection interval')
    monitor_parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU monitoring')
    
    # Shadow mode command
    shadow_parser = subparsers.add_parser('shadow', help='Manage shadow mode')
    shadow_parser.add_argument('--enable', action='store_true', help='Enable shadow mode')
    shadow_parser.add_argument('--sample-rate', type=float, help='Sample rate (0.0-1.0)')
    shadow_parser.add_argument('--stats', action='store_true', help='Show shadow mode statistics')
    
    # Configuration command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--validate', dest='validate_config', type=str, help='Validate config file')
    config_parser.add_argument('--generate', type=str, help='Generate default config at path')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create CLI instance
    cli = VantaCLI()
    
    # Load configuration
    cli.load_config(args.config)
    
    # Override config with command line args
    if args.log_level:
        cli.config.setdefault('logging', {})['level'] = args.log_level
    if args.device:
        cli.config.setdefault('ensemble', {})['device'] = args.device
    
    # Setup logging
    cli.setup_logging()
    
    # Route to command
    if args.command == 'demo':
        return cli.cmd_demo(args)
    elif args.command == 'train':
        return cli.cmd_train(args)
    elif args.command == 'infer':
        return cli.cmd_infer(args)
    elif args.command == 'validate':
        return cli.cmd_validate(args)
    elif args.command == 'monitor':
        return cli.cmd_monitor(args)
    elif args.command == 'shadow':
        return cli.cmd_shadow(args)
    elif args.command == 'config':
        return cli.cmd_config(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    import time
    exit(main())
