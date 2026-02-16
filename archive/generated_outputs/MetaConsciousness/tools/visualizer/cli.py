import argparse
import logging
import sys
from typing import List
from .core import ModuleVisualizer

def parse_args(args: List[str] = None):
    parser = argparse.ArgumentParser(description="Module Dependency Visualization Tool")
    parser.add_argument("root_dir", help="Root directory of the Python project")
    parser.add_argument("--output-dir", "-o", help="Output directory for generated files")
    parser.add_argument("--exclude", "-e", action="append", help="Patterns to exclude")
    parser.add_argument("--subpath", "-s", help="Analyze specific subpath only")
    parser.add_argument("--ignore-modules", "-I", action="append", help="Modules to ignore")
    parser.add_argument("--color-by", choices=['default', 'complexity', 'coupling', 'cohesion'],
                       default='default', help="Color nodes based on metric")
    parser.add_argument("--format", choices=['json', 'graphml'], help="Export format")
    parser.add_argument("--compare", action="store_true", help="Compare with previous")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    # Removed the -i shorthand to avoid conflict
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive mode")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Watch for changes and update visualization")
    parser.add_argument("--metrics", "-m", choices=['all', 'complexity', 'coupling', 'cohesion'],
                       default='all', help="Metrics to calculate")
    parser.add_argument("--threshold", "-t", type=float, default=0.7,
                       help="Threshold for metrics warnings")
    parser.add_argument("--export-metrics", action="store_true",
                       help="Export metrics to CSV")
    parser.add_argument("--community-detection", action="store_true",
                       help="Perform community detection")
    parser.add_argument("--validate", action="store_true",
                       help="Perform comprehensive dependency validation")
    parser.add_argument("--show-issues", action="store_true",
                       help="Show detailed dependency issues")
    
    return parser.parse_args(args)

def print_help():
    """Display help information for interactive mode."""
    print("\nAvailable commands:")
    print("  h                 - Show this help message")
    print("  q                 - Quit interactive mode")
    print("  r                 - Run analysis and compare with previous")
    print("  focus <module>    - Focus on a specific module and its dependencies")
    print("  metrics           - Show detailed metrics for all modules")
    print("  cycles            - Show detailed information about dependency cycles")
    print("  validate          - Validate project dependencies")
    print("  impact <module>   - Analyze impact of changing a module")
    print("  health            - Show module health scores")
    print("  api <module>      - Show API surface of a module")
    print("  export <format>   - Export graph (formats: json, graphml, png)")

def interactive_mode(visualizer: ModuleVisualizer):
    """Start interactive analysis mode."""
    print("\nInteractive Mode - Type 'h' for help, 'q' to quit")
    
    while True:
        try:
            command = input("\nCommand> ").strip()
            
            if command == 'q':
                break
            elif command == 'h':
                print_help()
            elif command == 'r':
                result = visualizer.run(compare=True)
                print(f"Analysis complete: {result['modules_count']} modules, {result['dependencies_count']} dependencies")
            elif command.startswith('focus '):
                parts = command.split(maxsplit=1)
                if len(parts) < 2:
                    print("Error: Missing module name. Usage: focus <module_name>")
                    continue
                    
                module = parts[1]
                result = visualizer.focus_module(module)
                
                if result:
                    print(f"Generated focused visualization for {module}")
                else:
                    print(f"Could not focus on module: {module}")
            elif command == 'metrics':
                metrics = visualizer.show_metrics()
                print("\nMetrics Summary:")
                for category, values in metrics.items():
                    if isinstance(values, dict):
                        print(f"  {category}: {len(values)} items")
                    else:
                        print(f"  {category}: {len(values) if hasattr(values, '__len__') else values}")
            elif command == 'cycles':
                cycles = visualizer.show_cycles()
                print(f"\nFound {len(cycles)} cycles:")
                for i, cycle_info in enumerate(cycles[:5]):
                    print(f"  {i+1}. {' -> '.join(cycle_info['cycle'])} (length: {cycle_info['length']})")
                if len(cycles) > 5:
                    print(f"  ... and {len(cycles)-5} more")
            elif command == 'validate':
                validation = visualizer.validate_project()
                print(f"\nValidation: {validation['summary']['errors']} errors, {validation['summary']['warnings']} warnings")
            else:
                print(f"Unknown command: {command}. Type 'h' for help.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error executing command: {e}")

def main():
    args = parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    try:
        visualizer = ModuleVisualizer(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            exclude_patterns=args.exclude,
            subpath=args.subpath,
            ignore_modules=args.ignore_modules,
            color_by=args.color_by
        )
        
        if args.validate:
            try:
                validation_results = visualizer.validate_project()
                print("\nValidation Results:")
                print(f"Analyzed {validation_results['summary']['modules_analyzed']} modules")
                print(f"Found {validation_results['summary']['errors']} errors and "
                      f"{validation_results['summary']['warnings']} warnings")
                
                if args.show_issues:
                    for issue in validation_results['dependency_issues']:
                        print(f"\n{issue['severity'].upper()}: {issue['type']}")
                        print(f"Module: {issue['module']}")
                        if 'dependency' in issue:
                            print(f"Dependency: {issue['dependency']}")
            except Exception as e:
                print(f"Error during validation: {e}")
        
        if args.interactive:
            interactive_mode(visualizer)
        else:
            result = visualizer.run(
                compare=args.compare,
                export_format=args.format
            )
            
            print("\nVisualization Results:")
            print(f"- Modules analyzed: {result.get('modules_count', 0)}")
            print(f"- Dependencies found: {result.get('dependencies_count', 0)}")
            try:
                print(f"- Cycles detected: {result.get('cycles_found', 0)}")
                print(f"- Orphans found: {result.get('orphans_found', 0)}")
                print(f"- Entry points found: {result.get('entry_points_found', 0)}")
            except Exception as e:
                print(f"- Warning: Some metrics could not be calculated: {e}")
            
            print(f"- Output directory: {result.get('output_dir', 'unknown')}")
            
            if 'error' in result:
                print(f"\nWarning: Process completed with errors: {result['error']}")
                print("Some outputs may be incomplete or missing.")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
