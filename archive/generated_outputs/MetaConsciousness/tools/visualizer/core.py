import os
import time
import logging
from typing import Dict, Any, Optional, List

from .graph_utils import GraphBuilder, CycleDetector
from .exporters import HTMLExporter, GraphExporter, TableExporter
from .analyzers import DependencyAnalyzer, MetricsCalculator, CodeQualityAnalyzer

logger = logging.getLogger(__name__)

class ModuleVisualizer:
    def __init__(self,
                 root_dir: str,
                 output_dir: str = None,
                 exclude_patterns: List[str] = None,
                 subpath: Optional[str] = None,
                 ignore_modules: Optional[List[str]] = None,
                 color_by: str = 'default'):
        
        self.root_dir = os.path.abspath(root_dir)
        self.output_dir = output_dir or os.path.join(self.root_dir, "module_viz")
        self.subpath_dir = os.path.abspath(os.path.join(self.root_dir, subpath)) if subpath else self.root_dir
        
        self.analyzer = DependencyAnalyzer(
            root_dir=self.root_dir,
            exclude_patterns=exclude_patterns,
            ignore_modules=ignore_modules
        )
        
        self.graph_builder = GraphBuilder()
        self.cycle_detector = CycleDetector()
        self.metrics = MetricsCalculator()
        
        self.html_exporter = HTMLExporter(self.output_dir)
        self.graph_exporter = GraphExporter(self.output_dir)
        self.table_exporter = TableExporter(self.output_dir)
        
        self.color_by = color_by
        self.modules = {}
        self.graph = None

    def run(self, compare: bool = False, export_format: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting module visualization for: {self.root_dir}")

        try:
            # Analyze dependencies
            self.modules = self.analyzer.scan_modules(self.subpath_dir)
            
            if not self.modules:
                logger.warning("No modules found to analyze. Check the path and exclude patterns.")
                return {
                    "modules_count": 0,
                    "dependencies_count": 0,
                    "output_dir": self.output_dir,
                    "png_path": None,
                    "html_path": None,
                    "table_path": None,
                    "export_path": None,
                    "cycles_found": 0,
                    "orphans_found": 0,
                    "entry_points_found": 0,
                    "comparison": None,
                    "duration_seconds": time.time() - start_time
                }
                
            dependencies = self.analyzer.analyze_dependencies(self.modules)
            
            # Build and analyze graph
            self.graph = self.graph_builder.build_graph(self.modules, dependencies)
            cycles = self.cycle_detector.detect_cycles(self.graph)
            metrics = self.metrics.calculate_metrics(self.graph)
            
            # Generate outputs
            visualization_result = self.graph_exporter.generate_visualization(
                self.graph, 
                self.color_by,
                metrics
            ) or {}
            
            html_path = None
            try:
                html_result = self.html_exporter.generate_report(
                    self.graph,
                    cycles,
                    metrics,
                    self.color_by
                ) or {}
                html_path = html_result.get("path")
            except Exception as e:
                logger.error(f"Error generating HTML report: {e}")
            
            table_path = None
            try:
                table_result = self.table_exporter.generate_table(
                    self.graph,
                    cycles,
                    metrics,
                    self.root_dir
                ) or {}
                table_path = table_result.get("path")
            except Exception as e:
                logger.error(f"Error generating table: {e}")

            # Handle export and comparison if requested
            export_path = None
            if export_format:
                try:
                    export_path = self.graph_exporter.export_graph(self.graph, format=export_format)
                except Exception as e:
                    logger.error(f"Error exporting graph: {e}")

            comparison_results = None
            if compare:
                try:
                    comparison_results = self.graph_exporter.compare_with_previous(self.graph)
                except Exception as e:
                    logger.error(f"Error comparing with previous: {e}")

        except Exception as e:
            logger.error(f"Error during visualization process: {e}", exc_info=True)
            # Return minimal results on error
            return {
                "modules_count": len(self.modules) if hasattr(self, 'modules') else 0,
                "dependencies_count": 0,
                "output_dir": self.output_dir,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "cycles_found": 0,
                "orphans_found": 0,
                "entry_points_found": 0,
                "png_path": None,
                "html_path": None,
                "table_path": None,
                "export_path": None
            }

        duration = time.time() - start_time
        
        return {
            "modules_count": len(self.modules),
            "dependencies_count": self.graph.number_of_edges() if self.graph else 0,
            "output_dir": self.output_dir,
            "png_path": visualization_result.get("png_path"),
            "html_path": html_path,
            "table_path": table_path,
            "export_path": export_path,
            "cycles_found": len(cycles),
            "orphans_found": len(metrics.get("orphans", [])) if isinstance(metrics.get("orphans"), (list, set)) else 0,
            "entry_points_found": len(metrics.get("entry_points", [])) if isinstance(metrics.get("entry_points"), (list, set)) else 0,
            "comparison": comparison_results,
            "duration_seconds": duration
        }

    def focus_module(self, module_name: str) -> Dict[str, Any]:
        """Analyze and visualize dependencies for a specific module."""
        if not self.graph or module_name not in self.graph:
            logger.error(f"Module {module_name} not found in the graph")
            return {}

        # Get module neighborhood
        predecessors = set(self.graph.predecessors(module_name))
        successors = set(self.graph.successors(module_name))
        related_modules = predecessors | successors | {module_name}

        # Create subgraph
        subgraph = self.graph.subgraph(related_modules)
        
        # Generate focused visualization
        return self.graph_exporter.generate_visualization(
            subgraph,
            self.color_by,
            self.metrics.calculate_metrics(subgraph)
        )

    def show_metrics(self) -> Dict[str, Any]:
        """Display detailed metrics for all modules."""
        if not self.graph:
            return {}

        metrics = self.metrics.calculate_metrics(self.graph)
        cycles = self.cycle_detector.detect_cycles(self.graph)
        centrality = self.graph_builder.calculate_centrality(self.graph)
        bottlenecks = self.graph_builder.find_bottlenecks(self.graph)

        return {
            'basic_metrics': metrics,
            'cycles': cycles,
            'centrality': centrality,
            'bottlenecks': bottlenecks
        }

    def show_cycles(self) -> List[Dict[str, Any]]:
        """Analyze and display detailed information about dependency cycles."""
        if not self.graph:
            return []

        cycles = self.cycle_detector.detect_cycles(self.graph)
        impacts = self.cycle_detector.analyze_cycle_impact(self.graph, cycles)
        
        return [{
            'cycle': cycle,
            'length': len(cycle),
            'impacts': {node: impacts[node] for node in cycle}
        } for cycle in cycles]

    def validate_project(self) -> Dict[str, Any]:
        """Perform comprehensive project validation."""
        if not self.modules:
            self.modules = self.analyzer.scan_modules(self.subpath_dir)
        
        validation_results = {
            'dependency_issues': self.analyzer.validate_dependencies(self.modules),
            'code_quality': self._validate_code_quality(),
            'architecture': self._validate_architecture(),
            'test_coverage': self._analyze_test_coverage()
        }

        # Add severity counts
        validation_results['summary'] = {
            'errors': len([i for i in validation_results['dependency_issues'] 
                         if i['severity'] == 'error']),
            'warnings': len([i for i in validation_results['dependency_issues'] 
                           if i['severity'] == 'warning']),
            'modules_analyzed': len(self.modules)
        }

        return validation_results

    def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality metrics."""
        quality_analyzer = CodeQualityAnalyzer()
        return quality_analyzer.analyze_code_quality(self.modules)

    def _validate_architecture(self) -> Dict[str, List[str]]:
        """Check architectural constraints."""
        violations = []
        
        # Check layer dependencies
        layers = ['presentation', 'domain', 'infrastructure']
        for module_name, module_info in self.modules.items():
            current_layer = next((l for l in layers if l in module_name), None)
            if current_layer:
                layer_idx = layers.index(current_layer)
                for imp in module_info['stats']['imports']:
                    imported_layer = next((l for l in layers if l in imp), None)
                    if imported_layer and layers.index(imported_layer) < layer_idx:
                        violations.append(f"Layer violation: {module_name} -> {imp}")

        return {'violations': violations}

    def _analyze_test_coverage(self) -> Dict[str, float]:
        """Analyze test coverage for modules."""
        import coverage
        test_coverage = {}
        
        try:
            cov = coverage.Coverage()
            for module_name, module_info in self.modules.items():
                if not module_name.startswith('test'):
                    cov.start()
                    try:
                        __import__(module_name)
                        test_coverage[module_name] = cov.get_data().line_percent(module_info['path'])
                    except ImportError:
                        test_coverage[module_name] = 0.0
                    finally:
                        cov.stop()
                        cov.erase()
        except Exception as e:
            logger.error(f"Error analyzing test coverage: {e}")
            
        return test_coverage
