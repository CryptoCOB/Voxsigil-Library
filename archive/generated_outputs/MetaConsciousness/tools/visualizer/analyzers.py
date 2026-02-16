import os
import ast
import logging
from typing import Dict, Set, List, Any, Optional, Tuple
import networkx as nx
from .utils import parse_python_file, calculate_module_complexity, is_excluded

logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    def __init__(self, root_dir: str, exclude_patterns: List[str] = None, ignore_modules: List[str] = None):
        self.root_dir = root_dir
        self.exclude_patterns = exclude_patterns or []
        self.ignore_modules = set(ignore_modules or [])
        
    def scan_modules(self, path: str) -> Dict[str, Dict[str, Any]]:
        modules = {}
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                if is_excluded(file_path, self.exclude_patterns):
                    continue

                rel_path = os.path.relpath(file_path, self.root_dir)
                module_name = rel_path.replace(os.path.sep, '.').replace('.py', '')
                
                if module_name in self.ignore_modules:
                    continue
                    
                stats = parse_python_file(file_path)
                modules[module_name] = {
                    'path': file_path,
                    'stats': stats,
                    'complexity': calculate_module_complexity(stats),
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path)
                }
                
        return modules

    def analyze_dependencies(self, modules: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        dependencies = {}
        for module_name, module_info in modules.items():
            deps = set()
            try:
                for imp in module_info['stats'].get('imports', []):
                    if not imp:  # Skip empty imports
                        continue
                    try:
                        base_module = imp.split('.')[0]
                        if base_module in modules and base_module != module_name:
                            deps.add(base_module)
                    except (AttributeError, IndexError) as e:
                        logger.warning(f"Error processing import '{imp}' in module '{module_name}': {e}")
            except Exception as e:
                logger.error(f"Error analyzing dependencies for module '{module_name}': {e}")
            
            dependencies[module_name] = deps
        return dependencies

    def validate_dependencies(self, modules: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate all module dependencies and return any issues found."""
        issues = []
        
        for module_name, module_info in modules.items():
            # Check if all imported modules exist
            for imp in module_info['stats']['imports']:
                base_module = imp.split('.')[0]
                if base_module not in modules and not self._is_standard_library(base_module):
                    issues.append({
                        'type': 'missing_dependency',
                        'module': module_name,
                        'dependency': base_module,
                        'severity': 'error'
                    })

            # Check for circular dependencies
            if self._has_circular_dependency(module_name, module_info['stats']['imports'], set()):
                issues.append({
                    'type': 'circular_dependency',
                    'module': module_name,
                    'severity': 'warning'
                })

        return issues

    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of Python's standard library."""
        import sys
        import pkgutil
        
        stdlib_modules = {mod.name for mod in pkgutil.iter_modules()}
        stdlib_modules.update(sys.stdlib_module_names)
        return module_name in stdlib_modules

    def _has_circular_dependency(self, current: str, imports: List[str], visited: Set[str]) -> bool:
        """Check for circular dependencies recursively."""
        if current in visited:
            return True
        
        visited.add(current)
        for imp in imports:
            if not imp:  # Skip empty imports
                continue
            try:
                base_module = imp.split('.')[0]
                # Make sure we don't have a direct reference to self.modules
                # Instead get the modules dict as a parameter or via closure
                modules_dict = getattr(self, 'modules', {})
                if base_module in modules_dict:
                    if self._has_circular_dependency(
                        base_module, 
                        modules_dict[base_module]['stats'].get('imports', []),
                        visited.copy()
                    ):
                        return True
            except Exception as e:
                logger.warning(f"Error checking circular dependency for '{imp}': {e}")
        return False

    def update_import_counts(self, import_details: Dict[Tuple[str, str], Dict[str, Any]], 
                           from_module: str, to_module: str):
        """Update the count of imports between modules."""
        edge = (from_module, to_module)
        if edge not in import_details:
            import_details[edge] = {"type": "direct", "count": 0}
        import_details[edge]["count"] += 1

class MetricsCalculator:
    def calculate_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        metrics = {
            "orphans": self._find_orphans(graph),
            "entry_points": self._find_entry_points(graph),
            "complexity": self._calculate_complexity(graph),
            "coupling": self._calculate_coupling(graph),
            "cohesion": self._calculate_cohesion(graph)
        }
        return metrics
        
    def _find_orphans(self, graph: nx.DiGraph) -> Set[str]:
        nodes = set(graph.nodes())
        return {n for n in nodes if graph.in_degree(n) == 0 and graph.out_degree(n) > 0}

    def _find_entry_points(self, graph: nx.DiGraph) -> Set[str]:
        nodes = set(graph.nodes())
        return {n for n in nodes if graph.in_degree(n) == 0}

    def _calculate_complexity(self, graph: nx.DiGraph) -> Dict[str, float]:
        return {
            node: data.get('complexity', 1.0)
            for node, data in graph.nodes(data=True)
        }

    def _calculate_coupling(self, graph: nx.DiGraph) -> Dict[str, float]:
        coupling = {}
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            coupling[node] = (in_degree + out_degree) / (2 * len(graph.nodes()))
        return coupling

    def _calculate_cohesion(self, graph: nx.DiGraph) -> Dict[str, float]:
        cohesion = {}
        for node in graph.nodes():
            neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
            if not neighbors:
                cohesion[node] = 0.0
                continue
            
            connections = 0
            possible_connections = len(neighbors) * (len(neighbors) - 1)
            
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and graph.has_edge(n1, n2):
                        connections += 1
                        
            cohesion[node] = connections / max(1, possible_connections)
        return cohesion

class CodeQualityAnalyzer:
    def __init__(self):
        self.code_smells = {
            'god_module': lambda stats: len(stats['classes']) + len(stats['functions']) > 20,
            'high_complexity': lambda stats: calculate_module_complexity(stats) > 50,
            'poor_cohesion': lambda stats: len(stats['imports']) > 15,
            'missing_docs': lambda stats: not stats['docstring']
        }

    def analyze_code_quality(self, modules: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        quality_metrics = {}
        for module_name, module_info in modules.items():
            metrics = self._calculate_module_quality(module_info)
            quality_metrics[module_name] = metrics
        return quality_metrics
    
    def _calculate_module_quality(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        stats = module_info['stats']
        smells = [name for name, check in self.code_smells.items() if check(stats)]
        
        return {
            'maintainability': self._calculate_maintainability_index(stats),
            'documentation_quality': self._analyze_documentation(stats),
            'code_smells': smells,
            'architecture_violations': self._check_architecture_rules(module_info)
        }

    def _calculate_maintainability_index(self, stats: Dict[str, Any]) -> float:
        # Implementation of maintainability index calculation
        pass

    def _calculate_test_coverage(self, module_path: str) -> float:
        # Implementation of test coverage calculation
        pass

    def _calculate_documentation_score(self, stats: Dict[str, Any]) -> float:
        # Implementation of documentation score calculation
        pass

    def _detect_code_smells(self, stats: Dict[str, Any]) -> List[str]:
        # Implementation of code smell detection
        pass

    def _analyze_documentation(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Analyze documentation quality."""
        doc_stats = {
            'has_module_doc': bool(stats['docstring']),
            'doc_coverage': sum(1 for f in stats['functions'] if f.get('docstring')) / len(stats['functions']) if stats['functions'] else 0,
            'doc_quality': self._assess_doc_quality(stats['docstring']) if stats['docstring'] else 0
        }
        return doc_stats

    def _check_architecture_rules(self, module_info: Dict[str, Any]) -> List[str]:
        """Check for architecture rule violations."""
        violations = []
        path = module_info['path']
        if 'tests' in path and '..' in module_info['stats'].get('imports', []):
            violations.append('test_isolation')
        if 'domain' in path and 'infrastructure' in str(module_info['stats'].get('imports', [])):
            violations.append('layering_violation')
        return violations

class ImpactAnalyzer:
    def analyze_change_impact(self, graph: nx.DiGraph, module_name: str) -> Dict[str, Any]:
        """Analyze the impact of changing a specific module."""
        impacted_modules = set()
        impact_levels = {"high": set(), "medium": set(), "low": set()}
        
        # Direct dependents (high impact)
        direct = set(graph.predecessors(module_name))
        impact_levels["high"] = direct
        impacted_modules.update(direct)
        
        # Indirect dependents (medium impact)
        for mod in direct:
            indirect = set(graph.predecessors(mod)) - impacted_modules
            impact_levels["medium"].update(indirect)
            impacted_modules.update(indirect)
        
        # Distantly related (low impact)
        all_connected = set(nx.descendants(graph, module_name))
        impact_levels["low"] = all_connected - impacted_modules
        
        return {
            "total_impacted": len(all_connected),
            "impact_levels": impact_levels,
            "risk_score": len(direct) * 3 + len(impact_levels["medium"]) * 2 + len(impact_levels["low"])
        }

class APIAnalyzer:
    def analyze_api_surface(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the public API surface of a module."""
        try:
            with open(module_info['path'], 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            
            public_api = {
                'classes': [],
                'functions': [],
                'constants': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if not node.name.startswith('_'):
                        api_type = 'classes' if isinstance(node, ast.ClassDef) else 'functions'
                        public_api[api_type].append({
                            'name': node.name,
                            'docstring': ast.get_docstring(node),
                            'line_number': node.lineno
                        })
            
            # Find constants (module-level assignments to non-private names)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            public_api['constants'].append({
                                'name': target.id,
                                'line_number': node.lineno
                            })
            
            return public_api
        except Exception as e:
            logger.error(f"Error analyzing API surface: {e}")
            return {'classes': [], 'functions': [], 'constants': []}

class TrendAnalyzer:
    def analyze_trends(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in code quality over time."""
        if not snapshots:
            return {"trends": {}, "alerts": [], "recommendations": []}
            
        trends = {
            'complexity': self._calculate_trend([s.get('complexity', 0) for s in snapshots]),
            'dependencies': self._calculate_trend([len(s.get('dependencies', {})) for s in snapshots]),
            'test_coverage': self._calculate_trend([s.get('test_coverage', 0) for s in snapshots]),
            'maintainability': self._calculate_trend([s.get('maintainability', 0) for s in snapshots])
        }
        
        return {
            'trends': trends,
            'alerts': self._generate_trend_alerts(trends),
            'recommendations': self._generate_trend_recommendations(trends)
        }
        
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        if not values or len(values) < 2:
            return {"direction": 0, "rate": 0, "current": values[-1] if values else 0}
            
        current = values[-1]
        previous = values[-2] 
        direction = 1 if current > previous else (-1 if current < previous else 0)
        rate = ((current - previous) / previous) if previous else 0
        
        return {"direction": direction, "rate": rate, "current": current}
        
    def _generate_trend_alerts(self, trends: Dict[str, Dict[str, float]]) -> List[str]:
        alerts = []
        if trends.get('complexity', {}).get('direction', 0) > 0 and trends.get('complexity', {}).get('rate', 0) > 0.1:
            alerts.append("Complexity increasing rapidly")
        if trends.get('test_coverage', {}).get('direction', 0) < 0:
            alerts.append("Test coverage decreasing")
        return alerts
        
    def _generate_trend_recommendations(self, trends: Dict[str, Dict[str, float]]) -> List[str]:
        recommendations = []
        if trends.get('complexity', {}).get('direction', 0) > 0:
            recommendations.append("Consider refactoring to reduce complexity")
        if trends.get('test_coverage', {}).get('direction', 0) < 0:
            recommendations.append("Improve test coverage")
        return recommendations

class DebtAnalyzer:
    def estimate_technical_debt(self, module_info: Dict[str, Any],
                              metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate technical debt in terms of effort and risk."""
        debt_factors = {
            'complex_code': self._estimate_complexity_debt(module_info),
            'missing_tests': self._estimate_testing_debt(metrics),
            'poor_structure': self._estimate_structural_debt(metrics),
            'documentation': self._estimate_documentation_debt(module_info)
        }
        
        total_hours = sum(factor.get('estimated_hours', 0) for factor in debt_factors.values())
        
        return {
            'total_debt_hours': total_hours,
            'estimated_cost': total_hours * 100,  # Assuming $100/hour
            'breakdown': debt_factors,
            'priority': 'high' if total_hours > 40 else 'medium' if total_hours > 20 else 'low'
        }
        
    def _estimate_complexity_debt(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        complexity = module_info.get('complexity', 0)
        if complexity > 50:
            return {'estimated_hours': 16, 'reason': 'Very high complexity'}
        elif complexity > 30:
            return {'estimated_hours': 8, 'reason': 'High complexity'}
        elif complexity > 15:
            return {'estimated_hours': 4, 'reason': 'Moderate complexity'}
        return {'estimated_hours': 0, 'reason': 'Low complexity'}
        
    def _estimate_testing_debt(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        test_coverage = metrics.get('test_coverage', 0)
        if test_coverage < 0.2:
            return {'estimated_hours': 12, 'reason': 'Very low test coverage'}
        elif test_coverage < 0.5:
            return {'estimated_hours': 6, 'reason': 'Low test coverage'}
        elif test_coverage < 0.7:
            return {'estimated_hours': 3, 'reason': 'Moderate test coverage'}
        return {'estimated_hours': 0, 'reason': 'Good test coverage'}
        
    def _estimate_structural_debt(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        coupling = metrics.get('coupling', 0)
        cohesion = metrics.get('cohesion', 0)
        if coupling > 0.7 and cohesion < 0.3:
            return {'estimated_hours': 10, 'reason': 'Poor structure (high coupling, low cohesion)'}
        elif coupling > 0.5 or cohesion < 0.4:
            return {'estimated_hours': 5, 'reason': 'Moderate structural issues'}
        return {'estimated_hours': 0, 'reason': 'Good structure'}
        
    def _estimate_documentation_debt(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        stats = module_info.get('stats', {})
        has_docstring = bool(stats.get('docstring'))
        if not has_docstring:
            return {'estimated_hours': 4, 'reason': 'Missing module documentation'}
        elif not stats.get('functions', []):
            return {'estimated_hours': 2, 'reason': 'Incomplete documentation'}
        return {'estimated_hours': 0, 'reason': 'Adequate documentation'}

class HealthAnalyzer:
    def calculate_health_score(self, module_info: Dict[str, Any], 
                             metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall health score for a module."""
        weights = {
            'complexity': 0.3,
            'documentation': 0.2,
            'test_coverage': 0.2,
            'coupling': 0.15,
            'cohesion': 0.15
        }
        
        scores = {
            'complexity': self._score_complexity(module_info.get('complexity', 0)),
            'documentation': self._score_documentation(module_info.get('stats', {})),
            'test_coverage': metrics.get('test_coverage', 0.0),
            'coupling': 1.0 - metrics.get('coupling', 0.0),
            'cohesion': metrics.get('cohesion', 0.0)
        }
        
        total_score = sum(score * weights[metric] for metric, score in scores.items())
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'health_status': 'good' if total_score > 0.7 else 'fair' if total_score > 0.4 else 'poor'
        }
        
    def _score_complexity(self, complexity: float) -> float:
        if complexity <= 0:
            return 1.0
        elif complexity > 50:
            return 0.0
        else:
            return max(0.0, 1.0 - (complexity / 50.0))
            
    def _score_documentation(self, stats: Dict[str, Any]) -> float:
        has_module_doc = bool(stats.get('docstring', ''))
        functions = stats.get('functions', [])
        classes = stats.get('classes', [])
        
        if not functions and not classes:
            return 1.0 if has_module_doc else 0.5
            
        return 1.0 if has_module_doc else 0.25

class ArchitectureAnalyzer:
    def detect_patterns(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Detect common architecture patterns in the codebase."""
        patterns = {
            "layered": self._detect_layered_architecture(graph),
            "facade": self._detect_facade_pattern(graph),
            "mediator": self._detect_mediator_pattern(graph),
            "singleton": self._detect_singleton_modules(graph)
        }
        return patterns

    def _detect_layered_architecture(self, graph: nx.DiGraph) -> List[str]:
        """Detect layered architecture pattern."""
        # Default layer categories to check for
        layers = {"presentation": [], "business": [], "data": []}
        
        for node in graph.nodes():
            node_name = str(node).lower()
            if "ui" in node_name or "view" in node_name or "controller" in node_name:
                layers["presentation"].append(node)
            elif "service" in node_name or "manager" in node_name or "business" in node_name:
                layers["business"].append(node)
            elif "repository" in node_name or "dao" in node_name or "model" in node_name:
                layers["data"].append(node)
        
        # Only return layers that have modules
        return [layer for layer, modules in layers.items() if modules]
        
    def _detect_facade_pattern(self, graph: nx.DiGraph) -> List[str]:
        """Detect facade pattern implementations."""
        # Facade modules typically:
        # 1. Have 'facade' in their name
        # 2. Have many outgoing connections but few incoming connections
        # 3. Act as simplified interface to a complex subsystem
        candidates = []
        
        for node in graph.nodes():
            node_name = str(node).lower()
            out_degree = graph.out_degree(node)
            in_degree = graph.in_degree(node)
            
            if "facade" in node_name:
                candidates.append(node)
            elif out_degree > 3 and in_degree < out_degree / 2:
                # Potential facade: many outgoing, fewer incoming
                candidates.append(node)
                
        return candidates
        
    def _detect_mediator_pattern(self, graph: nx.DiGraph) -> List[str]:
        """Detect mediator pattern implementations."""
        # Mediator modules typically:
        # 1. Have 'mediator' in their name
        # 2. Many modules connect to it
        # 3. Reduces direct connections between other components
        candidates = []
        
        for node in graph.nodes():
            node_name = str(node).lower()
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if "mediator" in node_name:
                candidates.append(node)
            elif in_degree > 2 and out_degree > 2:
                # Check if it reduces connections between its neighbors
                predecessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))
                
                if len(predecessors) > 1 and len(successors) > 1:
                    candidates.append(node)
                    
        return candidates
        
    def _detect_singleton_modules(self, graph: nx.DiGraph) -> List[str]:
        """Detect potential singleton implementations."""
        # Look for modules that:
        # 1. Have 'singleton' in their name
        # 2. Are imported by many other modules
        candidates = []
        
        for node in graph.nodes():
            node_name = str(node).lower()
            in_degree = graph.in_degree(node)
            
            if "singleton" in node_name or "instance" in node_name:
                candidates.append(node)
            elif in_degree > graph.number_of_nodes() / 3:
                # Used by a significant portion of the codebase
                candidates.append(node)
                
        return candidates