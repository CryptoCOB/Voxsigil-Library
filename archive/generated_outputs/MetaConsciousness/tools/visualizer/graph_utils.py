import networkx as nx
from typing import Dict, Set, List, Any, Optional, Tuple
import logging
import community  # for community detection
import numpy as np

logger = logging.getLogger(__name__)

class GraphBuilder:
    def build_graph(self, modules: Dict[str, Dict[str, Any]], dependencies: Dict[str, Set[str]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        
        # Add nodes with module info
        for module_name, module_info in modules.items():
            graph.add_node(module_name, **module_info)
            
        # Add edges from dependencies
        for module_name, deps in dependencies.items():
            for dep in deps:
                if dep in modules:  # Only add edges for known modules
                    graph.add_edge(module_name, dep)
        
        return graph

    def analyze_communities(self, graph: nx.DiGraph) -> Dict[str, int]:
        """Detect module communities using Louvain method."""
        undirected = graph.to_undirected()
        return community.best_partition(undirected)

    def calculate_centrality(self, graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics."""
        return {
            'degree': nx.degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph),
            'eigenvector': nx.eigenvector_centrality_numpy(graph)
        }

    def find_bottlenecks(self, graph: nx.DiGraph) -> List[str]:
        """Identify bottleneck modules."""
        betweenness = nx.betweenness_centrality(graph)
        mean = np.mean(list(betweenness.values()))
        std = np.std(list(betweenness.values()))
        return [node for node, score in betweenness.items() if score > mean + 2*std]

class CycleDetector:
    def detect_cycles(self, graph: nx.DiGraph) -> List[List[str]]:
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                logger.warning(f"Found {len(cycles)} circular dependencies")
                for cycle in cycles[:3]:  # Show first 3 cycles
                    logger.warning(f"Cycle: {' -> '.join(cycle + [cycle[0]])}")
            return cycles
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return []
            
    def get_cycle_stats(self, cycles: List[List[str]]) -> Dict[str, Any]:
        return {
            "total_cycles": len(cycles),
            "avg_cycle_length": sum(len(c) for c in cycles) / len(cycles) if cycles else 0,
            "max_cycle_length": max((len(c) for c in cycles), default=0),
            "modules_in_cycles": len(set(module for cycle in cycles for module in cycle))
        }

    def analyze_cycle_impact(self, graph: nx.DiGraph, cycles: List[List[str]]) -> Dict[str, Any]:
        """Analyze the impact of breaking cycles."""
        impacts = {}
        for cycle in cycles:
            for node in cycle:
                temp_graph = graph.copy()
                edges_to_remove = [(cycle[i], cycle[(i+1)%len(cycle)]) 
                                 for i in range(len(cycle))]
                temp_graph.remove_edges_from(edges_to_remove)
                impacts[node] = {
                    'affected_modules': len(nx.descendants(temp_graph, node)),
                    'complexity_reduction': len(list(nx.simple_cycles(temp_graph)))
                }
        return impacts
