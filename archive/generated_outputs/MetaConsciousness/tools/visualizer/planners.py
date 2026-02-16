from typing import Any, Dict, List, Set
import networkx as nx

class MigrationPlanner:
    def plan_dependency_migration(self, graph: nx.DiGraph, 
                                target_modules: Set[str]) -> Dict[str, Any]:
        """Plan the optimal order for migrating/updating dependencies."""
        order = []
        impacts = {}
        visited = set()
        
        # Topological sort for dependency order
        for module in nx.topological_sort(graph):
            if module in target_modules:
                dependencies = set(graph.successors(module))
                impacted = set(graph.predecessors(module))
                
                order.append({
                    'module': module,
                    'dependencies_to_update': dependencies - visited,
                    'impacted_modules': impacted,
                    'risk_level': len(impacted)
                })
                visited.add(module)
                
        return {
            'migration_order': order,
            'total_steps': len(order),
            'estimated_effort': sum(len(step['dependencies_to_update']) for step in order)
        }
