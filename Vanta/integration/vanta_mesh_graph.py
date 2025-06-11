from typing import Any, Dict, List


class VantaMeshGraph:
    """Generate a simple agent graph from the live registry."""

    def __init__(self, core):
        self.core = core

    def update(self) -> Dict[str, Any]:
        """Poll live agent state and emit a graph update."""
        agents: List[str] = []
        if getattr(self.core, "agent_registry", None):
            agents = [name for name, _ in self.core.agent_registry.get_all_agents()]
        edges = [{"from": name, "to": "UnifiedVantaCore"} for name in agents]
        graph = {"agents": agents, "edges": edges}
        if getattr(self.core, "event_bus", None):
            self.core.event_bus.emit("mesh_graph_update", graph)
        return graph
