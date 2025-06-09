import logging
from typing import Any, Dict, List, Optional, Tuple


class UnifiedAgentRegistry:
    """Registry for managing agents in the UnifiedVantaCore."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.agents = {}
        self.logger.info("UnifiedAgentRegistry initialized")

    def register_agent(
        self, name: str, agent: object, metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an agent by name with optional metadata."""
        if name in self.agents:
            self.logger.warning(f"Agent '{name}' already registered – replacing entry")
        self.agents[name] = {"agent": agent, "metadata": metadata or {}}
        self.logger.info(f"Agent '{name}' registered")

    def get_agent(self, name: str):
        """Retrieve an agent by name."""
        agent_entry = self.agents.get(name)
        return agent_entry["agent"] if agent_entry else None

    def get_all_agents(self) -> List[Tuple[str, object]]:
        """Get all registered agents as a list of (name, agent) tuples."""
        return [(name, entry["agent"]) for name, entry in self.agents.items()]

    def get_agents_by_capability(self, capability: str) -> List[Tuple[str, object]]:
        """Get all agents that have a specific capability."""
        matching_agents = []
        for name, entry in self.agents.items():
            metadata = entry.get("metadata", {})
            capabilities = metadata.get("capabilities", [])
            if capability in capabilities:
                matching_agents.append((name, entry["agent"]))
        return matching_agents
