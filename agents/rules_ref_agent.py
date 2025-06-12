from __future__ import annotations
import json
from pathlib import Path
from .base import BaseAgent, vanta_agent, CognitiveMeshRole

@vanta_agent(name="RulesRefAgent", subsystem="tabletop", mesh_role=CognitiveMeshRole.EVALUATOR)
class RulesRefAgent(BaseAgent):
    sigil = "📜📚"
    invocations = ["lookup spell", "lookup condition"]

    def __init__(self, vanta_core=None, srd_path: str = "srd.json"):
        super().__init__(vanta_core=vanta_core)
        self.data = {}
        path = Path(srd_path)
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                self.data = {}

    def handle_message(self, query: str) -> str:
        return self.data.get(query.lower(), "Unknown")
