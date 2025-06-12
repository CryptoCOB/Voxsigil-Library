from __future__ import annotations
import os
from .base import BaseAgent, vanta_agent, CognitiveMeshRole
from services.game_state_store import GameStateStore
from services.dice_roller_service import DiceRollerService
from services.inventory_manager import InventoryManager
from rules import rolls

@vanta_agent(name="GameMasterAgent", subsystem="tabletop", mesh_role=CognitiveMeshRole.PLANNER)
class GameMasterAgent(BaseAgent):
    sigil = "🎲👑"
    invocations = ["begin encounter", "narrate scene"]

    def __init__(self, vanta_core=None, campaign_id: str = "demo"):
        super().__init__(vanta_core=vanta_core)
        self.store = GameStateStore(campaign_id)
        self.dice = DiceRollerService()
        self.inventory = InventoryManager()

    def handle_message(self, message: str) -> str:
        """Very simple intent parser and dice roller."""
        if "roll" in message:
            parts = message.split()
            expr = next((p for p in parts if 'd' in p), 'd20')
            result = self.dice.roll(expr)
            return f"You rolled {result} on {expr}"
        return "The adventure continues..."
