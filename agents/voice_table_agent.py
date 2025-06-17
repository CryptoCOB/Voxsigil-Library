from __future__ import annotations
from .base import BaseAgent, vanta_agent, CognitiveMeshRole
from services.dice_roller_service import DiceRollerService

@vanta_agent(name="VoiceTableAgent", subsystem="tabletop", mesh_role=CognitiveMeshRole.GENERATOR)
class VoiceTableAgent(BaseAgent):
    sigil = "🎙️🗺️"
    invocations = ["speak narration", "listen command"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core=vanta_core)
        self.dice = DiceRollerService()

    def handle_message(self, message: str) -> str:
        if message.strip().lower() == "roll":
            return str(self.dice.roll("d20"))
        return "Acknowledged"
