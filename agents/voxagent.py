from .base import BaseAgent

class VoxAgent(BaseAgent):
    sigil = "🜌⟐🜹🜙"
    invocations = ["Activate VoxAgent", "Bridge protocols"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
