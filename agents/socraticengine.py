from .base import BaseAgent

class SocraticEngine(BaseAgent):
    sigil = "🜏🔍⟡🜒"
    invocations = ["Begin Socratic", "Initiate reflection"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
