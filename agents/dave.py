from .base import BaseAgent

class Dave(BaseAgent):
    sigil = "⚠️🧭🧱⛓️"
    invocations = ["Dave validate", "Run checks"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
