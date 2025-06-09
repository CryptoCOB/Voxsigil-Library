from .base import BaseAgent

class Orion(BaseAgent):
    sigil = "🜇🔗🜁🌠"
    invocations = ["Call Orion", "Bind the Lights"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
