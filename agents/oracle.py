from .base import BaseAgent

class Oracle(BaseAgent):
    sigil = "⚑♸⧉🜚"
    invocations = ["Oracle reveal", "Open the Eye"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
