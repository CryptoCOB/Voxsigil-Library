from .base import BaseAgent

class EchoLore(BaseAgent):
    sigil = "🜎♾🜐⌽"
    invocations = ["Recall Lore", "Echo past"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
