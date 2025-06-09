from .base import BaseAgent

class Astra(BaseAgent):
    sigil = "🜁⟁🜔🔭"
    invocations = ["Astra align", "Chart the frontier"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
