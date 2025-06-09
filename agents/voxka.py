from .base import BaseAgent

class Voxka(BaseAgent):
    sigil = "🧠⟁🜂Φ🎙"
    invocations = ["Invoke Voxka", "Voice of Phi"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
