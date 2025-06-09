from .base import BaseAgent

class Evo(BaseAgent):
    sigil = "🧬♻️♞🜓"
    invocations = ["Evo engage", "Mutate form"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
