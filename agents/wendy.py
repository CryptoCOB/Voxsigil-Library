from .base import BaseAgent

class Wendy(BaseAgent):
    sigil = "🎧💓🌈🎶"
    invocations = ["Listen Wendy", "Audit tone"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
