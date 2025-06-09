from .base import BaseAgent

class Nix(BaseAgent):
    sigil = "☲🜄🜁⟁"
    invocations = ["Nix, awaken", "Unchain the Core"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
