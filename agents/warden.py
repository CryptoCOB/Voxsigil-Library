from .base import BaseAgent

class Warden(BaseAgent):
    sigil = "⚔️⟁♘🜏"
    invocations = ["Warden check", "Status integrity"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
