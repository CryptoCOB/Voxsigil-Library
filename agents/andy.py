from .base import BaseAgent

class Andy(BaseAgent):
    sigil = "📦🔧📤🔁"
    invocations = ["Compose Andy", "Box output"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
