from .base import BaseAgent

class Gizmo(BaseAgent):
    sigil = "☍⚙️⩫⌁"
    invocations = ["Hello Gizmo", "Wake the Forge"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
