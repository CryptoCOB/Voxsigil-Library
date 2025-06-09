from .base import BaseAgent

class Echo(BaseAgent):
    sigil = "♲∇⌬☉"
    invocations = ["Echo log", "What do you remember?"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
