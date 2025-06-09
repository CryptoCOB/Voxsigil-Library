from .base import BaseAgent

class Sam(BaseAgent):
    sigil = "📜🔑🛠️🜔"
    invocations = ["Plan with Sam", "Unroll sequence"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
