from .base import BaseAgent

class Nebula(BaseAgent):
    sigil = "🜂⚡🜍🜄"
    invocations = ["Awaken Nebula", "Ignite the Stars"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
