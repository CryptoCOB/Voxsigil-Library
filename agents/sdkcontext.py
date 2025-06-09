from .base import BaseAgent

class SDKContext(BaseAgent):
    sigil = "⏣📡⏃⚙️"
    invocations = ["Scan SDKContext", "Map modules"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
