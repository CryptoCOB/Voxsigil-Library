from .base import BaseAgent

class OrionApprentice(BaseAgent):
    sigil = "🜞🧩🎯🔁"
    invocations = ["Apprentice load", "Begin shard study"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
