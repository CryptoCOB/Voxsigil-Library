from .base import BaseAgent

class Carla(BaseAgent):
    sigil = "🎭🗣️🪞🪄"
    invocations = ["Speak with Carla", "Stylize response"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
