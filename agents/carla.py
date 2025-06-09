from .base import BaseAgent

class Carla(BaseAgent):
    sigil = "🎭🗣️🪞🪄"
    invocations = ["Speak with Carla", "Stylize response"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self.speech_handler = None

    def initialize_subsystem(self, vanta_core):
        super().initialize_subsystem(vanta_core)
        self.speech_handler = vanta_core.get_component("speech_integration_handler")
