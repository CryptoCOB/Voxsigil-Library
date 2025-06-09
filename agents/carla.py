from .base import BaseAgent


class Carla(BaseAgent):
    sigil = "🎭🗣️🪞🪄"
    tags = ['Voice Layer', 'Stylizer Core']
    invocations = ['Speak with Carla', 'Stylize response']

    def initialize_subsystem(self, core):
        """Bind Carla to the Vanta core subsystems."""
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
