from .base import BaseAgent


class Andy(BaseAgent):
    sigil = "📦🔧📤🔁"
    tags = ['Composer', 'Output Synthesizer']
    invocations = ['Compose Andy', 'Box output']

    def initialize_subsystem(self, core):
        """Bind Andy to the Vanta core subsystems."""
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
