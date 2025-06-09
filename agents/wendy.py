from .base import BaseAgent


class Wendy(BaseAgent):
    sigil = "🎧💓🌈🎶"
    tags = ['Tonal Auditor', 'Emotional Oversight']
    invocations = ['Listen Wendy', 'Audit tone']

    def initialize_subsystem(self, core):
        """Bind Wendy to the Vanta core subsystems."""
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
