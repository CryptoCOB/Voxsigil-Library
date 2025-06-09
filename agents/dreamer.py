from .base import BaseAgent


class Dreamer(BaseAgent):
    sigil = "🧿🧠🧩♒"
    tags = ['Dream Generator', 'Dream-State Core']
    invocations = ['Enter Dreamer', 'Seed dream state']

    def initialize_subsystem(self, core):
        """Bind to the ART controller subsystem."""
        super().initialize_subsystem(core)
        self.subsystem = core.get_component("art_controller")

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
