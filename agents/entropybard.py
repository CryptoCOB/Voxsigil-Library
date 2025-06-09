from .base import BaseAgent


class EntropyBard(BaseAgent):
    sigil = "🜔🕊️⟁⧃"
    tags = ['Chaos Interpreter', 'Singularity Bard', 'None']
    invocations = ['Sing Bard', 'Unleash entropy']

    def initialize_subsystem(self, core):
        """Bind to the BLT encoder subsystem."""
        super().initialize_subsystem(core)
        self.subsystem = core.get_component("blt_encoder")

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
