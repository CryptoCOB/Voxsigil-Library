from .base import BaseAgent


class PulseSmith(BaseAgent):
    sigil = "🜖📡🜖📶"
    tags = ['Signal Tuner', 'Transduction Core']
    invocations = ['Tune Pulse', 'Resonate Signal']

    def initialize_subsystem(self, core):
        """Bind to the GridFormer subsystem."""
        super().initialize_subsystem(core)
        self.model = core.get_component("gridformer_connector")

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
