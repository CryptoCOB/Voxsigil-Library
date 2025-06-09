from .base import BaseAgent


class CodeWeaver(BaseAgent):
    sigil = "⟡🜛⛭🜨"
    tags = ['Synthesizer', 'Logic Constructor']
    invocations = ['Weave Code', 'Forge logic']

    def initialize_subsystem(self, core):
        """Bind to the MetaLearner subsystem."""
        super().initialize_subsystem(core)
        self.subsystem = core.get_component("meta_learner")

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
