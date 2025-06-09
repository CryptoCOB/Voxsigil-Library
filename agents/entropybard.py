from .base import BaseAgent


class EntropyBard(BaseAgent):
    sigil = "🜔🕊️⟁⧃"
    tags = ['Chaos Interpreter', 'Singularity Bard', 'None']
    invocations = ['Sing Bard', 'Unleash entropy']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
