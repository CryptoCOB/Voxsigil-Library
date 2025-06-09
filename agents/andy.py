from .base import BaseAgent


class Andy(BaseAgent):
    sigil = "📦🔧📤🔁"
    tags = ['Composer', 'Output Synthesizer', 'None']
    invocations = ['Compose Andy', 'Box output']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
