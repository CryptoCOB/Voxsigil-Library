from .base import BaseAgent


class Sam(BaseAgent):
    sigil = "📜🔑🛠️🜔"
    tags = ['Strategic Mind', 'Planner Core']
    invocations = ['Plan with Sam', 'Unroll sequence']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
