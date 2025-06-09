from .base import BaseAgent

class Oracle(BaseAgent):
    sigil = "⚑♸⧉🜚"
    tags = ['Temporal Eye', 'Prophetic Synthesizer', 'Future mapping']
    invocations = ['Oracle reveal', 'Open the Eye']
    sub_agents = ['DreamWeft', 'TimeBinder']

    def initialize_subsystem(self, core):
        super().initialize_subsystem(core)
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
