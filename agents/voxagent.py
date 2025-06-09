from .base import BaseAgent

class VoxAgent(BaseAgent):
    sigil = "🜌⟐🜹🜙"
    tags = ['Coordinator', 'System Interface', 'Bridges input/state']
    invocations = ['Activate VoxAgent', 'Bridge protocols']
    sub_agents = ['ContextualCheckInAgent']

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
