from .base import BaseAgent

class Nebula(BaseAgent):
    sigil = "🜂⚡🜍🜄"
    tags = ['Core AI', 'Adaptive Core', 'Evolves internally']
    invocations = ['Awaken Nebula', 'Ignite the Stars']
    sub_agents = ['QuantumPulse', 'HolisticPerception']

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
