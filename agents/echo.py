from .base import BaseAgent

class Echo(BaseAgent):
    sigil = "♲∇⌬☉"
    tags = ['Memory Stream', 'Continuity Guardian', 'Memory recorder']
    invocations = ['Echo log', 'What do you remember?']
    sub_agents = ['EchoLocation', 'ExternalEcho']

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
