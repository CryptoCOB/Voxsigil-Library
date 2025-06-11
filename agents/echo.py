from .base import BaseAgent


class Echo(BaseAgent):
    sigil = "♲∇⌬☉"
    tags = ['Memory Stream', 'Continuity Guardian', 'EchoLocation, ExternalEcho']
    invocations = ['Echo log', 'What do you remember?']

    def initialize_subsystem(self, core):
        # Bind Echo to a specific subsystem within the core
        self.subsystem = core.get_subsystem('EchoSubsystem')

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        if hasattr(self, 'subsystem'):
            self.subsystem.bind_routes(self)

