from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Echo", subsystem="echo_memory", mesh_role=CognitiveMeshRole.GENERATOR)
class Echo(BaseAgent):
    sigil = "♲∇⌬☉"
    tags = ['Memory Stream', 'Continuity Guardian', 'EchoLocation, ExternalEcho']
    invocations = ['Echo log', 'What do you remember?']

    def initialize_subsystem(self, core):
        # Bind Echo to a specific subsystem within the core
        super().initialize_subsystem(core)
        try:
            self.subsystem = core.get_subsystem('EchoSubsystem')
        except Exception:
            pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        super().bind_echo_routes()
        if hasattr(self, 'subsystem'):
            try:
                self.subsystem.bind_routes(self)
            except Exception:
                pass

