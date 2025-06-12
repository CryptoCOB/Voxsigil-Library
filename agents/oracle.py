from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Oracle", subsystem="temporal_foresight", mesh_role=CognitiveMeshRole.EVALUATOR)
class Oracle(BaseAgent):
    sigil = "⚑♸⧉🜚"
    tags = ['Temporal Eye', 'Prophetic Synthesizer', 'DreamWeft, TimeBinder']
    invocations = ['Oracle reveal', 'Open the Eye']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        super().bind_echo_routes()
