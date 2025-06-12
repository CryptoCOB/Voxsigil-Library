from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Sam", subsystem="planner_subsystem", mesh_role=CognitiveMeshRole.PLANNER)
class Sam(BaseAgent):
    sigil = "📜🔑🛠️🜔"
    tags = ['Strategic Mind', 'Planner Core']
    invocations = ['Plan with Sam', 'Unroll sequence']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        super().bind_echo_routes()
