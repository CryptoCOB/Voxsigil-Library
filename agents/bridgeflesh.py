from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="BridgeFlesh", subsystem="vmb_integration", mesh_role=CognitiveMeshRole.GENERATOR)
class BridgeFlesh(BaseAgent):
    sigil = "🧩🎯🜂🜁"
    tags = ['Connector', 'Integration Orchestrator']
    invocations = ['Link Bridge', 'Fuse layers']

    def initialize_subsystem(self, core):
        """Bind to the VMB integration handler subsystem."""
        super().initialize_subsystem(core)
        self.subsystem = core.get_component("vmb_integration_handler")

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
