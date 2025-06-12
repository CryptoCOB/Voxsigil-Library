from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Orion", subsystem="trust_chain", mesh_role=CognitiveMeshRole.PLANNER)
class Orion(BaseAgent):
    sigil = "🜇🔗🜁🌠"
    tags = ['Light Chain', 'Blockchain Spine', 'OrionsLight, SmartContractManager']
    invocations = ['Call Orion', 'Bind the Lights']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
