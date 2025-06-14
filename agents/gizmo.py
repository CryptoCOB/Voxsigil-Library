from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Gizmo", subsystem="forge_subsystem", mesh_role=CognitiveMeshRole.GENERATOR)
class Gizmo(BaseAgent):
    sigil = "☍⚙️⩫⌁"
    tags = ['Artifact Twin', 'Tactical Forge-Agent', 'PatchCrawler, LoopSmith']
    invocations = ['Hello Gizmo', 'Wake the Forge']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
