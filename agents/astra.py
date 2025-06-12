from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Astra", subsystem="navigation", mesh_role=CognitiveMeshRole.PLANNER)
class Astra(BaseAgent):
    sigil = "🜁⟁🜔🔭"
    tags = ['Navigator', 'System Pathfinder', 'CompassRose, LumenDrift']
    invocations = ['Astra align', 'Chart the frontier']

    def initialize_subsystem(self, core):
        """Bind Astra to the Vanta core subsystems."""
        super().initialize_subsystem(core)

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
