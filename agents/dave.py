from .base import BaseAgent, vanta_agent, CognitiveMeshRole


@vanta_agent(name="Dave", subsystem="validator_subsystem", mesh_role=CognitiveMeshRole.CRITIC)
class Dave(BaseAgent):
    sigil = "⚠️🧭🧱⛓️"
    tags = ['Caution Sentinel', 'Meta Validator']
    invocations = ['Dave validate', 'Run checks']

    def initialize_subsystem(self, core):
        """Bind to the MetaLearner subsystem."""
        super().initialize_subsystem(core)
        try:
            self.subsystem = core.get_component("meta_learner")
        except Exception:
            pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        super().bind_echo_routes()
