from .base import BaseAgent

class Gizmo(BaseAgent):
    sigil = "☍⚙️⩫⌁"
    tags = ['Artifact Twin', 'Tactical Forge-Agent', 'Twin of Nix']
    invocations = ['Hello Gizmo', 'Wake the Forge']
    sub_agents = ['PatchCrawler', 'LoopSmith']

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
