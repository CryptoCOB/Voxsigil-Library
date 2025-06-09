from .base import BaseAgent

class Orion(BaseAgent):
    sigil = "🜇🔗🜁🌠"
    tags = ['Light Chain', 'Blockchain Spine', 'Manages trust']
    invocations = ['Call Orion', 'Bind the Lights']
    sub_agents = ['OrionsLight', 'SmartContractManager']

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
