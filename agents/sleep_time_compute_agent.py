from .base import BaseAgent


class SleepTimeComputeAgent(BaseAgent):
    sigil = "🌒🧵🧠🜝"
    tags = ['Reflection Engine', 'Dream-State Scheduler']
    invocations = ['Sleep Compute', 'Dream consolidate']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass

class SleepTimeCompute(SleepTimeComputeAgent):
    """Alias to match AGENTS.md name."""
    pass

