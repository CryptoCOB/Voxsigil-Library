class BaseAgent:
    """Base class for all agents."""

    sigil: str = ""
    invocations: list[str] = []
    sub_agents: list[str] = []

    def initialize_subsystem(self, vanta_core):
        """Initialize any subsystem connections."""
        pass

    def on_gui_call(self, *args, **kwargs):
        """Handle GUI-triggered calls."""
        pass


class NullAgent(BaseAgent):
    """Fallback agent used when a real agent fails to load."""

    pass
