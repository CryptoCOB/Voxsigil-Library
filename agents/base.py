from ..UnifiedAsyncBus import AsyncMessage, MessageType


class BaseAgent:
    """Base class for all agents with async bus integration."""

    sigil: str = ""
    invocations: list[str] = []
    sub_agents: list[str] = []

    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core

    def initialize_subsystem(self, vanta_core):
        """Initialize subsystem and register with async bus."""
        self.vanta_core = vanta_core
        if vanta_core and hasattr(vanta_core, "async_bus"):
            try:
                vanta_core.async_bus.register_component(self.__class__.__name__)
                vanta_core.async_bus.subscribe(
                    self.__class__.__name__,
                    MessageType.USER_INTERACTION,
                    self.handle_message,
                )
            except Exception:
                pass

    def handle_message(self, message: AsyncMessage):
        """Handle messages from the async bus (override in subclasses)."""
        return None

    def on_gui_call(self, payload=None):
        """Publish a generic user interaction message on GUI trigger."""
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.USER_INTERACTION,
                self.__class__.__name__,
                payload,
            )
            self.vanta_core.async_bus.publish(msg)


class NullAgent(BaseAgent):
    """Fallback agent used when a real agent fails to load."""

    pass
