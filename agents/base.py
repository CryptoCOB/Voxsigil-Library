
import logging
import asyncio

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents with async bus integration."""

    sigil: str = ""
    invocations: Optional[list[str]] = None
    sub_agents: Optional[list[str]] = None

    def __init__(self, vanta_core=None):

        self.vanta_core = None
        # Use instance-specific lists to avoid shared mutable defaults
        if self.invocations is None:
            self.invocations = []
        else:
            self.invocations = list(self.invocations)
        if self.sub_agents is None:
            self.sub_agents = []
        else:
            self.sub_agents = list(self.sub_agents)
        if vanta_core is not None:
            self.initialize_subsystem(vanta_core)


    def initialize_subsystem(self, vanta_core):
        """Initialize subsystem, register with async bus and bind echo routes."""
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

        # Bind echo routes on the event bus if available
        if vanta_core and hasattr(vanta_core, "event_bus"):
            try:
                self.bind_echo_routes()
            except Exception:
                pass

    def bind_echo_routes(self) -> None:
        """Subscribe to class-specific echo events."""
        if not (self.vanta_core and hasattr(self.vanta_core, "event_bus")):
            return
        event_type = f"sigil_{self.__class__.__name__.lower()}_triggered"
        self.vanta_core.event_bus.subscribe(event_type, self.receive_echo)

    def receive_echo(self, event) -> None:
        """Handle echo events from the event bus."""
        logger.info(
            f"{self.__class__.__name__} received echo event: {event.get('type')}"
        )

    def handle_message(self, message: AsyncMessage):
        """Handle messages from the async bus (override in subclasses)."""

        logger.debug(
            f"{self.__class__.__name__} received {message.message_type.value}"
        )
        if (
            self.vanta_core
            and hasattr(self.vanta_core, "event_bus")
            and self.vanta_core.event_bus
        ):
            try:
                self.vanta_core.event_bus.emit(
                    "agent_message_received",
                    {
                        "agent": self.__class__.__name__,
                        "type": message.message_type.value,
                    },
                )
            except Exception:
                pass

        return None

    def on_gui_call(self, payload=None):
        """Publish a generic user interaction message on GUI trigger."""
        logger.info(f"GUI invoked {self.__class__.__name__} with payload={payload}")
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.USER_INTERACTION,
                self.__class__.__name__,
                payload,
            )
            asyncio.create_task(self.vanta_core.async_bus.publish(msg))

        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            try:
                self.vanta_core.event_bus.emit(
                    "gui_agent_invoked",
                    {"agent": self.__class__.__name__, "payload": payload},
                )
                self.vanta_core.event_bus.emit(
                    f"{self.__class__.__name__.lower()}_invoked",
                    {"origin": self.sigil, "payload": payload},
                )
                # stream output to GUI panels when available
                self.vanta_core.event_bus.emit(
                    "gui_console_output",
                    {"text": f"{self.__class__.__name__} invoked", "payload": payload},
                )
                self.vanta_core.event_bus.emit(
                    "gui_panel_output",
                    {
                        "panel": "AgentStatusPanel",
                        "agent": self.__class__.__name__,
                        "payload": payload,
                    },
                )
            except Exception:
                pass
        # 🧠 Codex BugPatch - Vanta Phase @2025-06-09



class NullAgent(BaseAgent):
    """Fallback agent used when a real agent fails to load."""

    pass
