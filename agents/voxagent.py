import asyncio
import logging

from middleware.blt_compression_middleware import compress_outbound
from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType

from .base import BaseAgent, CognitiveMeshRole, vanta_agent

logger = logging.getLogger(__name__)


@vanta_agent(
    name="VoxAgent", subsystem="system_interface", mesh_role=CognitiveMeshRole.EVALUATOR
)
class VoxAgent(BaseAgent):
    sigil = "🜌⟐🜹🜙"
    tags = ["Coordinator", "System Interface", "ContextualCheckInAgent"]
    invocations = ["Activate VoxAgent", "Bridge protocols"]

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self.outbox: list[str] = []

    @compress_outbound
    def send(self, message: str) -> None:
        """Send a message to the outbox with BLT compression."""
        self.outbox.append(message)
        if self.vanta_core:
            if (
                hasattr(self.vanta_core, "async_bus")
                and self.vanta_core.async_bus is not None
            ):
                msg = AsyncMessage(
                    MessageType.USER_INTERACTION, self.__class__.__name__, message
                )
                # Create task with error handling
                task = asyncio.create_task(self._safe_publish(msg))
                # Store task reference to prevent garbage collection
                if not hasattr(self, "_pending_tasks"):
                    self._pending_tasks = set()
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
                if hasattr(self.vanta_core, "send_to_mesh"):
                    self.vanta_core.send_to_mesh(self.__class__.__name__, message)

    async def _safe_publish(self, msg: AsyncMessage):
        """Safely publish message with error handling."""
        try:
            if self.vanta_core.async_bus is not None:
                await self.vanta_core.async_bus.publish(msg)
                logger.debug(
                    f"✅ Published message to async bus: {msg.payload[:50]}..."
                )
            else:
                logger.debug("Async bus not available, skipping message publish")
        except Exception as e:
            logger.error(f"❌ Failed to publish message to async bus: {e}")
            logger.debug(f"Publish error details: {e.__class__.__name__}: {e}")

    def respond(self) -> str:
        """Return the last sent message."""
        return self.outbox[-1] if self.outbox else ""
