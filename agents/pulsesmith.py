
import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)


class PulseSmith(BaseAgent):
    sigil = "🜖📡🜖📶"
    invocations = ["Tune Pulse", "Resonate Signal"]


    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self.gridformer = None

    def initialize_subsystem(self, vanta_core):
        super().initialize_subsystem(vanta_core)
        self.vanta_core = vanta_core
        self.gridformer = (
            vanta_core.get_component("gridformer_connector")
            or vanta_core.get_component("gridformer")
        )
        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.subscribe(
                "PulseSmith",
                MessageType.PROCESSING_REQUEST,
                self.handle_trace,
            )

    def handle_trace(self, message: AsyncMessage):
        if self.gridformer and hasattr(self.gridformer, "forward"):
            try:
                result = self.gridformer.forward(message.content)
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.PROCESSING_RESPONSE,
                        "PulseSmith",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"PulseSmith failed to process trace: {e}")

    def on_gui_call(self, trace=None):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.PROCESSING_REQUEST,
                "PulseSmith",
                trace,
            )
            self.vanta_core.async_bus.publish(msg)

