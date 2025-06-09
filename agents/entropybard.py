import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)

class EntropyBard(BaseAgent):
    sigil = "🜔🕊️⟁⧃"
    invocations = ["Sing Bard", "Unleash entropy"]

    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        self.rag_handler = None
        self.rag_interface = None

    def initialize_subsystem(self, vanta_core):
        self.vanta_core = vanta_core
        try:
            from rag_integration_handler import RagIntegrationHandler

            self.rag_handler = RagIntegrationHandler(vanta_core)
            self.rag_interface = self.rag_handler.initialize_rag_interface()
        except Exception as e:
            logger.warning(f"RAG subsystem unavailable: {e}")

        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.register_component("EntropyBard")
            vanta_core.async_bus.subscribe(
                "EntropyBard",
                MessageType.PROCESSING_REQUEST,
                self.handle_request,
            )

    def handle_request(self, message: AsyncMessage):
        if self.rag_interface and hasattr(self.rag_interface, "query"):
            try:
                result = self.rag_interface.query(message.content)
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.PROCESSING_RESPONSE,
                        "EntropyBard",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"EntropyBard processing error: {e}")

    def on_gui_call(self, query: str = ""):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.PROCESSING_REQUEST,
                "EntropyBard",
                query,
            )
            self.vanta_core.async_bus.publish(msg)
