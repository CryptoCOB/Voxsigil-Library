
import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)

class MirrorWarden(BaseAgent):
    sigil = "⚛️🜂🜍🝕"
    invocations = ["Check Mirror", "Guard reflections"]


    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        self.meta_learner = None

    def initialize_subsystem(self, vanta_core):
        self.vanta_core = vanta_core
        self.meta_learner = vanta_core.get_component("meta_learner")
        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.register_component("MirrorWarden")
            vanta_core.async_bus.subscribe(
                "MirrorWarden",
                MessageType.REASONING_REQUEST,
                self.handle_request,
            )

    def handle_request(self, message: AsyncMessage):
        if self.meta_learner and hasattr(self.meta_learner, "evaluate"):
            try:
                result = self.meta_learner.evaluate(message.content)
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.BRANCH_EVALUATION,
                        "MirrorWarden",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"MirrorWarden error: {e}")

    def on_gui_call(self, request=None):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.REASONING_REQUEST,
                "MirrorWarden",
                request,
            )
            self.vanta_core.async_bus.publish(msg)

