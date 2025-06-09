
import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)


class CodeWeaver(BaseAgent):
    sigil = "⟡🜛⛭🜨"
    invocations = ["Weave Code", "Forge logic"]


    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        self.meta_learner = None

    def initialize_subsystem(self, vanta_core):
        self.vanta_core = vanta_core
        self.meta_learner = vanta_core.get_component("meta_learner")
        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.register_component("CodeWeaver")
            vanta_core.async_bus.subscribe(
                "CodeWeaver",
                MessageType.REASONING_REQUEST,
                self.handle_request,
            )

    def handle_request(self, message: AsyncMessage):
        if self.meta_learner and hasattr(self.meta_learner, "plan"):
            try:
                result = self.meta_learner.plan(message.content)
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.BRANCH_UPDATE,
                        "CodeWeaver",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"CodeWeaver error: {e}")

    def on_gui_call(self, request=None):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.REASONING_REQUEST,
                "CodeWeaver",
                request,
            )
            self.vanta_core.async_bus.publish(msg)

