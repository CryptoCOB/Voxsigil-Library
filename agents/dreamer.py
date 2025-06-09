
import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType




class Dreamer(BaseAgent):
    sigil = "🧿🧠🧩♒"
    invocations = ["Enter Dreamer", "Seed dream state"]


    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self.art_controller = None

    def initialize_subsystem(self, vanta_core):
        super().initialize_subsystem(vanta_core)
        self.vanta_core = vanta_core
        self.art_controller = vanta_core.get_component("art_controller")
        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.subscribe(
                "Dreamer",
                MessageType.USER_INTERACTION,
                self.handle_prompt,
            )

    def handle_prompt(self, message: AsyncMessage):
        if self.art_controller and hasattr(self.art_controller, "generate"):
            try:
                result = self.art_controller.generate(message.content)
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.PROCESSING_RESPONSE,
                        "Dreamer",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"Dreamer error: {e}")

    def on_gui_call(self, prompt=None):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.USER_INTERACTION,
                "Dreamer",
                prompt,
            )
            self.vanta_core.async_bus.publish(msg)

