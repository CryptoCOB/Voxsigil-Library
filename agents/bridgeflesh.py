
import logging

from .base import BaseAgent
from ..UnifiedAsyncBus import AsyncMessage, MessageType




class BridgeFlesh(BaseAgent):
    sigil = "🧩🎯🜂🜁"
    invocations = ["Link Bridge", "Fuse layers"]

    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        self.vmb_handler = None

    def initialize_subsystem(self, vanta_core):
        self.vanta_core = vanta_core
        self.vmb_handler = vanta_core.get_component("vmb_integration_handler")
        if vanta_core and hasattr(vanta_core, "async_bus"):
            vanta_core.async_bus.register_component("BridgeFlesh")
            vanta_core.async_bus.subscribe(
                "BridgeFlesh",
                MessageType.SYSTEM_COMMAND,
                self.handle_command,
            )

    def handle_command(self, message: AsyncMessage):
        if self.vmb_handler and hasattr(self.vmb_handler, "initialize_vmb_system"):
            try:
                result = self.vmb_handler.initialize_vmb_system()
                if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                    resp = AsyncMessage(
                        MessageType.COMPONENT_STATUS,
                        "BridgeFlesh",
                        result,
                        target_ids=[message.sender_id],
                    )
                    self.vanta_core.async_bus.publish(resp)
            except Exception as e:
                logger.error(f"BridgeFlesh error: {e}")

    def on_gui_call(self):
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.SYSTEM_COMMAND,
                "BridgeFlesh",
                "initialize",
            )
            self.vanta_core.async_bus.publish(msg)

