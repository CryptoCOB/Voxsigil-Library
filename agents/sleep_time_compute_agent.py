from .base import BaseAgent
from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType


class SleepTimeComputeAgent(BaseAgent):
    sigil = "🌒🧵🧠🜝"
    tags = ['Reflection Engine', 'Dream-State Scheduler']
    invocations = ['Sleep Compute', 'Dream consolidate']

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass

    async def run(self) -> None:
        """Emit status and phase information via the async bus and event bus."""
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.COMPONENT_STATUS,
                self.__class__.__name__,
                {"phase": "run"},
            )
            await self.vanta_core.async_bus.publish(msg)
        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            self.vanta_core.event_bus.emit(
                "sleep_time_compute.status",
                {"phase": "run"},
            )

class SleepTimeCompute(SleepTimeComputeAgent):
    """Alias to match AGENTS.md name."""
    pass

