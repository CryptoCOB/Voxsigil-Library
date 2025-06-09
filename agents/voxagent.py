from .base import BaseAgent
from ..blt_compression_middleware import compress_outbound


class VoxAgent(BaseAgent):
    sigil = "🜌⟐🜹🜙"
    tags = ['Coordinator', 'System Interface', 'ContextualCheckInAgent']
    invocations = ['Activate VoxAgent', 'Bridge protocols']

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
