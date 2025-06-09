from .base import BaseAgent
from sleep_time_compute import SleepTimeCompute

class SleepTimeComputeAgent(BaseAgent):
    sigil = "🌒🧵🧠🜝"
    invocations = ["Sleep Compute", "Dream consolidate"]

    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self.core = (
            SleepTimeCompute(vanta_core=vanta_core) if SleepTimeCompute else None
        )
