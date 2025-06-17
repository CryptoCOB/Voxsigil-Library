#!/usr/bin/env python3
"""
Heartbeat Publisher - Generates system health metrics for the heartbeat monitor

Publishes real-time system metrics every 250ms:
- Event throughput (events/sec)
- GPU memory usage
- CPU usage
- Error rate per minute
"""

import asyncio
import logging
import time
from typing import Any, Dict

import psutil

from agents.base import BaseAgent
from Vanta.core.vanta_registration import vanta_core_module
from Vanta.interfaces.cognitive_mesh import CognitiveMeshRole

logger = logging.getLogger(__name__)


def get_gpu_memory_percent() -> float:
    """Get GPU memory usage percentage."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_stats(0)
            reserved = gpu_mem.get("reserved_bytes.all.current", 0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (reserved / total) * 100 if total > 0 else 0.0
    except Exception:
        pass
    return 0.0


@vanta_core_module(
    name="heartbeat_publisher",
    subsystem="monitoring",
    mesh_role=CognitiveMeshRole.PUBLISHER,
    description="Publishes real-time system health metrics for heartbeat monitor",
    capabilities=["system_metrics", "real_time_publishing", "health_monitoring"],
)
class HeartbeatPublisher(BaseAgent):
    """
    Publisher agent that generates system health metrics for the heartbeat monitor.

    Publishes metrics every 250ms including:
    - TPS (transactions per second)
    - GPU memory usage
    - CPU usage
    - Error rate per minute
    """

    def __init__(self, bus=None, period: float = 0.25):
        super().__init__(bus=bus)
        self.period = period
        self.logger = logging.getLogger("VoxSigil.Monitoring.Heartbeat")

        # Metrics tracking
        self.last_tick_time = time.time()
        self.event_count = 0
        self.error_count = 0
        self.last_error_reset = time.time()
        self.running = False

        # Subscribe to all events for counting
        if self.bus:
            self.bus.subscribe("*", self.count_event)

    async def initialize_agent(self):
        """Initialize the heartbeat publisher."""
        self.logger.info("Heartbeat publisher initialized")
        self.running = True

        # Start the publishing loop
        asyncio.create_task(self.publish_loop())

    async def publish_loop(self):
        """Main publishing loop."""
        while self.running:
            try:
                await self.publish_heartbeat()
                await asyncio.sleep(self.period)
            except Exception as e:
                self.logger.error(f"Error in heartbeat publishing loop: {e}")
                await asyncio.sleep(self.period)

    async def publish_heartbeat(self):
        """Publish current system metrics."""
        try:
            current_time = time.time()
            time_delta = current_time - self.last_tick_time

            # Calculate events per second
            tps = self.event_count / time_delta if time_delta > 0 else 0.0

            # Get system metrics
            gpu_percent = get_gpu_memory_percent()
            cpu_percent = psutil.cpu_percent()

            # Calculate errors per minute
            errors_per_min = self.error_count * (
                60.0 / max(current_time - self.last_error_reset, 1.0)
            )

            # Reset error count every minute
            if current_time - self.last_error_reset >= 60.0:
                self.error_count = 0
                self.last_error_reset = current_time

            # Publish metrics
            metrics = {
                "tps": tps,
                "gpu": gpu_percent,
                "cpu": cpu_percent,
                "errors": errors_per_min,
                "timestamp": current_time,
            }

            if self.bus:
                self.bus.publish("heartbeat", metrics)

            # Reset counters
            self.event_count = 0
            self.last_tick_time = current_time

            self.logger.debug(
                f"Published heartbeat: TPS={tps:.1f}, GPU={gpu_percent:.1f}%, CPU={cpu_percent:.1f}%, Errors={errors_per_min:.1f}/min"
            )

        except Exception as e:
            self.logger.error(f"Error publishing heartbeat: {e}")

    def count_event(self, topic: str, payload: Any):
        """Count events for TPS calculation."""
        self.event_count += 1

        # Check for errors
        if "error" in topic.lower() or (isinstance(payload, dict) and payload.get("error")):
            self.error_count += 1

    async def shutdown(self):
        """Shutdown the publisher."""
        self.running = False
        self.logger.info("Heartbeat publisher shutdown")

    def get_ui_spec(self) -> Dict[str, Any]:
        """Get UI specification."""
        return {
            "name": "HeartbeatPublisher",
            "type": "publisher",
            "publishes": ["heartbeat"],
            "enabled": True,
        }


# Singleton instance for global access
_heartbeat_publisher = None


def get_heartbeat_publisher(bus=None) -> HeartbeatPublisher:
    """Get or create the global heartbeat publisher instance."""
    global _heartbeat_publisher
    if _heartbeat_publisher is None:
        _heartbeat_publisher = HeartbeatPublisher(bus=bus)
    return _heartbeat_publisher


async def start_heartbeat_monitoring(bus=None, period: float = 0.25):
    """Start heartbeat monitoring with the specified period."""
    publisher = get_heartbeat_publisher(bus)
    publisher.period = period
    await publisher.initialize_agent()
    logger.info(f"Started heartbeat monitoring with {period}s period")
    return publisher
