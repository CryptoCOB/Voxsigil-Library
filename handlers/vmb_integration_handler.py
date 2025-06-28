"""
VMB Integration Handler for VantaCore

This module provides the integration between the VMB (VANTA Model Builder) system
and VantaCore, ensuring that VMB functionality is properly registered with
VantaCore and available to other components, including the GUI.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from Vanta.core.UnifiedVantaCore import UnifiedVantaCore, get_vanta_core

# Import VMB components with proper error handling
try:
    from vmb.vmb_activation import CopilotSwarm

    VMB_ACTIVATION_AVAILABLE = True
except ImportError:
    VMB_ACTIVATION_AVAILABLE = False
    CopilotSwarm = None

try:
    from vmb.vmb_production_executor import ProductionTaskExecutor

    VMB_PRODUCTION_AVAILABLE = True
except ImportError:
    VMB_PRODUCTION_AVAILABLE = False
    ProductionTaskExecutor = None

logger = logging.getLogger("VantaCore.VMB")


class VMBIntegrationHandler:
    """
    Handles integration of VMB (VANTA Model Builder) with VantaCore.
    This ensures that VMB functionality is properly registered with VantaCore
    and available for use by other components, including the GUI.
    """

    def __init__(self, vanta_core: Optional[UnifiedVantaCore] = None):
        """Initialize with optional VantaCore instance."""
        self.vanta_core = vanta_core if vanta_core else get_vanta_core()
        self.vmb_swarm = None
        self.production_executor = None
        self._vmb_initialized = False
        self._production_initialized = False

        # Default configuration
        self.config = {
            "sigil": "⟠∆∇𓂀",
            "agent_class": "CopilotSwarm",
            "swarm_variant": "RPG_Sentinel",
            "role_scope": ["planner", "validator", "executor", "summarizer"],
            "activation_mode": "VMB_Production",
        }

    async def initialize_vmb_system(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Initialize the VMB system based on the provided configuration.

        Args:
            config: Optional configuration for VMB

        Returns:
            Dict indicating initialization status of VMB components
        """
        if config is not None:
            self.config.update(config)

        results = {"vmb_initialized": False, "production_initialized": False}

        # First check if VMB components are already registered
        existing_vmb = self.vanta_core.get_component("vmb_swarm")
        existing_executor = self.vanta_core.get_component("vmb_executor")

        if existing_vmb:
            logger.info(f"Using existing VMB swarm: {type(existing_vmb).__name__}")
            self.vmb_swarm = existing_vmb
            self._vmb_initialized = True
            results["vmb_initialized"] = True

        if existing_executor:
            logger.info(
                f"Using existing VMB executor: {type(existing_executor).__name__}"
            )
            self.production_executor = existing_executor
            self._production_initialized = True
            results["production_initialized"] = True

        # Initialize VMB if not already registered
        if not self._vmb_initialized:
            results["vmb_initialized"] = await self._initialize_vmb_swarm()

        # Initialize Production Executor if not already registered
        if not self._production_initialized:
            results[
                "production_initialized"
            ] = await self._initialize_production_executor()

        # Register with VantaCore event bus for VMB-related events
        if hasattr(self.vanta_core, "event_bus"):
            self._register_event_handlers()

        return results

    async def _initialize_vmb_swarm(self) -> bool:
        """
        Initialize the VMB CopilotSwarm.

        Returns:
            True if initialization was successful, False otherwise
        """
        # Check if VMB is available
        if not VMB_ACTIVATION_AVAILABLE or CopilotSwarm is None:
            logger.warning("VMB CopilotSwarm not available")
            self._vmb_initialized = False
            return False

        try:
            # Create and initialize the VMB swarm
            self.vmb_swarm = CopilotSwarm(self.config)
            await self.vmb_swarm.initialize_swarm()

            # Register with VantaCore
            self.vanta_core.register_component(
                "vmb_swarm",
                self.vmb_swarm,
                {
                    "type": "vanta_model_builder",
                    "sigil": self.config.get("sigil", "⟠∆∇𓂀"),
                    "variant": self.config.get("swarm_variant", "RPG_Sentinel"),
                    "capabilities": [
                        "model_building",
                        "code_analysis",
                        "component_validation",
                        "error_detection",
                        "performance_monitoring",
                        "task_execution",
                    ],
                },
            )

            logger.info("VMB CopilotSwarm initialized and registered with VantaCore")
            self._vmb_initialized = True

            # Emit event for successful initialization
            self._publish_event(
                "vmb.swarm.initialized",
                {
                    "sigil": self.config.get("sigil", "⟠∆∇𓂀"),
                    "variant": self.config.get("swarm_variant", "RPG_Sentinel"),
                    "agents": list(getattr(self.vmb_swarm, "agents", {}).keys()),
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize VMB CopilotSwarm: {e}")
            self._vmb_initialized = False
            return False

    async def _initialize_production_executor(self) -> bool:
        """
        Initialize the VMB Production Executor.

        Returns:
            True if initialization was successful, False otherwise
        """
        # Check if VMB Production is available
        if not VMB_PRODUCTION_AVAILABLE or ProductionTaskExecutor is None:
            logger.warning("VMB Production Executor not available")
            self._production_initialized = False
            return False

        try:
            # Create and initialize the production executor
            self.production_executor = ProductionTaskExecutor()
            await self.production_executor.initialize()

            # Register with VantaCore
            self.vanta_core.register_component(
                "vmb_executor",
                self.production_executor,
                {
                    "type": "vanta_model_executor",
                    "sigil": self.config.get("sigil", "⟠∆∇𓂀"),
                    "capabilities": [
                        "task_execution",
                        "error_handling",
                        "performance_monitoring",
                        "result_analysis",
                    ],
                },
            )

            logger.info(
                "VMB Production Executor initialized and registered with VantaCore"
            )
            self._production_initialized = True

            # Emit event for successful initialization
            self._publish_event(
                "vmb.executor.initialized", {"sigil": self.config.get("sigil", "⟠∆∇𓂀")}
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize VMB Production Executor: {e}")
            self._production_initialized = False
            return False

    def _register_event_handlers(self) -> None:
        """Register event handlers for VMB-related events."""
        if not hasattr(self.vanta_core, "event_bus"):
            logger.warning(
                "Event bus not available, skipping event handler registration"
            )
            return

        # Register event handlers
        self.vanta_core.event_bus.subscribe(
            "vmb.task.execute", self._handle_execute_task_event
        )
        self.vanta_core.event_bus.subscribe(
            "vmb.swarm.status", self._handle_swarm_status_event
        )

        logger.info("VMB event handlers registered with VantaCore event bus")

    def _handle_execute_task_event(self, event: Dict[str, Any]) -> None:
        """
        Handle VMB task execution events.

        Args:
            event: The event data containing the task to execute
        """
        if not self._production_initialized or not self.production_executor:
            logger.error("Cannot execute task: VMB Production Executor not initialized")
            return

        # Extract task from event
        task = event.get("data", {})
        if not task:
            logger.warning("No task data provided in event")
            return

        # Execute task asynchronously
        asyncio.create_task(self._async_execute_task(task))

    async def _async_execute_task(self, task: Dict[str, Any]) -> None:
        """
        Execute a VMB task asynchronously.

        Args:
            task: The task to execute
        """
        try:
            # Execute the task            result = await self.production_executor.execute_production_task(task)

            # Publish result event
            self._publish_event(
                "vmb.task.result",
                {
                    "task": task,
                    "result": result,
                    "status": result.get("status", "unknown"),
                },
            )
        except Exception as e:
            logger.error(f"Error executing VMB task: {e}")
            # Publish error event
            self._publish_event("vmb.task.error", {"task": task, "error": str(e)})

    def _handle_swarm_status_event(self, event: Dict[str, Any]) -> None:
        """
        Handle VMB swarm status events.

        Args:
            event: The event data
        """
        if not self._vmb_initialized or not self.vmb_swarm:
            logger.error("Cannot provide status: VMB CopilotSwarm not initialized")
            return

        try:  # Get swarm status
            status = self.vmb_swarm.get_status()

            # Publish status event
            self._publish_event(
                "vmb.swarm.status.result",
                {
                    "status": status,
                    "active": status.get("active", False),
                    "agents": status.get("agents", {}),
                },
            )
        except Exception as e:
            logger.error(f"Error getting VMB swarm status: {e}")
            # Publish error event
            self._publish_event("vmb.swarm.status.error", {"error": str(e)})

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a VMB task.

        Args:
            task: The task to execute

        Returns:
            The task execution result
        """
        if not self._production_initialized or not self.production_executor:
            error_msg = "Cannot execute task: VMB Production Executor not initialized"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        try:
            # Execute the task
            result = await self.production_executor.execute_production_task(task)
            return result
        except Exception as e:
            error_msg = f"Error executing VMB task: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg, "task": task}

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of VMB components.

        Returns:
            Dictionary with status information
        """
        status = {
            "vmb_initialized": self._vmb_initialized,
            "production_initialized": self._production_initialized,
            "vmb_swarm_type": type(self.vmb_swarm).__name__ if self.vmb_swarm else None,
            "production_executor_type": type(self.production_executor).__name__
            if self.production_executor
            else None,
            "vmb_activation_available": VMB_ACTIVATION_AVAILABLE,
            "vmb_production_available": VMB_PRODUCTION_AVAILABLE,
            "sigil": self.config.get("sigil", "⟠∆∇𓂀"),
        }

        # Add VMB swarm status if available
        if (
            self._vmb_initialized
            and self.vmb_swarm
            and hasattr(self.vmb_swarm, "get_status")
        ):
            status["vmb_swarm_status"] = self.vmb_swarm.get_status()

        return status

    def _publish_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        Publish an event through VantaCore's event system, handling different event mechanisms.

        Args:
            event_name: The name of the event
            event_data: The event data
        """
        # Try to publish using event_bus if available (preferred method)
        if hasattr(self.vanta_core, "event_bus") and hasattr(
            self.vanta_core.event_bus, "emit"
        ):
            try:
                self.vanta_core.event_bus.emit(event_name, event_data)
                return
            except Exception as e:
                logger.warning(
                    f"Failed to emit event {event_name} through event_bus: {e}"
                )

        # Fall back to publish_event if available
        if hasattr(self.vanta_core, "publish_event"):
            try:
                self.vanta_core.publish_event(event_name, event_data)
                return
            except Exception as e:
                logger.warning(
                    f"Failed to publish event {event_name} through publish_event: {e}"
                )

        # Log if no event publishing mechanism is available
        logger.warning(
            f"Unable to publish event {event_name}: No event publishing mechanism available"
        )


def initialize_vmb_system(
    vanta_core: Optional[UnifiedVantaCore] = None,
    config: Optional[Dict[str, Any]] = None,
) -> VMBIntegrationHandler:
    """
    Initialize the VMB system with VantaCore.

    Args:
        vanta_core: Optional VantaCore instance
        config: Optional configuration for VMB

    Returns:
        The initialized VMB integration handler
    """
    handler = VMBIntegrationHandler(vanta_core)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(handler.initialize_vmb_system(config))
    finally:
        loop.close()

    return handler
