"""
Vanta Central Orchestrator
=========================

The central orchestrator that manages all modules in the VoxSigil Library.
Provides unified communication, fallback coordination, and system monitoring.

Key Responsibilities:
- Module registration and lifecycle management
- Request routing and load balancing
- Fallback coordination when services fail
- System-wide monitoring and observability
- Event-driven communication between modules
"""

# HOLO-1.5 Registration System
try:
    from core.base import CognitiveMeshRole, vanta_core_module
except ImportError:
    # Safe fallback decorator that accepts parameters
    def vanta_core_module(**kwargs):
        def decorator(cls):
            return cls

        return decorator

    class CognitiveMeshRole:
        ORCHESTRATOR = "orchestrator"


import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..interfaces import ModuleAdapterProtocol
from .fallback_implementations import fallback_registry


@vanta_core_module(
    name="vanta_orchestrator",
    subsystem="vanta_core",
    mesh_role=CognitiveMeshRole.ORCHESTRATOR,
    description="Central orchestrator for VantaCore module management and communication",
    capabilities=[
        "module_registry",
        "load_balancing",
        "fallback_coordination",
        "system_monitoring",
    ],
)
class VantaOrchestrator:
    """
    Central orchestrator managing all VoxSigil Library modules.
    """

    def __init__(self):
        self._modules: Dict[str, ModuleAdapterProtocol] = {}
        self._module_info: Dict[str, Dict[str, Any]] = {}
        self._capabilities: Dict[str, List[str]] = {}
        self._event_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._request_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "fallback_requests": 0,
                "avg_response_time": 0.0,
                "last_request": None,
            }
        )
        self._system_health: Dict[str, Any] = {
            "status": "initializing",
            "uptime_start": datetime.utcnow(),
            "total_modules": 0,
            "active_modules": 0,
            "failed_modules": 0,
        }

        self._logger = logging.getLogger(__name__)
        self._logger.info("Vanta Orchestrator initialized")

    async def register_module(
        self,
        module_id: str,
        module: ModuleAdapterProtocol,
        startup_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register and initialize a module with Vanta."""
        try:
            self._logger.info(f"Registering module: {module_id}")

            # Initialize module with Vanta client
            vanta_client = VantaClient(self, module_id)
            success = await module.initialize(vanta_client, startup_config or {})

            if not success:
                self._logger.error(f"Failed to initialize module: {module_id}")
                return False

            # Store module information
            self._modules[module_id] = module
            self._module_info[module_id] = module.module_info
            self._capabilities[module_id] = module.capabilities

            # Update system health
            self._system_health["total_modules"] += 1
            self._system_health["active_modules"] += 1

            # Perform initial health check
            health_status = await module.health_check()
            self._module_info[module_id]["health_status"] = health_status
            self._module_info[module_id]["registered_at"] = (
                datetime.utcnow().isoformat()
            )

            self._logger.info(f"Successfully registered module: {module_id}")

            # Publish module registration event
            await self._publish_system_event(
                "module_registered",
                {
                    "module_id": module_id,
                    "capabilities": module.capabilities,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            return True

        except Exception as e:
            self._logger.error(f"Error registering module {module_id}: {str(e)}")
            self._system_health["failed_modules"] += 1
            return False

    async def unregister_module(self, module_id: str) -> bool:
        """Unregister and shutdown a module."""
        try:
            if module_id not in self._modules:
                return False

            self._logger.info(f"Unregistering module: {module_id}")

            # Gracefully shutdown module
            module = self._modules[module_id]
            await module.shutdown()

            # Remove from registries
            del self._modules[module_id]
            del self._module_info[module_id]
            del self._capabilities[module_id]

            # Update system health
            self._system_health["active_modules"] -= 1

            # Publish module unregistration event
            await self._publish_system_event(
                "module_unregistered",
                {"module_id": module_id, "timestamp": datetime.utcnow().isoformat()},
            )

            self._logger.info(f"Successfully unregistered module: {module_id}")
            return True

        except Exception as e:
            self._logger.error(f"Error unregistering module {module_id}: {str(e)}")
            return False

    async def send_request(
        self,
        source_module: str,
        target_module: str,
        request_type: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Route request between modules with fallback support."""
        request_start = datetime.utcnow()

        try:
            # Validate modules
            if target_module not in self._modules:
                return await self._handle_request_with_fallback(
                    target_module,
                    request_type,
                    payload,
                    f"Target module {target_module} not found",
                )

            # Check if target module can handle request
            target_capabilities = self._capabilities.get(target_module, [])
            if request_type not in target_capabilities:
                return await self._handle_request_with_fallback(
                    target_module,
                    request_type,
                    payload,
                    f"Module {target_module} doesn't support {request_type}",
                )

            # Send request to target module
            module = self._modules[target_module]
            context = {
                "source_module": source_module,
                "target_module": target_module,
                "request_id": f"{source_module}_{target_module}_{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Execute with timeout
            result = await asyncio.wait_for(
                module.handle_request(request_type, payload, context), timeout=timeout
            )

            # Update statistics
            self._update_request_stats(target_module, request_start, True, False)

            return {
                "success": True,
                "result": result,
                "source_module": source_module,
                "target_module": target_module,
                "fallback_used": False,
            }

        except asyncio.TimeoutError:
            self._logger.warning(f"Request timeout: {source_module} -> {target_module}")
            return await self._handle_request_with_fallback(
                target_module,
                request_type,
                payload,
                f"Request timeout to {target_module}",
            )

        except Exception as e:
            self._logger.error(
                f"Request error: {source_module} -> {target_module}: {str(e)}"
            )
            return await self._handle_request_with_fallback(
                target_module, request_type, payload, f"Request failed: {str(e)}"
            )

    async def _handle_request_with_fallback(
        self,
        target_module: str,
        request_type: str,
        payload: Dict[str, Any],
        error_reason: str,
    ) -> Dict[str, Any]:
        """Handle request using fallback implementations."""
        try:
            # Determine service type from module name
            service_type = self._get_service_type(target_module)

            # Get appropriate fallback
            fallback = await fallback_registry.get_fallback(
                service_type, request_type, payload
            )

            if fallback:
                self._logger.info(
                    f"Using fallback for {target_module}: {fallback.fallback_type}"
                )

                context = {
                    "original_target": target_module,
                    "error_reason": error_reason,
                    "fallback_type": fallback.fallback_type,
                }

                result = await fallback.execute_fallback(request_type, payload, context)

                # Update statistics
                request_start = datetime.utcnow()
                self._update_request_stats(target_module, request_start, True, True)

                return {
                    "success": result.get("success", False),
                    "result": result.get("result"),
                    "fallback_used": True,
                    "fallback_type": fallback.fallback_type,
                    "original_error": error_reason,
                }

            else:
                self._logger.error(
                    f"No fallback available for {target_module}:{request_type}"
                )
                self._update_request_stats(
                    target_module, datetime.utcnow(), False, False
                )

                return {
                    "success": False,
                    "error": f"No fallback available: {error_reason}",
                    "fallback_used": False,
                }

        except Exception as e:
            self._logger.error(f"Fallback execution failed: {str(e)}")
            return {
                "success": False,
                "error": f"Fallback failed: {str(e)}",
                "fallback_used": False,
            }

    def _get_service_type(self, module_id: str) -> str:
        """Determine service type from module ID."""
        module_lower = module_id.lower()

        if "rag" in module_lower or "retrieval" in module_lower:
            return "rag"
        elif "llm" in module_lower or "language" in module_lower:
            return "llm"
        elif "memory" in module_lower or "storage" in module_lower:
            return "memory"
        elif "agent" in module_lower:
            return "agent"
        elif "model" in module_lower:
            return "model"
        else:
            return "generic"

    def _update_request_stats(
        self,
        module_id: str,
        request_start: datetime,
        success: bool,
        fallback_used: bool,
    ) -> None:
        """Update request statistics for monitoring."""
        stats = self._request_stats[module_id]

        stats["total_requests"] += 1
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1

        if fallback_used:
            stats["fallback_requests"] += 1

        # Update average response time
        response_time = (datetime.utcnow() - request_start).total_seconds()
        if stats["avg_response_time"] == 0:
            stats["avg_response_time"] = response_time
        else:
            # Exponential moving average
            stats["avg_response_time"] = (
                0.8 * stats["avg_response_time"] + 0.2 * response_time
            )

        stats["last_request"] = datetime.utcnow().isoformat()

    async def subscribe_to_events(
        self,
        module_id: str,
        event_types: List[str],
        handler: Callable[[Dict[str, Any]], None],
    ) -> str:
        """Subscribe module to system events."""
        subscription_id = f"{module_id}_{datetime.utcnow().timestamp()}"

        for event_type in event_types:
            self._event_subscribers[event_type].append(
                {
                    "subscription_id": subscription_id,
                    "module_id": module_id,
                    "handler": handler,
                }
            )

        self._logger.info(f"Module {module_id} subscribed to events: {event_types}")
        return subscription_id

    async def publish_event(
        self, source_module: str, event_type: str, event_data: Dict[str, Any]
    ) -> bool:
        """Publish event to subscribed modules."""
        try:
            if event_type not in self._event_subscribers:
                return True  # No subscribers, but not an error

            event = {
                "event_type": event_type,
                "source_module": source_module,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data,
            }

            # Notify all subscribers
            for subscriber in self._event_subscribers[event_type]:
                try:
                    await subscriber["handler"](event)
                except Exception as e:
                    self._logger.error(
                        f"Error notifying subscriber {subscriber['module_id']}: {str(e)}"
                    )

            return True

        except Exception as e:
            self._logger.error(f"Error publishing event {event_type}: {str(e)}")
            return False

    async def _publish_system_event(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> None:
        """Publish system-level events."""
        await self.publish_event("vanta_orchestrator", event_type, event_data)

    async def get_module_status(self, module_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific module."""
        if module_id not in self._modules:
            return {"error": f"Module {module_id} not found"}

        try:
            module = self._modules[module_id]
            health_status = await module.health_check()
            metrics = await module.get_metrics()

            return {
                "module_id": module_id,
                "module_info": self._module_info[module_id],
                "capabilities": self._capabilities[module_id],
                "health_status": health_status,
                "metrics": metrics,
                "request_stats": self._request_stats[module_id],
            }

        except Exception as e:
            return {"module_id": module_id, "error": f"Failed to get status: {str(e)}"}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health."""
        uptime = datetime.utcnow() - self._system_health["uptime_start"]

        # Perform health checks on all modules
        module_health = {}
        for module_id in self._modules:
            try:
                health = await self._modules[module_id].health_check()
                module_health[module_id] = health
            except Exception as e:
                module_health[module_id] = {"status": "error", "error": str(e)}

        # Calculate system health metrics
        healthy_modules = sum(
            1 for health in module_health.values() if health.get("status") == "healthy"
        )

        overall_health = (
            "healthy" if healthy_modules == len(self._modules) else "degraded"
        )
        if healthy_modules == 0:
            overall_health = "critical"

        return {
            "overall_health": overall_health,
            "uptime_seconds": uptime.total_seconds(),
            "total_modules": len(self._modules),
            "healthy_modules": healthy_modules,
            "module_health": module_health,
            "request_stats": dict(self._request_stats),
            "fallback_stats": fallback_registry.get_usage_stats(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown all modules and the orchestrator."""
        self._logger.info("Shutting down Vanta Orchestrator")

        # Shutdown all modules
        for module_id in list(self._modules.keys()):
            await self.unregister_module(module_id)

        self._system_health["status"] = "shutdown"
        self._logger.info("Vanta Orchestrator shutdown complete")


class VantaClient:
    """
    Client interface that modules use to communicate with Vanta.
    """

    def __init__(self, orchestrator: VantaOrchestrator, module_id: str):
        self._orchestrator = orchestrator
        self._module_id = module_id

    async def send_request(
        self,
        target_module: str,
        request_type: str,
        payload: Dict[str, Any],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Send request to another module through Vanta."""
        result = await self._orchestrator.send_request(
            self._module_id, target_module, request_type, payload
        )

        if callback:
            try:
                await callback(result)
            except Exception as e:
                logging.getLogger(__name__).error(f"Callback error: {str(e)}")

        return result

    async def subscribe_to_events(
        self, event_types: List[str], handler: Callable[[Dict[str, Any]], None]
    ) -> str:
        """Subscribe to system events."""
        return await self._orchestrator.subscribe_to_events(
            self._module_id, event_types, handler
        )

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Publish event to system."""
        return await self._orchestrator.publish_event(
            self._module_id, event_type, event_data
        )

    async def get_module_status(self, module_id: str) -> Dict[str, Any]:
        """Get status of another module."""
        return await self._orchestrator.get_module_status(module_id)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return await self._orchestrator.get_system_status()


# Global orchestrator instance
vanta_orchestrator = VantaOrchestrator()
