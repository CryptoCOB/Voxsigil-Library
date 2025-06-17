"""
Protocol Interface Definitions
=============================

Communication protocols and adapter interfaces for module integration
through Vanta as the central orchestrator.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class VantaProtocol(Protocol):
    """
    Core Vanta Communication Protocol

    Defines the contract for modules to communicate with Vanta
    as the central orchestrator.
    """

    async def register_module(
        self, module_id: str, module_info: Dict[str, Any], capabilities: List[str]
    ) -> bool:
        """Register a module with Vanta."""
        ...

    async def send_request(
        self,
        target_module: str,
        request_type: str,
        payload: Dict[str, Any],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Send request to another module through Vanta."""
        ...

    async def subscribe_to_events(
        self, event_types: List[str], handler: Callable[[Dict[str, Any]], None]
    ) -> str:
        """Subscribe to system events."""
        ...

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Publish event to system."""
        ...

    async def get_module_status(self, module_id: str) -> Dict[str, Any]:
        """Get status of a registered module."""
        ...


@runtime_checkable
class ModuleAdapterProtocol(Protocol):
    """
    Module Adapter Protocol

    Interface that each module must implement to integrate with Vanta.
    Provides standardized communication and lifecycle management.
    """

    @property
    def module_id(self) -> str:
        """Unique identifier for this module."""
        ...

    @property
    def module_info(self) -> Dict[str, Any]:
        """Module metadata and configuration."""
        ...

    @property
    def capabilities(self) -> List[str]:
        """List of capabilities this module provides."""
        ...

    async def initialize(self, vanta_client: VantaProtocol, config: Dict[str, Any]) -> bool:
        """Initialize module with Vanta client."""
        ...

    async def shutdown(self) -> bool:
        """Gracefully shutdown module."""
        ...

    async def handle_request(
        self, request_type: str, payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming request from Vanta."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """Get module performance metrics."""
        ...


@runtime_checkable
class IntegrationProtocol(Protocol):
    """
    Integration Protocol

    Defines contracts for cross-module integration patterns:
    - Data sharing and synchronization
    - Workflow coordination
    - Event-driven communication
    """

    async def share_data(
        self, data_id: str, data: Any, access_permissions: Dict[str, List[str]]
    ) -> bool:
        """Share data with specified access permissions."""
        ...

    async def access_shared_data(self, data_id: str, requester_id: str) -> Optional[Any]:
        """Access shared data if permissions allow."""
        ...

    async def coordinate_workflow(
        self, workflow_id: str, steps: List[Dict[str, Any]], coordination_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate multi-module workflow."""
        ...

    async def synchronize_state(
        self, state_key: str, state_data: Dict[str, Any], sync_targets: List[str]
    ) -> Dict[str, bool]:
        """Synchronize state across modules."""
        ...

    async def handle_integration_event(
        self, event: Dict[str, Any], integration_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle integration-specific events."""
        ...


@runtime_checkable
class FallbackProtocol(Protocol):
    """
    Fallback Implementation Protocol

    Defines contracts for fallback implementations when primary
    services are unavailable.
    """

    @property
    def fallback_type(self) -> str:
        """Type of fallback this implementation provides."""
        ...

    @property
    def reliability_score(self) -> float:
        """Reliability score (0.0 to 1.0) of this fallback."""
        ...

    async def can_handle_request(self, request_type: str, payload: Dict[str, Any]) -> bool:
        """Check if this fallback can handle the request."""
        ...

    async def execute_fallback(
        self, request_type: str, payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback implementation."""
        ...

    async def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage and performance metrics."""
        ...


@runtime_checkable
class ObservabilityProtocol(Protocol):
    """
    Observability Protocol

    Standardizes monitoring, logging, and telemetry across modules.
    """

    async def log_event(
        self, level: str, message: str, context: Dict[str, Any], tags: Optional[List[str]] = None
    ) -> None:
        """Log event with context and tags."""
        ...

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str],
        timestamp: Optional[float] = None,
    ) -> None:
        """Record performance metric."""
        ...

    async def trace_operation(
        self,
        operation_name: str,
        operation_data: Dict[str, Any],
        parent_trace_id: Optional[str] = None,
    ) -> str:
        """Start tracing an operation, return trace ID."""
        ...

    async def end_trace(self, trace_id: str, result: Dict[str, Any]) -> None:
        """End operation tracing."""
        ...

    async def get_observability_data(
        self, time_range: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get observability data for analysis."""
        ...


# --- Memory and Storage Protocol Interfaces ---


@runtime_checkable
class MemoryBraidInterface(Protocol):
    """
    Unified Memory Braid Protocol

    Standard interface for memory braiding operations across engines.
    Consolidates different memory braid implementations.
    """

    def store_mirrored_data(
        self,
        original_key: Any,
        mirrored_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store mirrored data with key and metadata."""
        ...

    def retrieve_mirrored_data(self, original_key: Any) -> Optional[Any]:
        """Retrieve mirrored data by key."""
        ...

    def get_braid_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory braid."""
        ...

    def adapt_behavior(self, context_key: str) -> Dict[str, Any]:
        """Adapt behavior based on context."""
        ...

    def store_braid_data(
        self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store braid data (alternate signature for compatibility)."""
        ...

    def retrieve_braid_data(self, key: str) -> Optional[Any]:
        """Retrieve braid data (alternate signature for compatibility)."""
        ...

    def imprint(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Imprint data with optional TTL (external echo layer compatibility)."""
        ...
