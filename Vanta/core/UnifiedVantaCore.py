"""
UnifiedVantaCore - Composition-Based Vanta Architecture

Provides a unified interface that orchestrates both the VantaCognitiveEngine
(advanced cognitive processing with Meta-Learning, BLT Encoding, Hybrid RAG)
and the VantaOrchestrationEngine (simple component management and event handling).

This composition approach allows for modular, maintainable integration of both
engines while preserving their individual strengths.
"""

import asyncio
import datetime
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from BLT import BLTEncoder
from BLT.hybrid_middleware import HybridMiddleware
from Vanta.interfaces.blt_encoder_interface import BaseBLTEncoder
from Vanta.interfaces.hybrid_middleware_interface import BaseHybridMiddleware
from Vanta.interfaces.real_supervisor_connector import RealSupervisorConnector
from Vanta.interfaces.supervisor_connector_interface import BaseSupervisorConnector

from ..integration.vanta_supervisor import VantaSupervisor
from .UnifiedAgentRegistry import UnifiedAgentRegistry

from .UnifiedAsyncBus import UnifiedAsyncBus

from .agents import (
    Phi,
    Voxka,
    Gizmo,
    Nix,
    Echo,
    Oracle,
    Astra,
    Warden,
    Nebula,
    Orion,
    Evo,
    OrionApprentice,
    SocraticEngine,
    Dreamer,
    EntropyBard,
    CodeWeaver,
    EchoLore,
    MirrorWarden,
    PulseSmith,
    BridgeFlesh,
    Sam,
    Dave,
    Carla,
    Andy,
    Wendy,
    VoxAgent,
    SDKContext,
    SleepTimeComputeAgent,
    NullAgent,
)

# Configure logger
logger = logging.getLogger("unified_vanta_core")

# Direct imports

# Set availability flags
AGENT_REGISTRY_AVAILABLE = True
VANTA_SUPERVISOR_AVAILABLE = True
BLT_COMPONENTS_AVAILABLE = True

# --- PYDANTIC VALIDATION ---


class ComponentRegistry:
    """Enhanced component registry with metadata and lifecycle management."""

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        self._component_health: Dict[str, str] = {}
        self._lock = threading.RLock()

    def register(
        self, name: str, component: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a component with optional metadata and health tracking."""
        with self._lock:
            if name in self._components:
                logger.warning(f"Component '{name}' already registered, overwriting")

            self._components[name] = component
            self._component_metadata[name] = metadata or {}
            self._component_metadata[name].update(
                {
                    "registered_at": datetime.datetime.now(),
                    "component_type": type(component).__name__,
                    "registration_count": self._component_metadata.get(name, {}).get(
                        "registration_count", 0
                    )
                    + 1,
                }
            )
            self._component_health[name] = "healthy"

            logger.info(f"Component '{name}' registered successfully")
            return True

    def get(self, name: str, default: Any = None) -> Any:
        """Get a component by name with health check."""
        with self._lock:
            component = self._components.get(name, default)
            if component is not None and name in self._component_health:
                # Simple health check - could be enhanced
                if hasattr(component, "is_healthy"):
                    self._component_health[name] = (
                        "healthy" if component.is_healthy() else "degraded"
                    )
            return component

    def unregister(self, name: str) -> bool:
        """Unregister a component."""
        with self._lock:
            if name in self._components:
                del self._components[name]
                del self._component_metadata[name]
                self._component_health.pop(name, None)
                logger.info(f"Component '{name}' unregistered")
                return True
            return False

    def list_components(self) -> List[str]:
        """List all registered component names."""
        with self._lock:
            return list(self._components.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a component."""
        with self._lock:
            return self._component_metadata.get(name)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status."""
        with self._lock:
            healthy_count = sum(
                1 for status in self._component_health.values() if status == "healthy"
            )
            return {
                "total_components": len(self._components),
                "healthy_components": healthy_count,
                "degraded_components": len(self._components) - healthy_count,
                "components": list(self._components.keys()),
                "component_health": self._component_health.copy(),
                "status": "active",
            }


class EventBus:
    """Enhanced event bus with history and filtering capabilities."""

    def __init__(self):
        self._subscribers: Dict[str, List[tuple[Callable, int]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        self._event_stats: Dict[str, int] = defaultdict(int)

    def subscribe(self, event_type: str, callback: Callable, priority: int = 0) -> None:
        """Subscribe to an event type with optional priority."""
        with self._lock:
            # Insert based on priority (higher priority first)
            self._subscribers[event_type].append((callback, priority))
            self._subscribers[event_type].sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Subscribed to event '{event_type}' with priority {priority}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            self._subscribers[event_type] = [
                (cb, prio)
                for cb, prio in self._subscribers[event_type]
                if cb != callback
            ]
            logger.debug(f"Unsubscribed from event '{event_type}'")

    def emit(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Emit an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.datetime.now(),
            "kwargs": kwargs,
        }

        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Update stats
            self._event_stats[event_type] += 1

            # Notify subscribers
            subscribers = self._subscribers[event_type]
            if not subscribers:
                logger.debug(
                    f"No subscribers for event '{event_type}'"
                )
            for callback, priority in subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        f"Error in event callback for '{event_type}': {e}"
                    )

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        with self._lock:
            return {
                "total_events": len(self._event_history),
                "event_type_counts": dict(self._event_stats),
                "recent_events": self._event_history[-10:]
                if self._event_history
                else [],
            }


def trace_event(message: str, category: str = "info", **kwargs):
    """Enhanced trace event function with categorization."""
    timestamp = datetime.datetime.now().isoformat()
    logger.info(f"[{category.upper()}] {timestamp}: {message}")
    if kwargs:
        logger.debug(f"Event data: {kwargs}")


class UnifiedVantaCore:
    """
    Unified VantaCore - Integrated Cognitive Orchestration Engine

    Combines simple component orchestration with advanced cognitive capabilities:
    - Component Registry & Event Bus (orchestration layer)
    - Meta-Learning & BLT Encoding (cognitive layer)
    - Progressive capability loading (scalable architecture)
    - Full VoxSigil integration (system layer)
    """

    _instance: Optional["UnifiedVantaCore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "UnifiedVantaCore":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_sigil_ref: Optional[str] = None,
        supervisor_connector: Optional[BaseSupervisorConnector] = None,
        blt_encoder: Optional[BaseBLTEncoder] = None,
        hybrid_middleware: Optional[BaseHybridMiddleware] = None,
        enable_cognitive_features: bool = True,
    ):
        """
        Initialize UnifiedVantaCore with progressive capability loading.

        Args:
            config_sigil_ref: Optional sigil reference for configuration
            supervisor_connector: Optional supervisor connector for advanced features
            blt_encoder: Optional BLT encoder for cognitive capabilities
            hybrid_middleware: Optional middleware for advanced processing
            enable_cognitive_features: Whether to enable advanced cognitive features
        """
        # Only initialize once (singleton pattern)
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.start_time = (
            datetime.datetime.now()
        )  # --- ORCHESTRATION LAYER (Always Available) ---
        self.registry = ComponentRegistry()
        self.event_bus = EventBus()
        self.async_bus = UnifiedAsyncBus(logger)
        try:
            asyncio.create_task(self.async_bus.start())
        except Exception as e:
            logger.error(f"Failed to start async bus: {e}")
        self._config: Dict[str, Any] = {}

        # --- AGENT MANAGEMENT LAYER (Step 1 of VANTA Integration Master Plan) ---
        # Initialize UnifiedAgentRegistry for agent management
        if AGENT_REGISTRY_AVAILABLE and UnifiedAgentRegistry is not None:
            self.agent_registry = UnifiedAgentRegistry(logger)
            logger.info("UnifiedAgentRegistry initialized for agent management")
        else:
            self.agent_registry = None
            logger.warning(
                "UnifiedAgentRegistry not available, agent management disabled"
            )

        # Initialize VantaSupervisor for agent management if available
        self.vanta_supervisor = None
        if VANTA_SUPERVISOR_AVAILABLE and VantaSupervisor is not None:
            try:
                self.vanta_supervisor = VantaSupervisor(self)
                logger.info(
                    "VantaSupervisor initialized for central agent orchestration"
                )
            except Exception as e:
                logger.error(f"Failed to initialize VantaSupervisor: {e}")
                self.vanta_supervisor = None

        # --- COGNITIVE LAYER (Conditional Loading) ---
        self.cognitive_enabled = False
        self.supervisor_connector: Optional[BaseSupervisorConnector] = None
        self.blt_encoder: Optional[BaseBLTEncoder] = None
        self.hybrid_middleware: Optional[BaseHybridMiddleware] = None

        # Meta-learning attributes (initialized only if cognitive features are enabled)
        self.meta_parameters: Dict[str, Any] = {}
        self.task_adaptation_profiles: Dict[str, Dict[str, Any]] = {}
        self.cross_task_knowledge_index: Dict[str, List[float]] = {}
        self.supervisor_registration_sigil: Optional[str] = None
        self.last_supervisor_health_check_status: str = "pending"

        logger.info("UnifiedVantaCore initialized with orchestration capabilities")

        # Load cognitive features if requested and components are available
        if enable_cognitive_features and config_sigil_ref:
            self._initialize_cognitive_layer(
                config_sigil_ref, supervisor_connector, blt_encoder, hybrid_middleware
            )

        # Register core components
        self._register_core_components()

        # Initialize speech integration (TTS/STT)
        self._initialize_speech_integration()

        # Initialize VMB integration (Step 16 of VANTA Integration Master Plan)
        self._initialize_vmb_integration()

        # Register core agents (Step 1 of VANTA Integration Master Plan)
        self._initialize_core_agents()


        # Map subsystems to guardian agents (Step 3 of VANTA Integration Master Plan)
        self._map_subsystems_to_guardians()


        self.get_agents_by_capability = (
            self.agent_registry.get_agents_by_capability
            if self.agent_registry
            else None
        )

        # Emit initialization event
        self.event_bus.emit(
            "vanta_core_initialized",
            {
                "cognitive_enabled": self.cognitive_enabled,
                "start_time": self.start_time,
                "components_available": BLT_COMPONENTS_AVAILABLE,
            },
        )

    def _initialize_cognitive_layer(
        self,
        config_sigil_ref: str,
        supervisor_connector: Optional[BaseSupervisorConnector],
        blt_encoder: Optional[BaseBLTEncoder],
        hybrid_middleware: Optional[BaseHybridMiddleware],
    ) -> None:
        """Initialize advanced cognitive capabilities."""
        try:
            # Set up supervisor connector
            if supervisor_connector:
                self.supervisor_connector = supervisor_connector
            elif BLT_COMPONENTS_AVAILABLE:
                self.supervisor_connector = RealSupervisorConnector()
            else:
                logger.warning("No supervisor connector available")
                return

            # Load configuration
            self.config_sigil_ref = config_sigil_ref
            self._config = self._load_configuration(config_sigil_ref)

            # Initialize meta-learning parameters
            self.meta_parameters = {
                "default_learning_rate": self._config.get(
                    "default_learning_rate", 0.05
                ),
                "default_exploration_rate": self._config.get(
                    "default_exploration_rate", 0.1
                ),
                "transfer_strength": self._config.get("transfer_strength", 0.3),
                "parameter_damping_factor": self._config.get(
                    "parameter_damping_factor", 0.7
                ),
                "similarity_threshold_for_transfer": self._config.get(
                    "similarity_threshold_for_transfer", 0.75
                ),
                "max_performance_history_per_task": self._config.get(
                    "max_performance_history_per_task", 50
                ),
                "min_perf_points_for_adaptation": self._config.get(
                    "min_perf_points_for_adaptation", 5
                ),
                "min_perf_points_for_global_opt": self._config.get(
                    "min_perf_points_for_global_opt", 10
                ),
            }

            # Set up BLT encoder
            if blt_encoder:
                self.blt_encoder = blt_encoder
            elif BLT_COMPONENTS_AVAILABLE:
                self.blt_encoder = BLTEncoder()

            # Set up hybrid middleware
            if hybrid_middleware:
                self.hybrid_middleware = hybrid_middleware
            elif BLT_COMPONENTS_AVAILABLE:

                class ConcreteHybridMiddleware(HybridMiddleware):
                    def get_middleware_capabilities(self) -> List[str]:
                        return ["basic_processing", "data_transformation"]

                    def configure_middleware(self, config: Dict[str, Any]) -> bool:
                        return True

                self.hybrid_middleware = ConcreteHybridMiddleware()

            self.cognitive_enabled = True
            logger.info("Cognitive layer initialized successfully")

            # Initialize supervisor integration
            self._initialize_supervisor_integration()

        except Exception as e:
            logger.error(f"Failed to initialize cognitive layer: {e}", exc_info=True)
            self.cognitive_enabled = False

    def _register_core_components(self) -> None:
        """Register core VantaCore components."""
        self.registry.register(
            "component_registry",
            self.registry,
            {"description": "Core component registry"},
        )
        self.registry.register(
            "event_bus", self.event_bus, {"description": "Core event bus"}
        )
        self.registry.register(
            "async_bus", self.async_bus, {"description": "Unified async bus"}
        )

        if self.cognitive_enabled:
            if self.supervisor_connector:
                self.registry.register(
                    "supervisor_connector",
                    self.supervisor_connector,
                    {"description": "VoxSigil supervisor interface"},
                )
            if self.blt_encoder:
                self.registry.register(
                    "blt_encoder",
                    self.blt_encoder,
                    {"description": "BLT encoding engine"},
                )
            if self.hybrid_middleware:
                self.registry.register(
                    "hybrid_middleware",
                    self.hybrid_middleware,
                    {"description": "Hybrid RAG middleware"},
                )

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """Get capabilities of a specific agent."""
        if not self.agent_registry:
            logger.warning("Agent registry not available")
            return []

        agent = self.agent_registry.get_agent(agent_name)
        if not agent:
            logger.warning(f"Agent '{agent_name}' not found")
            return []

        # Try to get capabilities from agent metadata in registry
        if (
            hasattr(self.agent_registry, "agents")
            and agent_name in self.agent_registry.agents
        ):
            metadata = self.agent_registry.agents[agent_name].get("metadata", {})
            if "capabilities" in metadata:
                return metadata["capabilities"]

        # Fallback: try to get capabilities directly from agent
        if hasattr(agent, "get_capabilities"):
            try:
                return agent.get_capabilities()
            except Exception as e:
                logger.error(f"Error getting capabilities from {agent_name}: {e}")
                return []

        logger.warning(f"Agent '{agent_name}' does not have capabilities defined")
        return []

    def query_deep_cognition_memory(
        self, query: str, task_sigil_ref: Optional[str] = None
    ) -> Optional[str]:
        """
        Query the deep cognition memory for relevant information.

        Args:
            query: The query string to search in memory
            task_sigil_ref: Optional reference to the task sigil for context

        Returns:
            A sigil reference to the retrieved information or None if not found
        """
        if not self.cognitive_enabled or not self.hybrid_middleware:
            logger.error("Cognitive features not enabled - cannot query memory")
            return None

        try:
            # Check if the middleware has query_memory method
            if hasattr(self.hybrid_middleware, "query_memory"):
                # Use getattr to safely call the method when it exists
                query_memory_func = getattr(self.hybrid_middleware, "query_memory")
                result_sigil = query_memory_func(query, task_sigil_ref)
            else:
                # Fallback to a generic interface method if available
                logger.warning("hybrid_middleware does not have query_memory method")
                return None

            if result_sigil:
                logger.info(f"Memory query successful: {result_sigil}")
                return result_sigil
            else:
                logger.warning("No relevant information found in memory")
                return None
        except Exception as e:
            logger.error(f"Error querying memory: {e}", exc_info=True)
            return None

    def _initialize_core_agents(self) -> None:
        """Initialize and register core agents (Step 1 of VANTA Integration Master Plan)."""
        if not self.agent_registry:
            logger.warning("Agent registry not available, cannot register core agents")
            return

        # Register VantaSupervisor as a core agent if available (Step 1 of VANTA Integration Master Plan)
        if self.vanta_supervisor:
            try:
                self.agent_registry.register_agent(
                    "vanta_supervisor",
                    self.vanta_supervisor,
                    {
                        "type": "supervisor_agent",
                        "capabilities": self.vanta_supervisor.get_capabilities(),
                        "priority": "critical",
                        "role": "central_agent_orchestrator",
                        "managed_by": "unified_vanta_core",
                    },
                )
                logger.info(
                    "VantaSupervisor registered as core agent with capabilities: %s",
                    self.vanta_supervisor.get_capabilities(),
                )
            except Exception as e:
                logger.error(f"Failed to register VantaSupervisor as core agent: {e}")

        # Register all defined agents dynamically from agents.__all__
        agent_classes = []
        try:
            from agents import __all__ as agent_names  # type: ignore
            import agents as agent_pkg

            for name in agent_names:
                if name in {"BaseAgent", "NullAgent"}:
                    continue
                cls = getattr(agent_pkg, name, None)
                if cls:
                    agent_classes.append(cls)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to gather agent classes dynamically: {e}")

        for cls in agent_classes:
            try:
                instance = cls()
            except Exception as e:
                logger.error(f"Failed to instantiate {cls.__name__}: {e}")
                instance = NullAgent()

            try:
                if hasattr(instance, "initialize_subsystem"):
                    instance.initialize_subsystem(self)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize {cls.__name__}: {e}")

            try:
                self.agent_registry.register_agent(
                    cls.__name__,
                    instance,
                    {
                        "sigil": getattr(cls, "sigil", ""),
                        "invocations": getattr(cls, "invocations", []),
                        "tags": getattr(cls, "tags", []),
                        "doc": getattr(cls, "__doc__", ""),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to register {cls.__name__}: {e}")


    def _map_subsystems_to_guardians(self) -> None:
        """Link key subsystems to their guardian agents (Step 3 of VANTA Integration Master Plan)."""
        if not self.agent_registry:
            return

        mapping = {
            "EntropyBard": "rag_interface",
            "PulseSmith": "gridformer_connector",
            "MirrorWarden": "meta_learner",
            "CodeWeaver": "meta_learner",
            "Dreamer": "art_controller",
            "BridgeFlesh": "vmb_integration_handler",
            "Carla": "speech_integration_handler",
            "Wendy": "speech_integration_handler",
        }

        for agent_name, component_key in mapping.items():
            agent = self.agent_registry.get_agent(agent_name)
            if not agent:
                continue
            try:
                subsystem = self.get_component(component_key)
                if hasattr(agent, "initialize_subsystem"):
                    agent.initialize_subsystem(self)
                if subsystem:
                    setattr(agent, "subsystem", subsystem)
            except Exception as e:
                logger.error(f"Failed to map {agent_name} to {component_key}: {e}")


    # --- AGENT MANAGEMENT METHODS (Step 1 of VANTA Integration Master Plan) ---

    def register_agent(
        self, name: str, agent: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register an agent in the system (public interface to agent registry)."""
        if not self.agent_registry:
            logger.warning("Agent registry not available, cannot register agent")
            return False

        try:
            self.agent_registry.register_agent(name, agent, metadata)
            self.event_bus.emit(
                "agent_registered",
                {"name": name, "type": type(agent).__name__, "metadata": metadata},
            )
            logger.info(f"Agent '{name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent '{name}': {e}")
            return False

    def get_agent(self, name: str) -> Any:
        """Retrieve an agent by name."""
        if not self.agent_registry:
            logger.warning("Agent registry not available")
            return None
        return self.agent_registry.get_agent(name)

    def get_all_agents(self) -> List[tuple[str, Any]]:
        """Get all registered agents."""
        if not self.agent_registry:
            logger.warning("Agent registry not available")
            return []
        return self.agent_registry.get_all_agents()

    def send_agent_message(self, agent_name: str, message: str, **kwargs) -> Any:
        """Send a message to a specific agent."""
        agent = self.get_agent(agent_name)
        if not agent:
            logger.warning(f"Agent '{agent_name}' not found")
            return None

        # Try to send message using common agent interfaces
        if hasattr(agent, "process_message"):
            return agent.process_message(message, **kwargs)
        elif hasattr(agent, "handle_message"):
            return agent.handle_message(message, **kwargs)
        elif hasattr(agent, "execute"):
            return agent.execute(message, **kwargs)
        else:
            logger.warning(
                f"Agent '{agent_name}' does not have a recognized message interface"
            )
            return None

    # --- ORCHESTRATION METHODS ---

    def register_component(
        self, name: str, component: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a component in the system."""
        success = self.registry.register(name, component, metadata)
        if success:
            self.event_bus.emit(
                "component_registered",
                {"name": name, "type": type(component).__name__, "metadata": metadata},
            )
        return success

    def get_component(self, name: str, default: Any = None) -> Any:
        """Get a component by name."""
        return self.registry.get(name, default)

    def emit_event(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Emit an event on the event bus."""
        self.event_bus.emit(event_type, data, **kwargs)

    def subscribe_to_event(
        self, event_type: str, callback: Callable, priority: int = 0
    ) -> None:
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, callback, priority)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        registry_status = self.registry.get_status()
        event_stats = self.event_bus.get_event_stats()

        status = {
            "vanta_core_version": "unified_v1.0",
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": (
                datetime.datetime.now() - self.start_time
            ).total_seconds(),
            "cognitive_enabled": self.cognitive_enabled,
            "blt_components_available": BLT_COMPONENTS_AVAILABLE,
            "registry": registry_status,
            "events": event_stats,
        }

        if self.cognitive_enabled:
            status.update(
                {
                    "supervisor_status": self.last_supervisor_health_check_status,
                    "registered_tasks": len(self.task_adaptation_profiles),
                    "knowledge_index_size": len(self.cross_task_knowledge_index),
                }
            )

        return status

    # --- COGNITIVE METHODS (Available only if cognitive layer is enabled) ---

    def _load_configuration(self, config_sigil_ref: str) -> Dict[str, Any]:
        """Load configuration from VoxSigil definition."""
        if not self.supervisor_connector:
            logger.warning(
                "No supervisor connector available for configuration loading"
            )
            return {}

        try:
            config_content = self.supervisor_connector.get_sigil_content_as_dict(
                config_sigil_ref
            )
            if not config_content:
                logger.error(f"Config sigil '{config_sigil_ref}' is empty or not found")
                return {}

            logger.info(f"Configuration loaded from '{config_sigil_ref}'")
            return config_content.get(
                "vanta_core_settings",
                config_content.get(
                    "custom_attributes_vanta_extensions", config_content
                ),
            )
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return {}

    def _initialize_supervisor_integration(self) -> None:
        """Initialize integration with the supervisor."""
        if not self.cognitive_enabled or not self.supervisor_connector:
            return

        try:
            success = self._register_with_supervisor()
            if success:
                logger.info("Successfully registered with supervisor")
                self._perform_supervisor_health_check()
            else:
                logger.warning("Failed to register with supervisor")
                self.last_supervisor_health_check_status = "degraded"
        except Exception as e:
            logger.error(f"Error during supervisor integration: {e}", exc_info=True)
            self.last_supervisor_health_check_status = "error"

    def _register_with_supervisor(self) -> bool:
        """Register VantaCore with the supervisor."""
        # Implementation would depend on supervisor interface
        # For now, simulate successful registration
        self.supervisor_registration_sigil = (
            f"vanta_core_registration_{int(time.time())}"
        )
        return True

    def _perform_supervisor_health_check(self) -> None:
        """Perform health check with supervisor."""
        if not self.supervisor_connector or not self.supervisor_registration_sigil:
            self.last_supervisor_health_check_status = "degraded"
            return

        try:
            # Simulate health check
            self.last_supervisor_health_check_status = "healthy"
            logger.info("Supervisor health check successful")
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            self.last_supervisor_health_check_status = "error"

    def process_input(
        self,
        input_data_sigil_ref: str,
        task_sigil_ref: str,
        task_description_sigil_ref: Optional[str] = None,
    ) -> Optional[str]:
        """Process input through the cognitive pipeline (requires cognitive layer)."""
        if not self.cognitive_enabled:
            logger.error("Cognitive features not enabled - cannot process input")
            return None

        if not self.supervisor_connector or not self.hybrid_middleware:
            logger.error("Required cognitive components not available")
            return None

        logger.info(
            f"Processing input '{input_data_sigil_ref}' for task '{task_sigil_ref}'"
        )

        # Emit processing start event
        self.event_bus.emit(
            "input_processing_started",
            {
                "input_ref": input_data_sigil_ref,
                "task_ref": task_sigil_ref,
                "description_ref": task_description_sigil_ref,
            },
        )

        try:
            # Implementation would include the full cognitive processing pipeline
            # For now, return a placeholder result
            result_sigil = f"result_{task_sigil_ref}_{int(time.time())}"

            self.event_bus.emit(
                "input_processing_completed",
                {
                    "input_ref": input_data_sigil_ref,
                    "task_ref": task_sigil_ref,
                    "result_ref": result_sigil,
                },
            )

            return result_sigil

        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            self.event_bus.emit(
                "input_processing_failed",
                {
                    "input_ref": input_data_sigil_ref,
                    "task_ref": task_sigil_ref,
                    "error": str(e),
                },
            )
            return None

    # --- AGENT COORDINATION METHODS (Step 3 of VANTA Integration Master Plan) ---

    def coordinate_multi_agent_task(
        self, task_description: str, required_capabilities: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate a multi-agent task that requires multiple agents working together.

        Args:
            task_description: Description of the task to be performed
            required_capabilities: List of capabilities needed for the task
            **kwargs: Additional task parameters

        Returns:
            Dict containing coordination results and agent responses
        """
        if not self.vanta_supervisor:
            logger.warning("VantaSupervisor not available for multi-agent coordination")
            return {"error": "VantaSupervisor not available"}

        logger.info(f"Coordinating multi-agent task: {task_description}")

        # Step 1: Discover agents with required capabilities
        suitable_agents = self.discover_agents_by_capabilities(required_capabilities)
        if not suitable_agents:
            logger.warning(
                f"No agents found with required capabilities: {required_capabilities}"
            )
            return {
                "error": "No suitable agents found",
                "required_capabilities": required_capabilities,
            }

        # Step 2: Plan task distribution through VantaSupervisor
        coordination_plan = {
            "task_description": task_description,
            "agent_assignments": {},
            "execution_order": [],
            "coordination_strategy": "sequential",  # Default strategy
        }

        # Assign agents to task components based on capabilities
        for capability in required_capabilities:
            capable_agents = [
                agent
                for agent in suitable_agents
                if capability in agent["capabilities"]
            ]
            if capable_agents:
                # Select the first available agent for this capability
                selected_agent = capable_agents[0]
                coordination_plan["agent_assignments"][capability] = selected_agent[
                    "name"
                ]
                if selected_agent["name"] not in coordination_plan["execution_order"]:
                    coordination_plan["execution_order"].append(selected_agent["name"])

        # Step 3: Execute coordinated task through VantaSupervisor
        coordination_result = {
            "task_id": f"coordination_{int(time.time())}",
            "plan": coordination_plan,
            "results": {},
            "status": "in_progress",
        }

        try:
            # Execute task using VantaSupervisor coordination
            supervisor_task = {
                "action": "coordinate_task",
                "task_description": task_description,
                "coordination_plan": coordination_plan,
                "kwargs": kwargs,
            }

            supervisor_result = self.vanta_supervisor.perform_task(supervisor_task)
            coordination_result["supervisor_result"] = supervisor_result
            coordination_result["status"] = (
                "completed" if "error" not in supervisor_result else "failed"
            )

            # Emit coordination event
            self.event_bus.emit(
                "multi_agent_task_coordinated",
                {
                    "task_id": coordination_result["task_id"],
                    "agents_involved": coordination_plan["execution_order"],
                    "status": coordination_result["status"],
                },
            )

            logger.info(
                f"Multi-agent task coordination completed: {coordination_result['task_id']}"
            )
            return coordination_result

        except Exception as e:
            logger.error(f"Error during multi-agent coordination: {e}")
            coordination_result["status"] = "error"
            coordination_result["error"] = str(e)
            return coordination_result

    def delegate_task_to_agent(
        self, agent_name: str, task: Dict[str, Any], priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Delegate a specific task to a named agent with priority and tracking.

        Args:
            agent_name: Name of the agent to receive the task
            task: Task details and parameters
            priority: Task priority level ("low", "normal", "high", "critical")

        Returns:
            Dict containing delegation result and task tracking info
        """
        if not self.vanta_supervisor:
            logger.warning("VantaSupervisor not available for task delegation")
            return {"error": "VantaSupervisor not available"}

        # Validate agent exists
        agent = self.get_agent(agent_name)
        if not agent:
            logger.warning(f"Cannot delegate task to unknown agent: {agent_name}")
            return {"error": f"Agent '{agent_name}' not found"}

        # Create task delegation with tracking
        delegation_id = f"delegation_{agent_name}_{int(time.time())}"
        delegation_details = {
            "delegation_id": delegation_id,
            "target_agent": agent_name,
            "task": task,
            "priority": priority,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "delegated",
        }

        try:
            # Execute delegation through VantaSupervisor
            delegation_task = {
                "action": "execute_task",
                "agent_name": agent_name,
                "task_detail": {
                    **task,
                    "delegation_id": delegation_id,
                    "priority": priority,
                },
            }

            result = self.vanta_supervisor.perform_task(delegation_task)
            delegation_details["result"] = result
            delegation_details["status"] = (
                "completed" if "error" not in result else "failed"
            )

            # Emit delegation event
            self.event_bus.emit(
                "task_delegated",
                {
                    "delegation_id": delegation_id,
                    "agent": agent_name,
                    "priority": priority,
                    "status": delegation_details["status"],
                },
            )

            logger.info(f"Task delegated to {agent_name}: {delegation_id}")
            return delegation_details

        except Exception as e:
            logger.error(f"Error delegating task to {agent_name}: {e}")
            delegation_details["status"] = "error"
            delegation_details["error"] = str(e)
            return delegation_details

    def discover_agents_by_capabilities(
        self, capabilities: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Discover agents that have specific capabilities.

        Args:
            capabilities: List of required capabilities

        Returns:
            List of agent info dictionaries with matching capabilities
        """
        if not self.agent_registry:
            logger.warning("Agent registry not available for capability discovery")
            return []

        discovered_agents = []
        all_agents = self.get_all_agents()

        for agent_name, agent in all_agents:
            agent_capabilities = []  # Try to get capabilities from agent metadata in registry
            if (
                hasattr(self.agent_registry, "agents")
                and self.agent_registry is not None
                and self.agent_registry is not None
                and hasattr(self.agent_registry, "agents")
                and agent_name in self.agent_registry.agents
            ):
                metadata = self.agent_registry.agents[agent_name].get("metadata", {})
                agent_capabilities = metadata.get("capabilities", [])

            # Fallback: try to get capabilities directly from agent
            if not agent_capabilities and hasattr(agent, "get_capabilities"):
                try:
                    agent_capabilities = agent.get_capabilities()
                except Exception as e:
                    logger.debug(f"Could not get capabilities from {agent_name}: {e}")

            # Check if agent has any of the required capabilities
            matching_capabilities = [
                cap for cap in capabilities if cap in agent_capabilities
            ]
            if matching_capabilities:
                discovered_agents.append(
                    {
                        "name": agent_name,
                        "agent": agent,
                        "capabilities": agent_capabilities,
                        "matching_capabilities": matching_capabilities,
                        "agent_type": type(agent).__name__,
                    }
                )

        logger.info(
            f"Discovered {len(discovered_agents)} agents with capabilities: {capabilities}"
        )
        return discovered_agents

    def route_task_by_capability(
        self, task: Dict[str, Any], required_capability: str
    ) -> Dict[str, Any]:
        """
        Route a task to the best available agent based on a specific capability.

        Args:
            task: Task to be routed
            required_capability: The capability needed to handle the task

        Returns:
            Dict containing routing result and execution details
        """
        if not self.vanta_supervisor:
            logger.warning("VantaSupervisor not available for capability routing")
            return {"error": "VantaSupervisor not available"}

        # Find agents with the required capability
        capable_agents = self.discover_agents_by_capabilities([required_capability])

        if not capable_agents:
            logger.warning(f"No agents found with capability: {required_capability}")
            return {
                "error": f"No agents available for capability: {required_capability}"
            }

        # Select the first available agent (could be enhanced with load balancing)
        selected_agent = capable_agents[0]
        logger.info(
            f"Routing task to {selected_agent['name']} for capability: {required_capability}"
        )

        # Route the task through delegation
        routing_result = self.delegate_task_to_agent(
            selected_agent["name"],
            {
                **task,
                "routed_for_capability": required_capability,
                "routing_timestamp": datetime.datetime.now().isoformat(),
            },
            priority="normal",
        )

        # Add routing metadata
        routing_result["routing_info"] = {
            "required_capability": required_capability,
            "selected_agent": selected_agent["name"],
            "available_agents": len(capable_agents),
            "routing_strategy": "first_available",
        }

        return routing_result

    def broadcast_message_to_agents(
        self, message: str, agent_filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Broadcast a message to multiple agents with optional filtering.

        Args:
            message: Message to broadcast
            agent_filter: Optional filter criteria (e.g., {"capability": "planning"})
            **kwargs: Additional message parameters

        Returns:
            Dict containing broadcast results from all agents
        """
        all_agents = self.get_all_agents()
        target_agents = []

        # Apply agent filter if specified
        if agent_filter:
            if "capability" in agent_filter:
                capable_agents = self.discover_agents_by_capabilities(
                    [agent_filter["capability"]]
                )
                target_agents = [
                    (agent["name"], agent["agent"]) for agent in capable_agents
                ]
            elif "type" in agent_filter:
                target_agents = [
                    (name, agent)
                    for name, agent in all_agents
                    if type(agent).__name__ == agent_filter["type"]
                ]
            else:
                target_agents = all_agents
        else:
            target_agents = all_agents

        broadcast_results = {
            "broadcast_id": f"broadcast_{int(time.time())}",
            "message": message,
            "target_count": len(target_agents),
            "responses": {},
            "successful_deliveries": 0,
            "failed_deliveries": 0,
        }

        # Send message to each target agent
        for agent_name, agent in target_agents:
            try:
                response = self.send_agent_message(agent_name, message, **kwargs)
                broadcast_results["responses"][agent_name] = {
                    "status": "delivered",
                    "response": response,
                }
                broadcast_results["successful_deliveries"] += 1
                logger.debug(f"Message delivered to {agent_name}")
            except Exception as e:
                broadcast_results["responses"][agent_name] = {
                    "status": "failed",
                    "error": str(e),
                }
                broadcast_results["failed_deliveries"] += 1
                logger.warning(f"Failed to deliver message to {agent_name}: {e}")

        # Emit broadcast event
        self.event_bus.emit(
            "message_broadcasted",
            {
                "broadcast_id": broadcast_results["broadcast_id"],
                "target_count": broadcast_results["target_count"],
                "successful_deliveries": broadcast_results["successful_deliveries"],
                "failed_deliveries": broadcast_results["failed_deliveries"],
            },
        )

        logger.info(
            f"Broadcast completed: {broadcast_results['successful_deliveries']}/{broadcast_results['target_count']} successful"
        )
        return broadcast_results

    def get_agent_coordination_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of agent coordination capabilities.

        Returns:
            Dict containing coordination system status and metrics
        """
        coordination_status = {
            "vanta_supervisor_available": self.vanta_supervisor is not None,
            "agent_registry_available": self.agent_registry is not None,
            "total_registered_agents": len(self.get_all_agents()),
            "coordination_capabilities": [],
            "agent_capabilities_summary": {},
        }

        if self.vanta_supervisor:
            coordination_status["vanta_supervisor_status"] = (
                self.vanta_supervisor.get_status()
            )
            coordination_status["coordination_capabilities"] = (
                self.vanta_supervisor.get_capabilities()
            )

        # Summarize agent capabilities
        all_agents = self.get_all_agents()
        capability_counts = defaultdict(int)

        for agent_name, agent in all_agents:
            agent_capabilities = []

            # Get capabilities from metadata or agent directly
            if (
                self.agent_registry is not None
                and hasattr(self.agent_registry, "agents")
                and agent_name in self.agent_registry.agents
            ):
                metadata = self.agent_registry.agents[agent_name].get("metadata", {})
                agent_capabilities = metadata.get("capabilities", [])
            elif hasattr(agent, "get_capabilities"):
                try:
                    agent_capabilities = agent.get_capabilities()
                except Exception:
                    agent_capabilities = []

            coordination_status["agent_capabilities_summary"][agent_name] = {
                "type": type(agent).__name__,
                "capabilities": agent_capabilities,
            }

            # Count capabilities
            for capability in agent_capabilities:
                capability_counts[capability] += 1

        coordination_status["capability_distribution"] = dict(capability_counts)

        return coordination_status

    # --- UTILITY METHODS ---

    def shutdown(self) -> None:
        """Gracefully shutdown VantaCore."""
        logger.info("Shutting down UnifiedVantaCore")

        self.event_bus.emit(
            "vanta_core_shutdown",
            {
                "uptime_seconds": (
                    datetime.datetime.now() - self.start_time
                ).total_seconds()
            },
        )

        # Stop async bus if running
        if hasattr(self, "async_bus") and self.async_bus.running:
            try:
                asyncio.run(self.async_bus.stop())
            except Exception as e:
                logger.error(f"Failed to stop async bus during shutdown: {e}")

        # Additional cleanup would go here
        logger.info("UnifiedVantaCore shutdown complete")

    def _initialize_speech_integration(self) -> None:
        """Initialize speech (TTS/STT) integration for VantaCore."""
        try:
            from handlers.speech_integration_handler import (
                initialize_speech_system,
            )

            # Initialize speech system with default settings
            speech_handler = initialize_speech_system(
                vanta_core=self,
                enable_tts=True,
                enable_stt=True,
                register_with_async_bus=True,
            )

            # Register the handler with VantaCore
            self.registry.register(
                "speech_integration_handler",
                speech_handler,
                {
                    "description": "Speech Integration Handler",
                    "capabilities": ["text_to_speech", "speech_to_text"],
                    "status": speech_handler.get_status(),
                },
            )

            logger.info("Speech integration initialized successfully")

        except ImportError:
            logger.warning("Speech integration handler not available")
        except Exception as e:
            logger.error(f"Failed to initialize speech integration: {e}")

    def _initialize_vmb_integration(self) -> None:
        """Initialize VMB (VANTA Model Builder) integration for VantaCore."""
        try:
            from handlers.vmb_integration_handler import initialize_vmb_system

            # Initialize VMB system with default settings
            vmb_handler = initialize_vmb_system(
                vanta_core=self,
                config={
                    "sigil": "⟠∆∇𓂀",
                    "agent_class": "CopilotSwarm",
                    "swarm_variant": "RPG_Sentinel",
                    "role_scope": ["planner", "validator", "executor", "summarizer"],
                    "activation_mode": "VMB_Production",
                },
            )

            # Register the handler with VantaCore
            self.registry.register(
                "vmb_integration_handler",
                vmb_handler,
                {
                    "description": "VMB Integration Handler",
                    "capabilities": [
                        "model_building",
                        "code_analysis",
                        "component_validation",
                        "error_detection",
                        "performance_monitoring",
                        "task_execution",
                    ],
                    "status": vmb_handler.get_status(),
                },
            )

            logger.info("VMB integration initialized successfully")

        except ImportError:
            logger.warning("VMB integration handler not available")
        except Exception as e:
            logger.error(f"Failed to initialize VMB integration: {e}")
    def bind_cross_system_link(self) -> None:
        """Placeholder for Nebula cross-system link integration."""
        pass





def get_vanta_core(**kwargs) -> UnifiedVantaCore:
    """Get the singleton VantaCore instance."""
    return UnifiedVantaCore(**kwargs)


def trace_vanta_event(message: str, category: str = "info", **kwargs):
    """Trace an event in VantaCore."""
    trace_event(message, category, **kwargs)
    # Also emit on event bus if VantaCore is initialized
    try:
        core = get_vanta_core()
        core.emit_event(
            "trace_event", {"message": message, "category": category, **kwargs}
        )
    except Exception:
        pass  # Ignore if VantaCore not yet initialized


# --- BACKWARD COMPATIBILITY ---
# Alias for existing code that imports VantaCore
VantaCore = UnifiedVantaCore

if __name__ == "__main__":
    # Demo usage
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

    print("🚀 UnifiedVantaCore Demo")

    # Basic orchestration features
    core = get_vanta_core()
    print(f"Status: {core.get_system_status()}")

    # Register a test component
    core.register_component("test_component", {"value": 42}, {"purpose": "demo"})

    # Subscribe to events
    def event_handler(event):
        print(f"Event: {event['type']} - {event.get('data', {})}")

    core.subscribe_to_event("component_registered", event_handler)

    # Register another component to trigger event
    core.register_component("another_component", lambda x: x * 2)

    print("✅ Demo complete - UnifiedVantaCore operational")
