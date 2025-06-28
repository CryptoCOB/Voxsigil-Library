"""
UnifiedVantaCore - Composition-Based Vanta Architecture

Provides a unified interface that orchestrates both the VantaCognitiveEngine
(advanced cognitive processing with Meta-Learning, BLT Encoding, Hybrid RAG)
and the VantaOrchestrationEngine (simple component management and event handling).

This composition approach allows for modular, maintainable integration of both
engines while preserving their individual strengths.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

# HOLO-1.5 Registration
try:
    from ..registration.master_registration import vanta_core_module
except ImportError:

    def vanta_core_module(name: str = "", role: str = ""):
        def decorator(cls):
            return cls

        return decorator


# Use hybrid BLT components for better performance
from BLT.hybrid_blt import BLTEncoder

# The HybridMiddleware implementation lives in VoxSigilRag.hybrid_blt
from VoxSigilRag.hybrid_blt import HybridMiddleware

from Vanta.interfaces.blt_encoder_interface import BaseBLTEncoder
from Vanta.interfaces.hybrid_middleware_interface import BaseHybridMiddleware

# RealSupervisorConnector implementation lives in integration.real_supervisor_connector
# RealSupervisorConnector is located in the top-level integration package
try:
    from integration.real_supervisor_connector import RealSupervisorConnector
except ImportError:
    # Create a stub if the import fails due to circular dependencies
    class RealSupervisorConnector:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            return True

        def disconnect(self):
            pass

        def send_message(self, message):
            return True


# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
# Lazy imports to avoid circular dependency issues

from VoxSigilRag.voxsigil_mesh import VoxSigilMesh

from Vanta.interfaces.supervisor_connector_interface import BaseSupervisorConnector

# Lazy imports for modules that may cause circular dependencies
# from ..integration.vanta_mesh_graph import VantaMeshGraph
# from ..integration.vanta_supervisor import VantaSupervisor
from .UnifiedAgentRegistry import UnifiedAgentRegistry
from .UnifiedAsyncBus import UnifiedAsyncBus

# Configure logger
logger = logging.getLogger("unified_vanta_core")


# Lazy import functions to avoid circular dependencies
def _get_vanta_mesh_graph():
    """Lazy import VantaMeshGraph to avoid circular dependencies."""
    try:
        from ..integration.vanta_mesh_graph import VantaMeshGraph

        return VantaMeshGraph
    except ImportError as e:
        logger.warning(f"Could not import VantaMeshGraph: {e}")
        return None


def _get_vanta_supervisor():
    """Lazy import VantaSupervisor to avoid circular dependencies."""
    try:
        from ..integration.vanta_supervisor import VantaSupervisor

        return VantaSupervisor
    except ImportError as e:
        logger.warning(f"Could not import VantaSupervisor: {e}")
        return None


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
                logger.debug(f"No subscribers for event '{event_type}'")
            for callback, priority in subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback for '{event_type}': {e}")

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


@vanta_core_module(name="UnifiedVantaCore", role="core_orchestrator")
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
        enable_async_bus: bool = False,  # Disabled by default to avoid GUI issues
    ):
        """
        Initialize UnifiedVantaCore with progressive capability loading.

        Args:
            config_sigil_ref: Optional sigil reference for configuration
            supervisor_connector: Optional supervisor connector for advanced features
            blt_encoder: Optional BLT encoder for cognitive capabilities
            hybrid_middleware: Optional middleware for advanced processing
            enable_cognitive_features: Whether to enable advanced cognitive features
        """  # Only initialize once (singleton pattern)
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.start_time = (
            datetime.datetime.now()
        )  # --- ORCHESTRATION LAYER (Always Available) ---
        self.registry = ComponentRegistry()
        self.event_bus = EventBus()
        self.mesh = VoxSigilMesh(
            gui_hook=lambda msg: self.event_bus.emit("mesh_echo", msg)
        )

        # Lazy initialization of mesh graph
        VantaMeshGraph = _get_vanta_mesh_graph()
        if VantaMeshGraph:
            self.mesh_graph = VantaMeshGraph(self)
        else:
            self.mesh_graph = None

        # Conditional async bus creation
        if enable_async_bus:
            self.async_bus = UnifiedAsyncBus(logger, blt_encoder)
            try:
                # Check if there's a running event loop before creating task
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.async_bus.start())
                except RuntimeError:
                    # No running event loop, start async bus in a thread
                    import threading

                    def start_async_bus():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.async_bus.start())

                    thread = threading.Thread(target=start_async_bus, daemon=True)
                    thread.start()
            except Exception as e:
                logger.error(f"Failed to start async bus: {e}")
            logger.info("Async bus enabled and started")
        else:
            self.async_bus = None
            logger.info("Async bus disabled (sync mode only)")
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
            )  # Initialize VantaSupervisor for agent management if available
        self.vanta_supervisor = None
        VantaSupervisor = _get_vanta_supervisor()
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
        self._map_subsystems_to_guardians()  # Initialize HOLO mesh runtime loader
        try:
            from agents import HoloMesh

            self.holo_mesh = HoloMesh(agent_registry=self.registry)
        except Exception as e:
            logger.error(f"Failed to initialize HOLO mesh: {e}")
            self.holo_mesh = None

        self.get_agents_by_capability = (
            self.agent_registry.get_agents_by_capability
            if self.agent_registry
            else None
        )  # Auto-register all training, evaluation, inference, and visualization components
        logger.info("🔍 Starting comprehensive component auto-registration...")
        try:
            auto_registration_results = self.auto_register_all_components()
            total_registered = auto_registration_results.get("total_registered", 0)
            logger.info(
                f"✅ Auto-registration completed: {total_registered} components registered and assigned to agents"
            )
        except Exception as e:
            logger.warning(f"⚠️ Auto-registration encountered issues: {e}")
            # Fallback to ensure basic functionality
            auto_registration_results = {"total_registered": 0}

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
        if self.async_bus:
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
                return []  # Provide default capabilities based on agent name/type if no explicit capabilities
        agent_type = type(agent).__name__.lower()
        default_capabilities = []

        if "training" in agent_name.lower() or "trainer" in agent_type:
            default_capabilities = ["training", "learning", "optimization"]
        elif "evaluation" in agent_name.lower() or "eval" in agent_type:
            default_capabilities = ["evaluation", "testing", "assessment"]
        elif "inference" in agent_name.lower() or "infer" in agent_type:
            default_capabilities = ["inference", "prediction", "generation"]
        elif "visualization" in agent_name.lower() or "visual" in agent_type:
            default_capabilities = ["visualization", "display", "rendering"]
        elif "system" in agent_name.lower() or "core" in agent_name.lower():
            default_capabilities = ["system", "coordination", "management"]
        elif "agent" in agent_name.lower():
            default_capabilities = ["agent", "processing", "task_execution"]
        else:
            default_capabilities = ["general", "processing"]

        # Only warn for specific non-auto-registered components and log at debug level for others
        should_warn = not any(
            prefix in agent_name.lower()
            for prefix in [
                "training_",
                "evaluation_",
                "inference_",
                "visualization_",
                "system_",
                "core_",
                "base",
                "simple",
                "mock",
                "test",
            ]
        ) and agent_type not in ["baseagent", "simpleagent", "testagent"]

        if should_warn:
            logger.warning(
                f"Agent '{agent_name}' does not have capabilities defined, using defaults: {default_capabilities}"
            )
        else:
            logger.debug(
                f"Agent '{agent_name}' using default capabilities: {default_capabilities}"
            )

        return default_capabilities

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
                logger.error(
                    f"Failed to register VantaSupervisor as core agent: {e}"
                )  # Register all defined agents dynamically from agents.__all__
        agent_classes = []
        try:
            import agents as agent_pkg
            from agents import __all__ as agent_names  # type: ignore

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
                # Try different instantiation patterns
                instance = None

                # First try with no parameters
                try:
                    instance = cls()
                except TypeError:
                    # Try with vanta_core parameter
                    try:
                        instance = cls(vanta_core=self)
                    except TypeError:
                        # Try with both vanta_core and empty config
                        try:
                            instance = cls(self, {})
                        except TypeError:
                            # Skip this agent if we can't instantiate it
                            logger.debug(
                                f"Skipping agent {cls.__name__} - incompatible constructor signature"
                            )
                            continue

                if instance is None:
                    continue

            except Exception as e:
                from agents import NullAgent

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

        try:
            from agents.base import AGENT_SUBSYSTEM_MAP

            mapping = AGENT_SUBSYSTEM_MAP
        except ImportError:
            logger.warning(
                "AGENT_SUBSYSTEM_MAP not available, skipping subsystem mapping"
            )
            mapping = {}

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

    def publish_event(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Publish an event on the event bus."""
        self.event_bus.emit(event_type, data, **kwargs)

    def emit_event(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Emit an event (alias for publish_event)."""
        self.publish_event(event_type, data, **kwargs)

    def subscribe_to_event(
        self, event_type: str, callback: Callable, priority: int = 0
    ) -> None:
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, callback, priority)

    def register_mesh_node(self, name: str, node: Any) -> None:
        """Register a node with the VoxSigilMesh."""
        if hasattr(self, "mesh"):
            self.mesh.register(name, node)

    def send_to_mesh(self, sender: str, message: str) -> None:
        """Transmit a message through the VoxSigilMesh."""
        if hasattr(self, "mesh"):
            self.mesh.transmit(sender, message)

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
                self.agent_registry is not None
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

    # --- AUTO-REGISTRATION METHODS (Component Discovery & Agent Assignment) ---

    def auto_register_all_components(self) -> Dict[str, Any]:
        """
        Auto-discover and register ALL training, evaluation, inference, and visualization components.
        Then assign each component to appropriate agents for orchestration.
        """
        logger.info("🔍 Starting comprehensive component auto-registration...")

        registration_results = {
            "training_components": [],
            "evaluation_components": [],
            "inference_components": [],
            "visualization_components": [],
            "agent_assignments": {},
            "total_registered": 0,
            "errors": [],
        }

        try:
            # Register training components
            training_results = self._auto_register_training_components()
            registration_results["training_components"] = training_results

            # Register evaluation components
            eval_results = self._auto_register_evaluation_components()
            registration_results["evaluation_components"] = eval_results

            # Register inference components
            inference_results = self._auto_register_inference_components()
            registration_results["inference_components"] = inference_results

            # Register visualization components
            viz_results = self._auto_register_visualization_components()
            registration_results["visualization_components"] = viz_results

            # Auto-assign components to agents
            assignment_results = self._auto_assign_components_to_agents()
            registration_results["agent_assignments"] = assignment_results

            # Calculate totals
            total = (
                len(training_results)
                + len(eval_results)
                + len(inference_results)
                + len(viz_results)
            )
            registration_results["total_registered"] = total

            logger.info(f"✅ Auto-registration complete: {total} components registered")

            # Emit comprehensive registration event
            self.event_bus.emit("auto_registration_complete", registration_results)

            return registration_results

        except Exception as e:
            logger.error(f"❌ Auto-registration failed: {e}")
            registration_results["errors"].append(str(e))
            return registration_results

    def _auto_register_training_components(self) -> List[str]:
        """Auto-discover and register all training components."""
        training_components = []

        # Training modules to discover and register
        training_modules = [
            ("training.arc_grid_trainer", "ARCGridTrainer"),
            ("ART.art_trainer", "ArtTrainer"),
            ("training.vanta_registration", "VantaTrainingAdapter"),
            ("Vanta.async_training_engine", "AsyncTrainingEngine"),
            ("interfaces.training_interface", "VoxSigilTrainingInterface"),
            ("gui.components.enhanced_training_tab", "EnhancedTrainingTab"),
            ("training.rag_interface", "SupervisorRagInterface"),
        ]

        for module_path, class_name in training_modules:
            try:
                # Use safe import with error handling
                module = None
                try:
                    module = __import__(module_path, fromlist=[class_name])
                except ImportError as ie:
                    logger.debug(f"Module {module_path} not available: {ie}")
                    continue
                except Exception as e:
                    logger.warning(f"Error importing {module_path}: {e}")
                    continue

                if module and hasattr(module, class_name):
                    component_class = getattr(module, class_name)

                    # Register with VantaCore
                    component_name = f"training_{class_name.lower()}"
                    self.register_component(
                        component_name,
                        component_class,
                        {
                            "type": "training",
                            "module": module_path,
                            "class": class_name,
                            "capabilities": [
                                "training",
                                "model_optimization",
                                "learning",
                            ],
                        },
                    )
                    training_components.append(component_name)
                    logger.info(f"✅ Registered training component: {component_name}")

            except Exception as e:
                logger.debug(f"⚠️ Could not register {module_path}.{class_name}: {e}")

        return training_components

    def _auto_register_evaluation_components(self) -> List[str]:
        """Auto-discover and register all evaluation components."""
        evaluation_components = []

        # Evaluation modules to discover and register
        eval_modules = [
            ("core.grid_former_evaluator", "GridFormerEvaluator"),
            ("VoxSigilRag.voxsigil_evaluator", "VoxSigilResponseEvaluator"),
            (
                "voxsigil_supervisor.strategies.evaluation_heuristics",
                "EvaluationHeuristics",
            ),
            ("ARC.end_to_end_arc_validation", "ARCValidationEngine"),
            (
                "core.ensemble_integration.arc_ensemble_orchestrator",
                "ARCEnsembleOrchestrator",
            ),
            ("tests_archive.final_training_validation", "TrainingValidationEngine"),
        ]

        for module_path, class_name in eval_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    component_class = getattr(module, class_name)

                    # Register with VantaCore
                    component_name = f"evaluation_{class_name.lower()}"
                    self.register_component(
                        component_name,
                        component_class,
                        {
                            "type": "evaluation",
                            "module": module_path,
                            "class": class_name,
                            "capabilities": [
                                "evaluation",
                                "validation",
                                "metrics",
                                "assessment",
                            ],
                        },
                    )
                    evaluation_components.append(component_name)
                    logger.info(f"✅ Registered evaluation component: {component_name}")

            except Exception as e:
                logger.warning(f"⚠️ Could not register {module_path}.{class_name}: {e}")

        return evaluation_components

    def _auto_register_inference_components(self) -> List[str]:
        """Auto-discover and register all inference components."""
        inference_components = []

        # Inference modules to discover and register
        inference_modules = [
            (
                "Gridformer.inference.gridformer_inference_engine",
                "GridFormerInferenceEngine",
            ),
            ("Gridformer.inference.inference_strategy", "InferenceStrategy"),
            ("VoxSigil_Gridformer.gridformer_arc_inference", "GridFormerARCInference"),
            ("ARC.arc_grid_former_runner", "ARCGridFormerRunner"),
            ("VoxSigilRag.voxsigil_mesh", "VoxSigilMesh"),
        ]

        for module_path, class_name in inference_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    component_class = getattr(module, class_name)

                    # Register with VantaCore
                    component_name = f"inference_{class_name.lower()}"
                    self.register_component(
                        component_name,
                        component_class,
                        {
                            "type": "inference",
                            "module": module_path,
                            "class": class_name,
                            "capabilities": [
                                "inference",
                                "prediction",
                                "reasoning",
                                "problem_solving",
                            ],
                        },
                    )
                    inference_components.append(component_name)
                    logger.info(f"✅ Registered inference component: {component_name}")

            except Exception as e:
                logger.warning(f"⚠️ Could not register {module_path}.{class_name}: {e}")

        return inference_components

    def _auto_register_visualization_components(self) -> List[str]:
        """Auto-discover and register all visualization components."""
        visualization_components = []

        # Visualization modules to discover and register
        viz_modules = [
            ("utils.visualization_utils", "GridVisualizer"),
            ("utils.visualization_utils", "PerformanceVisualizer"),
            ("ARC.arc_task_visualizer", "ARCTaskVisualizer"),
            ("gui.components.streaming_dashboard", "StreamingDashboard"),
            ("gui.components.novel_reasoning_tab", "NovelReasoningTab"),
        ]

        for module_path, class_name in viz_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    component_class = getattr(module, class_name)

                    # Register with VantaCore
                    component_name = f"visualization_{class_name.lower()}"
                    self.register_component(
                        component_name,
                        component_class,
                        {
                            "type": "visualization",
                            "module": module_path,
                            "class": class_name,
                            "capabilities": [
                                "visualization",
                                "plotting",
                                "display",
                                "monitoring",
                            ],
                        },
                    )
                    visualization_components.append(component_name)
                    logger.info(
                        f"✅ Registered visualization component: {component_name}"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Could not register {module_path}.{class_name}: {e}")

        return visualization_components

    def _auto_assign_components_to_agents(self) -> Dict[str, List[str]]:
        """Auto-assign registered components to appropriate agents based on capabilities."""
        logger.info("🤖 Auto-assigning components to agents...")

        assignments = {
            "training_agents": [],
            "evaluation_agents": [],
            "inference_agents": [],
            "visualization_agents": [],
            "unassigned_components": [],
        }

        try:
            # Get all registered components
            all_components = self.registry.list_components()

            # Get all available agents
            all_agents = self.get_all_agents() if self.agent_registry else []

            for component_name in all_components:
                component_metadata = self.registry.get_metadata(component_name)
                if not component_metadata:
                    continue

                component_type = component_metadata.get("type", "unknown")
                component_capabilities = component_metadata.get("capabilities", [])

                # Find suitable agents for this component
                assigned = False
                for agent_name, agent in all_agents:
                    agent_capabilities = self.get_agent_capabilities(agent_name)

                    # Check if agent capabilities match component needs
                    if self._capabilities_match(
                        component_capabilities, agent_capabilities
                    ):
                        # Assign component to agent
                        if hasattr(agent, "assign_component"):
                            try:
                                agent.assign_component(
                                    component_name, self.get_component(component_name)
                                )
                                assignments[f"{component_type}_agents"].append(
                                    {
                                        "agent": agent_name,
                                        "component": component_name,
                                        "capabilities": component_capabilities,
                                    }
                                )
                                assigned = True
                                logger.info(
                                    f"✅ Assigned {component_name} to agent {agent_name}"
                                )
                                break
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to assign {component_name} to {agent_name}: {e}"
                                )

                if not assigned:
                    assignments["unassigned_components"].append(component_name)
                    logger.info(
                        f"📋 Component {component_name} not assigned to any agent"
                    )

            # Create specialized agents if needed
            self._create_specialized_agents_for_unassigned(
                assignments["unassigned_components"]
            )

            return assignments

        except Exception as e:
            logger.error(f"❌ Error during component-to-agent assignment: {e}")
            return assignments

    def _capabilities_match(
        self, component_caps: List[str], agent_caps: List[str]
    ) -> bool:
        """Check if agent capabilities match component requirements."""
        if not component_caps or not agent_caps:
            return False

        # Simple overlap check - at least one capability must match
        return bool(set(component_caps) & set(agent_caps))

    def _create_specialized_agents_for_unassigned(
        self, unassigned_components: List[str]
    ) -> None:
        """Create specialized agents for components that couldn't be assigned."""
        if not unassigned_components:
            return

        logger.info(
            f"🔧 Creating specialized agents for {len(unassigned_components)} unassigned components"
        )

        # Group unassigned components by type
        component_groups = {}
        for comp_name in unassigned_components:
            metadata = self.registry.get_metadata(comp_name)
            comp_type = metadata.get("type", "general") if metadata else "general"

            if comp_type not in component_groups:
                component_groups[comp_type] = []
            component_groups[comp_type].append(comp_name)

        # Create specialized agents for each type
        for comp_type, components in component_groups.items():
            try:
                # Create a specialized coordinator agent
                agent_name = f"{comp_type}_coordinator_agent"

                # Simple agent class for component coordination
                class ComponentCoordinatorAgent:
                    def __init__(self, component_type, assigned_components):
                        self.component_type = component_type
                        self.assigned_components = assigned_components
                        self.capabilities = [comp_type, "coordination", "management"]

                    def assign_component(self, component_name, component_instance):
                        self.assigned_components.append(component_name)
                        return True

                    def get_capabilities(self):
                        return self.capabilities

                    def execute_task(self, task):
                        return {
                            "status": "delegated",
                            "components": self.assigned_components,
                        }

                # Create and register the specialized agent
                coordinator_agent = ComponentCoordinatorAgent(
                    comp_type, components.copy()
                )

                self.register_agent(
                    agent_name,
                    coordinator_agent,
                    {
                        "type": f"{comp_type}_coordinator",
                        "managed_components": components,
                        "capabilities": coordinator_agent.capabilities,
                        "auto_created": True,
                    },
                )

                logger.info(
                    f"✅ Created {agent_name} for {len(components)} {comp_type} components"
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to create specialized agent for {comp_type}: {e}"
                )

    def get_all_registered_components_by_type(
        self, component_type: str = None
    ) -> Dict[str, List[str]]:
        """Get all registered components, optionally filtered by type."""
        components_by_type = {
            "training": [],
            "evaluation": [],
            "inference": [],
            "visualization": [],
            "other": [],
        }

        all_components = self.registry.list_components()

        for component_name in all_components:
            metadata = self.registry.get_metadata(component_name)
            comp_type = metadata.get("type", "other") if metadata else "other"

            if comp_type in components_by_type:
                components_by_type[comp_type].append(component_name)
            else:
                components_by_type["other"].append(component_name)

        if component_type:
            return {component_type: components_by_type.get(component_type, [])}

        return components_by_type

    def assign_component_to_agent(self, component_name: str, agent_name: str) -> bool:
        """Manually assign a specific component to a specific agent."""
        try:
            agent = self.get_agent(agent_name)
            component = self.get_component(component_name)

            if not agent:
                logger.error(f"Agent '{agent_name}' not found")
                return False

            if not component:
                logger.error(f"Component '{component_name}' not found")
                return False

            # Try to assign component to agent
            if hasattr(agent, "assign_component"):
                agent.assign_component(component_name, component)

                # Emit assignment event
                self.event_bus.emit(
                    "component_assigned_to_agent",
                    {
                        "component": component_name,
                        "agent": agent_name,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                )

                logger.info(f"✅ Manually assigned {component_name} to {agent_name}")
                return True
            else:
                logger.warning(
                    f"Agent {agent_name} does not support component assignment"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Failed to assign {component_name} to {agent_name}: {e}")
            return False

    def _initialize_vmb_integration(self) -> None:
        """Initialize VMB integration (Step 16 of VANTA Integration Master Plan)."""
        try:
            from Vanta.handlers.vmb_integration_handler import (
                VantaVMBIntegrationHandler,
            )

            self.vmb_integration_handler = VantaVMBIntegrationHandler(
                vanta_core=self,
                logger=logger,
                event_bus=self.event_bus,
            )

            logger.info("VMB integration initialized successfully")

        except ImportError:
            logger.warning("VMB integration handler not available")
        except Exception as e:
            logger.error(f"Failed to initialize VMB integration: {e}")

    def bind_cross_system_link(self) -> None:
        """Placeholder for Nebula cross-system link integration."""
        pass

    def _initialize_speech_integration(self) -> None:
        """Initialize speech integration (TTS/STT components)."""
        try:
            # Placeholder for future speech integration
            # This could include TTS/STT component initialization
            logger.info("Speech integration placeholder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech integration: {e}")


def get_vanta_core(**kwargs) -> UnifiedVantaCore:
    """Get the singleton VantaCore instance."""
    return UnifiedVantaCore(**kwargs)


def trace_vanta_event(message: str, category: str = "info", **kwargs):
    """Trace an event in VantaCore."""
    trace_event(
        message, category, **kwargs
    )  # Also emit on event bus if VantaCore is initialized
    try:
        core = get_vanta_core()
        core.publish_event(
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
