# File: vanta_learning_manager.py
"""
VantaCore Learning and Reflection Management.

This module handles:
1. Learning mode activation and control.
2. Reflection cycles and insights generation.
3. Cross-component knowledge integration using VantaCore services.
4. Training awareness and coordination with other VantaCore components.
"""

import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Configure logger
logger = logging.getLogger("VantaCore.LearningManager")


# Protocol for Model interface
@runtime_checkable
class ModelInterface(Protocol):
    def get_available_models(self) -> List[str]: ...
    def get_model_status(self, model_name: str) -> Dict[str, Any]: ...
    def request_model_tuning(
        self, model_name: str, tuning_params: Optional[Dict[str, Any]] = None
    ) -> bool: ...
    def update_model_weights(
        self, model_name: str, weights: Dict[str, Any]
    ) -> bool: ...


# Model manager interface
@runtime_checkable
class ModelManagerProtocol(Protocol):
    """Model manager protocol for learning manager."""

    def training_job_active(self) -> bool: ...
    def get_available_models(self) -> List[str]: ...
    def get_model_status(self, model_name: str) -> Dict[str, Any]: ...
    def request_model_tuning(
        self, model_name: str, tuning_params: Optional[Dict[str, Any]] = None
    ) -> bool: ...
    def update_model_weights(
        self, model_name: str, weights: Dict[str, Any]
    ) -> bool: ...


# Default implementation of the ModelManager protocol
class DefaultModelManager(ModelManagerProtocol):
    def training_job_active(self) -> bool:
        return False

    def get_available_models(self) -> List[str]:
        return []

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        return {"status": "unknown", "name": model_name}

    def request_model_tuning(
        self, model_name: str, tuning_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        return False

    def update_model_weights(self, model_name: str, weights: Dict[str, Any]) -> bool:
        return False


# Helper functions
def safe_component_call(
    component_name_or_func: Any, method_or_args: Any = None, *args: Any, **kwargs: Any
) -> Any:
    logger.debug(
        f"[SafeCall] Call: {component_name_or_func} {method_or_args} {args} {kwargs}"
    )
    return None


def trace_event(
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    source_override: Optional[str] = None,
) -> None:
    logger.debug(
        f"[TraceEvent] Type:{event_type}, Meta:{metadata}, Src:{source_override}"
    )


# Registry-based VantaCore access - no more stubs needed
def get_vanta_core_instance():
    """Get the UnifiedVantaCore singleton instance."""
    try:
        from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

        return UnifiedVantaCore()
    except ImportError:
        try:
            from ..core.UnifiedVantaCore import UnifiedVantaCore

            return UnifiedVantaCore()
        except ImportError:
            logger.error(
                "UnifiedVantaCore not available - cannot initialize LearningManager"
            )
            raise ImportError("UnifiedVantaCore is required for LearningManager")


# Registry-based ModelManager access
def get_model_manager_from_registry(vanta_core):
    """Get ModelManager from UnifiedVantaCore registry."""
    model_manager = vanta_core.get_component("model_manager")
    if model_manager:
        return model_manager

    # Try alternative registry names
    model_manager = vanta_core.get_component("ModelManager")
    if model_manager:
        return model_manager

    # Return default implementation if not found in registry
    logger.warning("ModelManager not found in registry, using DefaultModelManager")
    return DefaultModelManager()


# Registry-based event publishing
def publish_learning_event(
    vanta_core, event_type: str, metadata: Optional[Dict[str, Any]] = None
):
    """Publish learning events through VantaCore registry."""
    if vanta_core and hasattr(vanta_core, "publish_event"):
        vanta_core.publish_event(f"learning_{event_type}", metadata or {})
    else:
        logger.debug(f"[LearningEvent] {event_type}: {metadata}")


# Registry-based component access helper
def get_component_from_registry(
    vanta_core, component_name: str, expected_interface=None
):
    """Get any component from UnifiedVantaCore registry with optional type checking."""
    if not vanta_core:
        return None

    component = vanta_core.get_component(component_name)
    if component and expected_interface:
        if isinstance(component, expected_interface):
            return component
        else:
            logger.warning(
                f"Component '{component_name}' found but doesn't implement expected interface"
            )
            return None
    return component


# --- Configuration Class ---
class LearningManagerConfig:
    def __init__(
        self,
        log_level: str = "INFO",
        enable_learning_mode: bool = True,
        learning_activation_checkins: int = 2,  # Number of "missed check-ins" to trigger learning mode
        learning_interval_s: int = 300,
        memory_compression_interval_s: int = 600,
        conversation_prune_interval_s: int = 3600,  # From CheckInManager, might be relevant
        # Names of components to fetch from VantaCore registry
        memory_component_name: str = "memory_service",
        performance_monitor_name: str = "performance_monitor_service",
        goal_system_name: str = "goal_system_service",
        pattern_analyzer_name: str = "pattern_analyzer_service",
        meta_core_name: str = "meta_core_service",  # If 'meta_core' is a Vanta component
        compression_engine_name: str = "compression_engine_service",
        meta_learner_name: str = "advanced_meta_learner",  # Added for cross-domain transfer
    ):
        self.log_level = log_level
        self.enable_learning_mode = enable_learning_mode
        self.learning_activation_checkins = learning_activation_checkins
        self.learning_interval_s = learning_interval_s
        self.memory_compression_interval_s = memory_compression_interval_s
        self.conversation_prune_interval_s = (
            conversation_prune_interval_s  # Kept for reference
        )

        self.memory_component_name = memory_component_name
        self.performance_monitor_name = performance_monitor_name
        self.goal_system_name = goal_system_name
        self.pattern_analyzer_name = pattern_analyzer_name
        self.meta_core_name = meta_core_name
        self.compression_engine_name = compression_engine_name
        self.meta_learner_name = meta_learner_name  # Store the meta learner name

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Interfaces for Dependencies (if not standard VantaCore components) ---
# These define what LearningManager expects from components it gets from VantaCore.
@runtime_checkable
class LearningMemoryInterface(Protocol):  # Simplified from MemoryClusterInterface
    def search_relevant(
        self, topic: str, limit: int
    ) -> list[dict[str, Any]] | None: ...
    def store_event(self, event_data: dict[str, Any], memory_type: str) -> None: ...
    def get_compression_candidates(self) -> list[dict[str, Any]] | None: ...
    def store_compressed(self, compressed_item: dict[str, Any]) -> None: ...
    def prepare_learning_session(self) -> None: ...


@runtime_checkable
class MetaLearnerInterface(Protocol):
    def get_heuristics(self) -> list[dict[str, Any]]: ...
    def update_heuristic(self, heuristic_id: str, updates: dict[str, Any]) -> None: ...
    def add_heuristic(self, heuristic_data: dict[str, Any]) -> Any: ...
    def get_transferable_knowledge(
        self,
    ) -> Optional[Any]: ...  # Cross-domain knowledge transfer
    def integrate_knowledge(
        self, knowledge: Any, source: Any
    ) -> None: ...  # Knowledge integration
    def get_performance_metrics(
        self, domain: Optional[str] = None
    ) -> dict[str, Any]: ...  # Performance tracking


@runtime_checkable
class LearningPatternAnalyzerInterface(Protocol):
    def get_patterns(self, topic: str) -> list[dict[str, Any]] | None: ...
    def perform_learning(
        self, topic: str, context: dict[str, Any]
    ) -> None: ...  # From original usage
    def reset_learning_state(self) -> None: ...
    def get_transferable_knowledge(self) -> Optional[Any]: ...  # For cross-component
    def integrate_knowledge(
        self, knowledge: Any, source: Any
    ) -> None: ...  # For cross-component


@runtime_checkable
class LearningGoalSystemInterface(Protocol):
    def get_topic_relevance(self, topic: str) -> float | None: ...  # Score 0-1
    def perform_learning(self, topic: str, context: dict[str, Any]) -> None: ...
    def get_state(self) -> dict[str, Any] | None: ...
    def get_transferable_knowledge(self) -> Any | None: ...
    def integrate_knowledge(self, knowledge: Any, source: Any) -> None: ...


@runtime_checkable
class LearningPerformanceMonitorInterface(Protocol):
    def get_topic_performance(
        self, topic: str
    ) -> float | None: ...  # Score 0-1 (1 is good)


@runtime_checkable
class LearningMetaCoreInterface(Protocol):  # If 'meta_core' exists
    def adjust_state(
        self, learning_mode: bool, awareness_delta: float | None = None
    ) -> None: ...
    def perform_learning(self, topic: str, context: dict[str, Any]) -> None: ...
    def get_state(self) -> dict[str, Any] | None: ...
    def get_transferable_knowledge(self) -> Any | None: ...
    def integrate_knowledge(self, knowledge: Any, source: Any) -> None: ...


@runtime_checkable
class LearningCompressionEngineInterface(Protocol):
    def compress(
        self, method_name: str, item: dict[str, Any]
    ) -> dict[str, Any] | None: ...  # Changed signature


@vanta_core_module(
    name="learning_manager",
    subsystem="learning_management",
    mesh_role=CognitiveMeshRole.MANAGER,
    description="VantaCore learning and reflection management with cross-component knowledge integration",
    capabilities=["learning_coordination", "reflection_cycles", "knowledge_integration", "training_awareness", "memory_compression"],
    cognitive_load=3.0,
    symbolic_depth=3,
    collaboration_patterns=["knowledge_transfer", "meta_learning", "reflection_based_learning"]
)
class LearningManager(BaseCore):
    COMPONENT_NAME = "learning_manager"

    def __init__(
        self,
        vanta_core: Any,  # Relaxed type, could be VantaCore or stub
        config: LearningManagerConfig,
        model_manager: ModelManagerProtocol,
    ):  # ModelManager is a firm dependency
        # Initialize BaseCore
        super().__init__(vanta_core, config.__dict__ if hasattr(config, '__dict__') else {})
        
        self.config = config

        # Duck-typing check: ensure model_manager has the required methods
        required_methods = [
            "training_job_active",
            "get_available_models",
            "get_model_status",
            "request_model_tuning",
            "update_model_weights",
        ]
        missing_methods = [
            method for method in required_methods if not hasattr(model_manager, method)
        ]
        if missing_methods:
            logger.error(
                "Model manager is missing required methods: "
                + ", ".join(missing_methods)
            )
            raise TypeError("Invalid model_manager provided to LearningManager.")
        self.model_manager = model_manager

        # Learning state
        self.learning_active: bool = False
        self.learning_metrics: Dict[str, Any] = {}  # Initialize learning_metrics
        self.missed_checkin_count: int = 0  # This concept might change with VantaCore
        self.last_learning_timestamp: float = 0.0
        self.last_compression_timestamp: float = 0.0
        self.learning_topics: list[str] = [
            "metacognition_vanta",
            "pattern_recognition_vanta",
            "symbolic_reasoning_vanta",
            "self_reflection_vanta",
            "abstraction_vanta",
            "causal_inference_vanta",
            "knowledge_integration_vanta",
            "learning_transfer_vanta",
        ]  # Made topic names more Vanta-generic

        # Threading
        self.compression_thread: threading.Thread | None = None
        self._stop_compression_flag = threading.Event()
        self.learning_thread: threading.Thread | None = None
        self._stop_learning_flag = threading.Event()
        self._lock = threading.RLock()  # For learning_active and missed_checkin_count        # Register with VantaCore registry
        self.vanta_core.register_component(
            self.COMPONENT_NAME, self, {"type": "learning_service"}
        )

        # Subscribe to relevant events through VantaCore registry
        self._setup_event_subscriptions()        
        logger.info(
            f"{self.COMPONENT_NAME} initialized. Learning Mode initially: {'Enabled' if self.config.enable_learning_mode else 'Disabled'}"
        )

    async def initialize(self) -> bool:
        """Initialize the LearningManager for BaseCore compliance."""
        try:
            # Ensure all internal components are ready
            if self.config.enable_learning_mode:
                logger.info("LearningManager initialized successfully with HOLO-1.5 enhancement")
            return True
        except Exception as e:
            logger.error(f"Error initializing LearningManager: {e}")
            return False

    def _setup_event_subscriptions(self):
        """Set up event subscriptions through VantaCore registry."""
        try:
            # Subscribe to system events that should trigger learning
            if hasattr(self.vanta_core, "subscribe_to_event"):
                self.vanta_core.subscribe_to_event(
                    "model_training_complete", self._on_model_training_complete
                )
                self.vanta_core.subscribe_to_event(
                    "memory_pressure", self._on_memory_pressure
                )
                self.vanta_core.subscribe_to_event(
                    "performance_degradation", self._on_performance_issue
                )
                logger.info("LearningManager subscribed to VantaCore events")
            else:
                logger.warning("VantaCore does not support event subscription")
        except Exception as e:
            logger.error(f"Error setting up event subscriptions: {e}")

    def _on_model_training_complete(self, event_data):
        """Handle model training completion events."""
        logger.info("Model training completed, considering learning cycle")
        publish_learning_event(self.vanta_core, "training_completed", event_data)

    def _on_memory_pressure(self, event_data):
        """Handle memory pressure events."""
        logger.info("Memory pressure detected, triggering compression")
        self._perform_memory_compression()

    def _on_performance_issue(self, event_data):
        """Handle performance degradation events."""
        logger.info("Performance issue detected, activating learning mode")
        self.set_learning_mode_active(True)  # Use existing method

    # Helper to get components from VantaCore, with type checking
    def _get_vanta_component(
        self, component_name_in_config: str, expected_protocol: type
    ) -> Any | None:
        actual_component_name = getattr(self.config, component_name_in_config, None)
        if not actual_component_name:
            logger.warning(
                f"Configuration key '{component_name_in_config}' not set in LearningManagerConfig."
            )
            return None

        component = self.vanta_core.get_component(actual_component_name)
        if component and isinstance(component, expected_protocol):
            return component
        elif component:
            logger.warning(
                f"Component '{actual_component_name}' from VantaCore registry is not of expected type '{expected_protocol.__name__}'."
            )
        else:
            logger.debug(
                f"Optional component '{actual_component_name}' not found in VantaCore registry for LearningManager."
            )
        return None

    def start_threads(self) -> None:
        """Starts both learning and compression background loops."""
        self.start_memory_compression_loop()
        self.start_learning_mode_loop()  # Renamed from start_learning_mode

    def stop_threads(self) -> None:
        """Stops both learning and compression background loops gracefully."""
        logger.info(f"{self.COMPONENT_NAME} stopping background threads...")
        if self.learning_thread and self.learning_thread.is_alive():
            self._stop_learning_flag.set()
            self.learning_thread.join(timeout=5.0)
            if self.learning_thread.is_alive():
                logger.warning("Learning loop thread did not stop cleanly.")
        self.learning_thread = None

        if self.compression_thread and self.compression_thread.is_alive():
            self._stop_compression_flag.set()
            if self.compression_thread.is_alive():
                logger.warning("Compression loop thread did not stop cleanly.")
            if self.compression_thread.is_alive():
                logger.warning("Compression loop thread did not stop cleanly.")
        self.compression_thread = None
        logger.info(f"{self.COMPONENT_NAME} background threads stopped.")

    def start_memory_compression_loop(self) -> None:
        if self.compression_thread and self.compression_thread.is_alive():
            logger.info("Memory compression loop already running.")
            return

        self._stop_compression_flag.clear()
        self.compression_thread = threading.Thread(
            target=self._compression_loop_run, daemon=True, name="LM_CompressionLoop"
        )
        self.compression_thread.start()
        logger.info("Memory compression loop started by LearningManager.")

    def _compression_loop_run(self):  # Renamed for clarity
        logger.info(f"{self.COMPONENT_NAME} compression loop started.")
        while not self._stop_compression_flag.is_set():
            try:
                if (
                    time.monotonic() - self.last_compression_timestamp
                    >= self.config.memory_compression_interval_s
                ):
                    self._perform_memory_compression()
                    self.last_compression_timestamp = time.monotonic()

                    self.vanta_core.publish_event(
                        f"{self.COMPONENT_NAME}.memory_compression.complete",
                        {"timestamp": time.time()},
                        source=self.COMPONENT_NAME,
                    )

                # Check flag more frequently for faster shutdown
                if self._stop_compression_flag.wait(timeout=10.0):
                    logger.info("Memory compression loop stopping due to flag...")
                    break

            except Exception as e:
                logger.error(f"Error in compression loop: {e}", exc_info=True)
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.memory_compression.error",
                    {"error": str(e), "timestamp": time.time()},
                    source=self.COMPONENT_NAME,
                )

                # Sleep longer on error to prevent rapid error cycles
                if self._stop_compression_flag.wait(timeout=60.0):
                    logger.info("Memory compression loop stopping after error...")
                    break

        logger.info(f"{self.COMPONENT_NAME} compression loop finished.")

    def start_learning_mode_loop(self) -> None:  # Renamed from start_learning_mode
        if not self.config.enable_learning_mode:
            logger.info("Learning mode disabled by config.")
            return
        if self.learning_thread and self.learning_thread.is_alive():
            logger.info("Learning mode loop already running.")
            return

        self._stop_learning_flag.clear()
        self.learning_thread = threading.Thread(
            target=self._learning_loop_run, daemon=True, name="LM_LearningLoop"
        )
        self.learning_thread.start()
        logger.info("Learning mode loop started by LearningManager.")

    def _learning_loop_run(self):  # Renamed for clarity
        logger.info(f"{self.COMPONENT_NAME} learning loop started.")
        while not self._stop_learning_flag.is_set():
            try:
                # Using VantaCore events or direct calls for "check-in" concept
                # This `missed_checkin_count` needs a VantaCore equivalent trigger if it's to be used.
                # For now, learning mode activation will be time-based or explicit.
                # We can simplify by having learning mode always "potentially active" and the cycle
                # runs at `learning_interval_s` if not explicitly disabled.
                # The explicit `set_learning_mode` becomes the main control.

                if self.learning_active and (
                    time.monotonic() - self.last_learning_timestamp
                    >= self.config.learning_interval_s
                ):
                    self._perform_learning_cycle()
                    self.last_learning_timestamp = time.monotonic()

                # Check flag more frequently
                if self._stop_learning_flag.wait(timeout=10.0):
                    break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                if self._stop_learning_flag.wait(timeout=60.0):
                    break
        logger.info(f"{self.COMPONENT_NAME} learning loop finished.")

    # _enter_learning_mode and _exit_learning_mode are now effectively controlled by set_learning_mode
    # and the learning_active flag within the loop.

    def _perform_learning_cycle(self) -> None:
        if not self.learning_active:  # Double check
            logger.debug("Learning cycle skipped: learning_mode not active.")
            return

        logger.info("Starting new learning cycle...")
        # Removed `trace_event` and using VantaCore's `publish_event`
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.learning_cycle.start",
            {},
            source=self.COMPONENT_NAME,
        )

        if self.model_manager.training_job_active():  # Depends on ModelManagerInterface
            logger.info(
                "Skipping learning cycle: ModelManager reports active training job."
            )
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.learning_cycle.skipped",
                {"reason": "model_training_active"},
                source=self.COMPONENT_NAME,
            )
            return

        selected_topic = ""  # Ensure it's defined for finally block
        try:
            selected_topic = self._select_learning_topic()
            if not selected_topic:
                logger.info("No learning topic selected. Ending cycle.")
                return

            context = self._gather_learning_context(selected_topic)
            self._perform_component_specific_learning(
                selected_topic, context
            )  # Renamed
            self._perform_cross_component_knowledge_integration()  # Ensure method is defined
            self._store_learning_cycle_results(selected_topic, context)  # Renamed
            logger.info(
                f"Learning cycle for topic '{selected_topic}' completed successfully."
            )
        except Exception as e:
            logger.error(
                f"Error during learning cycle (topic: {selected_topic}): {e}",
                exc_info=True,
            )
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.learning_cycle.error",
                {"topic": selected_topic, "error": str(e)},
                source=self.COMPONENT_NAME,
            )
        finally:
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.learning_cycle.end",
                {"topic": selected_topic, "timestamp": time.time()},
                source=self.COMPONENT_NAME,
            )

    def _select_learning_topic(self) -> str:
        logger.debug("Selecting learning topic...")
        topic_scores: dict[str, float] = {}
        perf_monitor = self._get_vanta_component(
            self.config.performance_monitor_name, LearningPerformanceMonitorInterface
        )
        goal_system = self._get_vanta_component(
            self.config.goal_system_name, LearningGoalSystemInterface
        )

        for topic in self.learning_topics:
            score = 0.5  # Base
            if perf_monitor:
                perf = perf_monitor.get_topic_performance(
                    topic
                )  # Assumes returns 0-1, 1=good
                if perf is not None:
                    score += (1.0 - perf) * 0.4  # Prioritize low-performing
            if goal_system:
                relevance = goal_system.get_topic_relevance(
                    topic
                )  # Assumes returns 0-1
                if relevance is not None:
                    score += relevance * 0.3
            topic_scores[topic] = score

        if not topic_scores:
            return (
                random.choice(self.learning_topics)
                if self.learning_topics
                else "general_learning"
            )  # Fallback

        selected_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        logger.info(
            f"Selected learning topic: '{selected_topic}' (Score: {topic_scores[selected_topic]:.2f})"
        )
        return selected_topic

    def _gather_learning_context(self, topic: str) -> dict[str, Any]:
        logger.debug(f"Gathering learning context for topic: {topic}")
        context: dict[str, Any] = {
            "topic": topic,
            "timestamp": time.time(),
            "related_memories": [],
            "relevant_patterns": [],
            "component_states": {},
        }

        memory_comp = self._get_vanta_component(
            self.config.memory_component_name, LearningMemoryInterface
        )
        pattern_comp = self._get_vanta_component(
            self.config.pattern_analyzer_name, LearningPatternAnalyzerInterface
        )

        if memory_comp:
            context["related_memories"] = (
                memory_comp.search_relevant(topic=topic, limit=5) or []
            )
        if pattern_comp:
            context["relevant_patterns"] = pattern_comp.get_patterns(topic=topic) or []

        # Get states of relevant components for context
        for comp_conf_name, comp_interface in [
            (self.config.memory_component_name, LearningMemoryInterface),
            (self.config.goal_system_name, LearningGoalSystemInterface),
            (self.config.pattern_analyzer_name, LearningPatternAnalyzerInterface),
            (
                self.config.meta_core_name,
                LearningMetaCoreInterface,
            ),
        ]:
            comp_instance = self._get_vanta_component(comp_conf_name, comp_interface)
            if comp_instance and hasattr(comp_instance, "get_state"):
                try:
                    context["component_states"][comp_conf_name] = (
                        comp_instance.get_state()
                    )
                except Exception as e_state:
                    context["component_states"][comp_conf_name] = {
                        "error": str(e_state),
                        "error_type": type(e_state).__name__,
                        "timestamp": time.time(),
                    }

        return context

    def _perform_component_specific_learning(
        self, topic: str, context: dict[str, Any]
    ) -> None:  # Renamed
        logger.debug(f"Performing component-specific learning for topic: {topic}")
        component_configs_for_learning = [
            (self.config.memory_component_name, LearningMemoryInterface),
            (self.config.pattern_analyzer_name, LearningPatternAnalyzerInterface),
            (self.config.goal_system_name, LearningGoalSystemInterface),
            (self.config.meta_core_name, LearningMetaCoreInterface),
        ]
        for comp_conf_name, comp_interface in component_configs_for_learning:
            comp_instance = self._get_vanta_component(comp_conf_name, comp_interface)
            if comp_instance and hasattr(comp_instance, "perform_learning"):
                try:
                    logger.debug(
                        f"Triggering perform_learning for component: {comp_conf_name}"
                    )
                    comp_instance.perform_learning(topic=topic, context=context)
                    self.vanta_core.publish_event(
                        f"{self.COMPONENT_NAME}.component_learning.complete",
                        {
                            "component": comp_conf_name,
                            "topic": topic,
                            "timestamp": time.time(),
                        },
                        source=self.COMPONENT_NAME,
                    )
                except Exception as e_learn:
                    logger.error(
                        f"Error during perform_learning for {comp_conf_name}: {e_learn}",
                        exc_info=True,
                    )
                    self.vanta_core.publish_event(
                        f"{self.COMPONENT_NAME}.component_learning.error",
                        {
                            "component": comp_conf_name,
                            "topic": topic,
                            "error": str(e_learn),
                            "error_type": type(e_learn).__name__,
                            "timestamp": time.time(),
                        },
                        source=self.COMPONENT_NAME,
                    )

                def _perform_cross_component_knowledge_integration(
                    self,
                ) -> None:  # Renamed
                    logger.debug("Performing cross-component knowledge integration.")

        component_names_for_cross_learn = [
            self.config.pattern_analyzer_name,
            self.config.goal_system_name,
            self.config.meta_core_name,  # if defined and capable
            self.config.meta_learner_name,  # Include the meta learner for cross-domain transfer
        ]

        learning_components: list[Any] = []
        meta_learner = (
            None  # Separate reference to meta learner for performance tracking
        )

        for name_key in component_names_for_cross_learn:
            if name_key == self.config.meta_learner_name:
                # For MetaLearner, use the specific interface
                comp = self._get_vanta_component(name_key, MetaLearnerInterface)
                if comp:
                    meta_learner = comp
                    learning_components.append((name_key, comp))
                    logger.debug(
                        f"Meta Learner {name_key} added to knowledge integration pool"
                    )
            else:
                # For other components, use the pattern analyzer interface
                comp = self._get_vanta_component(
                    name_key, LearningPatternAnalyzerInterface
                )
                if (
                    comp
                    and hasattr(comp, "get_transferable_knowledge")
                    and hasattr(comp, "integrate_knowledge")
                ):
                    learning_components.append((name_key, comp))
                    logger.debug(
                        f"Component {name_key} added to knowledge integration pool"
                    )

        if not learning_components:
            logger.info("No components available for knowledge integration.")
            return

        integration_results = []
        # For each component pair, transfer knowledge
        for source_name, source_comp in learning_components:
            source_knowledge = None
            try:
                source_knowledge = source_comp.get_transferable_knowledge()
                if source_knowledge is None:
                    logger.debug(f"No transferable knowledge from {source_name}")
                    continue

                for target_name, target_comp in learning_components:
                    if source_name == target_name:
                        continue  # Skip self-integration

                    try:
                        target_comp.integrate_knowledge(
                            knowledge=source_knowledge, source=source_name
                        )
                        integration_results.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "timestamp": time.time(),
                                "status": "success",
                            }
                        )
                        logger.debug(
                            f"Integrated knowledge from {source_name} into {target_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error integrating knowledge from {source_name} into {target_name}: {e}"
                        )
                        integration_results.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "timestamp": time.time(),
                                "status": "error",
                                "error": str(e),
                            }
                        )

            except Exception as e:
                logger.error(f"Error getting knowledge from {source_name}: {e}")

        # After knowledge integration, track performance metrics from the meta learner
        if meta_learner:
            try:
                # Get performance metrics from the meta learner
                performance_metrics = meta_learner.get_performance_metrics()
                logger.info(f"MetaLearner performance metrics: {performance_metrics}")

                # If we have a metrics attribute, store the performance metrics
                if hasattr(self, "learning_metrics"):
                    self.learning_metrics["meta_learner_performance"] = (
                        performance_metrics
                    )

                # Publish a learning event with the performance metrics
                publish_learning_event(
                    self.vanta_core, "meta_learner_performance", performance_metrics
                )
            except Exception as e:
                logger.error(
                    f"Error getting performance metrics from meta learner: {e}"
                )

        logger.info(
            f"Cross-component knowledge integration completed with {len(integration_results)} transfers"
        )
        [r for r in integration_results if r["status"] == "error"]

        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.knowledge_integration.complete",
            {
                "successful_integrations": [
                    r for r in integration_results if r["status"] == "success"
                ],
                "failed_integrations": [
                    r for r in integration_results if r["status"] == "error"
                ],
                "details": integration_results,
            },
            source=self.COMPONENT_NAME,
        )

    def _perform_cross_component_knowledge_integration(self) -> None:
        """Perform cross-component knowledge integration."""
        logger.debug("Performing cross-component knowledge integration.")
        # Add logic for cross-component knowledge integration here
        pass

    def _store_learning_cycle_results(
        self, topic: str, context: dict[str, Any]
    ) -> None:  # Renamed
        logger.debug(f"Storing learning cycle results for topic: {topic}")
        memory_comp = self._get_vanta_component(
            self.config.memory_component_name, LearningMemoryInterface
        )
        if memory_comp:
            try:
                memory_comp.store_event(
                    event_data={
                        "type": "learning_cycle_outcome",
                        "topic": topic,
                        "context_summary_keys": list(context.keys()),
                        "timestamp": time.time(),
                    },
                    memory_type="learning_insights",  # Assuming store_event takes a memory_type
                )
            except Exception as e_store:
                logger.error(f"Error storing learning results to memory: {e_store}")
        else:
            logger.warning(
                "Memory component not available to store learning cycle results."
            )

    def _perform_memory_compression(self) -> None:
        logger.info("Attempting memory compression cycle...")
        memory_comp = self._get_vanta_component(
            self.config.memory_component_name, LearningMemoryInterface
        )
        comp_engine = self._get_vanta_component(
            self.config.compression_engine_name, LearningCompressionEngineInterface
        )

        if not memory_comp or not comp_engine:
            logger.warning(
                "Memory component or CompressionEngine not available. Skipping memory compression."
            )
            return

        try:
            candidates = memory_comp.get_compression_candidates()
            if not candidates:
                logger.info("No candidates for memory compression.")
                return

            compressed_count = 0
            for item_to_compress in candidates:
                if not isinstance(item_to_compress, dict):
                    continue  # Expecting dicts
                compression_method_name = self._select_compression_method_for_item(
                    item_to_compress
                )  # Renamed
                compressed_data = comp_engine.compress(
                    method_name=compression_method_name, item=item_to_compress
                )  # Pass method name
                if compressed_data:
                    memory_comp.store_compressed(compressed_data)
                    compressed_count += 1
            logger.info(
                f"Memory compression cycle complete. Compressed {compressed_count} items."
            )
        except Exception as e:
            logger.error(f"Error during memory compression cycle: {e}", exc_info=True)

    def _select_compression_method_for_item(
        self, item: dict[str, Any]
    ) -> str:  # Renamed
        # Logic from original script - simple routing
        if "text_content" in item:
            return "semantic_text_compression"
        elif "numeric_data_array" in item:
            return "lossy_numeric_compression"
        elif "pattern_sequence" in item:
            return "sequential_pattern_compression"
        return "general_data_compression"  # Default

    def set_learning_mode_active(self, active_status: bool) -> None:  # Renamed
        """Explicitly enables or disables learning mode."""
        with self._lock:
            if self.learning_active == active_status:
                logger.info(
                    f"Learning mode is already {'active' if active_status else 'inactive'}."
                )
                return

            self.learning_active = active_status
            if self.learning_active:
                self.last_learning_timestamp = (
                    time.monotonic()
                )  # Reset timer for next cycle
                self._initialize_internal_learning_state()  # Renamed
                event_name = "learning_mode.activated"
            else:
                event_name = "learning_mode.deactivated"

        logger.info(f"Learning mode explicitly set to: {active_status}")
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.{event_name}",
            {"status": active_status, "trigger": "explicit_set"},
            source=self.COMPONENT_NAME,
        )

        # Notify 'meta_core' or similar component if it exists and is configured
        meta_core_comp = self._get_vanta_component(
            self.config.meta_core_name, LearningMetaCoreInterface
        )
        if meta_core_comp and hasattr(meta_core_comp, "adjust_state"):
            try:
                meta_core_comp.adjust_state(
                    learning_mode=active_status,
                    awareness_delta=0.2 if active_status else -0.1,
                )
            except Exception as e_mc_adj:
                logger.warning(
                    f"Failed to notify meta_core of learning mode change: {e_mc_adj}"
                )

    def _initialize_internal_learning_state(self) -> None:  # Renamed
        logger.debug("Initializing internal state for new learning session.")
        # Reset any session-specific learning metrics or states here.
        # Example: notify relevant components via VantaCore or direct calls
        memory_comp = self._get_vanta_component(
            self.config.memory_component_name, LearningMemoryInterface
        )
        if memory_comp and hasattr(memory_comp, "prepare_learning_session"):
            memory_comp.prepare_learning_session()

        pattern_comp = self._get_vanta_component(
            self.config.pattern_analyzer_name, LearningPatternAnalyzerInterface
        )
        if pattern_comp and hasattr(pattern_comp, "reset_learning_state"):
            pattern_comp.reset_learning_state()

        # Could also publish a VantaCore event
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.learning_state.initialized",
            {},
            source=self.COMPONENT_NAME,
        )
        logger.info("Internal learning state initialized for session.")

    def is_learning_mode_active(self) -> bool:  # Renamed
        return self.learning_active

    def get_status(self) -> dict[str, Any]:
        return {
            "component_name": self.COMPONENT_NAME,
            "learning_mode_enabled_in_config": self.config.enable_learning_mode,
            "learning_mode_currently_active": self.learning_active,
            "last_learning_cycle_timestamp": self.last_learning_timestamp,
            "time_since_last_learning_s": time.monotonic()
            - self.last_learning_timestamp
            if self.last_learning_timestamp > 0
            else -1,
            "next_learning_cycle_approx_in_s": max(
                0,
                self.config.learning_interval_s
                - (time.monotonic() - self.last_learning_timestamp),
            )
            if self.learning_active and self.last_learning_timestamp > 0
            else -1,
            "last_compression_timestamp": self.last_compression_timestamp,
            "next_compression_approx_in_s": max(
                0,
                self.config.memory_compression_interval_s
                - (time.monotonic() - self.last_compression_timestamp),
            )
            if self.last_compression_timestamp > 0
            else -1,
        }


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    main_logger_lm = logging.getLogger("LearningManagerExample")
    main_logger_lm.setLevel(logging.DEBUG)

    main_logger_lm.info("--- Starting Vanta Learning Manager Example ---")

    # 1. Initialize VantaCore using registry access
    vanta_system_lm = get_vanta_core_instance()  # Use registry-based access instead

    class MockLMModelManager(ModelManagerProtocol):
        def training_job_active(self) -> bool:
            return False

        def get_available_models(self) -> List[str]:
            return ["mock_model"]

        def get_model_status(self, model_name: str) -> Dict[str, Any]:
            return {"status": "ok", "name": model_name}

        def request_model_tuning(
            self, model_name: str, tuning_params: Optional[Dict[str, Any]] = None
        ) -> bool:
            return True  # Pretend tuning always succeeds

        def update_model_weights(
            self, model_name: str, weights: Dict[str, Any]
        ) -> bool:
            return True  # Pretend weight updates always succeed

    # Register it so LearningManager can fetch it if not directly passed (though here we will pass it)

    # 3. Create LearningManagerConfig
    lm_config = LearningManagerConfig(
        log_level="DEBUG",
        learning_interval_s=10,  # Short for demo
        memory_compression_interval_s=15,
        enable_learning_mode=True,  # Start active for demo
    )

    # 4. Mock other components LearningManager might try to use from VantaCore registry
    #    These will use the default Learning*Interface stubs if not found in VantaCore
    #    or if the found component doesn't match the expected Protocol.
    #    For this demo, we'll let LM use its internal understanding of these through safe_component_call
    #    or its new _get_vanta_component helper which will use defaults if not found.

    # Example of providing a component that LearningManager will use:
    class MyPatternAnalyzer(LearningPatternAnalyzerInterface):
        def get_patterns(self, topic):
            main_logger_lm.info(f"MyPatternAnalyzer: get_patterns for {topic}")
            return [{"id": "p1", "pattern_data": "..."}]

        def perform_learning(self, topic, context):
            main_logger_lm.info(f"MyPatternAnalyzer: perform_learning for {topic}")

        def reset_learning_state(self):
            main_logger_lm.info("MyPatternAnalyzer: reset_learning_state")

        def get_transferable_knowledge(self):
            main_logger_lm.info("MyPatternAnalyzer: get_transferable_knowledge")
            return {"pattern_insight": "example"}

        def integrate_knowledge(self, k, s):
            main_logger_lm.info(f"MyPatternAnalyzer: integrate_knowledge from {s}")

    vanta_system_lm.register_component(
        lm_config.pattern_analyzer_name, MyPatternAnalyzer()
    )

    # 5. Instantiate LearningManager
    learning_mgr = LearningManager(
        vanta_core=vanta_system_lm,
        config=lm_config,
        model_manager=DefaultModelManager(),
    )

    # 6. Start its background loops
    learning_mgr.start_threads()
    learning_mgr.set_learning_mode_active(
        True
    )  # Ensure it's active for the demo cycles

    try:
        main_logger_lm.info(
            "Learning Manager running. Test interval is short. Ctrl+C to stop."
        )
        for i in range(3):  # Observe for a few cycles
            time.sleep(lm_config.learning_interval_s + 2)
            status = learning_mgr.get_status()
            main_logger_lm.info(
                f"LM STATUS Cycle ~{i + 1}: Active={status.get('learning_mode_currently_active')}, NextLearnIn~{status.get('next_learning_cycle_approx_in_s'):.0f}s"
            )
            if not learning_mgr.is_learning_mode_active() and not (
                learning_mgr.compression_thread
                and learning_mgr.compression_thread.is_alive()
            ):
                main_logger_lm.warning("Learning Manager threads seem to have stopped.")
                break

    except KeyboardInterrupt:
        main_logger_lm.info("Keyboard interrupt by user.")
    finally:
        main_logger_lm.info("Stopping Learning Manager threads.")
        learning_mgr.stop_threads()
        # vanta_system_lm.shutdown() # If VantaCore had general shutdown
        main_logger_lm.info("--- Vanta Learning Manager Example Finished ---")
