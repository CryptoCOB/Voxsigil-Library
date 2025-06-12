# File: vanta_proactive_intelligence.py (Refactored from proactive_intelligence.py)
"""
Proactive intelligence and priority management for the VantaCore ecosystem.

This module handles:
1. Predictive action evaluation
2. Dynamic priority management
3. Action simulation and risk assessment (conceptual, depends on model_manager capabilities)
4. System state monitoring and prediction
"""

import asyncio  # For async bus operations
import logging
import time
from collections import deque
from typing import Any, Protocol, runtime_checkable

# Import unified interface definitions
from Vanta.interfaces.specialized_interfaces import ModelManagerInterface

# HOLO-1.5 Recursive Symbolic Cognition Mesh imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType

# Assuming vanta_core.py is accessible
from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaCore.ProactiveIntelligence")


# --- Configuration Class ---
class ProactiveIntelligenceConfig:
    def __init__(
        self,
        log_level: str = "INFO",
        simulation_depth: int = 3,
        risk_threshold: float = 0.7,
        priority_update_interval_s: int = 60,
        state_prediction_window_s: int = 300,
        action_history_maxlen: int = 100,
        recent_decisions_maxlen: int = 50,
        task_scheduler_component_name: str = "task_scheduler",
        metrics_store_component_name: str = "metrics_store",
        health_monitor_component_name: str = "health_monitor",
    ):
        self.log_level = log_level
        self.simulation_depth = simulation_depth
        self.risk_threshold = risk_threshold
        self.priority_update_interval_s = priority_update_interval_s
        self.state_prediction_window_s = state_prediction_window_s
        self.action_history_maxlen = action_history_maxlen
        self.recent_decisions_maxlen = recent_decisions_maxlen

        self.task_scheduler_component_name = task_scheduler_component_name
        self.metrics_store_component_name = metrics_store_component_name
        self.health_monitor_component_name = health_monitor_component_name

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Interface Definitions ---

@runtime_checkable
class TaskSchedulerInterface(
    Protocol
):  # Interface for task_scheduler component from VantaCore
    def get_scheduled_events(self, window_s: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class MetricsStoreInterface(Protocol):  # Interface for metrics_store component
    def get_metric_history(
        self, metric: str, minutes: int
    ) -> list[dict[str, Any]] | None: ...


@runtime_checkable
class HealthMonitorInterface(Protocol):  # Interface for health_monitor component
    def get_health(self) -> dict[str, Any] | None: ...


# --- Main ProactiveIntelligence Class ---

@vanta_core_module(
    name="proactive_intelligence",
    subsystem="intelligence_services",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="VantaCore proactive intelligence and priority management with predictive action evaluation and system state monitoring",
    capabilities=["action_evaluation", "priority_management", "system_state_prediction", "risk_assessment", "resource_monitoring"],
    cognitive_load=2.8,
    symbolic_depth=3,
    collaboration_patterns=["predictive_analysis", "intelligent_orchestration", "proactive_adaptation"]
)
class ProactiveIntelligence(BaseCore):
    COMPONENT_NAME = "proactive_intelligence"

    def __init__(
        self,
        vanta_core: VantaCore,
        config: ProactiveIntelligenceConfig,
        model_manager: ModelManagerInterface,
    ):
        # Call BaseCore constructor first
        super().__init__(vanta_core, config)
        
        # Store original config object
        self.config = config

        if not isinstance(model_manager, ModelManagerInterface):
            # Try to get from VantaCore registry if not provided or wrong type
            mm_from_registry = self.vanta_core.get_component("model_manager")
            if isinstance(mm_from_registry, ModelManagerInterface):
                self.model_manager = mm_from_registry
                logger.info("Using 'model_manager' from VantaCore registry.")
            else:
                # This is a critical dependency, should ideally raise error or have a more robust default.
                # For now, log a critical warning. VantaCore setup should ensure it.
                logger.critical(
                    "ModelManagerInterface not provided or found in VantaCore registry. ProactiveIntelligence may not function correctly."
                )                # You might create a very basic DefaultModelManager stub here if partial functionality is acceptable without it.
                # For completeness, we'll use the passed model_manager regardless, but may fail later operations.
                from core.learning_manager import DefaultModelManager
                self.model_manager = DefaultModelManager()
        else:
            self.model_manager = model_manager

        # State tracking
        self.last_priority_update: float = 0.0
        self.action_history: deque[dict[str, Any]] = deque(
            maxlen=self.config.action_history_maxlen
        )
        self.current_priorities: dict[str, float] = {}
        self.state_predictions: dict[
            float, dict[str, Any]
        ] = {}  # timestamp -> prediction
        self.recent_decisions: deque[dict[str, Any]] = deque(
            maxlen=self.config.recent_decisions_maxlen
        )

        self.vanta_core.register_component(
            self.COMPONENT_NAME, self, metadata={"type": "intelligence_service"}
        )
        logger.info("ProactiveIntelligence initialized and registered with VantaCore.")

    async def initialize(self) -> bool:
        """Initialize the ProactiveIntelligence component."""
        try:
            # Register async bus handlers for proactive intelligence events
            await self.vanta_core.async_bus.subscribe(
                MessageType.ACTION_EVALUATION, self.handle_action_evaluation
            )
            await self.vanta_core.async_bus.subscribe(
                MessageType.PRIORITY_UPDATE, self.handle_priority_update
            )
            
            logger.info("ProactiveIntelligence async handlers registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ProactiveIntelligence: {e}")
            return False

    # --- Async bus handlers ---
    def handle_action_evaluation(self, message: AsyncMessage) -> None:
        """Handle incoming ACTION_EVALUATION requests via async bus and respond with evaluation."""
        logger.info(
            f"ProactiveIntelligence: Received evaluation request: {message.content}"
        )
        # Expect content to be dict with keys 'action_type', 'action_params', 'context'
        req = message.content if isinstance(message.content, dict) else {}
        action_type = str(req.get("action_type", ""))  # Ensure string type
        params = req.get("action_params", {}) or {}
        context = req.get("context", {}) or {}
        if not action_type:
            logger.warning(
                "ProactiveIntelligence: Missing action_type in evaluation request."
            )
            return
        evaluation = self.evaluate_action(action_type, params, context)
        # Send evaluation back to requester
        asyncio.create_task(
            self.vanta_core.async_bus.publish(
                AsyncMessage(
                    MessageType.ACTION_EVALUATION,
                    self.COMPONENT_NAME,
                    evaluation,
                    target_ids=[message.sender_id],
                )
            )
        )

    def handle_priority_update(self, message: AsyncMessage) -> None:
        """Handle incoming PRIORITY_UPDATE requests via async bus and update priorities."""
        logger.info(
            f"ProactiveIntelligence: Received priority update request: {message.content}"
        )
        # Trigger priority recalculation and publication
        self.update_priorities()

    def _get_task_scheduler(self) -> TaskSchedulerInterface | None:
        scheduler = self.vanta_core.get_component(
            self.config.task_scheduler_component_name
        )
        if scheduler and isinstance(scheduler, TaskSchedulerInterface):
            return scheduler
        logger.warning(
            f"TaskScheduler component '{self.config.task_scheduler_component_name}' not found or not TaskSchedulerInterface."
        )
        return None

    def _get_metrics_store(self) -> MetricsStoreInterface | None:
        store = self.vanta_core.get_component(self.config.metrics_store_component_name)
        if store and isinstance(store, MetricsStoreInterface):
            return store
        logger.warning(
            f"MetricsStore component '{self.config.metrics_store_component_name}' not found or not MetricsStoreInterface."
        )
        return None

    def _get_health_monitor(self) -> HealthMonitorInterface | None:
        monitor = self.vanta_core.get_component(
            self.config.health_monitor_component_name
        )
        if monitor and isinstance(monitor, HealthMonitorInterface):
            return monitor
        logger.warning(
            f"HealthMonitor component '{self.config.health_monitor_component_name}' not found or not HealthMonitorInterface."
        )
        return None

    def evaluate_action(
        self, action_type: str, action_params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        evaluation = {
            "action_type": action_type,
            "params_preview": {k: str(v)[:50] for k, v in action_params.items()},
            "timestamp": time.time(),
            "risk_score": 0.0,
            "stability_score": 1.0,
            "urgency_score": 0.0,
            "memory_impact_estimate": 0.0,  # Renamed
            "recommended": True,
            "warnings": [],
            "suggestions": [],
        }

        # Route to specific evaluation method
        eval_method_name = f"_evaluate_{action_type.lower()}"
        if hasattr(self, eval_method_name) and callable(
            getattr(self, eval_method_name)
        ):
            getattr(self, eval_method_name)(action_params, context, evaluation)
        else:
            self._evaluate_generic_action(
                action_type, action_params, context, evaluation
            )  # Fallback

        self._record_action_evaluation(
            evaluation
        )  # Changed name for clarity        # Publish evaluation event to VantaCore
        self.vanta_core.publish_event(
            event_type=f"{self.COMPONENT_NAME}.action_evaluated",
            data=evaluation,
            source=self.COMPONENT_NAME,
        )
        return evaluation

    def _evaluate_model_change(
        self,
        params: dict[str, Any],
        _context: dict[str, Any],
        eval_results: dict[str, Any],
    ):
        logger.debug(f"Evaluating model_change: {params}")
        # Note: model_status could be used for future validation logic if needed
        # model_status = self.model_manager.get_status()  # Assumes ModelManagerInterface
        # current_models = model_status.get("active_models", {})

        if self.model_manager.training_job_active():
            eval_results["risk_score"] = 0.9
            eval_results["warnings"].append("Model change during active training")
            eval_results["recommended"] = False
            return

        target_model_name = params.get("model_name")  # Consistent naming
        if target_model_name:
            memory_req = self._estimate_model_memory(target_model_name)  # Placeholder
            available_mem_percent = self._get_available_memory_percent()  # Placeholder
            if available_mem_percent < 0.2 and memory_req > (
                available_mem_percent * 1000
            ):  # Arbitrary logic
                eval_results["risk_score"] = max(eval_results["risk_score"], 0.8)
                eval_results["warnings"].append(
                    f"Potentially insufficient memory for model '{target_model_name}'. Available: {available_mem_percent * 100:.1f}%"
                )

    def _evaluate_memory_action(
        self,
        params: dict[str, Any],
        _context: dict[str, Any],
        eval_results: dict[str, Any],
    ):
        logger.debug(f"Evaluating memory_action: {params}")
        action_subtype = params.get("subtype", "").lower()

        if action_subtype == "compression":
            data_size_mb = params.get("data_size_mb", 0)
            if data_size_mb > 100:  # 100MB
                eval_results["risk_score"] = max(eval_results["risk_score"], 0.6)
                eval_results["memory_impact_estimate"] = 0.8  # High impact potential
                eval_results["warnings"].append(
                    "Large data compression requested, monitor system performance."
                )
        elif action_subtype == "clear_cache" or action_subtype == "full_reset":
            eval_results["risk_score"] = 0.9
            eval_results["warnings"].append(
                f"Memory '{action_subtype}' is a high-risk operation."
            )
            eval_results["recommended"] = False

        health_monitor = self._get_health_monitor()
        if health_monitor:
            memory_health_data = (
                health_monitor.get_health()
            )  # Assumes general health might contain memory stats
            if (
                memory_health_data
                and memory_health_data.get("memory_status", "healthy") != "healthy"
            ):
                eval_results["risk_score"] = max(
                    eval_results["risk_score"], eval_results["risk_score"] + 0.2
                )
                eval_results["warnings"].append(
                    f"Memory system health issues reported: {memory_health_data.get('memory_details', 'N/A')}"
                )

    def _evaluate_system_command(
        self,
        params: dict[str, Any],
        _context: dict[str, Any],
        eval_results: dict[str, Any],
    ):
        logger.debug(f"Evaluating system_command: {params}")
        command = params.get("command_name", "").lower()

        high_risk_keywords = [
            "delete",
            "remove",
            "reset",
            "clear_all",
            "shutdown_critical",
            "stop_core",
        ]
        if any(keyword in command for keyword in high_risk_keywords):
            eval_results["risk_score"] = 0.8
            eval_results["warnings"].append("High-risk system command detected.")

        system_load_percent = self._get_system_load_percent()  # Placeholder
        if system_load_percent > 85:  # 85% load
            eval_results["risk_score"] = max(
                eval_results["risk_score"], eval_results["risk_score"] + 0.2
            )
            eval_results["warnings"].append(
                f"High system load ({system_load_percent:.1f}%)."
            )

        # Simplified dependency check - in reality this needs a dependency map
        required_deps = self._get_command_dependencies(command).get("requires", [])
        if required_deps:
            missing_deps = [
                dep
                for dep in required_deps
                if self.vanta_core.get_component(dep) is None
            ]
            if missing_deps:
                eval_results["risk_score"] = max(
                    eval_results["risk_score"], eval_results["risk_score"] + 0.3
                )
                eval_results["warnings"].append(
                    f"Command '{command}' has missing dependencies: {missing_deps}"
                )
        # Simplified dependency check - in reality this needs a dependency map

    def _evaluate_learning_mode(
        self,
        params: dict[str, Any],
        _context: dict[str, Any],
        eval_results: dict[str, Any],
    ):
        logger.debug(f"Evaluating learning_mode change: {params}")
        available_resources = self._get_available_resources_summary()  # Placeholder
        if (
            available_resources.get("cpu_available_percent", 100) < 30
            or available_resources.get("memory_available_gb", 100) < 2
        ):  # Example thresholds
            eval_results["risk_score"] = 0.7
            eval_results["warnings"].append(
                "Limited system resources for intensive learning."
            )

        critical_tasks_active = self._get_critical_tasks_status()  # Placeholder
        if critical_tasks_active:
            eval_results["risk_score"] = max(
                eval_results["risk_score"], eval_results["risk_score"] + 0.2
            )
            eval_results["warnings"].append(
                "Critical system tasks are currently active."
            )
            # available_resources.get("memory_available_gb", 100) < 2: # Example thresholds (removed stray line)

    def _evaluate_generic_action(
        self,
        action_type: str,
        params: dict[str, Any],
        _context: dict[str, Any],
        eval_results: dict[str, Any],
    ):
        logger.debug(f"Evaluating generic_action '{action_type}': {params}")
        similar_actions_history = self._find_similar_actions_in_history(
            action_type, params
        )
        if similar_actions_history:
            success_rate = self._calculate_success_rate_from_history(
                similar_actions_history
            )
            eval_results["risk_score"] = max(
                eval_results["risk_score"], 1.0 - success_rate
            )
            if success_rate < 0.5:
                eval_results["warnings"].append(
                    f"Low historical success rate ({success_rate:.2f}) for similar actions."
                )

        est_resource_impact = self._estimate_generic_resource_impact(
            action_type, params
        )  # Placeholder (0.0 to 1.0)
        eval_results["memory_impact_estimate"] = max(
            eval_results["memory_impact_estimate"], est_resource_impact * 0.5
        )  # Example mapping
        if est_resource_impact > 0.7:
            eval_results["warnings"].append("Action may have high resource impact.")

        if similar_actions_history:
            success_rate = self._calculate_success_rate_from_history(
                similar_actions_history
            )
            eval_results["risk_score"] = max(
                eval_results["risk_score"], 1.0 - success_rate
            )
            if success_rate < 0.5:
                eval_results["warnings"].append(
                    f"Low historical success rate ({success_rate:.2f}) for similar actions."
                )

        est_resource_impact = self._estimate_generic_resource_impact(
            action_type, params
        )  # Placeholder (0.0 to 1.0)
        eval_results["memory_impact_estimate"] = max(
            eval_results["memory_impact_estimate"], est_resource_impact * 0.5
        )  # Example mapping
        if est_resource_impact > 0.7:
            eval_results["warnings"].append("Action may have high resource impact.")

    def update_priorities(self) -> None:
        now = time.monotonic()
        if now - self.last_priority_update < self.config.priority_update_interval_s:
            return  # Update too soon

        self.last_priority_update = now
        current_sys_state = self._get_current_system_state_summary()  # Placeholder

        new_priorities = {
            "memory_management": self._calculate_memory_priority(current_sys_state),
            "model_health_monitoring": self._calculate_model_priority(
                current_sys_state
            ),
            "adaptive_learning_cycles": self._calculate_learning_priority(
                current_sys_state
            ),
            "system_maintenance_tasks": self._calculate_maintenance_priority(
                current_sys_state
            ),
        }

        self.current_priorities = self._smooth_priority_transition(
            self.current_priorities, new_priorities
        )

        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.priorities_updated",
            {"priorities": self.current_priorities},
            source=self.COMPONENT_NAME,
        )
        logger.info(f"System priorities updated: {self.current_priorities}")

    def predict_system_state(
        self, future_window_s: int | None = None
    ) -> dict[str, Any]:  # Renamed for clarity
        window_s = future_window_s or self.config.state_prediction_window_s
        current_sys_state = self._get_current_system_state_summary()

        prediction: dict[str, Any] = {
            "prediction_timestamp": time.time() + window_s,
            "confidence": 1.0,
            "predicted_state": current_sys_state.copy(),
        }

        task_scheduler = self._get_task_scheduler()
        if task_scheduler:
            scheduled_events = task_scheduler.get_scheduled_events(window_s)
            for event_details in scheduled_events:
                self._apply_event_impact_to_prediction(
                    prediction, event_details
                )  # Placeholder

        system_trends = self._analyze_system_trends()
        for trend_details in system_trends:
            self._apply_trend_impact_to_prediction(
                prediction, trend_details, window_s
            )  # Placeholder

        self.state_predictions[prediction["prediction_timestamp"]] = prediction
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.state_predicted",
            prediction,
            source=self.COMPONENT_NAME,
        )
        logger.info(
            f"System state predicted for T+{window_s}s. Confidence: {prediction['confidence']:.2f}"
        )
        return prediction

    # --- Placeholder/Helper methods for internal logic & component interaction ---
    def _get_current_system_state_summary(
        self,
    ) -> dict[str, Any]:  # Simplified from _get_current_state
        health_monitor = self._get_health_monitor()
        metrics_store = self._get_metrics_store()

        state: dict[str, Any] = {"timestamp": time.time()}
        try:
            state["model_status_summary"] = self.model_manager.get_status().get(
                "status", "unknown"
            )
        except Exception:
            state["model_status_summary"] = "error_fetching"

        if health_monitor:
            health_data = health_monitor.get_health()
            if health_data and isinstance(health_data, dict):
                state["overall_health"] = health_data.get("overall_status", "unknown")
            else:
                state["overall_health"] = "unknown"
        else:
            state["overall_health"] = "monitor_unavailable"

        if metrics_store:
            memory_history = metrics_store.get_metric_history(
                "memory_usage_percent", minutes=1
            )
            if memory_history and len(memory_history) > 0:
                # Use the latest value if available
                state["memory_usage_percent"] = memory_history[-1].get("value", 0)
            else:
                state["memory_usage_percent"] = 0
        else:
            state["memory_usage_percent"] = 0

        return state

    def _estimate_model_memory(self, _model_name: str) -> float:
        return 1000.0  # MB, Placeholder

    def _get_available_memory_percent(self) -> float:
        return 0.6  # 60% free, Placeholder

    def _get_system_load_percent(self) -> float:
        return 30.0  # 30% load, Placeholder

    def _get_command_dependencies(self, _cmd: str) -> dict[str, list]:
        return {"requires": []}  # Placeholder

    def _get_available_resources_summary(self) -> dict[str, float]:
        return {
            "cpu_available_percent": 70.0,
            "memory_available_gb": 16.0,
        }  # Placeholder

    def _get_critical_tasks_status(self) -> bool:
        return False  # Placeholder, no critical tasks active

    def _assess_learning_readiness(self, _ctx: dict[str, Any]) -> float:
        return 0.8  # Placeholder

    def _find_similar_actions_in_history(
        self, _at: str, _p: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return []  # Placeholder

    def _calculate_success_rate_from_history(
        self, _acts: list[dict[str, Any]]
    ) -> float:
        return 0.75  # Placeholder

    def _estimate_generic_resource_impact(self, _at: str, _p: dict[str, Any]) -> float:
        return 0.2  # Placeholder, low impact

    def _calculate_memory_priority(self, state: dict) -> float:
        return 1.0 - (
            state.get("memory_usage_percent", 50) / 100
        )  # Higher usage, lower prio for new mem tasks

    def _calculate_model_priority(self, state: dict) -> float:
        return 0.8 if state.get("model_status_summary") == "healthy" else 0.2

    def _calculate_learning_priority(self, _state: dict) -> float:
        return 0.6  # Medium default

    def _calculate_maintenance_priority(self, _state: dict) -> float:
        return 0.3  # Low default

    def _smooth_priority_transition(self, old_p: dict, new_p: dict) -> dict:
        # (Implementation from original is fine)
        smoothed = {}
        alpha = 0.3
        all_keys = set(old_p.keys()) | set(new_p.keys())
        for key in all_keys:
            old_val = old_p.get(
                key, new_p.get(key, 0.0)
            )  # Use new if old doesn't exist
            new_val = new_p.get(key, old_val)  # Use old if new doesn't exist
            smoothed[key] = (alpha * new_val) + ((1 - alpha) * old_val)
        return smoothed

    def _apply_event_impact_to_prediction(self, pred: dict, event: dict):
        pass  # Placeholder

    def _analyze_system_trends(self) -> list[dict[str, Any]]:
        return []  # Placeholder

    def _apply_trend_impact_to_prediction(self, pred: dict, trend: dict, win_s: int):
        pass  # Placeholder

    def _record_action_evaluation(self, evaluation: dict[str, Any]) -> None:
        self.action_history.append(evaluation)  # Keep evaluated action details
        self.recent_decisions.append(
            evaluation
        )  # Also keep in recent decisions for quicker access
        logger.debug(
            f"Action evaluation recorded: {evaluation.get('action_type')}, Risk: {evaluation.get('risk_score')}"
        )
        # Publishing event is handled by the calling method evaluate_action

    def get_current_priorities(self) -> dict[str, float]:
        return self.current_priorities.copy()

    def get_recent_decisions(self, limit: int = 10) -> list[dict[str, Any]]:
        return list(self.recent_decisions)[-limit:]

    def get_state_predictions(self) -> dict[float, dict[str, Any]]:
        now = time.time()  # Use monotonic time for future predictions
        self.state_predictions = {
            ts: pred for ts, pred in self.state_predictions.items() if ts > now
        }
        return self.state_predictions.copy()


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    main_logger_pi = logging.getLogger("ProactiveIntelExample")

    # 1. Initialize VantaCore
    vanta_system_pi = VantaCore()

    # 2. Mock ModelManager (would be a real component provided by VantaCore or injected)
    class MockModelManager(ModelManagerInterface):
        def get_status(self):
            return {"active_models": {"role_A": "model_X"}, "status": "healthy_mock"}

        def training_job_active(self):
            return False

    mock_model_mgr_instance = MockModelManager()
    vanta_system_pi.register_component(
        "model_manager", mock_model_mgr_instance
    )  # ProactiveIntelligence will get this

    # 3. Create ProactiveIntelligenceConfig
    pro_intel_config = ProactiveIntelligenceConfig(
        log_level="DEBUG", priority_update_interval_s=5
    )

    # 4. Instantiate ProactiveIntelligence
    # It will get model_manager from vanta_system_pi's registry.
    proactive_intel_instance = ProactiveIntelligence(
        vanta_core=vanta_system_pi,
        config=pro_intel_config,
        model_manager=mock_model_mgr_instance,
    )

    # 5. Simulate some usage
    main_logger_pi.info("--- Simulating Action Evaluations ---")
    eval_1 = proactive_intel_instance.evaluate_action(
        "model_change",
        {"model_name": "new_model_Y", "role_to_replace": "role_A"},
        {"current_tasks": 0},
    )
    main_logger_pi.info(
        f"Eval 1 (model_change): Risk {eval_1['risk_score']:.2f}, Recommended: {eval_1['recommended']}, Warnings: {eval_1['warnings']}"
    )

    eval_2 = proactive_intel_instance.evaluate_action(
        "memory_action",
        {"subtype": "compression", "data_size_mb": 150},
        {"system_stability": 0.9},
    )
    main_logger_pi.info(
        f"Eval 2 (memory_action): Risk {eval_2['risk_score']:.2f}, Recommended: {eval_2['recommended']}, Impact Est: {eval_2['memory_impact_estimate']:.2f}"
    )

    eval_3 = proactive_intel_instance.evaluate_action(
        "start_new_research", {"topic": "AGI_alignment"}, {"available_compute": 0.8}
    )
    main_logger_pi.info(
        f"Eval 3 (generic_action): Risk {eval_3['risk_score']:.2f}, Recommended: {eval_3['recommended']}"
    )

    main_logger_pi.info("\n--- Simulating Priority Updates & State Prediction ---")
    # Mock some other components ProactiveIntelligence might query via vanta_core.get_component()

    class MockHealthMonitor(HealthMonitorInterface):
        def get_health(self):
            return {
                "overall_status": "healthy",
                "memory_status": "healthy",
                "cpu_load_percent": 25.0,
            }

    vanta_system_pi.register_component(
        pro_intel_config.health_monitor_component_name, MockHealthMonitor()
    )

    class MockMetricsStore(MetricsStoreInterface):
        def get_metric_history(self, metric, minutes):
            if metric == "memory_usage_percent":
                return [
                    {"timestamp": time.time() - 60, "value": 40},
                    {"timestamp": time.time(), "value": 45},
                ]
            return []

    vanta_system_pi.register_component(
        pro_intel_config.metrics_store_component_name, MockMetricsStore()
    )

    class MockTaskScheduler(TaskSchedulerInterface):
        def get_scheduled_events(self, _):
            return [
                {
                    "event_type": "scheduled_backup",
                    "time_offset": 100,
                    "estimated_impact": "medium_load",
                }
            ]

    vanta_system_pi.register_component(
        pro_intel_config.task_scheduler_component_name, MockTaskScheduler()
    )

    prediction = proactive_intel_instance.predict_system_state(
        future_window_s=600
    )  # Predict 10 mins into future
    main_logger_pi.info(
        f"State Prediction for T+600s (keys): {list(prediction.get('predicted_state', {}).keys())}, Confidence: {prediction.get('confidence')}"
    )

    main_logger_pi.info("\n--- Recent Decisions (last 3) ---")
    for i, decision_eval in enumerate(
        proactive_intel_instance.get_recent_decisions(limit=3)
    ):
        main_logger_pi.info(
            f"  Decision {i + 1}: Action='{decision_eval.get('action_type')}', Risk='{decision_eval.get('risk_score'):.2f}', Recommended='{decision_eval.get('recommended')}'"
        )

    main_logger_pi.info("\n--- Proactive Intelligence VantaCore Example Finished ---")
