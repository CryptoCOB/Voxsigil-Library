
"""
Tree-of-Thought (ToT) Engine Module for NebulaCore
Encapsulates four specialist agent roles for structured reasoning:
 1. ThoughtSeeder       → Seed Phase
 2. BranchEvaluator     → Score Phase
 3. BranchValidator     → Prune Phase
 4. MetaLearningAgent   → Integrate Phase
"""
import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Protocol, Union, runtime_checkable

# Assuming vanta_echo_nebula_core.py and vanta_cat_engine.py (for DefaultNebulaMemoryBraid if used) are accessible
try:
    from vanta_echo_nebula_bus import NebulaCore as VantaEchoNebulaCore
except ImportError:
    # Minimal stub for VantaEchoNebulaCore when bus is not available
    class VantaEchoNebulaCore:
        def __init__(self):
            self.event_bus = None
            self.async_bus = None
            self.registry = {}

# NebulaCore type alias for compatibility
NebulaCore = VantaEchoNebulaCore

def _safe_publish_event(vanta_echo_nebula_core, event_name: str, payload: dict, source: str):
    """Safely publish event from thread context."""
    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        # Schedule the coroutine to run in the event loop
        asyncio.run_coroutine_threadsafe(vanta_echo_nebula_core.publish_event(event_name, payload, source), loop)
    except RuntimeError:
        # No event loop running, try to create one
        try:
            asyncio.run(vanta_echo_nebula_core.publish_event(event_name, payload, source))
        except Exception:
            # If all else fails, just log it
            pass

# If ToTEngine needs its own default MemoryBraid, it could import the one from vanta_cat_engine or define its own.
# For simplicity, we'll assume it might use a generic one or one provided by NebulaCore.
# from vanta_cat_engine import DefaultNebulaMemoryBraid, MemoryBraidInterface (if sharing defaults)

# --- Basic logging setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NebulaCore.ToTEngine")


# --- Configuration Class ---
class ToTEngineConfig:
    def __init__(
        self,
        interval_s: int = 300,
        log_level: str = "INFO",
        auto_connect_checkin_manager: bool = True,
        checkin_manager_component_name: str = "checkin_manager",
        auto_connect_external_inputs: bool = True,
        external_inputs_component_name: str = "external_inputs_bridge",
        default_memory_braid_config: dict[str, Any]
        | None = None,  # Config for its internal default braid
    ):
        if not isinstance(interval_s, int) or interval_s <= 0:
            logger.warning(
                f"Invalid ToTEngine interval_s: {interval_s}. Defaulting to 300."
            )
            self.interval_s: int = 300
        else:
            self.interval_s: int = interval_s

        self.log_level: str = log_level
        self.auto_connect_checkin_manager = auto_connect_checkin_manager
        self.checkin_manager_component_name = checkin_manager_component_name
        self.auto_connect_external_inputs = auto_connect_external_inputs
        self.external_inputs_component_name = external_inputs_component_name
        self.default_memory_braid_config = default_memory_braid_config or {}

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Specialist Agent Protocols ---
@runtime_checkable
class ThoughtSeeder(Protocol):
    def generate(
        self, context: Any, task_directives: dict[str, Any] | None = None
    ) -> list[Any]: ...  # Added task_directives
    def expand(
        self,
        branches: list[Any],
        context: Any,
        task_directives: dict[str, Any] | None = None,
    ) -> list[Any]: ...  # Added task_directives


@runtime_checkable
class BranchEvaluator(Protocol):
    def score(
        self,
        branches: list[Any],
        context: Any,
        task_directives: dict[str, Any] | None = None,
    ) -> dict[str, float]: ...  # Key is branch_id


@runtime_checkable
class BranchValidator(Protocol):
    def prune(
        self,
        branches: list[Any],
        scores: dict[str, float],
        task_directives: dict[str, Any] | None = None,
    ) -> list[Any]: ...


@runtime_checkable
class MetaLearningAgent(Protocol):
    def integrate(
        self,
        branches: list[Any],
        scores: dict[str, float],
        task_directives: dict[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class ContextProvider(Protocol):
    def get_current_context(self) -> Any: ...
    # Optional: A way to identify the provider for merging contexts
    def get_provider_name(self) -> str:
        return self.__class__.__name__


# --- NebulaCore Specific Default Memory Braid ---
# This could be imported if CATEngine's default is generic enough, or defined here.
# For this refactor, let's assume ToTEngine can have its own simple default if needed.
@runtime_checkable
class MemoryBraidInterface(Protocol):  # Redefining for clarity, can be common
    def store_braid_data(
        self, key: str, data: Any, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def retrieve_braid_data(self, key: str) -> Any | None: ...


class DefaultToTMemoryBraid(MemoryBraidInterface):
    def __init__(self):
        self._storage: dict[str, Any] = {}
        logger.info("DefaultToTMemoryBraid (in-memory) active.")

    def store_braid_data(
        self, key: str, data: Any, metadata: dict[str, Any] | None = None
    ):
        self._storage[key] = {"data": data, "metadata": metadata or {}}

    def retrieve_braid_data(self, key: str) -> Any | None:
        item = self._storage.get(key)
        return item["data"] if item else None


# --- Adapter for CheckinManager (or similar component from VantaEchoNebulaCore registry) ---
class NebulaRegisteredContextProvider(ContextProvider):
    """
    Adapter to use a component registered in NebulaCore as a ContextProvider.
    The registered component must have a method like `get_current_state()` or `get_context()`.
    """

    def __init__(
        self,
        vanta_echo_nebula_core: NebulaCore,
        component_name: str,
        context_method_name: str = "get_current_state",
    ):
        self.vanta_echo_nebula_core = vanta_echo_nebula_core
        self.component_name = component_name
        self.context_method_name = context_method_name
        self._component_instance: Any | None = None
        logger.info(
            f"NebulaRegisteredContextProvider adapter initialized for component '{component_name}'."
        )

    def _get_component(self) -> Any | None:
        if self._component_instance is None:
            self._component_instance = self.vanta_echo_nebula_core.get_component(
                self.component_name
            )
            if self._component_instance is None:
                logger.warning(
                    f"Component '{self.component_name}' not found in NebulaCore registry for context provision."
                )
        return self._component_instance

    def get_current_context(self) -> dict[str, Any]:
        component = self._get_component()
        if component and hasattr(component, self.context_method_name):
            try:
                context = getattr(component, self.context_method_name)()
                if isinstance(context, dict):  # Ensure it's a dict
                    context.update(
                        {
                            "_source_component": self.component_name,
                            "_retrieval_time": time.time(),
                        }
                    )
                    return context
                else:
                    logger.warning(
                        f"Context from '{self.component_name}.{self.context_method_name}()' is not a dict: {type(context)}"
                    )
                    return {
                        "data": context,
                        "_source_component": self.component_name,
                        "_retrieval_time": time.time(),
                    }
            except Exception as e:
                logger.error(
                    f"Error retrieving context from '{self.component_name}.{self.context_method_name}()': {e}"
                )
        else:
            logger.debug(
                f"Component '{self.component_name}' or method '{self.context_method_name}' not available for context."
            )

        return {  # Minimal fallback context if component/method fails
            "error": f"Failed to retrieve context from {self.component_name}",
            "timestamp": time.time(),
            "_source_component": f"{self.component_name}_error",
        }

    def get_provider_name(self) -> str:
        return self.component_name


class ToTEngine:
    COMPONENT_NAME = "tot_engine"

    def __init__(
        self,
        vanta_echo_nebula_core: NebulaCore,
        config: ToTEngineConfig,
        thought_seeder: ThoughtSeeder,  # Mandatory
        branch_evaluator: BranchEvaluator,  # Mandatory
        branch_validator: BranchValidator,  # Mandatory
        meta_learner: MetaLearningAgent,  # Mandatory
        context_providers: list[ContextProvider] | ContextProvider | None = None,
        result_callback: Callable[[Any], None] | None = None,
        memory_braid_instance: MemoryBraidInterface | None = None,
    ):
        self.vanta_echo_nebula_core = vanta_echo_nebula_core
        self.config = config
        logger.info(
            f"ToTEngine initializing via NebulaCore. Interval: {self.config.interval_s}s"
        )

        self.seeder = thought_seeder
        self.evaluator = branch_evaluator
        self.validator = branch_validator
        self.meta_learner = meta_learner

        self.result_callback = result_callback
        self.running = False
        self.thread: threading.Thread | None = None
        self.current_phase: str | None = None
        self.last_error: str | None = None

        self.active_context_providers: list[ContextProvider] = []
        if isinstance(context_providers, list):
            self.active_context_providers.extend(context_providers)
        elif context_providers is not None:
            self.active_context_providers.append(context_providers)

        # Memory Braid: Prioritize injected, then from VantaEchoNebulaCore registry, then default
        _braid = memory_braid_instance
        if not _braid:
            _braid = self.vanta_echo_nebula_core.get_component(
                "memory_braid"
            )  # Standard name for NebulaCore
        self.memory_braid: MemoryBraidInterface = _braid or DefaultToTMemoryBraid()

        # Auto-connect to components from VantaEchoNebulaCore registry to act as ContextProviders
        if self.config.auto_connect_checkin_manager:
            self._try_add_registered_context_provider(
                self.config.checkin_manager_component_name
            )

        if self.config.auto_connect_external_inputs:
            self._try_add_registered_context_provider(
                self.config.external_inputs_component_name,
                context_method_name="get_external_inputs",
            )

        # Register self with NebulaCore
        self.vanta_echo_nebula_core.register_component(
            self.COMPONENT_NAME, self, metadata={"type": "reasoning_engine_tot"}
        )
        connected_provider_names = [
            p.get_provider_name()
            for p in self.active_context_providers
            if hasattr(p, "get_provider_name")
        ]
        logger.info(
            f"ToTEngine registered with NebulaCore. Context Providers: {connected_provider_names if connected_provider_names else 'None configured'}"
        )

    def _try_add_registered_context_provider(
        self, component_name: str, context_method_name: str = "get_current_state"
    ):
        """Attempts to get a component from VantaEchoNebulaCore registry and wrap it as a ContextProvider."""
        # Avoid adding if a provider for this component name (or similar role) already exists
        if any(
            getattr(p, "component_name", None) == component_name
            for p in self.active_context_providers
        ):
            logger.debug(
                f"Context provider for '{component_name}' seems to be already configured."
            )
            return

        component_instance = self.vanta_echo_nebula_core.get_component(component_name)
        if component_instance:
            if isinstance(
                component_instance, ContextProvider
            ):  # If it already is a ContextProvider
                self.active_context_providers.append(component_instance)
                logger.info(
                    f"Added '{component_name}' (already ContextProvider) from VantaEchoNebulaCore registry to context providers."
                )
            elif hasattr(
                component_instance, context_method_name
            ):  # If it has the required method
                adapter = NebulaRegisteredContextProvider(
                    self.vanta_echo_nebula_core, component_name, context_method_name
                )
                self.active_context_providers.append(adapter)
                logger.info(
                    f"Wrapped and added '{component_name}' from VantaEchoNebulaCore registry as a context provider."
                )
            else:
                logger.warning(
                    f"Component '{component_name}' from VantaEchoNebulaCore registry does not conform to ContextProvider or lack method '{context_method_name}'."
                )
        else:
            logger.info(
                f"Optional context provider component '{component_name}' not found in NebulaCore registry."
            )

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self._run_loop, daemon=True, name="ToTEngineLoop"
            )
            self.thread.start()
            logger.info("ToT Engine started its processing loop.")
            try:
                import asyncio
                asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                    f"{self.COMPONENT_NAME}.started",
                    {"interval_s": self.config.interval_s},
                    self.COMPONENT_NAME
                ))
            except Exception:
                pass  # Non-critical

    def stop(self) -> None:
        if self.running:
            logger.info("ToT Engine stopping...")
            self.running = False
            if self.thread and self.thread.is_alive():
                try:
                    self.thread.join(timeout=max(1.0, self.config.interval_s + 5))
                except Exception as e:
                    logger.error(f"Error joining ToT Engine thread: {e}")
            if self.thread and self.thread.is_alive():
                logger.warning("ToT Engine thread did not stop cleanly.")
            self.thread = None
            self.current_phase = None
            logger.info("ToT Engine stopped.")
            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                f"{self.COMPONENT_NAME}.stopped", {}, source=self.COMPONENT_NAME
            ))

    def _run_loop(self) -> None:
        logger.info(f"{self.COMPONENT_NAME}._run_loop started.")
        while self.running:
            cycle_start_time = time.monotonic()
            self.last_error = None
            logger.info(f"{self.COMPONENT_NAME}: Starting new ToT cycle.")

            current_context = None  # Define before try block
            task_directives_for_cycle: dict[str, Any] = {
                "max_branches": 10,
                "max_depth": 3,
            }  # Example

            try:
                self.current_phase = "ContextRetrieval"
                if not current_context or current_context.get(
                    "error"
                ):  # Check if context retrieval failed
                    logger.warning(
                        "Failed to retrieve valid context. Skipping current ToT cycle."
                    )
                    try:
                        import asyncio
                        asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                            f"{self.COMPONENT_NAME}.cycle_skipped",
                            {"reason": "context_retrieval_failed"},
                            self.COMPONENT_NAME
                        ))
                    except Exception:
                        pass  # Non-critical
                    time.sleep(self.config.interval_s)
                    continue

                self.log_operation(
                    "context_retrieved",
                    {"context_sources": current_context.get("_sources", [])},
                )

                self.current_phase = "Seed"
                logger.debug(f"Entering {self.current_phase} phase.")
                branches = self.seeder.generate(
                    current_context, task_directives_for_cycle
                )
                if not branches:
                    logger.info(
                        "Seeder generated no initial branches. Cycle might end early."
                    )  # Allow empty branches to proceed
                self.log_operation("branches_seeded", {"count": len(branches)})

                self.current_phase = "Score"
                logger.debug(f"Entering {self.current_phase} phase.")
                scores = self.evaluator.score(
                    branches, current_context, task_directives_for_cycle
                )  # branch_id -> score
                self.log_operation("branches_scored", {"count": len(scores)})

                self.current_phase = "Prune"
                logger.debug(f"Entering {self.current_phase} phase.")
                surviving_branches = self.validator.prune(
                    branches, scores, task_directives_for_cycle
                )
                if not surviving_branches:
                    logger.info(
                        "Pruned all branches. Cycle ends."
                    )  # End cycle if all pruned
                self.log_operation(
                    "branches_pruned", {"survived_count": len(surviving_branches)}
                )
                if not surviving_branches:
                    time.sleep(self.config.interval_s)
                    continue  # Skip rest of cycle

                self.current_phase = "Expand"
                logger.debug(f"Entering {self.current_phase} phase.")
                expanded_branches = self.seeder.expand(
                    surviving_branches, current_context, task_directives_for_cycle
                )
                self.log_operation(
                    "branches_expanded", {"count": len(expanded_branches)}
                )

                # Potentially re-score expanded branches before integration
                self.current_phase = "RescoreExpanded"
                logger.debug(f"Entering {self.current_phase} phase.")
                expanded_scores = self.evaluator.score(
                    expanded_branches, current_context, task_directives_for_cycle
                )  # branch_id -> score
                self.log_operation(
                    "expanded_branches_rescored", {"count": len(expanded_scores)}
                )

                self.current_phase = "Integrate"
                logger.debug(f"Entering {self.current_phase} phase.")
                integration_result = self.meta_learner.integrate(
                    expanded_branches, expanded_scores, task_directives_for_cycle
                )
                self.log_operation(
                    "integration_complete",
                    {"result_type": type(integration_result).__name__},
                )

                self.current_phase = "Publish"
                self._publish_result(integration_result)

            except Exception as e:
                self.last_error = (
                    f"Error during {self.current_phase or 'unknown'} phase: {str(e)}"
                )
                logger.exception(
                    f"ToT cycle failed during {self.current_phase or 'unknown'} phase."
                )
                asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                    f"{self.COMPONENT_NAME}.cycle_error",
                    {"phase": self.current_phase, "error": str(e)},
                    source=self.COMPONENT_NAME,
                ))
            finally:
                self.current_phase = "Idle"  # Ensure current_phase is reset
                cycle_duration = time.monotonic() - cycle_start_time
                logger.info(
                    f"{self.COMPONENT_NAME}: ToT cycle finished in {cycle_duration:.2f}s."
                )
                self.vanta_echo_nebula_core.publish_event(
                    f"{self.COMPONENT_NAME}.cycle_complete",
                    {
                        "duration_s": cycle_duration,
                        "error": self.last_error is not None,
                    },
                    source=self.COMPONENT_NAME,
                )

                wait_time = max(0.1, self.config.interval_s - cycle_duration)
                if self.running:
                    time.sleep(wait_time)
        logger.info(f"{self.COMPONENT_NAME}._run_loop finished.")

    def _get_current_context_for_cycle(self) -> dict[str, Any]:
        """Retrieves and merges context from all active context providers."""
        merged_context: dict[str, Any] = {"_timestamp": time.time(), "_sources": []}
        if not self.active_context_providers:
            logger.warning(
                "No context providers configured for ToTEngine. Using minimal context."
            )
            merged_context["_warning"] = "No context providers."
            return merged_context

        for provider in self.active_context_providers:
            try:
                provider_name = getattr(
                    provider, "get_provider_name", lambda: provider.__class__.__name__
                )()
                context_data = provider.get_current_context()
                if isinstance(context_data, dict):
                    # Merge, potentially prefixing keys from specific providers if they overlap
                    for k, v in context_data.items():
                        if k in merged_context and k not in [
                            "_timestamp",
                            "_sources",
                        ]:  # Avoid overwriting meta keys
                            merged_context[f"{provider_name}__{k}"] = v
                        else:
                            merged_context[k] = v
                    if provider_name not in merged_context["_sources"]:
                        merged_context["_sources"].append(provider_name)
                else:
                    merged_context[provider_name] = (
                        context_data  # Store as is if not dict
                    )
                    if provider_name not in merged_context["_sources"]:
                        merged_context["_sources"].append(provider_name)
            except Exception as e:
                provider_name_err = getattr(
                    provider, "get_provider_name", lambda: provider.__class__.__name__
                )()
                logger.error(
                    f"Failed to get context from provider '{provider_name_err}': {e}"
                )
                merged_context[f"_error_{provider_name_err}"] = str(e)

        if not merged_context[
            "_sources"
        ]:  # If all providers failed or none were effectively there
            merged_context["_warning"] = (
                "All context providers failed or returned no data."
            )
        return merged_context

    def _publish_result(self, result: Any) -> None:
        # (Implementation from original is fine, use NebulaCore event bus)
        if self.result_callback:
            try:
                self.result_callback(result)
                logger.debug("Published ToT result via callback.")
            except Exception as e:
                logger.error(f"Error executing result callback: {e}")
        else:
            try:
                result_str = json.dumps(
                    result,
                    default=lambda o: f"<non-serializable_type_{type(o).__name__}>",
                )
            except TypeError:
                result_str = str(result)
            logger.info(
                f"ToT cycle result (no callback, logged): {result_str[:500]}..."
            )  # Log snippet
        try:
            import asyncio
            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                f"{self.COMPONENT_NAME}.result",
                {"result_summary": str(result)[:200]},
                self.COMPONENT_NAME
            ))
        except Exception as e:
            logger.debug(f"Failed to publish result event: {e}")

    def get_status(self) -> dict[str, Any]:
        # (Implementation from original is fine)
        return {
            "engine_name": self.COMPONENT_NAME,
            "running": self.running,
            "current_phase": self.current_phase,
            "interval_s": self.config.interval_s,
            "last_error": self.last_error,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "context_providers_count": len(self.active_context_providers),
        }

    def log_operation(
        self, operation_name: str, details: Union[dict[str, Any], None] = None
    ) -> None:
        """Logs an operation using NebulaCore's event bus and standard logger for ToT specific events."""
        log_details = details or {}
        logger.debug(f"ToT Operation: '{operation_name}', Details: {log_details}")
        # Use asyncio.create_task to handle async publish_event from sync context
        try:
            import asyncio
            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                f"{self.COMPONENT_NAME}.operation.{operation_name}",
                log_details,
                self.COMPONENT_NAME
            ))
        except Exception as e:
            logger.debug(f"Failed to publish event: {e}")
        # If memory braid is used, log to it as well
        if self.memory_braid and hasattr(self.memory_braid, "store_braid_data"):
            try:
                braid_key = f"tot_op_{operation_name}_{time.time_ns()}"
                braid_data = {"phase": self.current_phase, **log_details}
                self.memory_braid.store_braid_data(braid_key, braid_data)
            except Exception as e:
                logger.warning(f"Failed to log ToT operation to memory_braid: {e}")

    def diagnose(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        # (Simplified version for ToT, CATEngine had a more elaborate one)
        self.log_operation(
            "diagnose_start", {"context_keys": list(context.keys()) if context else []}
        )
        status = self.get_status()
        health = self._calculate_health_metrics()  # Use ToT specific health
        issues = self._identify_issues(health)
        recs = self._generate_recommendations(issues)

        diag = {
            "timestamp": time.time(),
            "engine": self.COMPONENT_NAME,
            "overall_health_score": health.get("overall_score", 0.5),
            "current_phase": status.get("current_phase", "unknown"),
            "identified_issues": issues,
            "suggested_actions": recs,
            "is_running": status.get("running", False),
        }
        self.log_operation(
            "diagnose_complete", {"overall_score": diag["overall_health_score"]}
        )
        asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
            f"{self.COMPONENT_NAME}.diagnosis_results", diag, source=self.COMPONENT_NAME
        ))
        logger.info(
            f"ToT Engine Diagnosis: Score={diag['overall_health_score']:.2f}, Issues={len(issues)}"
        )
        return diag

    def _calculate_health_metrics(
        self,
    ) -> dict[str, float]:  # ToT specific health calculation
        health = {
            "overall_score": 0.75,
            "cycle_throughput": 0.8,
            "error_rate": 0.05,
            "context_quality": 0.7,
        }
        if not self.running:
            health["overall_score"] = 0.2
            health["cycle_throughput"] = 0.0
        if self.last_error:
            health["overall_score"] = 0.2
            health["error_rate"] = 0.8

        # Placeholder for context quality: if no providers, context quality is low
        if not self.active_context_providers:
            health["context_quality"] = 0.2
        # Recalculate overall score based on components
        health["overall_score"] = (
            health["cycle_throughput"] * 0.4
            + (1.0 - health["error_rate"]) * 0.4
            + health["context_quality"] * 0.2
        )
        return health

    def _identify_issues(self, health: dict[str, float]) -> list[str]:  # ToT specific
        issues = []
        if not self.running:
            issues.append("Engine is not running.")
        if self.last_error:
            issues.append(f"Last error: {self.last_error}")
        if health.get("overall_score", 1.0) < 0.5:
            issues.append("Overall health is low.")
        if health.get("cycle_throughput", 1.0) < 0.3:
            issues.append("Cycle throughput is low (engine might be stuck or slow).")
        if health.get("error_rate", 0.0) > 0.5:
            issues.append("High error rate in cycles.")
        if health.get("context_quality", 1.0) < 0.3:
            issues.append("Context quality is low, check context providers.")
        return issues

    def _generate_recommendations(self, issues: list[str]) -> list[str]:  # ToT specific
        recs = []
        if not issues:
            return ["Engine healthy. Monitor performance."]
        if "Engine is not running." in issues:
            recs.append("Try restarting the ToTEngine.")
        if any("Last error" in i for i in issues):
            recs.append("Review engine logs for error details.")
        if any("Overall health" in i for i in issues):
            recs.append("Diagnose specialist agents and context providers.")
        if any("throughput" in i for i in issues):
            recs.append(
                "Check specialist agent performance or increase cycle interval."
            )
        if any("Context quality" in i for i in issues):
            recs.append(
                "Verify NebulaCore context provider components (e.g., 'checkin_manager', 'external_inputs_bridge') are registered and functional."
            )
        return recs if recs else ["Address identified issues based on logs."]
        # --- Production-grade additions / utilities ---

        def _safe_call(self, fn: Callable[..., Any], *args, **kwargs) -> tuple[bool, Any]:
            """
            Execute fn with args/kwargs catching exceptions and returning (ok, result_or_exception).
            Also logs and publishes the failure to NebulaCore event bus.
            """
            try:
                result = fn(*args, **kwargs)
                return True, result
            except Exception as exc:
                err = str(exc)
                logger.exception(f"Safe call failure in {fn.__name__}: {err}")
                try:
                    asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                        f"{self.COMPONENT_NAME}.internal_error",
                        {"function": fn.__name__, "error": err},
                        source=self.COMPONENT_NAME,
                    ))
                except Exception:
                    logger.debug("Failed to publish internal_error event.")
                return False, exc

        def run_once(self, task_directives_override: dict[str, Any] | None = None) -> dict[str, Any]:
            """
            Execute a single ToT cycle synchronously. Returns a structured report useful for tests and telemetry.
            This method consolidates safe execution and ensures context retrieval is performed.
            """
            report: dict[str, Any] = {"timestamp": time.time(), "phase_timings": {}, "error": None}
            cycle_start = time.monotonic()

            task_directives = {
                "max_branches": 10,
                "max_depth": 3,
            }
            if task_directives_override:
                task_directives.update(task_directives_override)

            try:
                self.current_phase = "ContextRetrieval"
                ctx_ok, current_context_or_err = self._safe_call(self._get_current_context_for_cycle)
                if not ctx_ok:
                    report["error"] = f"context_retrieval_error: {current_context_or_err}"
                    self.log_operation("cycle_skipped", {"reason": "context_retrieval_failed"})
                    return report
                current_context = current_context_or_err
                if not current_context or current_context.get("error"):
                    report["error"] = "invalid_context"
                    self.log_operation("cycle_skipped", {"reason": "invalid_context"})
                    return report

                t0 = time.monotonic()
                self.current_phase = "Seed"
                ok, branches_or_exc = self._safe_call(self.seeder.generate, current_context, task_directives)
                if not ok:
                    raise branches_or_exc
                branches = branches_or_exc or []
                report["phase_timings"]["seed"] = time.monotonic() - t0
                self.log_operation("branches_seeded", {"count": len(branches)})

                t0 = time.monotonic()
                self.current_phase = "Score"
                ok, scores_or_exc = self._safe_call(self.evaluator.score, branches, current_context, task_directives)
                if not ok:
                    raise scores_or_exc
                scores = scores_or_exc or {}
                report["phase_timings"]["score"] = time.monotonic() - t0
                self.log_operation("branches_scored", {"count": len(scores)})

                t0 = time.monotonic()
                self.current_phase = "Prune"
                ok, survivors_or_exc = self._safe_call(self.validator.prune, branches, scores, task_directives)
                if not ok:
                    raise survivors_or_exc
                surviving_branches = survivors_or_exc or []
                report["phase_timings"]["prune"] = time.monotonic() - t0
                self.log_operation("branches_pruned", {"survived_count": len(surviving_branches)})
                if not surviving_branches:
                    report["result_summary"] = "all_pruned"
                    return report

                t0 = time.monotonic()
                self.current_phase = "Expand"
                ok, expanded_or_exc = self._safe_call(self.seeder.expand, surviving_branches, current_context, task_directives)
                if not ok:
                    raise expanded_or_exc
                expanded_branches = expanded_or_exc or []
                report["phase_timings"]["expand"] = time.monotonic() - t0
                self.log_operation("branches_expanded", {"count": len(expanded_branches)})

                t0 = time.monotonic()
                self.current_phase = "RescoreExpanded"
                ok, expanded_scores_or_exc = self._safe_call(self.evaluator.score, expanded_branches, current_context, task_directives)
                if not ok:
                    raise expanded_scores_or_exc
                expanded_scores = expanded_scores_or_exc or {}
                report["phase_timings"]["rescore"] = time.monotonic() - t0
                self.log_operation("expanded_branches_rescored", {"count": len(expanded_scores)})

                t0 = time.monotonic()
                self.current_phase = "Integrate"
                ok, integration_or_exc = self._safe_call(self.meta_learner.integrate, expanded_branches, expanded_scores, task_directives)
                if not ok:
                    raise integration_or_exc
                integration_result = integration_or_exc
                report["phase_timings"]["integrate"] = time.monotonic() - t0
                self.log_operation("integration_complete", {"result_type": type(integration_result).__name__})

                t0 = time.monotonic()
                self.current_phase = "Publish"
                self._publish_result(integration_result)
                report["phase_timings"]["publish"] = time.monotonic() - t0

                report["result"] = integration_result
                report["result_summary"] = str(integration_result)[:500]

                # Persist cycle summary to memory braid for audit/traceability
                try:
                    braid_key = f"tot_cycle_{time.time_ns()}"
                    braid_payload = {
                        "report": {k: v for k, v in report.items() if k != "result"},
                        "result_type": type(integration_result).__name__,
                        "context_sources": current_context.get("_sources", []),
                    }
                    if hasattr(self.memory_braid, "store_braid_data"):
                        self.memory_braid.store_braid_data(braid_key, braid_payload)
                except Exception as e:
                    logger.warning(f"Failed to persist ToT cycle summary to memory braid: {e}")

                return report
            except Exception as e:
                err = str(e)
                report["error"] = err
                logger.exception(f"run_once failed during phase {self.current_phase}: {err}")
                try:
                    asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                        f"{self.COMPONENT_NAME}.run_once_error",
                        {"phase": self.current_phase, "error": err},
                        source=self.COMPONENT_NAME,
                    ))
                except Exception:
                    logger.debug("Failed to publish run_once_error event.")
                return report
            finally:
                self.current_phase = "Idle"
                report["duration_s"] = time.monotonic() - cycle_start
                self.log_operation("cycle_complete", {"duration_s": report["duration_s"], "error": report.get("error") is not None})

        async def start_async(self, loop_interval: float | None = None) -> None:
            """
            Async-friendly runner: will repeatedly run run_once inside the current asyncio loop.
            Use this instead of start() when integrating into an async application event loop.
            """
            if loop_interval is None:
                loop_interval = float(self.config.interval_s)
            if self.running:
                logger.info("ToTEngine already running.")
                return
            self.running = True
            try:
                import asyncio
                asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                    f"{self.COMPONENT_NAME}.started_async",
                    {"interval_s": loop_interval},
                    self.COMPONENT_NAME
                ))
            except Exception:
                pass  # Non-critical
            logger.info("ToTEngine async loop started.")
            try:
                while self.running:
                    cycle_start = time.monotonic()
                    await self.vanta_echo_nebula_core.loop.run_in_executor(None, self.run_once)
                    elapsed = time.monotonic() - cycle_start
                    await asyncio.sleep(max(0.01, loop_interval - elapsed))
            except asyncio.CancelledError:
                logger.info("ToTEngine async loop cancelled.")
            except Exception as e:
                logger.exception(f"Unexpected error in async loop: {e}")
            finally:
                self.running = False
                try:
                    import asyncio
                    asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                        f"{self.COMPONENT_NAME}.stopped_async",
                        {},
                        self.COMPONENT_NAME
                    ))
                except Exception:
                    pass  # Non-critical
                logger.info("ToTEngine async loop stopped.")

        def save_state(self, key: str | None = None) -> dict[str, Any]:
            """
            Save minimal runtime state to memory braid or NebulaCore persistent store.
            Returns the stored metadata.
            """
            key = key or f"{self.COMPONENT_NAME}_state_{int(time.time())}"
            state = {
                "running": self.running,
                "current_phase": self.current_phase,
                "last_error": self.last_error,
                "timestamp": time.time(),
                "context_providers": [getattr(p, "get_provider_name", lambda: p.__class__.__name__)() for p in self.active_context_providers],
            }
            try:
                if hasattr(self.memory_braid, "store_braid_data"):
                    self.memory_braid.store_braid_data(key, state, metadata={"type": "engine_state"})
                else:
                    asyncio.create_task(self.vanta_echo_nebula_core.publish_event(f"{self.COMPONENT_NAME}.state_saved", {"key": key}, source=self.COMPONENT_NAME))
                return {"key": key, "stored": True, "state": state}
            except Exception as e:
                logger.exception(f"Failed to save state: {e}")
                return {"key": key, "stored": False, "error": str(e)}

        def load_state(self, key: str) -> dict[str, Any] | None:
            """
            Load saved state from memory braid if available. This is non-destructive and best-effort.
            """
            try:
                if hasattr(self.memory_braid, "retrieve_braid_data"):
                    data = self.memory_braid.retrieve_braid_data(key)
                    if isinstance(data, dict):
                        # Only restore safe runtime fields
                        self.current_phase = data.get("current_phase", self.current_phase)
                        self.last_error = data.get("last_error", self.last_error)
                        # Do not auto-start running state for safety
                    return data
                else:
                    logger.warning("No memory_braid retrieval available to load state.")
                    return None
            except Exception as e:
                logger.exception(f"Failed to load state {key}: {e}")
                return None

        def start_watchdog(self, interval_s: int = 30) -> None:
            """
            Lightweight watchdog that logs and publishes if the ToT worker thread isn't alive.
            It does not attempt to forcibly restart the thread, only notifies operators.
            """
            def _watchdog_loop():
                logger.info("ToTEngine watchdog started.")
                while self.running:
                    try:
                        thread_alive = self.thread.is_alive() if self.thread else False
                        if not thread_alive:
                            logger.warning("ToTEngine worker thread is not alive.")
                            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(f"{self.COMPONENT_NAME}.watchdog.thread_dead", {}, source=self.COMPONENT_NAME))
                        time.sleep(max(1, interval_s))
                    except Exception as e:
                        logger.exception(f"Watchdog encountered error: {e}")
                        time.sleep(max(1, interval_s))
                logger.info("ToTEngine watchdog stopped.")

            wd = threading.Thread(target=_watchdog_loop, name="ToTWatchdog", daemon=True)
            wd.start()
            # store for introspection if needed
            self._watchdog_thread = wd