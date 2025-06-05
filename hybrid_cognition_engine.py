# File: vanta_hybrid_cognition_engine.py (Refactored from MetaConsciousness/agent/vanta/hybrid_cognition_engine.py)
"""
Hybrid Cognition Engine Module for VantaCore

Fuses capabilities of Tree-of-Thought (ToTEngine) and Categorize-Analyze-Test (CATEngine)
for advanced cognitive reasoning, adapted for the VantaCore framework.
"""

import logging
import random
import threading
import time
import uuid
from typing import Any, Callable  # Added Callable and Protocol

# Assuming vanta_core.py, vanta_tot_engine.py, vanta_cat_engine.py are accessible
from Vanta.core.UnifiedVantaCore import (
    UnifiedVantaCore as VantaCore,
)  # VantaCore integration

from .cat_engine import CATEngine  # Use the VantaCore-adapted CATEngine
from .tot_engine import ToTEngine  # Use the VantaCore-adapted ToTEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaCore.HybridCognition")  # Logger for this engine


# --- Configuration Class ---
class HybridCognitionConfig:
    def __init__(
        self,
        interval_s: int = 300,
        fusion_mode: str = "parallel",  # "parallel", "sequential", "adaptive"
        log_level: str = "INFO",
        # Configs for sub-engine interactions, if HybridCognition drives them explicitly
        tot_cycle_timeout_factor: float = 0.8,
        cat_cycle_timeout_factor: float = 0.8,
    ):
        if not isinstance(interval_s, int) or interval_s <= 0:
            logger.warning(
                f"Invalid HybridCognition interval_s: {interval_s}. Defaulting to 300."
            )
            self.interval_s: int = 300
        else:
            self.interval_s: int = interval_s

        if fusion_mode not in ["parallel", "sequential", "adaptive"]:
            logger.warning(
                f"Invalid fusion_mode: {fusion_mode}. Defaulting to parallel."
            )
            self.fusion_mode: str = "parallel"
        else:
            self.fusion_mode: str = fusion_mode

        self.log_level: str = log_level
        self.tot_cycle_timeout_factor = tot_cycle_timeout_factor
        self.cat_cycle_timeout_factor = cat_cycle_timeout_factor

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Hybrid Thought Branch Data Structure ---
class HybridThoughtBranch:
    def __init__(
        self,
        branch_id: str | None = None,  # Allow None to auto-generate
        content: str = "",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ):
        self.branch_id: str = branch_id or f"hybrid_branch_{uuid.uuid4().hex[:8]}"
        self.content: str = content
        self.confidence: float = max(0.0, min(1.0, confidence))
        self.metadata: dict[str, Any] = metadata or {}
        self.creation_time: float = time.time()
        self.last_updated: float = time.time()
        self.category_tags: set[str] = set()
        self.uncertainty_metrics: dict[str, float] = {}
        self.test_results: list[dict[str, Any]] = []
        self.sub_branches: list[str] = []  # List of branch_ids
        self.parent_branch_id: str | None = None
        self.connected_to: list[str] = []  # Not used in original but kept for structure

    def update_confidence(self, new_confidence: float) -> None:
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.last_updated = time.time()

    def add_test_result(self, test_data: dict[str, Any]) -> None:
        self.test_results.append(test_data)
        self.last_updated = time.time()

    def add_category_tag(self, tag: str) -> None:
        self.category_tags.add(tag)
        self.last_updated = time.time()

    def update_uncertainty(self, metric_name: str, value: float) -> None:
        self.uncertainty_metrics[metric_name] = value
        self.last_updated = time.time()

    def add_sub_branch(self, branch_id: str) -> None:
        if branch_id not in self.sub_branches:
            self.sub_branches.append(branch_id)
        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, set) else v) for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridThoughtBranch":
        branch = cls(
            branch_id=data.get("branch_id"),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.5),
            metadata=data.get("metadata", {}),
        )
        for key, value in data.items():
            if key == "category_tags" and isinstance(value, list):
                setattr(branch, key, set(value))
            elif hasattr(branch, key) and key not in [
                "branch_id",
                "content",
                "confidence",
                "metadata",
            ]:
                setattr(branch, key, value)
        return branch


# --- Hybrid Cognition Engine ---
class HybridCognitionEngine:
    COMPONENT_NAME = "hybrid_cognition_engine"

    def __init__(
        self,
        vanta_core: VantaCore,  # Mandatory VantaCore instance
        config: HybridCognitionConfig,
        tot_engine_instance: ToTEngine,  # Must be VantaCore-adapted ToTEngine
        cat_engine_instance: CATEngine,  # Must be VantaCore-adapted CATEngine
        result_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.vanta_core = vanta_core
        self.config = config
        logger.info(
            f"HybridCognitionEngine initializing via VantaCore. Fusion Mode: {self.config.fusion_mode}, Interval: {self.config.interval_s}s"
        )

        if not isinstance(tot_engine_instance, ToTEngine):
            raise TypeError(
                "tot_engine_instance must be an instance of VantaCore-adapted ToTEngine."
            )
        if not isinstance(cat_engine_instance, CATEngine):
            raise TypeError(
                "cat_engine_instance must be an instance of VantaCore-adapted CATEngine."
            )

        self.tot_engine = tot_engine_instance
        self.cat_engine = cat_engine_instance

        self.result_callback = result_callback

        self.branches: dict[str, HybridThoughtBranch] = {}
        self.active_branch_id: str | None = None
        self.branch_categories: dict[str, list[str]] = {}
        self.contradiction_mappings: dict[str, dict[str, float]] = {}

        self.running = False
        self.thread: threading.Thread | None = None
        self.current_phase: str | None = None
        self.last_error: str | None = None
        self.cycle_count: int = 0

        self._last_tot_result: dict[
            str, Any
        ] = {}  # Store last results from sub-engines
        self._last_cat_result: dict[str, Any] = {}

        # Register with VantaCore
        self.vanta_core.register_component(
            self.COMPONENT_NAME, self, metadata={"type": "cognitive_fusion_engine"}
        )
        logger.info("HybridCognitionEngine instance registered with VantaCore.")

    def start(self) -> None:
        if not self.running:
            # Ensure sub-engines are started if they have their own loops
            if hasattr(self.tot_engine, "start") and not self.tot_engine.running:
                self.tot_engine.start()
            if hasattr(self.cat_engine, "start") and not self.cat_engine.running:
                self.cat_engine.start()

            self.running = True
            self.thread = threading.Thread(
                target=self._run_fusion_loop, daemon=True, name="HybridCognitionLoop"
            )
            self.thread.start()
            logger.info("HybridCognitionEngine started its fusion loop.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.started",
                {"fusion_mode": self.config.fusion_mode},
                source=self.COMPONENT_NAME,
            )

    def stop(self) -> None:
        if self.running:
            logger.info("HybridCognitionEngine stopping...")
            self.running = False
            if self.thread and self.thread.is_alive():
                try:
                    self.thread.join(
                        timeout=max(1.0, self.config.interval_s + 5)
                    )  # Wait longer
                except Exception as e:
                    logger.error(f"Error joining HybridCognitionLoop thread: {e}")
            if self.thread and self.thread.is_alive():
                logger.warning("HybridCognitionLoop thread did not stop cleanly.")
            self.thread = None
            self.current_phase = None

            # Stop sub-engines if they were started by this engine (or manage externally)
            if hasattr(self.tot_engine, "stop") and self.tot_engine.running:
                self.tot_engine.stop()
            if hasattr(self.cat_engine, "stop") and self.cat_engine.running:
                self.cat_engine.stop()

            logger.info("HybridCognitionEngine stopped.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.stopped", {}, source=self.COMPONENT_NAME
            )

    def _run_fusion_loop(self) -> None:
        logger.info(f"{self.COMPONENT_NAME}._run_fusion_loop started.")
        while self.running:
            cycle_start_time = time.monotonic()
            self.cycle_count += 1
            self.last_error = None
            logger.info(
                f"{self.COMPONENT_NAME}: Starting fusion cycle #{self.cycle_count}"
            )

            try:
                if self.config.fusion_mode == "parallel":
                    self._run_parallel_fusion()
                elif self.config.fusion_mode == "sequential":
                    self._run_sequential_fusion()
                elif self.config.fusion_mode == "adaptive":
                    self._run_adaptive_fusion()
                else:
                    logger.warning(
                        f"Unknown fusion mode '{self.config.fusion_mode}', defaulting to parallel."
                    )
                    self._run_parallel_fusion()

                if not self.running:
                    break  # Check if stopped during fusion phase

                self.current_phase = "ResultIntegration"
                fusion_output = self._integrate_engine_results()
                if self.result_callback:
                    self.result_callback(fusion_output)
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.fusion_result",
                    {"fusion_id": fusion_output.get("fusion_id")},
                    source=self.COMPONENT_NAME,
                )

            except Exception as e:
                self.last_error = (
                    f"Error during {self.current_phase or 'unknown'} phase: {str(e)}"
                )
                logger.exception(
                    f"Fusion cycle failed during {self.current_phase or 'unknown'} phase"
                )
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.cycle_error",
                    {"phase": self.current_phase, "error": str(e)},
                    source=self.COMPONENT_NAME,
                )
            finally:
                self.current_phase = "Idle"
                cycle_duration = time.monotonic() - cycle_start_time
                logger.info(
                    f"{self.COMPONENT_NAME}: Fusion cycle #{self.cycle_count} finished in {cycle_duration:.2f}s. Last error: {self.last_error or 'None'}"
                )
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.cycle_complete",
                    {
                        "cycle_num": self.cycle_count,
                        "duration_s": cycle_duration,
                        "error": self.last_error is not None,
                    },
                    source=self.COMPONENT_NAME,
                )

                wait_time = max(0.1, self.config.interval_s - cycle_duration)
                if self.running:
                    time.sleep(wait_time)
        logger.info(f"{self.COMPONENT_NAME}._run_fusion_loop finished.")

    def _run_parallel_fusion(self) -> None:
        logger.debug("Parallel fusion: Triggering/getting ToT and CAT states.")
        self.current_phase = "ParallelFusion_ToT"
        self._last_tot_result = self._execute_tot_cycle()  # Gets current state/output
        if not self.running:
            return

        self.current_phase = "ParallelFusion_CAT"
        self._last_cat_result = self._execute_cat_cycle()  # Gets current state/output
        logger.debug("Parallel fusion: ToT and CAT states obtained.")

    def _run_sequential_fusion(self) -> None:
        logger.debug("Sequential fusion: ToT -> CAT.")
        self.current_phase = "SequentialFusion_ToT"
        tot_output = self._execute_tot_cycle()
        self._last_tot_result = tot_output
        if not self.running:
            return

        self.current_phase = "SequentialFusion_CAT"
        # Pass tot_output as potential input to CAT cycle
        self._last_cat_result = self._execute_cat_cycle(tot_input=tot_output)
        logger.debug("Sequential fusion: ToT and CAT cycles executed.")

    def _run_adaptive_fusion(self) -> None:
        logger.debug("Adaptive fusion: Deciding execution order.")
        # Simplified adaptive strategy: alternate or use placeholder metric
        avg_uncertainty = self._calculate_average_branch_uncertainty()
        if (
            self.cycle_count % 2 == 0 or avg_uncertainty < 0.5
        ):  # Example: if low uncertainty or even cycle, run ToT first
            logger.info(
                f"Adaptive: ToT first (cycle {self.cycle_count}, uncertainty {avg_uncertainty:.2f})"
            )
            self.current_phase = "AdaptiveFusion_ToT"
            tot_output = self._execute_tot_cycle()
            self._last_tot_result = tot_output
            if not self.running:
                return
            self.current_phase = "AdaptiveFusion_CAT"
            self._last_cat_result = self._execute_cat_cycle(tot_input=tot_output)
        else:  # High uncertainty or odd cycle, run CAT first
            logger.info(
                f"Adaptive: CAT first (cycle {self.cycle_count}, uncertainty {avg_uncertainty:.2f})"
            )
            self.current_phase = "AdaptiveFusion_CAT"
            cat_output = self._execute_cat_cycle()
            self._last_cat_result = cat_output
            if not self.running:
                return
            self.current_phase = "AdaptiveFusion_ToT"
            self._last_tot_result = self._execute_tot_cycle(cat_input=cat_output)
        logger.debug("Adaptive fusion: Execution complete based on strategy.")

    def _calculate_average_branch_uncertainty(self) -> float:
        if not self.branches:
            return 0.5  # Default if no branches
        total_uncertainty_sum = 0.0
        metric_count = 0
        for branch in self.branches.values():
            for u_val in branch.uncertainty_metrics.values():
                total_uncertainty_sum += u_val
                metric_count += 1
        return total_uncertainty_sum / metric_count if metric_count > 0 else 0.5

    def _execute_tot_cycle(
        self, cat_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # If ToTEngine runs its own loop, this method primarily gets its latest state/output.
        # If HybridCognition needs to *drive* ToT synchronously, ToTEngine would need a
        # `process_one_cycle(input_context, directives)` method.
        # Assuming ToT runs independently for now, just get its status.
        logger.debug(
            f"Executing ToT cycle (or getting status). CAT input provided: {cat_input is not None}"
        )
        if (
            hasattr(self.tot_engine, "diagnose") and cat_input
        ):  # Use CAT input to inform ToT diagnosis/next step
            return self.tot_engine.diagnose(
                context=cat_input
            )  # Example of passing data
        return (
            self.tot_engine.get_status()
            if hasattr(self.tot_engine, "get_status")
            else {"status": "unknown_tot"}
        )

    def _execute_cat_cycle(
        self, tot_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        logger.debug(
            f"Executing CAT cycle (or getting status). ToT input provided: {tot_input is not None}"
        )
        if hasattr(self.cat_engine, "diagnose") and tot_input:
            return self.cat_engine.diagnose(context=tot_input)
        return (
            self.cat_engine.get_status()
            if hasattr(self.cat_engine, "get_status")
            else {"status": "unknown_cat"}
        )

    def _integrate_engine_results(self) -> dict[str, Any]:
        logger.debug("Integrating ToT and CAT results...")
        tot_summary = self._last_tot_result or {}
        cat_summary = self._last_cat_result or {}

        fusion_id = f"hybrid_fusion_{uuid.uuid4().hex[:8]}"

        # Example: Create a new HybridThoughtBranch based on dominant output or combined insights
        # This part is highly dependent on the actual outputs of ToT and CAT
        # For now, let's create a summary branch

        new_branch_content = f"Fusion Cycle {self.cycle_count}: ToT insights (phase {tot_summary.get('current_phase', 'N/A')}), CAT insights (phase {cat_summary.get('current_phase', 'N/A')})."

        # Confidence could be an average or weighted score
        tot_health = tot_summary.get(
            "overall_health_score", tot_summary.get("overall_score", 0.5)
        )  # Handle diff status keys
        cat_health = cat_summary.get(
            "overall_health_score", cat_summary.get("overall_score", 0.5)
        )
        hybrid_confidence = (tot_health + cat_health) / 2.0

        new_branch = HybridThoughtBranch(
            content=new_branch_content,
            confidence=hybrid_confidence,
            metadata={
                "fusion_id": fusion_id,
                "tot_summary": tot_summary,
                "cat_summary": cat_summary,
            },
        )

        # Add tags based on sub-engine states or outputs
        if tot_summary.get("current_phase") == "Integrate":
            new_branch.add_category_tag("ToT_Integrated")
        if cat_summary.get("current_phase") == "Test":
            new_branch.add_category_tag("CAT_Tested")

        self.branches[new_branch.branch_id] = new_branch
        self.active_branch_id = (
            new_branch.branch_id
        )  # Mark this new fusion as active for now

        logger.info(
            f"Integration complete. New hybrid branch '{new_branch.branch_id}' created with confidence {hybrid_confidence:.2f}."
        )

        # Publish VantaCore event for the new hybrid branch
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.new_hybrid_branch",
            new_branch.to_dict(),
            source=self.COMPONENT_NAME,
        )

        return {
            "fusion_id": fusion_id,
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "active_branch_id": self.active_branch_id,
            "branch_count": len(self.branches),
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "engine_name": self.COMPONENT_NAME,
            "running": self.running,
            "current_phase": self.current_phase,
            "cycle_count": self.cycle_count,
            "fusion_mode": self.config.fusion_mode,
            "tot_status_brief": (
                self.tot_engine.get_status()
                if hasattr(self.tot_engine, "get_status")
                else {}
            ).get("current_phase", "N/A"),
            "cat_status_brief": (
                self.cat_engine.get_status()
                if hasattr(self.cat_engine, "get_status")
                else {}
            ).get("current_phase", "N/A"),
            "active_hybrid_branch_id": self.active_branch_id,
            "hybrid_branch_count": len(self.branches),
            "last_error": self.last_error,
            "thread_alive": self.thread.is_alive() if self.thread else False,
        }

    def get_all_branches(self) -> dict[str, dict[str, Any]]:
        return {bid: b.to_dict() for bid, b in self.branches.items()}

    def get_branch(self, branch_id: str) -> dict[str, Any] | None:
        b = self.branches.get(branch_id)
        return b.to_dict() if b else None

    def get_branches_by_category(self, category: str) -> list[dict[str, Any]]:
        return [
            b.to_dict() for b in self.branches.values() if category in b.category_tags
        ]

    def export_state(self) -> dict[str, Any]:
        return {
            "branches": self.get_all_branches(),
            "active_branch_id": self.active_branch_id,
            "cycle_count": self.cycle_count,
            "fusion_mode": self.config.fusion_mode,
            "timestamp": time.time(),
            "version": "1.0_vanta",
        }

    def import_state(self, state: dict[str, Any]) -> bool:
        try:
            self.branches = {
                bid: HybridThoughtBranch.from_dict(bdata)
                for bid, bdata in state.get("branches", {}).items()
            }
            self.active_branch_id = state.get("active_branch_id")
            self.cycle_count = state.get("cycle_count", 0)
            self.fusion_mode = state.get(
                "fusion_mode", self.config.fusion_mode
            )  # Use current config if not in state
            logger.info(
                f"Imported HybridCognitionEngine state. Branches: {len(self.branches)}."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False

    def diagnose(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.current_phase = "HybridDiagnose"
        logger.info(f"{self.COMPONENT_NAME}: Running diagnosis.")
        # Removed MetaConsciousness trace event, VantaCore publish_event for tracing.
        self.vanta_core.publish_event(
            event_type=f"{self.COMPONENT_NAME}.diagnose.start",
            data={"context_provided": context is not None},
            source=self.COMPONENT_NAME,
        )

        tot_diagnosis = (
            self.tot_engine.diagnose(context)
            if hasattr(self.tot_engine, "diagnose")
            else {}
        )
        cat_diagnosis = (
            self.cat_engine.diagnose(context)
            if hasattr(self.cat_engine, "diagnose")
            else {}
        )

        # Simplified hybrid health assessment
        hybrid_health_score = (
            tot_diagnosis.get(
                "overall_health_score", tot_diagnosis.get("overall_score", 0.5)
            )
            + cat_diagnosis.get(
                "overall_health_score", cat_diagnosis.get("overall_score", 0.5)
            )
        ) / 2.0

        issues = tot_diagnosis.get(
            "identified_issues", tot_diagnosis.get("issues", [])
        ) + cat_diagnosis.get("identified_issues", cat_diagnosis.get("issues", []))
        if hybrid_health_score < 0.5 and "Overall hybrid health is low." not in issues:
            issues.append("Overall hybrid health is low.")

        recs = tot_diagnosis.get(
            "suggested_actions", tot_diagnosis.get("recommendations", [])
        ) + cat_diagnosis.get(
            "suggested_actions", cat_diagnosis.get("recommendations", [])
        )
        if not recs:
            recs.append("Monitor overall system health and sub-engine diagnostics.")

        diagnosis = {
            "timestamp": time.time(),
            "engine": self.COMPONENT_NAME,
            "overall_hybrid_health_score": hybrid_health_score,
            "sub_engine_diagnostics": {
                "tot_engine": tot_diagnosis,
                "cat_engine": cat_diagnosis,
            },
            "identified_issues_hybrid": list(set(issues)),  # Unique issues
            "suggested_actions_hybrid": list(set(recs)),  # Unique recs
            "active_hybrid_branches": len(self.branches),
        }

        # Removed MetaConsciousness heartbeat, replaced with VantaCore event.
        heartbeat_status = "normal"
        if hybrid_health_score < 0.3:
            heartbeat_status = "error"
        elif hybrid_health_score < 0.6:
            heartbeat_status = "warning"

        self.vanta_core.publish_event(
            event_type=f"{self.COMPONENT_NAME}.heartbeat",
            data={
                "status": heartbeat_status,
                "pulse_value": hybrid_health_score,
                "branch_count": len(self.branches),
                "active_branch_id": self.active_branch_id,
                "fusion_interval_s": self.config.interval_s,
                "issues_count": len(issues),
            },
            source=self.COMPONENT_NAME,
        )
        logger.info(
            f"Hybrid Diagnosis complete. Overall Score: {hybrid_health_score:.2f}"
        )
        self.current_phase = "Idle"
        return diagnosis

    # Internal health/issue/recommendation stubs specific to HybridCognition if needed
    # (Using sub-engine diagnostics for now)


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    main_logger = logging.getLogger("HybridCogExample")
    main_logger.setLevel(logging.DEBUG)
    main_logger.info("--- Starting Hybrid Cognition Engine VantaCore Example ---")

    # 1. Initialize VantaCore
    vanta_system = VantaCore()

    # 2. Create Configs for sub-engines and hybrid engine
    from .cat_engine import CATEngineConfig  # Assuming these are now importable
    from .tot_engine import (
        BranchEvaluator,
        BranchValidator,
        ContextProvider,
        MetaLearningAgent,
        ThoughtSeeder,
        ToTEngineConfig,
    )  # For ToT mock agents

    cat_config_inst = CATEngineConfig(
        interval_s=7, log_level="DEBUG"
    )  # Quick cycles for demo
    tot_config_inst = ToTEngineConfig(interval_s=7, log_level="DEBUG")
    hybrid_config_inst = HybridCognitionConfig(
        interval_s=15, fusion_mode="adaptive", log_level="DEBUG"
    )

    # 3. Create mock/default instances for ToT specialist agents (as ToTEngine requires them)
    class MockToTSeeder(ThoughtSeeder):
        def generate(self, c, td=None):
            return [
                {"id": f"seed_{uuid.uuid4().hex[:4]}", "content": "initial ToT idea"}
                for _ in range(2)
            ]

        def expand(self, b, c, td=None):
            return (
                b
                + [
                    {
                        "id": f"exp_{uuid.uuid4().hex[:4]}",
                        "content": "expanded ToT idea",
                    }
                ]
                if b
                else []
            )

    class MockToTEvaluator(BranchEvaluator):
        def score(self, b, c, td=None):
            return {
                item.get("id", "X"): random.random()
                for item in b
                if isinstance(item, dict)
            }

    class MockToTValidator(BranchValidator):
        def prune(self, b, s, td=None):
            return [
                item
                for item in b
                if isinstance(item, dict) and s.get(item.get("id", "X"), 0) > 0.3
            ]

    class MockToTMetaLearner(MetaLearningAgent):
        def integrate(self, b, s, td=None):
            return {
                "final_tot_summary": "integrated ToT results",
                "best_branch": b[0] if b else None,
            }

    class MockContextProvider(ContextProvider):
        def get_current_context(self):
            return {
                "source_example_context_provider": True,
                "random_val": random.random(),
            }

        def get_provider_name(self):
            return "MockContextProvider"

    # 4. Instantiate ToTEngine and CATEngine (they will use internal defaults if not all deps are given)
    #    VantaCore instance is passed to them.
    #    They will register themselves with vanta_system.

    # Explicitly create and register mock dependencies for ToT to ensure it finds them
    # (since ToTEngine's default creation for these mandatory agents might not exist)
    vanta_system.register_component("default_thought_seeder", MockToTSeeder())
    vanta_system.register_component("default_branch_evaluator", MockToTEvaluator())
    # ... (need to decide if ToT creates these if not in registry or if they are mandatory)
    # For now, assuming ToTEngine __init__ parameters are mandatory for specialists.

    tot_engine_inst = ToTEngine(
        vanta_core=vanta_system,
        config=tot_config_inst,
        thought_seeder=MockToTSeeder(),
        branch_evaluator=MockToTEvaluator(),
        branch_validator=MockToTValidator(),
        meta_learner=MockToTMetaLearner(),
        context_providers=[MockContextProvider()],
    )
    cat_engine_inst = CATEngine(
        vanta_core=vanta_system, config=cat_config_inst
    )  # Will use its defaults

    # 5. Instantiate HybridCognitionEngine
    hybrid_engine_inst = HybridCognitionEngine(
        vanta_core=vanta_system,
        config=hybrid_config_inst,
        tot_engine_instance=tot_engine_inst,
        cat_engine_instance=cat_engine_inst,
        result_callback=lambda res: main_logger.info(
            f"HybridCallback: Fusion ID {res.get('fusion_id')}, Active Branch: {res.get('active_branch_id')}"
        ),
    )

    # 6. Start the Hybrid Engine (which should start its sub-engines)
    hybrid_engine_inst.start()

    try:
        main_logger.info(
            "Hybrid Cognition Engine running. Test interval is short. Ctrl+C to stop."
        )
        for i in range(3):  # Let it run for a few fusion cycles
            time.sleep(hybrid_config_inst.interval_s + 3)
            status = hybrid_engine_inst.get_status()
            main_logger.info(
                f"HYBRID STATUS Cycle ~{i + 1}: Phase='{status.get('current_phase')}', Branches='{status.get('hybrid_branch_count')}'"
            )
            if not hybrid_engine_inst.running:
                main_logger.warning("Hybrid Engine stopped prematurely.")
                break

        hybrid_diag = hybrid_engine_inst.diagnose()
        main_logger.info(
            f"Hybrid Diagnosis: Score={hybrid_diag.get('overall_hybrid_health_score', 0.0):.2f}, Issues: {hybrid_diag.get('identified_issues_hybrid')}"
        )

        # Show some branches if any were created
        all_hybrid_branches = hybrid_engine_inst.get_all_branches()
        main_logger.info(f"Total Hybrid Branches: {len(all_hybrid_branches)}")
        for b_id, b_data in list(all_hybrid_branches.items())[
            :2
        ]:  # Print first 2 branches
            main_logger.info(
                f"  Branch {b_id}: {b_data.get('content', '')[:50]}... (Conf: {b_data.get('confidence', 0):.2f})"
            )

    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt by user.")
    finally:
        main_logger.info("Attempting to stop Hybrid Cognition Engine.")
        hybrid_engine_inst.stop()
        # ToTEngine and CATEngine should be stopped by HybridCognitionEngine's stop method.
        # vanta_system.shutdown() # If VantaCore needs explicit shutdown
        main_logger.info("--- Hybrid Cognition Engine VantaCore Example Finished ---")
