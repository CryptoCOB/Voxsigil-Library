# File: vanta_hybrid_cognition_engine.py (Refactored from MetaConsciousness/agent/vanta/hybrid_cognition_engine.py)
"""
Hybrid Cognition Engine Module for NebulaCore

Fuses capabilities of Tree-of-Thought (ToTEngine) and Categorize-Analyze-Test (CATEngine)
for advanced cognitive reasoning, adapted for NebulaCore framework.
"""

import asyncio
import logging
import random
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from cat_engine import CATEngine  # Use the NebulaCore-adapted CATEngine
from tot_engine import ToTEngine  # Use the NebulaCore-adapted ToTEngine
# Assuming vanta_echo_nebula_core.py, vanta_tot_engine.py, vanta_cat_engine.py are accessible
try:
    from vanta_echo_nebula_bus import NebulaCore as VantaEchoNebulaCore
except ImportError:
    # Minimal stub for VantaEchoNebulaCore when bus is not available
    class VantaEchoNebulaCore:
        def __init__(self):
            self.event_bus = None
            self.async_bus = None
            self.registry = {}
            self.logger = None

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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NebulaCore.HybridCognition")  # Logger for this engine


# --- Cognitive Task Selection Framework ---

class TaskPriority(Enum):
    """Priority levels for cognitive task selection"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1

class EthicsFramework(Enum):
    """Ethical frameworks for compute allocation decisions"""
    UTILITARIAN = "maximize_overall_benefit"  # Greatest good for greatest number
    DEONTOLOGICAL = "respect_worker_autonomy"  # Respect worker rights and choices
    VIRTUE_ETHICS = "promote_network_virtues"  # Honesty, fairness, sustainability
    CARE_ETHICS = "nurture_relationships"  # Maintain healthy network relationships

@dataclass
class WorkerProfile:
    """Profile of a compute worker for cognitive task selection"""
    worker_id: str
    capabilities: List[str]
    region: str
    performance_history: Dict[str, float]
    availability_score: float
    energy_efficiency: float
    network_reputation: float
    ethical_preferences: List[str]
    current_load: float
    preferred_task_types: List[str]
    sustainability_score: float  # Environmental impact consideration

@dataclass
class TaskProfile:
    """Profile of a compute task for cognitive optimization"""
    task_id: str
    task_type: str
    priority: TaskPriority
    resource_requirements: Dict[str, float]
    deadline: Optional[float]
    ethical_considerations: List[str]
    preferred_regions: List[str]
    minimum_quality_threshold: float
    reward_amount: float
    environmental_impact: str  # "low", "medium", "high"
    data_sensitivity: str  # "public", "private", "confidential"

@dataclass
class AllocationDecision:
    """Result of cognitive task allocation"""
    task_id: str
    selected_worker: Optional[str]
    allocation_score: float
    reasoning: str
    ethical_framework_used: EthicsFramework
    alternative_workers: List[Tuple[str, float]]  # worker_id, score
    confidence: float
    estimated_completion_time: float
    quality_prediction: float

class CognitiveTaskSelector:
    """Cognitive system for intelligent task-worker allocation"""
    
    def __init__(self):
        self.worker_profiles: Dict[str, WorkerProfile] = {}
        self.task_history: Dict[str, Dict[str, Any]] = {}
        self.ethical_weights: Dict[EthicsFramework, float] = {
            EthicsFramework.UTILITARIAN: 0.4,
            EthicsFramework.DEONTOLOGICAL: 0.3,
            EthicsFramework.VIRTUE_ETHICS: 0.2,
            EthicsFramework.CARE_ETHICS: 0.1
        }
        self.learning_rate = 0.1
        
    def register_worker(self, profile: WorkerProfile):
        """Register a new worker with the cognitive selector"""
        self.worker_profiles[profile.worker_id] = profile
        logger.info(f"Registered worker {profile.worker_id} with capabilities: {profile.capabilities}")
    
    def update_worker_performance(self, worker_id: str, task_type: str, 
                                performance_score: float, completion_time: float):
        """Update worker performance based on task completion"""
        if worker_id in self.worker_profiles:
            profile = self.worker_profiles[worker_id]
            
            # Update performance history with exponential smoothing
            current_score = profile.performance_history.get(task_type, 0.5)
            new_score = (1 - self.learning_rate) * current_score + self.learning_rate * performance_score
            profile.performance_history[task_type] = new_score
            
            # Update overall reputation
            profile.network_reputation = sum(profile.performance_history.values()) / len(profile.performance_history)
            
            logger.debug(f"Updated {worker_id} performance for {task_type}: {new_score:.3f}")
    
    def select_optimal_worker(self, task: TaskProfile) -> AllocationDecision:
        """Use cognitive heuristics to select optimal worker for task"""
        if not self.worker_profiles:
            return AllocationDecision(
                task_id=task.task_id,
                selected_worker=None,
                allocation_score=0.0,
                reasoning="No workers available",
                ethical_framework_used=EthicsFramework.UTILITARIAN,
                alternative_workers=[],
                confidence=0.0,
                estimated_completion_time=0.0,
                quality_prediction=0.0
            )
        
        # Score all eligible workers
        worker_scores = {}
        for worker_id, profile in self.worker_profiles.items():
            score = self._calculate_worker_score(task, profile)
            worker_scores[worker_id] = score
        
        # Sort by score and select best
        sorted_workers = sorted(worker_scores.items(), key=lambda x: x[1], reverse=True)
        best_worker_id, best_score = sorted_workers[0] if sorted_workers else (None, 0.0)
        
        # Determine ethical framework used
        primary_framework = self._select_ethical_framework(task)
        
        # Generate reasoning
        reasoning = self._generate_allocation_reasoning(task, best_worker_id, primary_framework)
        
        # Calculate confidence based on score distribution
        confidence = self._calculate_allocation_confidence(worker_scores)
        
        # Predict completion time and quality
        if best_worker_id:
            best_profile = self.worker_profiles[best_worker_id]
            estimated_time = self._estimate_completion_time(task, best_profile)
            quality_pred = self._predict_quality(task, best_profile)
        else:
            estimated_time = 0.0
            quality_pred = 0.0
        
        return AllocationDecision(
            task_id=task.task_id,
            selected_worker=best_worker_id,
            allocation_score=best_score,
            reasoning=reasoning,
            ethical_framework_used=primary_framework,
            alternative_workers=sorted_workers[1:6],  # Top 5 alternatives
            confidence=confidence,
            estimated_completion_time=estimated_time,
            quality_prediction=quality_pred
        )
    
    def _calculate_worker_score(self, task: TaskProfile, worker: WorkerProfile) -> float:
        """Calculate comprehensive score for worker-task pairing"""
        scores = {}
        
        # Capability match score (40% weight)
        capability_score = self._score_capability_match(task, worker)
        scores['capability'] = capability_score * 0.4
        
        # Performance history score (25% weight)
        performance_score = worker.performance_history.get(task.task_type, 
                                                         worker.network_reputation)
        scores['performance'] = performance_score * 0.25
        
        # Availability and load score (15% weight)
        availability_score = worker.availability_score * (1.0 - worker.current_load)
        scores['availability'] = availability_score * 0.15
        
        # Regional preference score (10% weight)
        region_score = 1.0 if worker.region in task.preferred_regions else 0.7
        scores['region'] = region_score * 0.1
        
        # Ethical alignment score (5% weight)
        ethics_score = self._score_ethical_alignment(task, worker)
        scores['ethics'] = ethics_score * 0.05
        
        # Environmental sustainability score (5% weight)
        sustainability_score = worker.sustainability_score
        if task.environmental_impact == "low":
            sustainability_score *= 1.2  # Boost for low-impact tasks
        scores['sustainability'] = sustainability_score * 0.05
        
        total_score = sum(scores.values())
        
        logger.debug(f"Worker {worker.worker_id} score for task {task.task_id}: {total_score:.3f} "
                    f"(capability: {scores['capability']:.3f}, performance: {scores['performance']:.3f}, "
                    f"availability: {scores['availability']:.3f}, region: {scores['region']:.3f}, "
                    f"ethics: {scores['ethics']:.3f}, sustainability: {scores['sustainability']:.3f})")
        
        return total_score
    
    def _score_capability_match(self, task: TaskProfile, worker: WorkerProfile) -> float:
        """Score how well worker capabilities match task requirements"""
        required_capabilities = set(task.resource_requirements.keys())
        worker_capabilities = set(worker.capabilities)
        
        if not required_capabilities:
            return 1.0
        
        match_score = len(required_capabilities.intersection(worker_capabilities)) / len(required_capabilities)
        
        # Bonus for having preferred task types
        if task.task_type in worker.preferred_task_types:
            match_score = min(1.0, match_score + 0.1)
        
        return match_score
    
    def _score_ethical_alignment(self, task: TaskProfile, worker: WorkerProfile) -> float:
        """Score ethical alignment between task and worker"""
        if not task.ethical_considerations or not worker.ethical_preferences:
            return 0.5  # Neutral score
        
        alignment_count = len(set(task.ethical_considerations).intersection(
                                set(worker.ethical_preferences)))
        max_possible = max(len(task.ethical_considerations), len(worker.ethical_preferences))
        
        return alignment_count / max_possible if max_possible > 0 else 0.5
    
    def _select_ethical_framework(self, task: TaskProfile) -> EthicsFramework:
        """Select appropriate ethical framework for task allocation"""
        if task.priority == TaskPriority.CRITICAL:
            return EthicsFramework.UTILITARIAN  # Maximize overall benefit for critical tasks
        elif task.data_sensitivity == "confidential":
            return EthicsFramework.DEONTOLOGICAL  # Respect worker autonomy for sensitive data
        elif "sustainability" in task.ethical_considerations:
            return EthicsFramework.VIRTUE_ETHICS  # Promote network virtues
        else:
            return EthicsFramework.CARE_ETHICS  # Nurture network relationships
    
    def _generate_allocation_reasoning(self, task: TaskProfile, worker_id: Optional[str], 
                                     framework: EthicsFramework) -> str:
        """Generate human-readable reasoning for allocation decision"""
        if not worker_id:
            return "No suitable worker found for task requirements"
        
        worker = self.worker_profiles[worker_id]
        
        base_reason = f"Selected worker {worker_id} based on {framework.value} principle. "
        
        if framework == EthicsFramework.UTILITARIAN:
            base_reason += f"Worker has {worker.network_reputation:.2f} reputation and optimal resource match."
        elif framework == EthicsFramework.DEONTOLOGICAL:
            base_reason += "Worker preferences align with task ethics and maintains autonomy."
        elif framework == EthicsFramework.VIRTUE_ETHICS:
            base_reason += f"Worker demonstrates network virtues with {worker.sustainability_score:.2f} sustainability score."
        else:  # CARE_ETHICS
            base_reason += "Selection maintains healthy network relationships and worker well-being."
        
        return base_reason
    
    def _calculate_allocation_confidence(self, worker_scores: Dict[str, float]) -> float:
        """Calculate confidence in allocation decision based on score distribution"""
        if len(worker_scores) < 2:
            return 0.5
        
        scores = list(worker_scores.values())
        scores.sort(reverse=True)
        
        # Confidence based on gap between best and second-best
        score_gap = scores[0] - scores[1]
        confidence = min(1.0, 0.5 + score_gap * 2.0)  # Scale gap to confidence
        
        return confidence
    
    def _estimate_completion_time(self, task: TaskProfile, worker: WorkerProfile) -> float:
        """Estimate task completion time based on worker profile"""
        base_time = 300.0  # 5 minutes base time
        
        # Adjust based on worker performance history
        performance_factor = worker.performance_history.get(task.task_type, 0.5)
        time_factor = 2.0 - performance_factor  # Better performance = faster completion
        
        # Adjust based on current load
        load_factor = 1.0 + worker.current_load
        
        estimated_time = base_time * time_factor * load_factor
        
        return estimated_time
    
    def _predict_quality(self, task: TaskProfile, worker: WorkerProfile) -> float:
        """Predict quality of task completion based on worker profile"""
        base_quality = worker.performance_history.get(task.task_type, worker.network_reputation)
        
        # Adjust for worker specialization
        if task.task_type in worker.preferred_task_types:
            base_quality = min(1.0, base_quality + 0.1)
        
        # Adjust for current load (high load might reduce quality)
        load_penalty = worker.current_load * 0.1
        
        predicted_quality = max(0.0, base_quality - load_penalty)
        
        return predicted_quality

class AgentGuidedAllocation:
    """Agent-based guidance for compute allocation decisions"""
    
    def __init__(self, task_selector: CognitiveTaskSelector):
        self.task_selector = task_selector
        self.agent_preferences: Dict[str, Dict[str, Any]] = {}
        self.allocation_policies: Dict[str, Callable] = {}
    
    def register_agent_preferences(self, agent_id: str, preferences: Dict[str, Any]):
        """Register agent preferences for compute allocation"""
        self.agent_preferences[agent_id] = preferences
        logger.info(f"Registered allocation preferences for agent {agent_id}")
    
    def guide_allocation(self, task: TaskProfile, guiding_agents: List[str]) -> AllocationDecision:
        """Use agent guidance to influence allocation decisions"""
        # Get base allocation decision
        base_decision = self.task_selector.select_optimal_worker(task)
        
        # Apply agent guidance
        guided_decision = self._apply_agent_guidance(base_decision, task, guiding_agents)
        
        return guided_decision
    
    def _apply_agent_guidance(self, base_decision: AllocationDecision, 
                            task: TaskProfile, agents: List[str]) -> AllocationDecision:
        """Apply agent preferences to modify allocation decision"""
        # Collect agent preferences relevant to this allocation
        relevant_preferences = []
        for agent_id in agents:
            if agent_id in self.agent_preferences:
                prefs = self.agent_preferences[agent_id]
                if self._preferences_apply_to_task(prefs, task):
                    relevant_preferences.append((agent_id, prefs))
        
        if not relevant_preferences:
            return base_decision  # No relevant agent guidance
        
        # Modify allocation based on agent preferences
        modified_decision = base_decision
        modification_reasons = []
        
        for agent_id, prefs in relevant_preferences:
            if "ethical_priority" in prefs:
                # Agent prioritizes certain ethical considerations
                ethical_boost = prefs["ethical_priority"]
                if ethical_boost in task.ethical_considerations:
                    modified_decision.allocation_score *= 1.1
                    modification_reasons.append(f"Agent {agent_id} ethical priority boost")
            
            if "sustainability_weight" in prefs:
                # Agent emphasizes environmental sustainability
                if base_decision.selected_worker:
                    worker = self.task_selector.worker_profiles[base_decision.selected_worker]
                    sustainability_factor = worker.sustainability_score * prefs["sustainability_weight"]
                    modified_decision.allocation_score = (
                        modified_decision.allocation_score * 0.8 + sustainability_factor * 0.2
                    )
                    modification_reasons.append(f"Agent {agent_id} sustainability weighting")
            
            if "region_preference" in prefs:
                # Agent has regional preferences
                preferred_regions = prefs["region_preference"]
                if (base_decision.selected_worker and 
                    self.task_selector.worker_profiles[base_decision.selected_worker].region in preferred_regions):
                    modified_decision.allocation_score *= 1.05
                    modification_reasons.append(f"Agent {agent_id} regional preference")
        
        # Update reasoning to include agent guidance
        if modification_reasons:
            modified_decision.reasoning += f" Agent guidance: {'; '.join(modification_reasons)}"
        
        return modified_decision
    
    def _preferences_apply_to_task(self, preferences: Dict[str, Any], task: TaskProfile) -> bool:
        """Check if agent preferences are relevant to this task"""
        # Check task type relevance
        if "task_types" in preferences:
            if task.task_type not in preferences["task_types"]:
                return False
        
        # Check priority relevance
        if "min_priority" in preferences:
            if task.priority.value < preferences["min_priority"]:
                return False
        
        return True


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
        vanta_echo_nebula_core: NebulaCore,  # Mandatory NebulaCore instance
        config: HybridCognitionConfig,
        tot_engine_instance: ToTEngine,  # Must be NebulaCore-adapted ToTEngine
        cat_engine_instance: CATEngine,  # Must be NebulaCore-adapted CATEngine
        result_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.vanta_echo_nebula_core = vanta_echo_nebula_core
        self.config = config
        logger.info(
            f"HybridCognitionEngine initializing via NebulaCore. Fusion Mode: {self.config.fusion_mode}, Interval: {self.config.interval_s}s"
        )

        if not isinstance(tot_engine_instance, ToTEngine):
            raise TypeError(
                "tot_engine_instance must be an instance of NebulaCore-adapted ToTEngine."
            )
        if not isinstance(cat_engine_instance, CATEngine):
            raise TypeError(
                "cat_engine_instance must be an instance of NebulaCore-adapted CATEngine."
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
        
        # Initialize cognitive task selection components
        self.task_selector = CognitiveTaskSelector()
        self.agent_guided_allocation = AgentGuidedAllocation(self.task_selector)
        self.active_allocations: Dict[str, AllocationDecision] = {}
        self.allocation_history: List[Dict[str, Any]] = []

        # Register with NebulaCore
        self.vanta_echo_nebula_core.register_component(
            self.COMPONENT_NAME, self, metadata={"type": "cognitive_fusion_engine"}
        )
        logger.info("HybridCognitionEngine instance registered with NebulaCore.")

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
            try:
                _safe_publish_event(
                    self.vanta_echo_nebula_core,
                    f"{self.COMPONENT_NAME}.started",
                    {"fusion_mode": self.config.fusion_mode},
                    self.COMPONENT_NAME
                )
            except Exception:
                pass  # Non-critical

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
            try:
                import asyncio
                asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                    f"{self.COMPONENT_NAME}.stopped", {}, self.COMPONENT_NAME
                ))
            except Exception:
                pass  # Non-critical

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
                try:
                    _safe_publish_event(
                        self.vanta_echo_nebula_core,
                        f"{self.COMPONENT_NAME}.fusion_result",
                        {"fusion_id": fusion_output.get("fusion_id")},
                        self.COMPONENT_NAME
                    )
                except Exception:
                    pass  # Non-critical

            except Exception as e:
                self.last_error = (
                    f"Error during {self.current_phase or 'unknown'} phase: {str(e)}"
                )
                logger.exception(
                    f"Fusion cycle failed during {self.current_phase or 'unknown'} phase"
                )
                try:
                    import asyncio
                    asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                        f"{self.COMPONENT_NAME}.cycle_error",
                        {"phase": self.current_phase, "error": str(e)},
                        self.COMPONENT_NAME
                    ))
                except Exception:
                    pass  # Non-critical
            finally:
                self.current_phase = "Idle"
                cycle_duration = time.monotonic() - cycle_start_time
                logger.info(
                    f"{self.COMPONENT_NAME}: Fusion cycle #{self.cycle_count} finished in {cycle_duration:.2f}s. Last error: {self.last_error or 'None'}"
                )
                try:
                    _safe_publish_event(
                        self.vanta_echo_nebula_core,
                        f"{self.COMPONENT_NAME}.cycle_complete",
                        {
                            "cycle_num": self.cycle_count,
                            "duration_s": cycle_duration,
                            "error": self.last_error is not None,
                        },
                        self.COMPONENT_NAME
                    )
                except Exception:
                    pass  # Non-critical

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
        # Simplified adaptive strategy alternating based on uncertainty
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

        # Publish NebulaCore event for the new hybrid branch
        try:
            _safe_publish_event(
                self.vanta_echo_nebula_core,
                f"{self.COMPONENT_NAME}.new_hybrid_branch",
                new_branch.to_dict(),
                self.COMPONENT_NAME
            )
        except Exception:
            pass  # Non-critical

        return {
            "fusion_id": fusion_id,
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "active_branch_id": self.active_branch_id,
            "branch_count": len(self.branches),
        }

    def fuse_from_results(
        self, tot_result: dict[str, Any] | None, cat_result: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Integrate externally provided ToT and CAT results once."""
        self._last_tot_result = tot_result or {}
        self._last_cat_result = cat_result or {}
        self.cycle_count += 1
        return self._integrate_engine_results()

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
            "worker_profiles": {wid: {
                "worker_id": wp.worker_id,
                "capabilities": wp.capabilities,
                "region": wp.region,
                "performance_history": wp.performance_history,
                "network_reputation": wp.network_reputation,
                "current_load": wp.current_load
            } for wid, wp in self.task_selector.worker_profiles.items()}
        }
    
    def allocate_task_cognitively(self, task_profile: TaskProfile, 
                                 guiding_agents: Optional[List[str]] = None) -> AllocationDecision:
        """Use cognitive reasoning to allocate task to optimal worker"""
        logger.info(f"Cognitively allocating task {task_profile.task_id} (type: {task_profile.task_type})")
        
        try:
            # Use agent guidance if provided
            if guiding_agents:
                decision = self.agent_guided_allocation.guide_allocation(task_profile, guiding_agents)
                logger.info(f"Agent-guided allocation for {task_profile.task_id}: "
                           f"worker {decision.selected_worker} (score: {decision.allocation_score:.3f})")
            else:
                decision = self.task_selector.select_optimal_worker(task_profile)
                logger.info(f"Cognitive allocation for {task_profile.task_id}: "
                           f"worker {decision.selected_worker} (score: {decision.allocation_score:.3f})")
            
            # Store allocation for tracking
            self.active_allocations[task_profile.task_id] = decision
            
            # Record allocation in history
            self.allocation_history.append({
                "timestamp": time.time(),
                "task_id": task_profile.task_id,
                "worker_id": decision.selected_worker,
                "allocation_score": decision.allocation_score,
                "confidence": decision.confidence,
                "ethical_framework": decision.ethical_framework_used.value,
                "reasoning": decision.reasoning
            })
            
            # Publish allocation event
            try:
                _safe_publish_event(
                    self.vanta_echo_nebula_core,
                    f"{self.COMPONENT_NAME}.task_allocated",
                    {
                        "task_id": task_profile.task_id,
                        "worker_id": decision.selected_worker,
                        "allocation_score": decision.allocation_score,
                        "confidence": decision.confidence,
                        "ethical_framework": decision.ethical_framework_used.value
                    },
                    self.COMPONENT_NAME
                )
            except Exception:
                pass  # Non-critical
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to allocate task {task_profile.task_id}: {e}")
            return AllocationDecision(
                task_id=task_profile.task_id,
                selected_worker=None,
                allocation_score=0.0,
                reasoning=f"Allocation failed: {str(e)}",
                ethical_framework_used=EthicsFramework.UTILITARIAN,
                alternative_workers=[],
                confidence=0.0,
                estimated_completion_time=0.0,
                quality_prediction=0.0
            )
    
    def register_worker_profile(self, profile: WorkerProfile):
        """Register a new worker with cognitive task selection"""
        self.task_selector.register_worker(profile)
        
        # Create a hybrid branch for worker registration
        worker_branch = HybridThoughtBranch(
            content=f"Worker {profile.worker_id} registered with capabilities: {profile.capabilities}",
            confidence=profile.network_reputation,
            metadata={
                "worker_id": profile.worker_id,
                "event_type": "worker_registration",
                "capabilities": profile.capabilities,
                "region": profile.region
            }
        )
        worker_branch.add_category_tag("worker_management")
        worker_branch.add_category_tag(f"region_{profile.region}")
        
        self.branches[worker_branch.branch_id] = worker_branch
        
        logger.info(f"Registered worker {profile.worker_id} in cognitive system")
    
    def update_task_completion(self, task_id: str, worker_id: str, 
                             performance_score: float, completion_time: float):
        """Update cognitive system with task completion feedback"""
        if task_id in self.active_allocations:
            task_type = "unknown"
            
            # Extract task type from allocation history
            for record in self.allocation_history:
                if record["task_id"] == task_id:
                    # Would need to store task_type in history
                    break
            
            # Update worker performance
            self.task_selector.update_worker_performance(
                worker_id, task_type, performance_score, completion_time
            )
            
            # Create completion branch
            completion_branch = HybridThoughtBranch(
                content=f"Task {task_id} completed by {worker_id} with score {performance_score:.3f}",
                confidence=performance_score,
                metadata={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "performance_score": performance_score,
                    "completion_time": completion_time,
                    "event_type": "task_completion"
                }
            )
            completion_branch.add_category_tag("task_completion")
            completion_branch.add_category_tag(f"worker_{worker_id}")
            
            self.branches[completion_branch.branch_id] = completion_branch
            
            # Remove from active allocations
            del self.active_allocations[task_id]
            
            logger.info(f"Updated completion for task {task_id}: performance {performance_score:.3f}")
    
    def register_agent_guidance(self, agent_id: str, preferences: Dict[str, Any]):
        """Register agent preferences for compute allocation guidance"""
        self.agent_guided_allocation.register_agent_preferences(agent_id, preferences)
        
        # Create branch for agent registration
        agent_branch = HybridThoughtBranch(
            content=f"Agent {agent_id} registered allocation preferences",
            confidence=0.8,
            metadata={
                "agent_id": agent_id,
                "preferences": preferences,
                "event_type": "agent_guidance_registration"
            }
        )
        agent_branch.add_category_tag("agent_guidance")
        agent_branch.add_category_tag(f"agent_{agent_id}")
        
        self.branches[agent_branch.branch_id] = agent_branch
        
        logger.info(f"Registered guidance preferences for agent {agent_id}")
    
    def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get metrics about cognitive task allocation performance"""
        total_workers = len(self.task_selector.worker_profiles)
        active_tasks = len(self.active_allocations)
        completed_tasks = len(self.allocation_history)
        
        # Calculate average allocation confidence
        if self.allocation_history:
            avg_confidence = sum(record["confidence"] for record in self.allocation_history) / len(self.allocation_history)
        else:
            avg_confidence = 0.0
        
        # Calculate ethical framework distribution
        framework_counts = {}
        for record in self.allocation_history:
            framework = record["ethical_framework"]
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
        return {
            "total_registered_workers": total_workers,
            "active_task_allocations": active_tasks,
            "completed_task_allocations": completed_tasks,
            "average_allocation_confidence": avg_confidence,
            "ethical_framework_distribution": framework_counts,
            "cognitive_branches_created": len([b for b in self.branches.values() 
                                             if "cognitive" in b.category_tags or 
                                                "worker_management" in b.category_tags or
                                                "task_completion" in b.category_tags]),
            "agent_guidance_registrations": len([b for b in self.branches.values() 
                                               if "agent_guidance" in b.category_tags])
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
        # Removed MetaConsciousness trace event, NebulaCore publish_event for tracing.
        try:
            import asyncio
            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                f"{self.COMPONENT_NAME}.diagnose.start",
                {"context_provided": context is not None},
                self.COMPONENT_NAME
            ))
        except Exception:
            pass  # Non-critical

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

        # Removed MetaConsciousness heartbeat, replaced with NebulaCore event.
        heartbeat_status = "normal"
        if hybrid_health_score < 0.3:
            heartbeat_status = "error"
        elif hybrid_health_score < 0.6:
            heartbeat_status = "warning"

        try:
            import asyncio
            asyncio.create_task(self.vanta_echo_nebula_core.publish_event(
                f"{self.COMPONENT_NAME}.heartbeat",
                {
                    "status": heartbeat_status,
                    "pulse_value": hybrid_health_score,
                    "branch_count": len(self.branches),
                    "active_branch_id": self.active_branch_id,
                    "fusion_interval_s": self.config.interval_s,
                    "issues_count": len(issues),
                },
                self.COMPONENT_NAME
            ))
        except Exception:
            pass  # Non-critical
        logger.info(
            f"Hybrid Diagnosis complete. Overall Score: {hybrid_health_score:.2f}"
        )
        self.current_phase = "Idle"
        return diagnosis

    # Internal health/issue/recommendation stubs specific to HybridCognition if needed
    # (Using sub-engine diagnostics for now)


# --- Example Usage (Adapted for NebulaCore) ---
if __name__ == "__main__":
    main_logger = logging.getLogger("HybridCogExample")
    main_logger.setLevel(logging.DEBUG)
    main_logger.info("--- Starting Hybrid Cognition Engine NebulaCore Example ---")

    # 1. Initialize NebulaCore
    vanta_system = NebulaCore()

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
    #    NebulaCore instance is passed to them.
    #    They will register themselves with vanta_system.

    # Explicitly create and register mock dependencies for ToT to ensure it finds them
    # (since ToTEngine's default creation for these mandatory agents might not exist)
    vanta_system.register_component("default_thought_seeder", MockToTSeeder())
    vanta_system.register_component("default_branch_evaluator", MockToTEvaluator())
    # ... (need to decide if ToT creates these if not in registry or if they are mandatory)
    # For now, assuming ToTEngine __init__ parameters are mandatory for specialists.

    tot_engine_inst = ToTEngine(
        vanta_echo_nebula_core=vanta_system,
        config=tot_config_inst,
        thought_seeder=MockToTSeeder(),
        branch_evaluator=MockToTEvaluator(),
        branch_validator=MockToTValidator(),
        meta_learner=MockToTMetaLearner(),
        context_providers=[MockContextProvider()],
    )
    cat_engine_inst = CATEngine(
        vanta_echo_nebula_core=vanta_system, config=cat_config_inst
    )  # Will use its defaults

    # 5. Instantiate HybridCognitionEngine
    hybrid_engine_inst = HybridCognitionEngine(
        vanta_echo_nebula_core=vanta_system,
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
        # vanta_system.shutdown() # If NebulaCore needs explicit shutdown
        main_logger.info("--- Hybrid Cognition Engine NebulaCore Example Finished ---")
