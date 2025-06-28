"""
ARC Ensemble Orchestrator - Main Coordination System

Orchestrates the complete ensemble of novel reasoning paradigms for
ARC-like task solving. Integrates all components: efficiency modules,
reasoning components, and meta-control systems into a cohesive pipeline.

Key Features:
- Multi-agent ensemble coordination
- Dynamic pipeline configuration based on problem complexity
- Real-time resource allocation and load balancing
- Consensus building from multiple reasoning approaches
- Integration with HOLO-1.5 cognitive mesh for orchestration

Addresses the full spectrum of LLM limitations through coordinated
application of novel paradigms.

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


# Define fallback classes first
@dataclass
class EffortBudget:
    max_compute: float = 1.0
    max_memory: float = 1.0
    max_time: float = 60.0


@dataclass
class ComplexityMeasurement:
    visual_complexity: float = 0.5
    logical_complexity: float = 0.5
    overall_complexity: float = 0.5


class EffortController:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}


class ComplexityMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}


try:
    from ...agents.base import BaseAgent, CognitiveMeshRole, vanta_agent
    from ..novel_efficiency import (
        AdaptiveMemoryManager,
    )
    from ..novel_reasoning import (
        AKOrNBindingNetwork,
        LogicalReasoningEngine,
        SPLRSpikingNetwork,
    )

    # Try to import meta_control - override fallbacks if available
    try:
        from ..meta_control import (
            ComplexityMeasurement,
            ComplexityMonitor,
            EffortBudget,
            EffortController,
        )
    except ImportError:
        pass  # Use fallback classes defined above

    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    HOLO_AVAILABLE = False

    def vanta_agent(*args, **kwargs):
        """
        Fallback vanta_agent decorator that accepts all keyword arguments
        but doesn't do anything with them (for non-HOLO environments)
        """

        def decorator(cls):
            # Store the agent metadata as class attributes for potential use
            if kwargs:
                cls._vanta_agent_config = kwargs
            return cls

        # Handle both @vanta_agent and @vanta_agent(...) usage
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Direct decoration: @vanta_agent
            return decorator(args[0])
        else:
            # Parametrized decoration: @vanta_agent(...)
            return decorator

        return decorator

    class CognitiveMeshRole:
        ORCHESTRATOR = "orchestrator"
        COORDINATOR = "coordinator"

    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages in the ARC processing pipeline"""

    INITIALIZATION = "initialization"
    ENCODING = "encoding"
    BINDING = "binding"
    REASONING = "reasoning"
    PATTERN_LEARNING = "pattern_learning"
    GRAPH_REASONING = "graph_reasoning"
    CONSENSUS = "consensus"
    OUTPUT_GENERATION = "output_generation"


class EnsembleStrategy(Enum):
    """Strategies for ensemble coordination"""

    SEQUENTIAL = "sequential"  # Process stages sequentially
    PARALLEL = "parallel"  # Process compatible stages in parallel
    ADAPTIVE = "adaptive"  # Adapt strategy based on complexity
    HIERARCHICAL = "hierarchical"  # Hierarchical processing with feedback


@dataclass
class ProcessingResult:
    """Result from a processing stage or component"""

    stage: ProcessingStage
    component_name: str
    output: Any
    confidence: float
    processing_time: float
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleState:
    """Current state of the ensemble processing"""

    current_stage: ProcessingStage
    stage_results: Dict[ProcessingStage, List[ProcessingResult]] = field(
        default_factory=dict
    )
    global_context: Dict[str, Any] = field(default_factory=dict)
    effort_budget: Optional[EffortBudget] = None
    complexity_measurement: Optional[ComplexityMeasurement] = None
    start_time: float = field(default_factory=time.time)

    def add_result(self, result: ProcessingResult):
        """Add a processing result to the state"""
        if result.stage not in self.stage_results:
            self.stage_results[result.stage] = []
        self.stage_results[result.stage].append(result)

    def get_best_result(self, stage: ProcessingStage) -> Optional[ProcessingResult]:
        """Get best result for a given stage based on confidence"""
        results = self.stage_results.get(stage, [])
        if not results:
            return None
        return max(results, key=lambda r: r.confidence)


class AgentContract(ABC):
    """
    Abstract interface contract for ensemble agents

    Defines the standard interface that all reasoning agents
    must implement for ensemble integration.
    """

    @abstractmethod
    async def process(
        self, input_data: Any, context: Dict[str, Any]
    ) -> ProcessingResult:
        """Process input and return result"""
        pass

    @abstractmethod
    async def get_confidence(self, output: Any) -> float:
        """Get confidence score for output"""
        pass

    @abstractmethod
    async def estimate_resource_needs(self, input_data: Any) -> Dict[str, float]:
        """Estimate resource requirements"""
        pass

    @abstractmethod
    def get_supported_stages(self) -> List[ProcessingStage]:
        """Get list of processing stages this agent supports"""
        pass


class SPLREncoderAgent(AgentContract):
    """Agent wrapper for SPLR Spiking Network encoding"""

    def __init__(self, spiking_network: "SPLRSpikingNetwork"):
        self.spiking_network = spiking_network
        self.component_name = "SPLR_Encoder"

    async def process(
        self, input_data: torch.Tensor, context: Dict[str, Any]
    ) -> ProcessingResult:
        """Process grid input through spiking network"""
        start_time = time.time()

        try:
            output, network_state = self.spiking_network.forward(input_data)
            confidence = await self.get_confidence(output)

            processing_time = time.time() - start_time

            return ProcessingResult(
                stage=ProcessingStage.ENCODING,
                component_name=self.component_name,
                output=output,
                confidence=confidence,
                processing_time=processing_time,
                resource_usage={"memory": 0.3, "compute": 0.4},
                metadata={"network_state": network_state},
            )
        except Exception as e:
            logger.error(f"SPLR Encoder processing failed: {e}")
            return ProcessingResult(
                stage=ProcessingStage.ENCODING,
                component_name=self.component_name,
                output=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                resource_usage={"memory": 0.1, "compute": 0.1},
            )

    async def get_confidence(self, output: torch.Tensor) -> float:
        """Estimate confidence based on output consistency"""
        if output is None:
            return 0.0

        # Simple confidence based on output variance
        output_variance = torch.var(output)
        confidence = float(
            torch.exp(-output_variance)
        )  # Lower variance = higher confidence
        return max(0.1, min(confidence, 1.0))

    async def estimate_resource_needs(
        self, input_data: torch.Tensor
    ) -> Dict[str, float]:
        """Estimate resource needs for spiking network processing"""
        input_size = input_data.numel()

        return {
            "memory": min(input_size / 10000, 1.0),
            "compute": min(input_size / 5000, 1.0),
            "time": min(input_size / 1000, 1.0),
        }

    def get_supported_stages(self) -> List[ProcessingStage]:
        """SPLR supports encoding stage"""
        return [ProcessingStage.ENCODING]


class AKOrNBinderAgent(AgentContract):
    """Agent wrapper for AKOrN binding network"""

    def __init__(self, binding_network: "AKOrNBindingNetwork"):
        self.binding_network = binding_network
        self.component_name = "AKOrN_Binder"

    async def process(
        self, input_data: torch.Tensor, context: Dict[str, Any]
    ) -> ProcessingResult:
        """Process visual features through oscillatory binding"""
        start_time = time.time()

        try:
            binding_result = self.binding_network.forward(input_data)
            confidence = await self.get_confidence(binding_result)

            processing_time = time.time() - start_time

            return ProcessingResult(
                stage=ProcessingStage.BINDING,
                component_name=self.component_name,
                output=binding_result,
                confidence=confidence,
                processing_time=processing_time,
                resource_usage={"memory": 0.4, "compute": 0.5},
                metadata={"num_objects": len(binding_result.bound_objects)},
            )
        except Exception as e:
            logger.error(f"AKOrN Binder processing failed: {e}")
            return ProcessingResult(
                stage=ProcessingStage.BINDING,
                component_name=self.component_name,
                output=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                resource_usage={"memory": 0.1, "compute": 0.1},
            )

    async def get_confidence(self, binding_result) -> float:
        """Estimate confidence based on binding quality"""
        if binding_result is None:
            return 0.0

        # Confidence based on binding strength and number of objects
        binding_confidence = float(binding_result.binding_confidence)
        num_objects = len(binding_result.bound_objects)

        # Penalty for too many or too few objects
        object_penalty = 1.0
        if num_objects == 0:
            object_penalty = 0.1
        elif num_objects > 10:
            object_penalty = 0.7

        return binding_confidence * object_penalty

    async def estimate_resource_needs(
        self, input_data: torch.Tensor
    ) -> Dict[str, float]:
        """Estimate resource needs for oscillatory binding"""
        batch_size, channels, height, width = input_data.shape
        spatial_size = height * width

        return {
            "memory": min(spatial_size / 5000, 1.0),
            "compute": min(spatial_size / 2000, 1.0),
            "time": min(spatial_size / 1000, 1.0),
        }

    def get_supported_stages(self) -> List[ProcessingStage]:
        """AKOrN supports binding stage"""
        return [ProcessingStage.BINDING]


class LNUReasonerAgent(AgentContract):
    """Agent wrapper for Logical Neural Units"""

    def __init__(self, reasoning_engine: "LogicalReasoningEngine"):
        self.reasoning_engine = reasoning_engine
        self.component_name = "LNU_Reasoner"

    async def process(
        self, input_data: Any, context: Dict[str, Any]
    ) -> ProcessingResult:
        """Process logical reasoning"""
        start_time = time.time()

        try:
            # Extract logical state from context or create from input
            if "logical_state" in context:
                logical_state = context["logical_state"]
            else:
                # Create initial logical state from input
                if isinstance(input_data, torch.Tensor):
                    batch_size = input_data.shape[0] if len(input_data.shape) > 1 else 1
                    truth_values = (
                        torch.rand(batch_size, 64) * 0.5
                    )  # Initialize with low confidence
                else:
                    truth_values = torch.rand(1, 64) * 0.5

                from ..novel_reasoning.logical_neural_units import create_logical_state

                logical_state = create_logical_state(truth_values)

            # Perform reasoning
            goal_propositions = context.get("goal_propositions", None)
            final_state = await self.reasoning_engine.reason(
                logical_state, goal_propositions
            )

            confidence = await self.get_confidence(final_state)
            processing_time = time.time() - start_time

            return ProcessingResult(
                stage=ProcessingStage.REASONING,
                component_name=self.component_name,
                output=final_state,
                confidence=confidence,
                processing_time=processing_time,
                resource_usage={"memory": 0.3, "compute": 0.6},
                metadata={
                    "reasoning_steps": self.reasoning_engine.cognitive_metrics[
                        "reasoning_steps"
                    ]
                },
            )
        except Exception as e:
            logger.error(f"LNU Reasoner processing failed: {e}")
            return ProcessingResult(
                stage=ProcessingStage.REASONING,
                component_name=self.component_name,
                output=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                resource_usage={"memory": 0.1, "compute": 0.1},
            )

    async def get_confidence(self, logical_state) -> float:
        """Estimate confidence based on logical state"""
        if logical_state is None:
            return 0.0

        # Confidence based on truth values and symbolic depth
        mean_truth = float(torch.mean(logical_state.truth_values))
        confidence_factor = self.reasoning_engine.cognitive_metrics.get(
            "confidence_level", 0.5
        )

        return (mean_truth + confidence_factor) / 2

    async def estimate_resource_needs(self, input_data: Any) -> Dict[str, float]:
        """Estimate resource needs for logical reasoning"""
        return {"memory": 0.4, "compute": 0.7, "time": 0.6}

    def get_supported_stages(self) -> List[ProcessingStage]:
        """LNU supports reasoning stage"""
        return [ProcessingStage.REASONING]


class ConsensusBuilder:
    """
    Builds consensus from multiple agent outputs

    Combines results from different reasoning approaches to produce
    final ensemble output with confidence estimation.
    """

    def __init__(self, fusion_strategy: str = "weighted_average"):
        self.fusion_strategy = fusion_strategy

    async def build_consensus(
        self, stage_results: Dict[ProcessingStage, List[ProcessingResult]]
    ) -> ProcessingResult:
        """
        Build consensus from multiple processing results

        Args:
            stage_results: Results from different processing stages

        Returns:
            consensus_result: Final consensus result
        """
        start_time = time.time()

        # Get final stage results for consensus
        final_stage_results = []
        for stage in [
            ProcessingStage.REASONING,
            ProcessingStage.PATTERN_LEARNING,
            ProcessingStage.GRAPH_REASONING,
        ]:
            if stage in stage_results:
                final_stage_results.extend(stage_results[stage])

        if not final_stage_results:
            logger.warning("No results available for consensus building")
            return ProcessingResult(
                stage=ProcessingStage.CONSENSUS,
                component_name="ConsensusBuilder",
                output=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                resource_usage={"memory": 0.1, "compute": 0.1},
            )

        # Apply fusion strategy
        if self.fusion_strategy == "weighted_average":
            consensus_output = await self._weighted_average_fusion(final_stage_results)
        elif self.fusion_strategy == "majority_vote":
            consensus_output = await self._majority_vote_fusion(final_stage_results)
        elif self.fusion_strategy == "confidence_ranking":
            consensus_output = await self._confidence_ranking_fusion(
                final_stage_results
            )
        else:
            consensus_output = await self._simple_average_fusion(final_stage_results)

        # Calculate consensus confidence
        weights = [r.confidence for r in final_stage_results]
        if sum(weights) > 0:
            consensus_confidence = sum(
                w * r.confidence for w, r in zip(weights, final_stage_results)
            ) / sum(weights)
        else:
            consensus_confidence = 0.0

        processing_time = time.time() - start_time

        return ProcessingResult(
            stage=ProcessingStage.CONSENSUS,
            component_name="ConsensusBuilder",
            output=consensus_output,
            confidence=consensus_confidence,
            processing_time=processing_time,
            resource_usage={"memory": 0.2, "compute": 0.3},
            metadata={
                "fusion_strategy": self.fusion_strategy,
                "num_inputs": len(final_stage_results),
                "input_confidences": [r.confidence for r in final_stage_results],
            },
        )

    async def _weighted_average_fusion(
        self, results: List[ProcessingResult]
    ) -> torch.Tensor:
        """Weighted average fusion based on confidence scores"""
        weights = torch.tensor([r.confidence for r in results])
        weights = (
            weights / torch.sum(weights)
            if torch.sum(weights) > 0
            else torch.ones_like(weights) / len(weights)
        )

        # Convert outputs to tensors for averaging
        tensor_outputs = []
        for result in results:
            if isinstance(result.output, torch.Tensor):
                tensor_outputs.append(result.output)
            elif hasattr(result.output, "truth_values"):  # Logical state
                tensor_outputs.append(result.output.truth_values)
            else:
                # Convert to tensor representation
                tensor_outputs.append(torch.tensor(0.5))  # Fallback

        if tensor_outputs:
            # Ensure all tensors have same shape for averaging
            target_shape = tensor_outputs[0].shape
            aligned_outputs = []
            for output in tensor_outputs:
                if output.shape == target_shape:
                    aligned_outputs.append(output)
                else:
                    # Reshape or pad as needed
                    aligned_outputs.append(torch.zeros(target_shape))

            if aligned_outputs:
                stacked_outputs = torch.stack(aligned_outputs)
                weighted_average = torch.sum(
                    stacked_outputs * weights.unsqueeze(-1).unsqueeze(-1), dim=0
                )
                return weighted_average

        return torch.tensor(0.5)  # Fallback

    async def _majority_vote_fusion(
        self, results: List[ProcessingResult]
    ) -> torch.Tensor:
        """Majority vote fusion for discrete outputs"""
        # Simplified majority vote - convert outputs to binary decisions
        decisions = []
        for result in results:
            if isinstance(result.output, torch.Tensor):
                decision = (result.output > 0.5).float()
            else:
                decision = torch.tensor(1.0 if result.confidence > 0.5 else 0.0)
            decisions.append(decision)

        if decisions:
            stacked_decisions = torch.stack(decisions)
            majority_vote = torch.mean(stacked_decisions, dim=0)
            return (majority_vote > 0.5).float()

        return torch.tensor(0.0)

    async def _confidence_ranking_fusion(
        self, results: List[ProcessingResult]
    ) -> torch.Tensor:
        """Use highest confidence result"""
        best_result = max(results, key=lambda r: r.confidence)

        if isinstance(best_result.output, torch.Tensor):
            return best_result.output
        elif hasattr(best_result.output, "truth_values"):
            return best_result.output.truth_values
        else:
            return torch.tensor(best_result.confidence)

    async def _simple_average_fusion(
        self, results: List[ProcessingResult]
    ) -> torch.Tensor:
        """Simple average of all outputs"""
        return await self._weighted_average_fusion(results)


@vanta_agent(role=CognitiveMeshRole.ORCHESTRATOR)
class ARCEnsembleOrchestrator(BaseAgent):
    """
    Main ARC Ensemble Orchestrator

    Coordinates the complete ensemble of novel reasoning paradigms
    for ARC-like task solving. Implements dynamic orchestration
    based on problem complexity and resource constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Orchestration parameters
        self.max_processing_time = config.get("max_processing_time", 60.0)
        self.ensemble_strategy = EnsembleStrategy(
            config.get("ensemble_strategy", "adaptive")
        )
        self.enable_parallelization = config.get("enable_parallelization", True)

        # Core components (will be initialized in async_init)
        self.effort_controller: Optional[EffortController] = None
        self.complexity_monitor: Optional[ComplexityMonitor] = None
        self.memory_manager: Optional[AdaptiveMemoryManager] = None

        # Agent registry
        self.agents: Dict[str, AgentContract] = {}
        self.stage_agents: Dict[ProcessingStage, List[AgentContract]] = {}

        # Consensus building
        self.consensus_builder = ConsensusBuilder(
            config.get("fusion_strategy", "weighted_average")
        )

        # Processing state
        self.current_ensemble_state: Optional[EnsembleState] = None

        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "orchestration_efficiency": 0.0,
            "consensus_quality": 0.0,
            "resource_utilization": 0.0,
            "pipeline_completion_rate": 0.0,
        }

    async def async_init(self):
        """Initialize the ensemble orchestrator"""
        if HOLO_AVAILABLE:
            await super().async_init()

        # Initialize core components
        from ..meta_control import create_complexity_monitor, create_effort_controller

        effort_config = self.config.get("effort_controller", {})
        self.effort_controller = await create_effort_controller(effort_config)

        complexity_config = self.config.get("complexity_monitor", {})
        self.complexity_monitor = await create_complexity_monitor(complexity_config)

        logger.info("ARC Ensemble Orchestrator initialized with HOLO-1.5 integration")

    def register_agent(self, agent: AgentContract, name: str):
        """Register an agent with the orchestrator"""
        self.agents[name] = agent

        # Update stage mappings
        for stage in agent.get_supported_stages():
            if stage not in self.stage_agents:
                self.stage_agents[stage] = []
            self.stage_agents[stage].append(agent)

        logger.info(
            f"Registered agent '{name}' supporting stages: {agent.get_supported_stages()}"
        )

    async def process_arc_task(self, arc_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete ARC task through ensemble pipeline

        Args:
            arc_task: Dictionary containing:
                - input_grid: Input ARC grid (numpy array)
                - target_grid: Optional target grid (numpy array)
                - predicted_grid: Optional predicted grid (numpy array)
                - task_complexity: Task complexity assessment

        Returns:
            paradigm_results: Dictionary with results from each paradigm
        """
        start_time = time.time()

        # Extract grids from task dictionary
        input_grid = torch.tensor(arc_task["input_grid"], dtype=torch.float32)
        target_grid = None
        if "target_grid" in arc_task:
            # Convert the target_grid to a float32 tensor and store it back
            target_grid = torch.tensor(arc_task["target_grid"], dtype=torch.float32)
            arc_task["target_grid"] = target_grid

        context = {
            "task_complexity": arc_task.get("task_complexity", 0.5),
            "predicted_grid": arc_task.get("predicted_grid"),
        }

        # Initialize ensemble state
        self.current_ensemble_state = EnsembleState(
            current_stage=ProcessingStage.INITIALIZATION, start_time=start_time
        )

        try:
            # Stage 1: Complexity Assessment and Effort Allocation
            await self._stage_complexity_assessment(input_grid, context)

            # Stage 2: Pipeline Execution
            await self._execute_processing_pipeline(input_grid, context)

            # Stage 3: Consensus Building
            final_result = await self._build_final_consensus()

            # Update cognitive metrics
            await self._update_cognitive_metrics(final_result)

            logger.info(
                f"ARC task processing completed in {time.time() - start_time:.2f}s "
                f"with confidence {final_result.confidence:.3f}"
            )

            return final_result

        except Exception as e:
            logger.error(f"ARC task processing failed: {e}")
            return ProcessingResult(
                stage=ProcessingStage.OUTPUT_GENERATION,
                component_name="ARCEnsembleOrchestrator",
                output=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                resource_usage={"memory": 0.1, "compute": 0.1},
            )

    async def _stage_complexity_assessment(
        self, input_grid: torch.Tensor, context: Dict[str, Any]
    ):
        """Stage 1: Assess complexity and allocate effort budget"""
        self.current_ensemble_state.current_stage = ProcessingStage.INITIALIZATION

        # Assess problem complexity
        complexity_measurement = await self.complexity_monitor.assess_complexity(
            input_grid
        )
        self.current_ensemble_state.complexity_measurement = complexity_measurement

        # Allocate effort budget based on complexity
        effort_budget = await self.effort_controller.allocate_effort_budget(
            input_grid, complexity_measurement.overall_complexity
        )
        self.current_ensemble_state.effort_budget = effort_budget

        # Start complexity monitoring
        await self.complexity_monitor.start_monitoring(input_grid)

        logger.info(
            f"Complexity assessed: {complexity_measurement.overall_complexity:.3f}, "
            f"budget allocated: {effort_budget.total_budget:.2f}"
        )

    async def _execute_processing_pipeline(
        self, input_grid: torch.Tensor, context: Dict[str, Any]
    ):
        """Stage 2: Execute the processing pipeline"""

        pipeline_stages = [
            ProcessingStage.ENCODING,
            ProcessingStage.BINDING,
            ProcessingStage.REASONING,
            ProcessingStage.PATTERN_LEARNING,
            ProcessingStage.GRAPH_REASONING,
        ]

        current_input = input_grid

        for stage in pipeline_stages:
            if stage not in self.stage_agents:
                logger.warning(f"No agents available for stage {stage}")
                continue

            self.current_ensemble_state.current_stage = stage

            # Get effort allocation for this stage
            stage_effort = self._get_stage_effort_allocation(stage)

            if stage_effort <= 0:
                logger.info(f"Skipping stage {stage} due to zero effort allocation")
                continue

            # Process stage with available agents
            stage_results = await self._process_stage(
                stage, current_input, context, stage_effort
            )

            # Add results to ensemble state
            for result in stage_results:
                self.current_ensemble_state.add_result(result)

            # Update input for next stage based on best result
            best_result = self.current_ensemble_state.get_best_result(stage)
            if best_result and best_result.output is not None:
                if isinstance(best_result.output, torch.Tensor):
                    current_input = best_result.output
                # Handle other output types as needed

            # Check for early termination
            if await self._should_terminate_early():
                logger.info(f"Early termination triggered at stage {stage}")
                break

        # Stop complexity monitoring
        self.complexity_monitor.stop_monitoring()

    async def _process_stage(
        self,
        stage: ProcessingStage,
        input_data: torch.Tensor,
        context: Dict[str, Any],
        effort_allocation: float,
    ) -> List[ProcessingResult]:
        """Process a single pipeline stage with available agents"""
        stage_agents = self.stage_agents.get(stage, [])

        if not stage_agents:
            return []

        # Decide whether to run agents in parallel or sequential
        if self.enable_parallelization and len(stage_agents) > 1:
            # Run compatible agents in parallel
            tasks = []
            for agent in stage_agents[
                :2
            ]:  # Limit parallelism to avoid resource exhaustion
                task = asyncio.create_task(agent.process(input_data, context))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and return valid results
            valid_results = []
            for result in results:
                if isinstance(result, ProcessingResult):
                    valid_results.append(result)
                else:
                    logger.error(f"Agent processing failed: {result}")

            return valid_results
        else:
            # Sequential processing
            results = []
            for agent in stage_agents:
                try:
                    result = await agent.process(input_data, context)
                    results.append(result)

                    # Early stopping if high confidence achieved
                    if result.confidence > 0.9:
                        break

                except Exception as e:
                    logger.error(f"Agent processing failed: {e}")

            return results

    def _get_stage_effort_allocation(self, stage: ProcessingStage) -> float:
        """Get effort allocation for a specific processing stage"""
        if not self.current_ensemble_state.effort_budget:
            return 0.2  # Default allocation

        budget = self.current_ensemble_state.effort_budget

        stage_allocations = {
            ProcessingStage.ENCODING: budget.spiking_processing,
            ProcessingStage.BINDING: budget.oscillatory_binding,
            ProcessingStage.REASONING: budget.logical_reasoning,
            ProcessingStage.PATTERN_LEARNING: budget.pattern_learning,
            ProcessingStage.GRAPH_REASONING: budget.graph_reasoning,
        }

        return stage_allocations.get(stage, 0.1)

    async def _should_terminate_early(self) -> bool:
        """Check if processing should terminate early"""
        if not self.current_ensemble_state.effort_budget:
            return False

        elapsed_time = time.time() - self.current_ensemble_state.start_time

        # Time-based termination
        if elapsed_time > self.current_ensemble_state.effort_budget.max_time_seconds:
            return True

        # Confidence-based termination
        recent_results = []
        for stage_results in self.current_ensemble_state.stage_results.values():
            recent_results.extend(
                stage_results[-1:]
            )  # Get most recent result per stage

        if recent_results:
            avg_confidence = np.mean([r.confidence for r in recent_results])
            if (
                avg_confidence
                > self.current_ensemble_state.effort_budget.early_termination_threshold
            ):
                return True

        return False

    async def _build_final_consensus(self) -> ProcessingResult:
        """Stage 3: Build final consensus from all results"""
        self.current_ensemble_state.current_stage = ProcessingStage.CONSENSUS

        consensus_result = await self.consensus_builder.build_consensus(
            self.current_ensemble_state.stage_results
        )

        # Generate final output
        self.current_ensemble_state.current_stage = ProcessingStage.OUTPUT_GENERATION

        final_result = ProcessingResult(
            stage=ProcessingStage.OUTPUT_GENERATION,
            component_name="ARCEnsembleOrchestrator",
            output=consensus_result.output,
            confidence=consensus_result.confidence,
            processing_time=time.time() - self.current_ensemble_state.start_time,
            resource_usage={
                "memory": np.mean(
                    [
                        r.resource_usage.get("memory", 0)
                        for results in self.current_ensemble_state.stage_results.values()
                        for r in results
                    ]
                ),
                "compute": np.mean(
                    [
                        r.resource_usage.get("compute", 0)
                        for results in self.current_ensemble_state.stage_results.values()
                        for r in results
                    ]
                ),
            },
            metadata={
                "consensus_metadata": consensus_result.metadata,
                "total_stages_processed": len(
                    self.current_ensemble_state.stage_results
                ),
                "ensemble_strategy": self.ensemble_strategy.value,
            },
        )

        return final_result

    async def _update_cognitive_metrics(self, final_result: ProcessingResult):
        """Update cognitive metrics based on processing results"""
        total_time = final_result.processing_time
        target_time = (
            self.current_ensemble_state.effort_budget.max_time_seconds
            if self.current_ensemble_state.effort_budget
            else 30.0
        )

        self.cognitive_metrics["orchestration_efficiency"] = min(
            target_time / max(total_time, 0.1), 1.0
        )
        self.cognitive_metrics["consensus_quality"] = final_result.confidence
        self.cognitive_metrics["resource_utilization"] = (
            final_result.resource_usage.get("compute", 0.5)
        )
        self.cognitive_metrics["pipeline_completion_rate"] = (
            len(self.current_ensemble_state.stage_results) / 5.0
        )  # 5 main stages

    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with lower efficiency and more resource usage
        efficiency_load = 1.0 - self.cognitive_metrics["orchestration_efficiency"]
        resource_load = self.cognitive_metrics["resource_utilization"]
        complexity_load = 0.5  # Base complexity load for orchestration

        return min(
            efficiency_load * 0.4 + resource_load * 0.3 + complexity_load * 0.3, 1.0
        )

    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # Orchestrator has very high symbolic depth - it reasons about reasoning
        base_depth = 6  # Meta-meta-cognitive orchestration
        agent_bonus = min(len(self.agents), 3)  # Bonus for managing multiple agents
        stage_bonus = len(self.stage_agents)  # Bonus for managing multiple stages
        return base_depth + agent_bonus + stage_bonus

    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "ARCEnsembleOrchestrator",
            "cognitive_metrics": self.cognitive_metrics,
            "registered_agents": list(self.agents.keys()),
            "stage_coverage": {
                stage.value: len(agents) for stage, agents in self.stage_agents.items()
            },
            "ensemble_strategy": self.ensemble_strategy.value,
            "current_state": {
                "stage": self.current_ensemble_state.current_stage.value
                if self.current_ensemble_state
                else "none",
                "num_results": len(self.current_ensemble_state.stage_results)
                if self.current_ensemble_state
                else 0,
            },
        }

    async def build_consensus(
        self, data: Union[Dict[str, Any], Dict[ProcessingStage, List[ProcessingResult]]]
    ) -> Dict[str, Any]:
        """
        Build consensus from either training context or processing results.

        Args:
            data: Either training context dict or stage results dict

        Returns:
            Consensus result as dictionary
        """
        # Handle training context (from async trainer)
        if isinstance(data, dict) and "current_accuracy" in data:
            return await self._build_training_consensus(data)

        # Handle standard processing results
        elif isinstance(data, dict) and any(
            isinstance(k, ProcessingStage) for k in data.keys()
        ):
            consensus_result = await self.consensus_builder.build_consensus(data)
            return {
                "confidence": consensus_result.confidence,
                "output": consensus_result.output,
                "processing_time": consensus_result.processing_time,
                "metadata": consensus_result.metadata,
            }

        # Fallback for unknown format
        else:
            logger.warning(f"Unknown consensus data format: {type(data)}")
            return {"confidence": 0.5, "output": None, "processing_time": 0.0}

    async def _build_training_consensus(
        self, training_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build consensus for training continuation decisions.

        Args:
            training_context: Dictionary with training metrics

        Returns:
            Training consensus decision
        """
        current_accuracy = training_context.get("current_accuracy", 0.0)
        best_accuracy = training_context.get("best_accuracy", 0.0)
        improvement_trend = training_context.get("improvement_trend", 0.0)
        recent_losses = training_context.get("recent_losses", [])

        # Calculate training confidence based on multiple factors
        accuracy_factor = min(current_accuracy / 0.85, 1.0)  # Target 85% accuracy
        improvement_factor = max(
            0.0, min(improvement_trend * 10, 1.0)
        )  # Positive improvement

        # Loss stability factor
        loss_stability = 1.0
        if len(recent_losses) >= 2:
            loss_variance = np.var(recent_losses)
            loss_stability = max(
                0.1, 1.0 - loss_variance
            )  # Lower variance = more stable

        # Progress factor (distance from best)
        progress_factor = 1.0 - abs(current_accuracy - best_accuracy)

        # Weighted consensus confidence
        consensus_confidence = (
            0.4 * accuracy_factor
            + 0.3 * improvement_factor
            + 0.2 * loss_stability
            + 0.1 * progress_factor
        )

        return {
            "confidence": consensus_confidence,
            "should_continue": consensus_confidence > 0.3,
            "factors": {
                "accuracy_factor": accuracy_factor,
                "improvement_factor": improvement_factor,
                "loss_stability": loss_stability,
                "progress_factor": progress_factor,
            },
            "recommendation": "continue" if consensus_confidence > 0.3 else "stop",
        }

    def process_arc_task_sync(self, arc_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_arc_task for non-async contexts.

        Args:
            arc_task: Dictionary containing ARC task data

        Returns:
            Paradigm results dictionary
        """
        try:
            # Try to run async method in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context but called the sync method
                logger.warning(
                    "process_arc_task_sync called from async context - use process_arc_task instead"
                )
                # Create a task that will be executed later
                future = asyncio.ensure_future(self.process_arc_task(arc_task))
                return {"_async_task": future, "confidence": 0.5}
            else:
                # Run in new event loop
                return loop.run_until_complete(self.process_arc_task(arc_task))
        except RuntimeError:
            # No event loop running, create new one
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(self.process_arc_task(arc_task))
                return result
            finally:
                new_loop.close()
        except Exception as e:
            logger.error(f"Sync processing failed: {e}")
            return {
                "logical_neural_units": {"confidence": 0.0, "error": str(e)},
                "akonr_binding": {"confidence": 0.0, "error": str(e)},
                "spiking_networks": {"confidence": 0.0, "error": str(e)},
                "deltanet_attention": {"confidence": 0.0, "error": str(e)},
            }

    def build_consensus_sync(
        self, data: Union[Dict[str, Any], Dict[ProcessingStage, List[ProcessingResult]]]
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for build_consensus for non-async contexts.
        """
        try:
            # Try to run async method in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context but called the sync method
                logger.warning(
                    "build_consensus_sync called from async context - use build_consensus instead"
                )
                future = asyncio.ensure_future(self.build_consensus(data))
                return {"_async_task": future, "confidence": 0.5}
            else:
                # Run in new event loop
                return loop.run_until_complete(self.build_consensus(data))
        except RuntimeError:
            # No event loop running, create new one
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(self.build_consensus(data))
                return result
            finally:
                new_loop.close()
        except Exception as e:
            logger.error(f"Sync consensus building failed: {e}")
            return {"confidence": 0.5, "should_continue": True, "error": str(e)}


# Factory function
async def create_arc_ensemble(config: Dict[str, Any]) -> ARCEnsembleOrchestrator:
    """Factory function to create and initialize ARC Ensemble Orchestrator"""
    orchestrator = ARCEnsembleOrchestrator(config)
    await orchestrator.async_init()
    return orchestrator


# Export main classes
__all__ = [
    "ARCEnsembleOrchestrator",
    "AgentContract",
    "SPLREncoderAgent",
    "AKOrNBinderAgent",
    "LNUReasonerAgent",
    "ConsensusBuilder",
    "ProcessingStage",
    "EnsembleStrategy",
    "ProcessingResult",
    "EnsembleState",
    "create_arc_ensemble",
]
