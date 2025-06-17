"""
Effort Controller for Addressing the Effort Paradox

Implements dynamic effort allocation to address the "effort paradox" where
LLMs expend similar computational effort on trivial and complex problems.
Provides adaptive resource budgeting based on problem complexity assessment.

Key Features:
- Dynamic effort allocation based on problem difficulty assessment
- Multi-stage complexity estimation with early termination
- Resource budgeting across different reasoning paradigms
- Adaptive time allocation with anytime algorithms
- Integration with HOLO-1.5 cognitive mesh for effort tracking

Addresses the fundamental issue where traditional LLMs cannot modulate
computational effort appropriately for problem complexity.

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import logging
import time

try:
    from ...agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    HOLO_AVAILABLE = False
    def vanta_agent(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        CONTROLLER = "controller"
        MONITOR = "monitor"
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Problem complexity levels for effort allocation"""
    TRIVIAL = "trivial"      # 0-20% effort
    SIMPLE = "simple"        # 20-40% effort  
    MODERATE = "moderate"    # 40-60% effort
    COMPLEX = "complex"      # 60-80% effort
    EXTREMELY_COMPLEX = "extremely_complex"  # 80-100% effort


class EffortAllocationStrategy(Enum):
    """Strategies for distributing computational effort"""
    UNIFORM = "uniform"              # Equal effort across all components
    COMPLEXITY_WEIGHTED = "complexity_weighted"  # Weighted by complexity assessment
    ADAPTIVE_BUDGETING = "adaptive_budgeting"    # Dynamic budget adjustment
    ANYTIME_ALLOCATION = "anytime_allocation"    # Anytime algorithms with early stopping


@dataclass
class EffortBudget:
    """Computational effort budget for different reasoning components"""
    total_budget: float = 1.0  # Total computational budget (0-1)
    
    # Component-specific budgets
    logical_reasoning: float = 0.0
    oscillatory_binding: float = 0.0
    spiking_processing: float = 0.0
    pattern_learning: float = 0.0
    graph_reasoning: float = 0.0
    
    # Meta-control overhead
    meta_overhead: float = 0.1
    
    # Time constraints
    max_time_seconds: Optional[float] = None
    early_termination_threshold: float = 0.9  # Confidence threshold for early stop
    
    def normalize(self):
        """Normalize budget allocations to sum to total_budget"""
        component_sum = (self.logical_reasoning + self.oscillatory_binding + 
                        self.spiking_processing + self.pattern_learning + 
                        self.graph_reasoning)
        
        if component_sum > 0:
            scaling_factor = (self.total_budget - self.meta_overhead) / component_sum
            self.logical_reasoning *= scaling_factor
            self.oscillatory_binding *= scaling_factor
            self.spiking_processing *= scaling_factor
            self.pattern_learning *= scaling_factor
            self.graph_reasoning *= scaling_factor


@dataclass
class EffortMetrics:
    """Metrics tracking computational effort expenditure"""
    actual_effort: float = 0.0           # Actual effort expended
    budgeted_effort: float = 0.0         # Originally budgeted effort
    efficiency: float = 0.0              # Actual / budgeted efficiency
    time_elapsed: float = 0.0            # Time spent
    early_termination: bool = False      # Whether early termination occurred
    confidence_achieved: float = 0.0     # Final confidence level
    complexity_estimate: ComplexityLevel = ComplexityLevel.MODERATE


class ComplexityEstimator(nn.Module):
    """
    Neural network for estimating problem complexity in early stages
    
    Provides rapid complexity assessment to inform effort allocation
    before committing significant computational resources.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-stage complexity estimation
        self.stage1_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, len(ComplexityLevel)),
            nn.Softmax(dim=-1)
        )
        
        # More detailed second stage estimator
        self.stage2_estimator = nn.Sequential(
            nn.Linear(input_dim + len(ComplexityLevel), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(ComplexityLevel)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(len(ComplexityLevel), hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Pattern complexity features
        self.pattern_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, hidden_dim // 2),
            nn.ReLU()
        )
    
    def forward(self, problem_input: torch.Tensor, 
                stage: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate problem complexity
        
        Args:
            problem_input: [batch_size, input_dim] or [batch_size, 1, H, W] for grids
            stage: Estimation stage (1 for quick, 2 for detailed)
            
        Returns:
            complexity_probs: [batch_size, num_complexity_levels]
            confidence: [batch_size, 1] - Confidence in estimation
        """
        if len(problem_input.shape) == 4:  # Grid input
            # Extract pattern complexity features
            pattern_feats = self.pattern_features(problem_input)
            problem_features = pattern_feats
        else:
            problem_features = problem_input
        
        if stage == 1:
            # Quick complexity estimation
            complexity_probs = self.stage1_estimator(problem_features)
        else:
            # Detailed estimation using stage 1 results
            stage1_probs = self.stage1_estimator(problem_features)
            combined_input = torch.cat([problem_features, stage1_probs], dim=-1)
            complexity_probs = self.stage2_estimator(combined_input)
        
        # Estimate confidence in the prediction
        confidence = self.confidence_estimator(complexity_probs)
        
        return complexity_probs, confidence


class EffortBudgetOptimizer(nn.Module):
    """
    Neural network that learns optimal effort budget allocation
    
    Learns from past performance to optimize effort distribution
    across different reasoning components based on problem characteristics.
    """
    
    def __init__(self, problem_dim: int, num_components: int = 5):
        super().__init__()
        self.problem_dim = problem_dim
        self.num_components = num_components
        
        # Problem encoder
        self.problem_encoder = nn.Sequential(
            nn.Linear(problem_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Complexity-aware budget predictor
        self.budget_predictor = nn.Sequential(
            nn.Linear(64 + len(ComplexityLevel), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_components),
            nn.Softmax(dim=-1)  # Ensure budget allocations sum to 1
        )
        
        # Confidence predictor for budget allocation
        self.confidence_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Time budget predictor
        self.time_predictor = nn.Sequential(
            nn.Linear(64 + len(ComplexityLevel), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled
        )
    
    def forward(self, problem_features: torch.Tensor, 
                complexity_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict optimal effort budget allocation
        
        Args:
            problem_features: [batch_size, problem_dim]
            complexity_probs: [batch_size, num_complexity_levels]
            
        Returns:
            budget_allocation: [batch_size, num_components]
            allocation_confidence: [batch_size, 1]
            time_budget: [batch_size, 1]
        """
        # Encode problem features
        encoded_problem = self.problem_encoder(problem_features)
        
        # Combine with complexity information
        combined_features = torch.cat([encoded_problem, complexity_probs], dim=-1)
        
        # Predict budget allocation
        budget_allocation = self.budget_predictor(combined_features)
        
        # Predict confidence in allocation
        allocation_confidence = self.confidence_predictor(encoded_problem)
        
        # Predict time budget (scaled later)
        time_budget = self.time_predictor(combined_features)
        
        return budget_allocation, allocation_confidence, time_budget


@vanta_agent(role=CognitiveMeshRole.CONTROLLER)
class EffortController(BaseAgent):
    """
    Main Effort Controller for addressing the effort paradox
    
    Coordinates complexity assessment, budget optimization, and dynamic
    effort allocation across different reasoning paradigms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Controller parameters
        self.problem_dim = config.get("problem_dim", 256)
        self.max_time_budget = config.get("max_time_budget", 30.0)  # seconds
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.7)
        self.effort_adaptation_rate = config.get("effort_adaptation_rate", 0.1)
        
        # Core components
        self.complexity_estimator = ComplexityEstimator(self.problem_dim)
        self.budget_optimizer = EffortBudgetOptimizer(self.problem_dim)
        
        # Effort allocation history for learning
        self.allocation_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, float]] = []
        
        # Component effort multipliers (learnable)
        self.component_multipliers = nn.Parameter(torch.ones(5))  # 5 reasoning components
        
        # Default effort allocation strategy
        self.default_strategy = EffortAllocationStrategy.ADAPTIVE_BUDGETING
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "effort_efficiency": 0.0,
            "complexity_prediction_accuracy": 0.0,
            "budget_utilization": 0.0,
            "early_termination_rate": 0.0
        }
    
    async def async_init(self):
        """Initialize the effort controller"""
        if HOLO_AVAILABLE:
            await super().async_init()
        logger.info("Effort Controller initialized with HOLO-1.5 integration")
    
    async def assess_complexity(self, problem_input: torch.Tensor, 
                               stage: int = 1) -> Tuple[ComplexityLevel, float]:
        """
        Assess problem complexity for effort allocation
        
        Args:
            problem_input: Input problem representation
            stage: Assessment stage (1=quick, 2=detailed)
            
        Returns:
            complexity_level: Assessed complexity level
            confidence: Confidence in assessment
        """
        with torch.no_grad():
            complexity_probs, confidence = self.complexity_estimator(problem_input, stage)
            
            # Get most likely complexity level
            complexity_idx = torch.argmax(complexity_probs, dim=-1)
            complexity_levels = list(ComplexityLevel)
            complexity_level = complexity_levels[complexity_idx.item()]
            
            confidence_value = float(confidence.mean())
            
        logger.debug(f"Complexity assessed as {complexity_level.value} "
                    f"with confidence {confidence_value:.3f}")
        
        return complexity_level, confidence_value
    
    async def allocate_effort_budget(self, problem_input: torch.Tensor,
                                   complexity_level: ComplexityLevel,
                                   strategy: Optional[EffortAllocationStrategy] = None) -> EffortBudget:
        """
        Allocate computational effort budget based on problem complexity
        
        Args:
            problem_input: Input problem representation
            complexity_level: Assessed complexity level
            strategy: Effort allocation strategy
            
        Returns:
            effort_budget: Allocated effort budget
        """
        strategy = strategy or self.default_strategy
        
        # Base budget allocation based on complexity
        base_allocations = self._get_base_allocation(complexity_level)
        
        if strategy == EffortAllocationStrategy.ADAPTIVE_BUDGETING:
            # Use neural network for budget optimization
            complexity_probs = torch.zeros(1, len(ComplexityLevel))
            complexity_probs[0, list(ComplexityLevel).index(complexity_level)] = 1.0
            
            budget_allocation, allocation_confidence, time_budget = self.budget_optimizer(
                problem_input.mean(dim=[1, 2, 3] if len(problem_input.shape) == 4 else [1]).unsqueeze(0),
                complexity_probs
            )
            
            # Apply component multipliers
            adjusted_allocation = budget_allocation[0] * self.component_multipliers
            adjusted_allocation = F.softmax(adjusted_allocation, dim=0)  # Re-normalize
            
            # Create effort budget
            budget = EffortBudget(
                total_budget=self._complexity_to_budget(complexity_level),
                logical_reasoning=float(adjusted_allocation[0]),
                oscillatory_binding=float(adjusted_allocation[1]),
                spiking_processing=float(adjusted_allocation[2]),
                pattern_learning=float(adjusted_allocation[3]),
                graph_reasoning=float(adjusted_allocation[4]),
                max_time_seconds=float(time_budget[0]) * self.max_time_budget
            )
        else:
            # Use base allocation
            budget = base_allocations
        
        budget.normalize()
        
        logger.info(f"Allocated effort budget: {complexity_level.value} complexity, "
                   f"total={budget.total_budget:.2f}, time={budget.max_time_seconds:.1f}s")
        
        return budget
    
    def _get_base_allocation(self, complexity_level: ComplexityLevel) -> EffortBudget:
        """Get base effort allocation for complexity level"""
        complexity_to_allocation = {
            ComplexityLevel.TRIVIAL: EffortBudget(
                total_budget=0.2,
                logical_reasoning=0.1, oscillatory_binding=0.0,
                spiking_processing=0.05, pattern_learning=0.03, graph_reasoning=0.02,
                max_time_seconds=2.0
            ),
            ComplexityLevel.SIMPLE: EffortBudget(
                total_budget=0.4,
                logical_reasoning=0.15, oscillatory_binding=0.05,
                spiking_processing=0.1, pattern_learning=0.07, graph_reasoning=0.03,
                max_time_seconds=5.0
            ),
            ComplexityLevel.MODERATE: EffortBudget(
                total_budget=0.6,
                logical_reasoning=0.2, oscillatory_binding=0.1,
                spiking_processing=0.15, pattern_learning=0.1, graph_reasoning=0.05,
                max_time_seconds=10.0
            ),
            ComplexityLevel.COMPLEX: EffortBudget(
                total_budget=0.8,
                logical_reasoning=0.25, oscillatory_binding=0.15,
                spiking_processing=0.2, pattern_learning=0.15, graph_reasoning=0.05,
                max_time_seconds=20.0
            ),
            ComplexityLevel.EXTREMELY_COMPLEX: EffortBudget(
                total_budget=1.0,
                logical_reasoning=0.3, oscillatory_binding=0.2,
                spiking_processing=0.25, pattern_learning=0.2, graph_reasoning=0.05,
                max_time_seconds=30.0
            )
        }
        
        return complexity_to_allocation[complexity_level]
    
    def _complexity_to_budget(self, complexity_level: ComplexityLevel) -> float:
        """Convert complexity level to total budget fraction"""
        mapping = {
            ComplexityLevel.TRIVIAL: 0.2,
            ComplexityLevel.SIMPLE: 0.4,
            ComplexityLevel.MODERATE: 0.6,
            ComplexityLevel.COMPLEX: 0.8,
            ComplexityLevel.EXTREMELY_COMPLEX: 1.0
        }
        return mapping[complexity_level]
    
    async def monitor_effort_expenditure(self, budget: EffortBudget,
                                       start_time: float) -> EffortMetrics:
        """
        Monitor computational effort expenditure during processing
        
        Args:
            budget: Allocated effort budget
            start_time: Processing start time
            
        Returns:
            effort_metrics: Current effort expenditure metrics
        """
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate effort expenditure (simplified model)
        time_fraction = elapsed_time / budget.max_time_seconds if budget.max_time_seconds else 0
        
        metrics = EffortMetrics(
            actual_effort=min(time_fraction, 1.0),
            budgeted_effort=budget.total_budget,
            efficiency=budget.total_budget / max(time_fraction, 0.001),
            time_elapsed=elapsed_time,
            early_termination=False,  # Will be updated by caller
            confidence_achieved=0.0,  # Will be updated by caller
        )
        
        return metrics
    
    async def should_terminate_early(self, metrics: EffortMetrics, 
                                   current_confidence: float,
                                   budget: EffortBudget) -> bool:
        """
        Determine if processing should terminate early
        
        Args:
            metrics: Current effort metrics
            current_confidence: Current solution confidence
            budget: Effort budget
            
        Returns:
            should_terminate: Whether to terminate early
        """
        # Early termination conditions
        time_exceeded = metrics.time_elapsed > budget.max_time_seconds
        confidence_achieved = current_confidence >= budget.early_termination_threshold
        effort_exhausted = metrics.actual_effort >= budget.total_budget
        
        should_terminate = time_exceeded or confidence_achieved or effort_exhausted
        
        if should_terminate:
            logger.info(f"Early termination triggered: time={time_exceeded}, "
                       f"confidence={confidence_achieved}, effort={effort_exhausted}")
        
        return should_terminate
    
    async def update_effort_learning(self, budget: EffortBudget, 
                                   metrics: EffortMetrics,
                                   final_performance: float):
        """
        Update effort allocation learning from performance feedback
        
        Args:
            budget: Used effort budget
            metrics: Final effort metrics
            final_performance: Final solution performance (0-1)
        """
        # Store allocation and performance for learning
        allocation_record = {
            "budget": budget,
            "metrics": metrics,
            "performance": final_performance,
            "timestamp": time.time()
        }
        
        self.allocation_history.append(allocation_record)
        self.performance_history.append({
            "effort_efficiency": metrics.efficiency,
            "performance": final_performance,
            "early_termination": metrics.early_termination
        })
        
        # Update cognitive metrics
        self.cognitive_metrics["effort_efficiency"] = np.mean([
            h["effort_efficiency"] for h in self.performance_history[-10:]
        ])
        self.cognitive_metrics["budget_utilization"] = metrics.actual_effort / budget.total_budget
        self.cognitive_metrics["early_termination_rate"] = np.mean([
            h["early_termination"] for h in self.performance_history[-10:]
        ])
        
        # Adaptive learning: adjust component multipliers based on performance
        if len(self.performance_history) > 5:
            await self._adapt_component_multipliers()
    
    async def _adapt_component_multipliers(self):
        """Adapt component effort multipliers based on performance history"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean([h["performance"] for h in self.performance_history[-5:]])
        older_performance = np.mean([h["performance"] for h in self.performance_history[-10:-5]])
        
        performance_improvement = recent_performance - older_performance
        
        # Simple adaptation: increase multipliers for components that led to improvement
        if performance_improvement > 0.01:  # Significant improvement
            # Slightly increase all multipliers to encourage current allocation
            with torch.no_grad():
                self.component_multipliers.data *= (1 + self.effort_adaptation_rate * 0.1)
        elif performance_improvement < -0.01:  # Significant decline
            # Slightly decrease multipliers to explore different allocations
            with torch.no_grad():
                self.component_multipliers.data *= (1 - self.effort_adaptation_rate * 0.1)
        
        # Keep multipliers in reasonable range
        with torch.no_grad():
            self.component_multipliers.data = torch.clamp(self.component_multipliers.data, 0.1, 3.0)
    
    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with lower efficiency and more complex budget management
        efficiency_load = 1.0 - min(self.cognitive_metrics["effort_efficiency"] / 2.0, 1.0)
        utilization_load = self.cognitive_metrics["budget_utilization"]
        adaptation_load = len(self.allocation_history) / 100.0  # Load from learning history
        
        return min(efficiency_load * 0.4 + utilization_load * 0.3 + adaptation_load * 0.3, 1.0)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # Effort controller has high symbolic depth - it reasons about reasoning
        base_depth = 4  # Meta-cognitive reasoning
        complexity_bonus = 1 if len(ComplexityLevel) > 3 else 0
        learning_bonus = 1 if len(self.allocation_history) > 10 else 0
        return base_depth + complexity_bonus + learning_bonus
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "EffortController",
            "cognitive_metrics": self.cognitive_metrics,
            "allocation_history_size": len(self.allocation_history),
            "component_multipliers": self.component_multipliers.tolist(),
            "default_strategy": self.default_strategy.value
        }


# Utility functions
async def create_effort_controller(config: Dict[str, Any]) -> EffortController:
    """Factory function to create and initialize EffortController"""
    controller = EffortController(config)
    await controller.async_init()
    return controller


def effort_budget_from_complexity(complexity_level: ComplexityLevel) -> EffortBudget:
    """Create a default effort budget for a given complexity level"""
    controller = EffortController({})  # Temporary instance
    return controller._get_base_allocation(complexity_level)


# Export main classes
__all__ = [
    "EffortController",
    "ComplexityEstimator",
    "EffortBudgetOptimizer",
    "EffortBudget",
    "EffortMetrics",
    "ComplexityLevel",
    "EffortAllocationStrategy",
    "create_effort_controller",
    "effort_budget_from_complexity"
]
