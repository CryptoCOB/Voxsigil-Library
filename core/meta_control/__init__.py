"""
Meta-Control Systems for Novel LLM Paradigms

This module implements meta-cognitive control systems to address fundamental
LLM limitations like the "effort paradox" and complexity management:

- Effort Controller: Addresses effort paradox with dynamic resource allocation
- Complexity Monitor: Real-time complexity assessment and adaptation
- Resource Allocator: Dynamic computational budgeting across paradigms
- Meta-Learning Controller: Adaptive strategy selection and learning

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration.
"""

from .effort_controller import (
    EffortController, ComplexityEstimator, EffortBudgetOptimizer,
    EffortBudget, EffortMetrics, ComplexityLevel, EffortAllocationStrategy,
    create_effort_controller, effort_budget_from_complexity
)

from .complexity_monitor import (
    ComplexityMonitor, VisualComplexityAnalyzer, LogicalComplexityAnalyzer,
    ResourceRequirementPredictor, ComplexityMeasurement, ComplexityProfile,
    ComplexityDimension, ComplexityTrend, create_complexity_monitor,
    analyze_complexity_dimensions
)

__all__ = [
    # Effort Controller
    "EffortController",
    "ComplexityEstimator",
    "EffortBudgetOptimizer",
    "EffortBudget",
    "EffortMetrics",
    "ComplexityLevel",
    "EffortAllocationStrategy",
    "create_effort_controller",
    "effort_budget_from_complexity",
    
    # Complexity Monitor
    "ComplexityMonitor",
    "VisualComplexityAnalyzer",
    "LogicalComplexityAnalyzer",
    "ResourceRequirementPredictor",
    "ComplexityMeasurement",
    "ComplexityProfile",
    "ComplexityDimension",
    "ComplexityTrend",
    "create_complexity_monitor",
    "analyze_complexity_dimensions"
]

# Version and compatibility info
__version__ = "1.0.0"
__holo_compatible__ = "1.5.0"
__paradigms__ = [
    "meta_cognitive_control",
    "effort_paradox_mitigation", 
    "complexity_management",
    "dynamic_resource_allocation",
    "adaptive_strategy_selection",
    "real_time_monitoring",
    "computational_budgeting"
]
