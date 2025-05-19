# filepath: c:\Users\16479\Desktop\ARC2025\voxsigil_supervisor\strategies\__init__.py
# voxsigil_supervisor/strategies/__init__.py
"""
Strategy modules for the VoxSigil Supervisor.

This sub-package contains strategy implementations for various aspects
of the supervisor's operation, including execution flow, evaluation,
and routing.
"""

# Base interfaces
from .scaffold_router import ScaffoldRouter as BaseScaffoldRouter
from .evaluation_heuristics import ResponseEvaluator as BaseEvaluationHeuristics
from .retry_policy import RetryPolicy as BaseRetryPolicy

# Optional, if implemented
try:
    from .execution_strategy import BaseExecutionStrategy
except ImportError:
    # Define a dummy class if not implemented yet
    class BaseExecutionStrategy:
        pass

__all__ = [
    "BaseScaffoldRouter",
    "BaseEvaluationHeuristics",
    "BaseRetryPolicy",
    "BaseExecutionStrategy"
]