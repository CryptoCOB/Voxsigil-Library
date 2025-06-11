"""
Specialized Interface Definitions
================================

Extended interfaces for specific modules and capabilities in the
VoxSigil Library. These build upon base interfaces to provide
specialized functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from .base_interfaces import (
    BaseRagInterface, 
    BaseLlmInterface, 
    BaseMemoryInterface,
    BaseAgentInterface,
    BaseModelInterface
)


class MetaLearnerInterface(BaseAgentInterface):
    """
    Meta-Learning Agent Interface
    
    Extends base agent with meta-learning capabilities:
    - Learning from task patterns
    - Strategy adaptation and optimization
    - Cross-domain knowledge transfer
    """
    
    @abstractmethod
    async def learn_from_experience(
        self,
        task_history: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn patterns from task execution history."""
        pass
    
    @abstractmethod
    async def adapt_strategy(
        self,
        current_strategy: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt strategy based on performance."""
        pass
    
    @abstractmethod
    async def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Transfer knowledge between domains."""
        pass
    
    @abstractmethod
    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Get meta-learning performance metrics."""
        pass


class ModelManagerInterface(BaseModelInterface):
    """
    Advanced Model Management Interface
    
    Extends base model interface with advanced management:
    - Multi-model orchestration
    - Dynamic model switching
    - Performance optimization
    """
    
    @abstractmethod
    async def register_model(
        self,
        model_id: str,
        model_config: Dict[str, Any]
    ) -> bool:
        """Register a new model for management."""
        pass
    
    @abstractmethod
    async def switch_model(
        self,
        model_id: str,
        transition_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Switch to a different registered model."""
        pass
    
    @abstractmethod
    async def get_model_performance(
        self,
        model_id: str,
        metric_types: List[str]
    ) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        pass
    
    @abstractmethod
    async def optimize_model_selection(
        self,
        task_requirements: Dict[str, Any]
    ) -> str:
        """Select optimal model for task requirements."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available registered models."""
        pass


class BLTInterface(BaseAgentInterface):
    """
    BLT (Build, Learn, Test) Module Interface
    
    Specialized interface for BLT module capabilities:
    - Iterative development cycles
    - Automated testing and validation
    - Performance monitoring
    """
    
    @abstractmethod
    async def build_component(
        self,
        specification: Dict[str, Any],
        build_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a component from specification."""
        pass
    
    @abstractmethod
    async def learn_from_feedback(
        self,
        component_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from component performance feedback."""
        pass
    
    @abstractmethod
    async def test_component(
        self,
        component_id: str,
        test_suite: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test component with provided test suite."""
        pass
    
    @abstractmethod
    async def get_build_metrics(
        self,
        component_id: str
    ) -> Dict[str, Any]:
        """Get build and performance metrics."""
        pass


class ARCInterface(BaseAgentInterface):
    """
    ARC (Abstraction and Reasoning Corpus) Interface
    
    Specialized interface for ARC module capabilities:
    - Pattern recognition and abstraction
    - Reasoning task execution
    - Solution generation and validation
    """
    
    @abstractmethod
    async def analyze_pattern(
        self,
        input_grid: List[List[int]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze pattern in input grid."""
        pass
    
    @abstractmethod
    async def generate_solution(
        self,
        problem: Dict[str, Any],
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate solution for ARC problem."""
        pass
    
    @abstractmethod
    async def validate_solution(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate solution against problem."""
        pass
    
    @abstractmethod
    async def get_reasoning_trace(
        self,
        problem_id: str
    ) -> List[Dict[str, Any]]:
        """Get reasoning trace for problem solving."""
        pass


class ARTInterface(BaseAgentInterface):
    """
    ART (Autonomous Reasoning Tool) Interface
    
    Specialized interface for ART module capabilities:
    - Autonomous reasoning and planning
    - Tool synthesis and composition
    - Adaptive problem solving
    """
    
    @abstractmethod
    async def reason_about_problem(
        self,
        problem_description: str,
        available_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Reason about problem and approach."""
        pass
    
    @abstractmethod
    async def synthesize_tool(
        self,
        tool_specification: Dict[str, Any],
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synthesize new tool from specification."""
        pass
    
    @abstractmethod
    async def compose_tools(
        self,
        tool_ids: List[str],
        composition_strategy: str
    ) -> Dict[str, Any]:
        """Compose multiple tools into workflow."""
        pass
    
    @abstractmethod
    async def adapt_reasoning(
        self,
        feedback: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt reasoning approach based on feedback."""
        pass


class MiddlewareInterface(ABC):
    """
    Middleware Integration Interface
    
    Interface for middleware components that connect modules:
    - Request/response handling
    - Data transformation and routing
    - Cross-module communication
    """
    
    @abstractmethod
    async def process_request(
        self,
        request: Dict[str, Any],
        source_module: str,
        target_module: str
    ) -> Dict[str, Any]:
        """Process inter-module request."""
        pass
    
    @abstractmethod
    async def transform_data(
        self,
        data: Any,
        source_format: str,
        target_format: str
    ) -> Any:
        """Transform data between module formats."""
        pass
    
    @abstractmethod
    async def route_message(
        self,
        message: Dict[str, Any],
        routing_rules: Dict[str, Any]
    ) -> List[str]:
        """Route message to appropriate modules."""
        pass
    
    @abstractmethod
    async def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware performance statistics."""
        pass
