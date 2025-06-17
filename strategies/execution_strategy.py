# voxsigil_supervisor/strategies/execution_strategy.py
"""
Defines the base interface for execution strategies in the VoxSigil Supervisor.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.
    Implementations will handle the specific flow of execution for complex
    reasoning scaffolds, such as multi-step reasoning or iterative refinement.
    """
    
    @abstractmethod
    def execute_scaffold(self,
                        query: str,
                        scaffold: Dict[str, Any],
                        context: Dict[str, Any],
                        llm_interface: Any,
                        rag_interface: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a reasoning scaffold with the given LLM and RAG interfaces.
        
        Args:
            query: The original user query
            scaffold: The reasoning scaffold to execute
            context: The context including RAG results
            llm_interface: The LLM interface to use
            rag_interface: The RAG interface to use
            
        Returns:
            Tuple containing the final response and execution metadata
        """
        raise NotImplementedError
    
    @abstractmethod
    def process_intermediate_step(self,
                                 step_result: Any,
                                 execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an intermediate step in the execution.
        
        Args:
            step_result: The result of the current step
            execution_state: The current execution state
            
        Returns:
            Updated execution state
        """
        raise NotImplementedError
    
    @abstractmethod
    def should_continue_execution(self,
                                 execution_state: Dict[str, Any]) -> bool:
        """
        Determine if execution should continue.
        
        Args:
            execution_state: The current execution state
            
        Returns:
            Boolean indicating whether to continue execution
        """
        raise NotImplementedError
