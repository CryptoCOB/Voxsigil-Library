#!/usr/bin/env python
"""
hybrid_middleware_interface.py - Interface definition for HybridMiddleware

This file defines the base interface that must be implemented by any Hybrid Middleware
that will be used with the VantaCore system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseHybridMiddleware(ABC):
    """
    Abstract Base Class for the Hybrid RAG and Cognitive Operations Middleware.
    Defines the interface for VantaCore to delegate complex task processing.
    """

    @abstractmethod
    def find_similar_examples(self, example_data: Any) -> Optional[Dict[str, Any]]:
        """
        Finds similar examples based on the provided data.

        Args:
            example_data (Any): Data to find similar examples for.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of similar examples or None if no examples are found.
        """
        pass

    @abstractmethod
    def process_arc_task(
        self,
        input_data_sigil_ref: str,
        task_definition_sigil_ref: str,
        task_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Processes an ARC (Abstract Reasoning Challenge) task or a similar complex task.

        This method is expected to handle:
        - Retrieval of task data and definitions using the provided sigil references.
        - Application of reasoning, planning, and problem-solving strategies (potentially involving LLMs, RAG, etc.).
        - Generation of a solution.
        - Generation of a performance metric for the attempt.

        Args:
            input_data_sigil_ref (str): Sigil reference to the input data for the task.
            task_definition_sigil_ref (str): Sigil reference to the task definition (e.g., ARC problem).
            task_parameters (Optional[Dict[str, Any]]):
                Additional parameters to guide task processing. This can include adaptive parameters
                from VantaCore (e.g., 'effective_temperature', 'max_solution_attempts', 'learning_context').

        Returns:
            Tuple[Optional[str], Optional[str]]:
                A tuple containing:
                - The sigil_ref of the generated solution (or None if no solution was found).
                - The sigil_ref of the performance metric for this processing attempt (or None).
                  The performance metric sigil should contain at least an "achieved_performance" field (float, 0.0 to 1.0).
        """
        pass

    @abstractmethod
    def get_middleware_capabilities(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the capabilities of this middleware instance.
        This could include supported task types, integrated models, etc.

        Returns:
            Dict[str, Any]: A dictionary of capabilities.
        """
        pass

    @abstractmethod
    def configure_middleware(self, config: Dict[str, Any]) -> bool:
        """
        Configures the middleware with new settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            bool: True if configuration was successful, False otherwise.
        """
        pass
