"""
Base LLM Interface for ARC Tasks

This module defines the abstract base class for LLM interfaces used in ARC tasks.
It provides a standard interface that different LLM implementations can inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BaseLlmInterface(ABC):
    """
    Abstract base class for LLM interfaces used in ARC tasks.

    This class defines the standard interface that all LLM implementations
    should implement to be compatible with the ARC system.
    """

    @abstractmethod
    def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        task_requirements: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        system_prompt_override: Optional[str] = None,
        use_global_system_prompt: bool = True,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generate a response from the LLM.

        Args:
            messages: Either a string prompt or a list of message dictionaries
            task_requirements: Optional requirements for the task (used for parameter settings)
            temperature: Optional temperature parameter for controlling randomness
            system_prompt_override: Optional system prompt to use instead of the default
            use_global_system_prompt: Whether to use the global system prompt
            **kwargs: Additional parameters for the LLM

        Returns:
            A tuple containing:
            - response_text: The generated text response
            - model_info: Information about the model used (name, provider, etc.)
            - response_metadata: Additional metadata about the response
        """
        pass

    @abstractmethod
    def select_model(
        self, task_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select and return model information.

        Args:
            task_requirements: Optional task requirements that might influence model selection

        Returns:
            Dictionary containing model information such as name, provider, capabilities, etc.
        """
        pass

    def build_prompt(
        self,
        query: str,
        context: str,
        scaffold: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a structured prompt from query, context, and optional components.

        This is a default implementation that can be overridden by subclasses.

        Args:
            query: The main query or question
            context: Contextual information to include
            scaffold: Optional reasoning scaffold or template
            history: Optional conversation history

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if scaffold:
            prompt_parts.append(f"<<REASONING_SCAFFOLD>>\n{scaffold}\n")

        if context:
            prompt_parts.append(f"<<CONTEXT>>\n{context}\n")

        if history:
            prompt_parts.append("<<HISTORY>>")
            for entry in history[-3:]:  # Last 3 entries
                role = entry.get("role", "unknown").upper()
                content = entry.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")

        prompt_parts.append(f"<<QUERY>>\n{query}")

        return "\n".join(prompt_parts)

    def validate_response(
        self, response: str, expected_format: Optional[str] = None
    ) -> bool:
        """
        Validate a response from the LLM.

        This is a default implementation that can be overridden by subclasses.

        Args:
            response: The response to validate
            expected_format: Optional expected format (e.g., "json", "grid")

        Returns:
            True if response is valid, False otherwise
        """
        if not response or not isinstance(response, str):
            return False

        if expected_format == "json":
            try:
                import json

                json.loads(response.strip())
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        elif expected_format == "grid":
            # Basic grid validation - check for nested list structure
            try:
                import json

                data = json.loads(response.strip())
                return isinstance(data, list) and all(
                    isinstance(row, list) for row in data
                )
            except (json.JSONDecodeError, ValueError):
                return False

        return True  # Default to valid if no specific format required

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this LLM interface.

        Returns:
            Dictionary describing the capabilities of this interface
        """
        return {
            "supports_streaming": False,
            "supports_embeddings": False,
            "supports_chat": True,
            "supports_completion": True,
            "max_context_length": None,
            "model_types": ["text_generation"],
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM interface.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "unknown",
            "timestamp": None,
            "error": "Health check not implemented",
        }
