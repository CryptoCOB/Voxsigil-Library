#!/usr/bin/env python
"""
supervisor_connector_interface.py - Interface definition for Supervisor Connector

This file defines the base interface that must be implemented by any Supervisor Connector
that will be used with the VantaCore system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseSupervisorConnector(ABC):
    """
    Abstract Base Class for a VoxSigil Supervisor Connector.
    Defines the interface for VantaCore to interact with supervisor services.
    """

    @abstractmethod
    def get_sigil_content_as_dict(self, sigil_ref: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the content of a sigil and parses it as a dictionary.

        Args:
            sigil_ref (str): The reference ID of the sigil.

        Returns:
            Optional[Dict[str, Any]]: The sigil content as a dictionary, or None if not found or parsing fails.
        """
        pass

    @abstractmethod
    def get_sigil_content_as_text(self, sigil_ref: str) -> Optional[str]:
        """
        Retrieves the content of a sigil as raw text.

        Args:
            sigil_ref (str): The reference ID of the sigil.

        Returns:
            Optional[str]: The sigil content as text, or None if not found.
        """
        pass

    @abstractmethod
    def store_sigil_content(
        self, sigil_ref: str, content: Any, content_type: str = "application/json"
    ) -> bool:
        """
        Stores content into a sigil. (Replaces create_sigil and update_sigil for simplicity if sigil_ref is known)
        If sigil_ref does not exist, it could be created. Or use a separate create_sigil if distinction is important.

        Args:
            sigil_ref (str): The reference ID of the sigil to store content into.
            content (Any): The content to store.
            content_type (str): The MIME type of the content.

        Returns:
            bool: True if storage was successful, False otherwise.
        """
        pass

    @abstractmethod
    def create_sigil(
        self,
        desired_sigil_ref: str,
        initial_content: Any,
        sigil_type: str,
        tags: Optional[List[str]] = None,
        related_sigils: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Requests the supervisor to create a new sigil.

        Args:
            desired_sigil_ref (str): The desired sigil reference (may be modified by supervisor).
            initial_content (Any): Initial content for the sigil.
            sigil_type (str): The type of sigil to create (e.g., 'data', 'config', 'module_registration', 'performance_metric').
            tags (Optional[List[str]]): Optional tags for categorization.
            related_sigils (Optional[List[str]]): Optional related sigil references.

        Returns:
            Optional[str]: The sigil_ref of the newly created sigil, or None on failure.
        """
        pass

    @abstractmethod
    def register_module_with_supervisor(
        self,
        module_name: str,
        module_capabilities: Dict[str, Any],
        requested_sigil_ref: Optional[str] = None,
    ) -> Optional[str]:
        """
        Registers a module (like VantaCore) with the supervisor.

        Args:
            module_name (str): The name of the module.
            module_capabilities (Dict[str, Any]): A dictionary describing the module's capabilities.
            requested_sigil_ref (Optional[str]): An optional requested sigil_ref for the registration.

        Returns:
            Optional[str]: A registration sigil_ref if successful, None otherwise.
        """
        pass

    @abstractmethod
    def perform_health_check(self, module_registration_sigil_ref: str) -> bool:
        """
        Performs a health check with the supervisor for the registered module.

        Args:
            module_registration_sigil_ref (str): The sigil_ref obtained during module registration.

        Returns:
            bool: True if the health check is successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_module_health(self, registration_sigil_ref: str) -> Dict[str, Any]:
        """
        Gets detailed health information for a registered module.

        Args:
            registration_sigil_ref (str): The sigil_ref of the module registration.

        Returns:
            Dict[str, Any]: Health status dictionary with 'status' and 'details' fields.
        """
        pass

    @abstractmethod
    def search_sigils(
        self, query_criteria: Dict[str, Any], max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for sigils matching the given criteria.

        Args:
            query_criteria (Dict[str, Any]): Search criteria (e.g., {'prefix': 'SigilRef:Vanta'}).
            max_results (Optional[int]): Maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: List of matching sigil information dictionaries.
        """
        pass

    @abstractmethod
    def find_similar_examples(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find similar examples using semantic search/embeddings.

        Args:
            query (Dict[str, Any]): Query parameters including:
                - embedding: List of floats representing the embedding vector
                - max_results: Maximum number of results to return
                - min_similarity: Minimum similarity threshold
                - collection: Collection/category to search in

        Returns:
            List[Dict[str, Any]]: List of similar examples with similarity scores
        """
        pass

    @abstractmethod
    def call_llm(self, llm_params: Dict[str, Any]) -> Any:
        """
        Call a Large Language Model through the supervisor.

        Args:
            llm_params (Dict[str, Any]): LLM parameters including:
                - model: Model name to use
                - prompt: Input prompt
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - format: Expected response format

        Returns:
            Any: LLM response (usually string or parsed JSON)
        """
        pass
