#!/usr/bin/env python3
"""
memory_service_connector.py: Service connector for UnifiedMemoryInterface.

This module provides a connector to register the UnifiedMemoryInterface
with the UnifiedVantaCore and expose memory services to other components.
"""

import logging
from typing import Any, Dict, List, Optional

from Vanta.core.UnifiedMemoryInterface import UnifiedMemoryInterface
from Vanta.core.UnifiedVantaCore import get_vanta_core

logger = logging.getLogger("Vanta.MemoryServiceConnector")


class MemoryServiceConnector:
    """
    Connector to register UnifiedMemoryInterface with UnifiedVantaCore
    and provide memory services to other components.
    """

    def __init__(self, config=None):
        """
        Initialize the MemoryServiceConnector.

        Args:
            config: Configuration object with memory settings
        """
        self.config = config
        self.vanta_core = get_vanta_core()
        self.memory_interface = None

        # Initialize and register memory interface
        self._initialize_memory_interface()

    def _initialize_memory_interface(self):
        """Initialize the UnifiedMemoryInterface and register it with VantaCore."""
        try:
            self.memory_interface = UnifiedMemoryInterface(
                vanta_core=self.vanta_core, config=self.config
            )

            # Register memory service with VantaCore
            if self.vanta_core:
                self.vanta_core.register_component(
                    "memory_service",
                    self,
                    meta={
                        "type": "memory_service",
                        "provides": [
                            "store",
                            "retrieve",
                            "retrieve_similar",
                            "update",
                            "imprint",
                            "log_event",
                            "get_interaction_history",
                        ],
                    },
                )
                logger.info("MemoryServiceConnector registered with VantaCore")
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedMemoryInterface: {e}")

    # Public API methods

    def store(
        self,
        key: str,
        value: Any,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Store data in the memory system.

        Args:
            key: Unique identifier for the data
            value: The data to store
            namespace: Optional namespace to organize memory
            metadata: Optional metadata to associate with the value
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            Unique identifier for the stored data
        """
        if self.memory_interface:
            meta = metadata or {}
            return self.memory_interface.store(
                key, value, namespace, meta, ttl_seconds
            )
        logger.error("Memory interface not initialized")
        return ""

    def retrieve(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve data from the memory system.

        Args:
            key: Key to retrieve
            namespace: Optional namespace

        Returns:
            Retrieved value or None if not found
        """
        if self.memory_interface:
            return self.memory_interface.retrieve(key, namespace)
        logger.error("Memory interface not initialized")
        return None

    def retrieve_similar(
        self, query: str, limit: int = 3, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve data with similar keys from the memory system.

        Args:
            query: Query to find similar entries for
            limit: Maximum number of results to return
            namespace: Optional namespace

        Returns:
            List of similar entries
        """
        if self.memory_interface:
            return self.memory_interface.retrieve_similar(query, limit, namespace)
        logger.error("Memory interface not initialized")
        return []

    def update(self, interaction_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.

        Args:
            interaction_id: ID of the interaction to update
            updates: Dictionary of updates to apply

        Returns:
            Boolean indicating success
        """
        if self.memory_interface:
            return self.memory_interface.update(interaction_id, updates)
        logger.error("Memory interface not initialized")
        return False

    def imprint(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Directly imprint data in the MemoryBraid system.

        Args:
            key: Key for the memory entry
            value: Value to store
            ttl_seconds: Optional time-to-live in seconds
        """
        if self.memory_interface:
            self.memory_interface.imprint(key, value, ttl_seconds)
        else:
            logger.error("Memory interface not initialized")

    def log_event(self, task_id: str, event_type: str, event_data: Any) -> None:
        """
        Log an event to EchoMemory.

        Args:
            task_id: Identifier for the task
            event_type: Type of event
            event_data: Data associated with the event
        """
        if self.memory_interface:
            self.memory_interface.log_event(task_id, event_type, event_data)
        else:
            logger.error("Memory interface not initialized")

    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent interaction history.

        Args:
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interaction dictionaries
        """
        if self.memory_interface:
            return self.memory_interface.get_interaction_history(limit)
        logger.error("Memory interface not initialized")
        return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory subsystems.

        Returns:
            Dictionary of statistics
        """
        if self.memory_interface:
            return self.memory_interface.get_memory_stats()
        logger.error("Memory interface not initialized")
        return {
            "file_memory": {"interactions": 0},
            "memory_braid": {"semantic_size": 0, "episodic_size": 0},
            "echo_memory": {"log_size": 0, "total_logged": 0},
        }


def initialize_memory_service(config=None):
    """
    Initialize and return a MemoryServiceConnector instance.

    Args:
        config: Configuration object with memory settings

    Returns:
        MemoryServiceConnector instance
    """
    return MemoryServiceConnector(config)
