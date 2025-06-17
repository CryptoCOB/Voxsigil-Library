#!/usr/bin/env python3
"""
UnifiedMemoryInterface.py: A central interface for all memory operations in the VoxSigil system.

This module provides a unified interface to integrate various memory subsystems:
- BaseMemoryInterface and JsonFileMemoryInterface (from both Vanta and VoxSigil)
- MemoryBraid for hybrid memory capabilities
- EchoMemory for structured event logging

The unified interface allows seamless interaction with multiple memory systems through
a single consistent API, supporting both direct access and component-based communication.
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import memory components
from interfaces.memory_interface import JsonFileMemoryInterface
from memory.echo_memory import EchoMemory
from memory.memory_braid import MemoryBraid
from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType

# Setup the logger
logger = logging.getLogger("Vanta.UnifiedMemoryInterface")


class UnifiedMemoryInterface:
    """
    Central interface for all memory operations in the VoxSigil system.

    This class provides:
    1. A unified API for memory operations across different memory systems
    2. Automatic routing of memory operations to appropriate subsystems
    3. Synchronization between different memory stores
    4. Memory operation event broadcasting for system-wide awareness
    5. Fallback mechanisms for memory subsystem failures
    """

    def __init__(self, vanta_core=None, config=None):
        """
        Initialize the UnifiedMemoryInterface with all memory subsystems.

        Args:
            vanta_core: Reference to UnifiedVantaCore for component registration
            config: Configuration object with memory settings
        """
        self.vanta_core = vanta_core
        self.config = config
        self._lock = threading.RLock()
        self._event_listeners = defaultdict(list)

        # Default configuration
        self.default_namespace = "default"
        self.default_ttl_seconds = 3600  # 1 hour
        self.memory_dir = None

        # Apply configuration if provided
        if config:
            self.default_namespace = getattr(
                config, "default_memory_namespace", self.default_namespace
            )
            self.default_ttl_seconds = getattr(
                config, "memory_ttl_seconds", self.default_ttl_seconds
            )
            memory_dir = getattr(config, "memory_dir", None)
            if memory_dir:
                self.memory_dir = Path(memory_dir)
                self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory subsystems
        self._initialize_memory_subsystems()

        # Register with VantaCore if available
        if self.vanta_core:
            try:
                self.vanta_core.register_component(
                    "unified_memory", self, metadata={"type": "memory_service"}
                )
                if hasattr(self.vanta_core, "async_bus"):
                    self.vanta_core.async_bus.register_component("unified_memory")
                    self.vanta_core.async_bus.subscribe(
                        "unified_memory",
                        MessageType.MEMORY_OPERATION,
                        self.handle_memory_operation,
                    )
                logger.info("UnifiedMemoryInterface registered with VantaCore")
            except Exception as e:
                logger.error(f"Failed to register UnifiedMemoryInterface with VantaCore: {e}")

    def _initialize_memory_subsystems(self):
        """Initialize all memory subsystems with appropriate configuration."""
        # Initialize subsystems with exception handling
        try:
            # Initialize JsonFileMemoryInterface
            if self.memory_dir:
                self.file_memory = JsonFileMemoryInterface(memory_dir=self.memory_dir)
            else:
                self.file_memory = JsonFileMemoryInterface()
            logger.info(
                f"JsonFileMemoryInterface initialized with memory_dir: {self.file_memory.memory_dir}"
            )

            # Initialize MemoryBraid
            self.memory_braid = MemoryBraid(
                vanta_core=None,  # We'll manage the registration ourselves
                max_episodic_len=getattr(self.config, "max_episodic_memory_len", 128),
                default_semantic_ttl_seconds=self.default_ttl_seconds,
                auto_decay_on_imprint=True,
            )
            logger.info("MemoryBraid initialized")

            # Initialize EchoMemory
            self.echo_memory = EchoMemory(
                vanta_core=None,  # We'll manage the registration ourselves
                max_log_size=getattr(self.config, "max_echo_memory_size", 10000),
                enable_persistence=getattr(self.config, "enable_echo_persistence", False),
                persistence_path=getattr(self.config, "echo_persistence_path", None),
            )
            logger.info("EchoMemory initialized")

        except Exception as e:
            logger.error(f"Error initializing memory subsystems: {e}")
            # Create fallback implementations if initialization fails
            if not hasattr(self, "file_memory"):
                self.file_memory = JsonFileMemoryInterface()
            if not hasattr(self, "memory_braid"):
                self.memory_braid = MemoryBraid()
            if not hasattr(self, "echo_memory"):
                self.echo_memory = EchoMemory()

    def handle_memory_operation(self, message: AsyncMessage):
        """
        Handle memory operation messages from the async bus.

        Args:
            message: AsyncMessage containing memory operation details
        """
        try:
            content = message.content or {}
            operation = content.get("operation")

            if operation == "store":
                self.store(
                    content.get("key", "unknown"),
                    content.get("value"),
                    namespace=content.get("namespace", self.default_namespace),
                    metadata=content.get("metadata"),
                    ttl_seconds=content.get("ttl_seconds", self.default_ttl_seconds),
                )
            elif operation == "retrieve":
                result = self.retrieve(
                    content.get("key"),
                    namespace=content.get("namespace", self.default_namespace),
                )
                # Send result back if reply_to is specified
                reply_to = getattr(message, "reply_to", None)
                if reply_to:
                    self.vanta_core.async_bus.publish(
                        AsyncMessage(
                            MessageType.MEMORY_RESULT,
                            "unified_memory",
                            {"key": content.get("key"), "result": result},
                            target_ids=[reply_to],
                        )
                    )
            elif operation == "retrieve_similar":
                result = self.retrieve_similar(
                    content.get("query"),
                    limit=content.get("limit", 3),
                    namespace=content.get("namespace", self.default_namespace),
                )
                # Send result back if reply_to is specified
                reply_to = getattr(message, "reply_to", None)
                if reply_to:
                    self.vanta_core.async_bus.publish(
                        AsyncMessage(
                            MessageType.MEMORY_RESULT,
                            "unified_memory",
                            {"query": content.get("query"), "results": result},
                            target_ids=[reply_to],
                        )
                    )
            elif operation == "log_event":
                self.log_event(
                    content.get("task_id", "unknown"),
                    content.get("event_type", "SYSTEM_EVENT"),
                    content.get("event_data", {}),
                )
            elif operation == "imprint":
                self.imprint(
                    content.get("key"),
                    content.get("value"),
                    ttl_seconds=content.get("ttl_seconds", self.default_ttl_seconds),
                )
        except Exception as e:
            logger.error(f"Error handling memory operation: {e}")

    def store(
        self,
        key: str,
        value: Any,
        namespace: str = None,
        metadata: Dict[str, Any] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Store data in the memory system.

        Args:
            key: Unique identifier for the data
            value: The data to store
            namespace: Optional namespace to organize memory (default: self.default_namespace)
            metadata: Optional metadata to associate with the value
            ttl_seconds: Optional time-to-live in seconds (default: self.default_ttl_seconds)

        Returns:
            Unique identifier for the stored data
        """
        if namespace is None:
            namespace = self.default_namespace

        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        # Prepare interaction data for JsonFileMemoryInterface
        interaction_data = {
            "query": key,
            "response": value,
            "timestamp": datetime.now().isoformat(),
            "namespace": namespace,
        }

        if metadata:
            interaction_data["metadata"] = metadata

        # Generate a unique ID
        interaction_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        interaction_data["id"] = interaction_id

        # Store in JsonFileMemoryInterface
        success_file = self.file_memory.store_interaction(interaction_data)

        # Also imprint in MemoryBraid
        full_key = f"{namespace}:{key}" if namespace else key
        self.memory_braid.imprint(full_key, value, ttl_seconds=ttl_seconds)

        # Log the memory operation
        self.echo_memory.log(
            task_id="memory_operation",
            event_type="MEMORY_STORE",
            event_data={"key": key, "namespace": namespace, "id": interaction_id},
        )

        # Emit event for listeners
        self._emit_event(
            "store",
            {
                "key": key,
                "namespace": namespace,
                "id": interaction_id,
                "success": success_file,
            },
        )

        # Broadcast event via async bus if available
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            self.vanta_core.async_bus.publish(
                AsyncMessage(
                    MessageType.MEMORY_OPERATION,
                    "unified_memory",
                    {
                        "operation": "store",
                        "key": key,
                        "namespace": namespace,
                        "id": interaction_id,
                        "success": success_file,
                    },
                )
            )

        return interaction_id

    def retrieve(self, key: str, namespace: str = None) -> Optional[Any]:
        """
        Retrieve data from the memory system.

        Args:
            key: Key to retrieve
            namespace: Optional namespace (default: self.default_namespace)

        Returns:
            Retrieved value or None if not found
        """
        if namespace is None:
            namespace = self.default_namespace

        # Try MemoryBraid first (faster semantic memory)
        full_key = f"{namespace}:{key}" if namespace else key
        result = self.memory_braid.recall_semantic(full_key)

        # If not found in MemoryBraid, try JsonFileMemoryInterface
        if result is None:
            similar_interactions = self.file_memory.retrieve_similar_interactions(key, limit=1)
            if similar_interactions and similar_interactions[0].get("query") == key:
                result = similar_interactions[0].get("response")

                # If found in file memory but not in semantic memory, add it to semantic memory
                if result is not None:
                    self.memory_braid.imprint(
                        full_key, result, ttl_seconds=self.default_ttl_seconds
                    )

        # Log the retrieval operation
        self.echo_memory.log(
            task_id="memory_operation",
            event_type="MEMORY_RETRIEVE",
            event_data={
                "key": key,
                "namespace": namespace,
                "found": result is not None,
            },
        )

        # Emit event for listeners
        self._emit_event(
            "retrieve",
            {"key": key, "namespace": namespace, "found": result is not None},
        )

        return result

    def retrieve_similar(
        self, query: str, limit: int = 3, namespace: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve data with similar keys from the memory system.

        Args:
            query: Query to find similar entries for
            limit: Maximum number of results to return
            namespace: Optional namespace (default: self.default_namespace)

        Returns:
            List of similar entries
        """
        if namespace is None:
            namespace = self.default_namespace

        # Retrieve from JsonFileMemoryInterface
        similar_interactions = self.file_memory.retrieve_similar_interactions(query, limit=limit)

        # Filter by namespace if specified
        if namespace:
            similar_interactions = [
                interaction
                for interaction in similar_interactions
                if interaction.get("namespace", self.default_namespace) == namespace
            ]

        # Log the retrieval operation
        self.echo_memory.log(
            task_id="memory_operation",
            event_type="MEMORY_RETRIEVE_SIMILAR",
            event_data={
                "query": query,
                "namespace": namespace,
                "count": len(similar_interactions),
            },
        )

        return similar_interactions

    def update(self, interaction_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.

        Args:
            interaction_id: ID of the interaction to update
            updates: Dictionary of updates to apply

        Returns:
            Boolean indicating success
        """
        # Update in JsonFileMemoryInterface
        success = self.file_memory.update_interaction(interaction_id, updates)

        # If successful and updates contain response, update MemoryBraid
        if success and "response" in updates:
            # Need to retrieve the full interaction to get the key
            interaction = self.file_memory.retrieve_interaction_by_id(interaction_id)
            if interaction:
                key = interaction.get("query")
                namespace = interaction.get("namespace", self.default_namespace)
                full_key = f"{namespace}:{key}" if namespace else key
                self.memory_braid.imprint(full_key, updates["response"])

        # Log the update operation
        self.echo_memory.log(
            task_id="memory_operation",
            event_type="MEMORY_UPDATE",
            event_data={"id": interaction_id, "success": success},
        )

        return success

    def imprint(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Directly imprint data in the MemoryBraid system.

        Args:
            key: Key for the memory entry
            value: Value to store
            ttl_seconds: Optional time-to-live in seconds
        """
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        # Imprint in MemoryBraid
        self.memory_braid.imprint(key, value, ttl_seconds=ttl_seconds)

        # Log the operation
        self.echo_memory.log(
            task_id="memory_operation",
            event_type="MEMORY_IMPRINT",
            event_data={"key": key},
        )

    def recall_episodic(self, limit: int = 5) -> List[Tuple[str, Any, float]]:
        """
        Retrieve recent episodic memories.

        Args:
            limit: Maximum number of memories to retrieve

        Returns:
            List of (key, value, timestamp) tuples
        """
        return self.memory_braid.recall_episodic_recent(limit=limit)

    def log_event(self, task_id: str, event_type: str, event_data: Any) -> None:
        """
        Log an event to EchoMemory.

        Args:
            task_id: Identifier for the task
            event_type: Type of event
            event_data: Data associated with the event
        """
        self.echo_memory.log(task_id, event_type, event_data)

    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent interaction history.

        Args:
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interaction dictionaries
        """
        return self.file_memory.get_interaction_history(limit=limit)

    def recall_echo_by_task(
        self, task_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve echo logs for a specific task.

        Args:
            task_id: Task identifier
            limit: Optional limit on the number of logs to retrieve

        Returns:
            List of echo log entries
        """
        return self.echo_memory.recall_by_task_id(task_id, limit=limit)

    def add_event_listener(self, event_type: str, listener: Callable) -> None:
        """
        Add a listener for memory events.

        Args:
            event_type: Type of event to listen for (e.g., "store", "retrieve")
            listener: Callback function to invoke when event occurs
        """
        with self._lock:
            self._event_listeners[event_type].append(listener)

    def remove_event_listener(self, event_type: str, listener: Callable) -> bool:
        """
        Remove an event listener.

        Args:
            event_type: Type of event
            listener: Listener to remove

        Returns:
            Boolean indicating whether the listener was found and removed
        """
        with self._lock:
            if (
                event_type in self._event_listeners
                and listener in self._event_listeners[event_type]
            ):
                self._event_listeners[event_type].remove(listener)
                return True
            return False

    def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered listeners.

        Args:
            event_type: Type of event
            event_data: Data associated with the event
        """
        with self._lock:
            listeners = self._event_listeners.get(event_type, []).copy()

        for listener in listeners:
            try:
                listener(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory subsystems.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "file_memory": {
                "interactions": len(self.file_memory._get_index().get("interactions", [])),
                "last_updated": self.file_memory._get_index().get("last_updated"),
            },
            "memory_braid": {
                "semantic_size": self.memory_braid.get_semantic_memory_size(),
                "episodic_size": self.memory_braid.get_episodic_memory_size(),
            },
            "echo_memory": {
                "log_size": self.echo_memory.get_log_size(),
                "total_logged": self.echo_memory.stats.get("total_logged", 0),
                "cleanup_count": self.echo_memory.stats.get("cleanup_count", 0),
            },
        }
        return stats
