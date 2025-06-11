# echo_memory.py
"""
EchoMemory: A structured event logging and recall system for reasoning processes.

This module provides a sophisticated logging system that maintains structured
event histories for complex reasoning tasks, with support for task correlation,
event categorization, and efficient retrieval.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType  # add imports

logger_echo = logging.getLogger("VoxSigilSupervisor.EchoMemory")
if not logger_echo.hasHandlers() and not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger_echo.addHandler(handler)
    logger_echo.setLevel(logging.INFO)


class EchoMemory:
    """
    A structured event logging system designed for complex reasoning processes.

    EchoMemory maintains chronological logs of events associated with tasks,
    allowing for detailed tracing of reasoning pipelines, dialectic processes,
    and error conditions. It supports efficient retrieval by task ID and
    automatic memory management.
    """

    def __init__(
        self,
        vanta_core=None,  # UnifiedVantaCore instance for registration
        max_log_size: int = 10000,
        auto_cleanup_threshold: float = 0.8,
        enable_persistence: bool = False,
        persistence_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize EchoMemory with configurable parameters.

        Args:
            max_log_size: Maximum number of log entries to maintain
            auto_cleanup_threshold: Fraction of max_log_size that triggers cleanup (0.0-1.0)
            enable_persistence: Whether to persist logs to disk
            persistence_path: Path for persistent storage (defaults to temp if enabled)
        """
        self.vanta_core = vanta_core  # store core reference
        self.max_log_size = max(100, max_log_size)  # Minimum 100 entries
        self.auto_cleanup_threshold = max(0.1, min(1.0, auto_cleanup_threshold))
        self.enable_persistence = enable_persistence

        # Core data structures
        self._logs: deque = deque(maxlen=self.max_log_size)
        self._task_index: Dict[str, List[int]] = defaultdict(
            list
        )  # task_id -> [entry_indices]
        self._event_type_index: Dict[str, List[int]] = defaultdict(
            list
        )  # event_type -> [entry_indices]
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_logged": 0,
            "cleanup_count": 0,
            "last_cleanup_time": None,
            "created_time": time.time(),
        }

        # Persistence setup
        if self.enable_persistence:
            if persistence_path:
                self.persistence_path = Path(persistence_path)
            else:
                import tempfile

                self.persistence_path = (
                    Path(tempfile.gettempdir()) / "echo_memory_logs.jsonl"
                )
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            logger_echo.info(f"EchoMemory persistence enabled: {self.persistence_path}")
        else:
            self.persistence_path = None

        logger_echo.info(
            f"EchoMemory initialized. Max size: {self.max_log_size}, "
            f"Cleanup threshold: {self.auto_cleanup_threshold:.1%}, "
            f"Persistence: {'enabled' if self.enable_persistence else 'disabled'}"
        )
        # Register with UnifiedVantaCore
        if self.vanta_core:
            try:
                self.vanta_core.register_component(
                    "echo_memory", self, {"type": "echo_memory"}
                )
                # Subscribe to memory operations
                if hasattr(self.vanta_core, "async_bus"):
                    self.vanta_core.async_bus.register_component("echo_memory")
                    self.vanta_core.async_bus.subscribe(
                        "echo_memory",
                        MessageType.MEMORY_OPERATION,
                        self.handle_memory_operation,
                    )
            except Exception as e:
                logger_echo.warning(f"Failed to register EchoMemory: {e}")

    def log(self, task_id: str, event_type: str, event_data: Any) -> None:
        """
        Log a structured event associated with a task.

        Args:
            task_id: Unique identifier for the task (allows correlation)
            event_type: Category/type of the event (e.g., "REASONING_PIPELINE", "ERROR")
            event_data: Data associated with the event (will be JSON-serialized for persistence)
        """
        if not isinstance(task_id, str) or not task_id.strip():
            logger_echo.warning(
                "Invalid task_id provided to log(). Using 'unknown_task'."
            )
            task_id = "unknown_task"

        if not isinstance(event_type, str) or not event_type.strip():
            logger_echo.warning(
                "Invalid event_type provided to log(). Using 'UNKNOWN_EVENT'."
            )
            event_type = "UNKNOWN_EVENT"

        timestamp = time.time()

        # Create log entry
        entry = {
            "timestamp": timestamp,
            "task_id": task_id.strip(),
            "event_type": event_type.strip(),
            "event_data": event_data,
            "entry_id": self.stats["total_logged"],  # Unique entry identifier
        }

        with self._lock:
            # Add to main log
            self._logs.append(entry)
            entry_index = len(self._logs) - 1

            # Update indices
            self._task_index[task_id].append(entry_index)
            self._event_type_index[event_type].append(entry_index)

            # Update statistics
            self.stats["total_logged"] += 1

            # Check if cleanup is needed
            if len(self._logs) >= self.max_log_size * self.auto_cleanup_threshold:
                self._cleanup_old_entries()

        # Persist if enabled
        if self.enable_persistence:
            self._persist_entry(entry)

        logger_echo.debug(
            f"Logged event: task_id='{task_id}', type='{event_type}', data_size={len(str(event_data))}"
        )

    def handle_memory_operation(self, message: AsyncMessage):
        """Handle incoming MEMORY_OPERATION messages by logging them."""
        try:
            content = message.content or {}
            self.log(
                content.get("task_id", "unknown"),
                content.get("event_type", "MEMORY_OPERATION"),
                content.get("data"),
            )
        except Exception as e:
            logger_echo.error(f"EchoMemory failed to handle memory operation: {e}")

    def get_log_size(self) -> int:
        """
        Get the current number of log entries.

        Returns:
            Number of entries currently stored
        """
        with self._lock:
            return len(self._logs)

    def recall_by_task_id(
        self, task_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve log entries for a specific task.

        Args:
            task_id: Task identifier to search for
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of log entries for the specified task, ordered by timestamp (newest first)
        """
        if not isinstance(task_id, str):
            logger_echo.warning(f"Invalid task_id type for recall: {type(task_id)}")
            return []

        with self._lock:
            if task_id not in self._task_index:
                logger_echo.debug(f"No entries found for task_id: '{task_id}'")
                return []

            # Get all entries for this task
            entry_indices = self._task_index[task_id]
            entries = []

            for idx in entry_indices:
                if idx < len(
                    self._logs
                ):  # Ensure index is still valid after deque operations
                    entries.append(dict(self._logs[idx]))  # Return a copy

            # Sort by timestamp (newest first)
            entries.sort(key=lambda x: x["timestamp"], reverse=True)

            # Apply limit if specified
            if limit is not None and limit > 0:
                entries = entries[:limit]

            logger_echo.debug(
                f"Recalled {len(entries)} entries for task_id: '{task_id}'"
            )
            return entries

    def recall_by_event_type(
        self, event_type: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve log entries for a specific event type.

        Args:
            event_type: Event type to search for
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of log entries for the specified event type, ordered by timestamp (newest first)
        """
        if not isinstance(event_type, str):
            logger_echo.warning(
                f"Invalid event_type type for recall: {type(event_type)}"
            )
            return []

        with self._lock:
            if event_type not in self._event_type_index:
                logger_echo.debug(f"No entries found for event_type: '{event_type}'")
                return []

            # Get all entries for this event type
            entry_indices = self._event_type_index[event_type]
            entries = []

            for idx in entry_indices:
                if idx < len(self._logs):
                    entries.append(dict(self._logs[idx]))

            # Sort by timestamp (newest first)
            entries.sort(key=lambda x: x["timestamp"], reverse=True)

            # Apply limit if specified
            if limit is not None and limit > 0:
                entries = entries[:limit]

            logger_echo.debug(
                f"Recalled {len(entries)} entries for event_type: '{event_type}'"
            )
            return entries

    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent log entries across all tasks.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent log entries, ordered by timestamp (newest first)
        """
        with self._lock:
            if not self._logs:
                return []

            # Get the last 'limit' entries
            recent = (
                list(self._logs)[-limit:]
                if limit < len(self._logs)
                else list(self._logs)
            )

            # Reverse to get newest first
            recent.reverse()

            return [dict(entry) for entry in recent]

    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tasks and their event counts.

        Returns:
            Dictionary with task summaries and statistics
        """
        with self._lock:
            task_summary = {}

            for task_id, entry_indices in self._task_index.items():
                valid_entries = [idx for idx in entry_indices if idx < len(self._logs)]
                if valid_entries:
                    entries = [self._logs[idx] for idx in valid_entries]
                    event_types = {}

                    for entry in entries:
                        event_type = entry["event_type"]
                        event_types[event_type] = event_types.get(event_type, 0) + 1

                    first_timestamp = min(entry["timestamp"] for entry in entries)
                    last_timestamp = max(entry["timestamp"] for entry in entries)

                    task_summary[task_id] = {
                        "total_events": len(valid_entries),
                        "event_types": event_types,
                        "first_logged": first_timestamp,
                        "last_logged": last_timestamp,
                        "duration_seconds": last_timestamp - first_timestamp,
                    }

            return {
                "tasks": task_summary,
                "total_tasks": len(task_summary),
                "total_entries": len(self._logs),
                "stats": dict(self.stats),
            }

    def clear_task_logs(self, task_id: str) -> int:
        """
        Remove all log entries for a specific task.

        Args:
            task_id: Task identifier to clear

        Returns:
            Number of entries removed
        """
        if not isinstance(task_id, str):
            logger_echo.warning(f"Invalid task_id type for clear: {type(task_id)}")
            return 0

        with self._lock:
            if task_id not in self._task_index:
                return 0

            # This is complex with deque - for now, we'll mark entries as cleared
            # In a production system, you might want to use a different data structure
            entries_to_remove = []

            for i, entry in enumerate(self._logs):
                if entry["task_id"] == task_id:
                    entries_to_remove.append(i)

            # Remove from back to front to maintain indices
            for i in reversed(entries_to_remove):
                # Since we're using deque, we can't easily remove by index
                # Instead, we'll mark the entry as cleared
                if i < len(self._logs):
                    self._logs[i] = {
                        "timestamp": self._logs[i]["timestamp"],
                        "task_id": "CLEARED",
                        "event_type": "CLEARED",
                        "event_data": {"original_task_id": task_id},
                        "entry_id": self._logs[i]["entry_id"],
                    }

            # Clear from indices
            del self._task_index[task_id]

            # Clean up event type index
            for event_type, indices in self._event_type_index.items():
                self._event_type_index[event_type] = [
                    idx for idx in indices if idx not in entries_to_remove
                ]

            logger_echo.info(
                f"Cleared {len(entries_to_remove)} entries for task_id: '{task_id}'"
            )
            return len(entries_to_remove)

    def _cleanup_old_entries(self) -> None:
        """
        Internal method to clean up old entries when approaching max capacity.
        """
        if len(self._logs) < self.auto_cleanup_threshold * self.max_log_size:
            return

        # Calculate how many entries to remove (remove oldest 25%)
        entries_to_remove = max(1, int(self.max_log_size * 0.25))

        logger_echo.info(
            f"Starting cleanup: removing {entries_to_remove} oldest entries"
        )

        # Since we're using deque with maxlen, it will automatically remove oldest
        # But we need to update our indices

        # For simplicity in this implementation, we'll rebuild indices
        # In production, you'd want a more efficient approach
        self._rebuild_indices()

        self.stats["cleanup_count"] += 1
        self.stats["last_cleanup_time"] = time.time()

        logger_echo.info(f"Cleanup completed. Current size: {len(self._logs)}")

    def _rebuild_indices(self) -> None:
        """
        Rebuild the task and event type indices.
        """
        self._task_index.clear()
        self._event_type_index.clear()

        for i, entry in enumerate(self._logs):
            if entry["task_id"] != "CLEARED":  # Skip cleared entries
                self._task_index[entry["task_id"]].append(i)
                self._event_type_index[entry["event_type"]].append(i)

    def _persist_entry(self, entry: Dict[str, Any]) -> None:
        """
        Persist a single log entry to disk.

        Args:
            entry: Log entry to persist
        """
        if self.persistence_path is None:
            logger_echo.error("Cannot persist entry: persistence_path is None")
            return

        try:
            with open(self.persistence_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            logger_echo.error(f"Failed to persist log entry: {e}")

    def export_logs(
        self,
        output_path: Union[str, Path],
        task_id: Optional[str] = None,
        event_type: Optional[str] = None,
        format_type: str = "jsonl",
    ) -> bool:
        """
        Export logs to a file.

        Args:
            output_path: Path for the output file
            task_id: Optional task ID filter
            event_type: Optional event type filter
            format_type: Export format ("jsonl" or "json")

        Returns:
            True if export was successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine which entries to export
            if task_id:
                entries = self.recall_by_task_id(task_id)
            elif event_type:
                entries = self.recall_by_event_type(event_type)
            else:
                with self._lock:
                    entries = [
                        dict(entry)
                        for entry in self._logs
                        if entry["task_id"] != "CLEARED"
                    ]

            # Export in requested format
            with open(output_path, "w", encoding="utf-8") as f:
                if format_type.lower() == "json":
                    json.dump(
                        {
                            "export_metadata": {
                                "timestamp": time.time(),
                                "total_entries": len(entries),
                                "filters": {
                                    "task_id": task_id,
                                    "event_type": event_type,
                                },
                            },
                            "entries": entries,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                else:  # jsonl
                    for entry in entries:
                        json.dump(entry, f, ensure_ascii=False, default=str)
                        f.write("\n")

            logger_echo.info(f"Exported {len(entries)} entries to {output_path}")
            return True

        except Exception as e:
            logger_echo.error(f"Failed to export logs: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the EchoMemory instance.

        Returns:
            Dictionary containing various statistics
        """
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.stats["created_time"]

            # Calculate memory usage estimation
            estimated_size_bytes = (
                sum(len(str(entry)) for entry in self._logs) * 2
            )  # Rough estimate

            return {
                **self.stats,
                "current_size": len(self._logs),
                "max_size": self.max_log_size,
                "capacity_used": len(self._logs) / self.max_log_size,
                "uptime_seconds": uptime,
                "unique_tasks": len(self._task_index),
                "unique_event_types": len(self._event_type_index),
                "estimated_memory_bytes": estimated_size_bytes,
                "persistence_enabled": self.enable_persistence,
                "persistence_path": str(self.persistence_path)
                if self.persistence_path
                else None,
            }


# Convenience functions for common usage patterns
def create_echo_memory(config: Optional[Dict[str, Any]] = None) -> EchoMemory:
    """
    Factory function to create an EchoMemory instance with common configurations.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured EchoMemory instance
    """
    config = config or {}

    return EchoMemory(
        max_log_size=config.get("max_log_size", 10000),
        auto_cleanup_threshold=config.get("auto_cleanup_threshold", 0.8),
        enable_persistence=config.get("enable_persistence", False),
        persistence_path=config.get("persistence_path"),
    )


def create_arc_echo_memory() -> EchoMemory:
    """
    Create an EchoMemory instance optimized for ARC reasoning tasks.

    Returns:
        EchoMemory configured for ARC usage
    """
    return EchoMemory(
        max_log_size=5000,  # Moderate size for reasoning tasks
        auto_cleanup_threshold=0.75,
        enable_persistence=True,  # Enable persistence for ARC analysis
        persistence_path=None,  # Use default temp location
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    )
    logger_echo.info("--- Running EchoMemory Example ---")

    # Create EchoMemory instance
    echo = EchoMemory(max_log_size=100, enable_persistence=False)

    # Example ARC reasoning log sequence
    task_id = "arc_task_001"

    echo.log(
        task_id,
        "REASONING_PIPELINE",
        {"step": "solve_task_start", "scaffold_used": "ARC_DECOMPOSITION"},
    )
    echo.log(
        task_id,
        "REASONING_PIPELINE",
        {"step": "scaffold_applied_successfully", "output_type": "dict"},
    )
    echo.log(
        task_id,
        "REASONING_PIPELINE",
        {
            "step": "reasoner_solved",
            "solution_grid_preview": "[[1,2],[3,4]]...",
            "trace_summary": ["step1", "step2", "step3"],
        },
    )
    echo.log(
        task_id,
        "FULL_REASONER_TRACE",
        {
            "trace": [
                "detailed",
                "reasoning",
                "steps",
                "with",
                "intermediate",
                "results",
            ]
        },
    )

    # Example Hegelian dialectic logs
    dialectic_task = "hegelian_task_002"
    echo.log(
        dialectic_task,
        "DIALECTIC_PIPELINE",
        {"step": "dialectic_pass_start", "attempt": 1, "thesis_preview": "input_grid"},
    )
    echo.log(
        dialectic_task,
        "DIALECTIC_PIPELINE",
        {
            "step": "antithesis_generated",
            "antithesis_preview": "generated_contradiction",
        },
    )
    echo.log(
        dialectic_task,
        "DIALECTIC_PIPELINE",
        {"step": "synthesis_achieved", "synthesis_preview": "resolved_result"},
    )

    # Example error case
    error_task = "error_task_003"
    echo.log(
        error_task,
        "DIALECTIC_ERROR",
        {"attempt": 1, "error": "Failed to load scaffold 'MISSING_SCAFFOLD'"},
    )

    # Demonstrate retrieval
    print(f"\nEcho log size: {echo.get_log_size()}")

    print(f"\nLogs for {task_id}:")
    for entry in echo.recall_by_task_id(task_id, limit=2):
        print(f"  [{entry['event_type']}] {entry['event_data']}")

    print("\nRecent REASONING_PIPELINE events:")
    for entry in echo.recall_by_event_type("REASONING_PIPELINE", limit=3):
        print(f"  Task: {entry['task_id']} - {entry['event_data']['step']}")

    print("\nTask Summary:")
    summary = echo.get_task_summary()
    for task, info in summary["tasks"].items():
        print(
            f"  {task}: {info['total_events']} events, types: {list(info['event_types'].keys())}"
        )

    print("\nEchoMemory Statistics:")
    stats = echo.get_statistics()
    print(f"  Total logged: {stats['total_logged']}")
    print(f"  Unique tasks: {stats['unique_tasks']}")
    print(f"  Capacity used: {stats['capacity_used']:.1%}")

    logger_echo.info("--- EchoMemory Example Finished ---")
