# voxsigil_supervisor/memory_braid.py
"""
MemoryBraid: A hybrid memory system combining episodic, semantic,
             and time-to-live (TTL) characteristics.
"""

import json
import logging
import time  # For TTL using monotonic clock
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType  # add at top

logger_braid = logging.getLogger("VoxSigilSupervisor.MemoryBraid")
if not logger_braid.hasHandlers() and not logging.getLogger().hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger_braid.addHandler(handler)
    logger_braid.setLevel(logging.INFO)


class MemoryBraid:
    """
    A hybrid memory system composed of:
    1. Episodic Memory: A fixed-size queue (deque) of recent (key, value, timestamp) imprints.
    2. Semantic Memory: A dictionary for direct key-based recall.
    3. TTL Mechanism: Entries in semantic memory can expire after a time-to-live.
    """

    def __init__(
        self,
        vanta_core=None,  # UnifiedVantaCore instance for registration
        max_episodic_len: int = 128,
        default_semantic_ttl_seconds: Optional[int] = 3600,  # Default 1 hour
        auto_decay_on_imprint: bool = True,
        auto_decay_on_recall: bool = False,
    ):  # Decay on recall can be costly
        """
        Initializes the MemoryBraid.

        Args:
            max_episodic_len: Maximum number of entries in the episodic memory queue.
            default_semantic_ttl_seconds: Default TTL for entries in semantic memory (in seconds).
                                          None means entries do not expire by default.
            auto_decay_on_imprint: If True, run decay process after each imprint.
            auto_decay_on_recall: If True, run decay process before each recall.
                                   Can impact recall performance.
        """
        if max_episodic_len <= 0:
            logger_braid.warning(
                "max_episodic_len must be positive. Setting to default 128."
            )
            max_episodic_len = 128
        self._episodic: Deque[Tuple[str, Any, float]] = deque(
            maxlen=max_episodic_len
        )  # Stores (key, value, imprint_time)
        self._semantic: Dict[
            str, Tuple[Any, float]
        ] = {}  # Stores {key: (value, expiry_time_monotonic)}

        self.default_semantic_ttl_seconds = default_semantic_ttl_seconds
        if (
            self.default_semantic_ttl_seconds is not None
            and self.default_semantic_ttl_seconds <= 0
        ):
            logger_braid.warning(
                "default_semantic_ttl_seconds must be positive if set. Disabling default TTL."
            )
            self.default_semantic_ttl_seconds = None

        self.auto_decay_on_imprint = auto_decay_on_imprint
        self.auto_decay_on_recall = auto_decay_on_recall

        logger_braid.info(
            f"MemoryBraid initialized. Episodic max_len: {max_episodic_len}, "
            f"Default Semantic TTL: {self.default_semantic_ttl_seconds}s, "
            f"Auto-decay (imprint/recall): {self.auto_decay_on_imprint}/{self.auto_decay_on_recall}"
        )
        self.vanta_core = vanta_core
        # Register with UnifiedVantaCore
        if self.vanta_core:
            try:
                self.vanta_core.register_component(
                    "memory_braid", self, {"type": "memory_braid"}
                )
                if hasattr(self.vanta_core, "async_bus"):
                    self.vanta_core.async_bus.register_component("memory_braid")
                    self.vanta_core.async_bus.subscribe(
                        "memory_braid",
                        MessageType.MEMORY_OPERATION,
                        self.handle_memory_operation,
                    )
            except Exception as e:
                logger_braid.warning(f"Failed to register MemoryBraid: {e}")

    def imprint(self, key: str, value: Any, ttl_seconds: Optional[int] = -1) -> None:
        """
        Imprints a key-value pair into both episodic and semantic memory.

        Args:
            key: The key for the memory entry. Must be a string.
            value: The value to store.
            ttl_seconds: Time-to-live for this specific entry in semantic memory (in seconds).
                         - If None, entry does not expire (persists until manually removed or overwritten).
                         - If -1 (default), uses the instance's default_semantic_ttl_seconds.
                         - If 0, entry is not added to semantic memory (or immediately expires).
        """
        if not isinstance(key, str) or not key.strip():
            logger_braid.error("Invalid key for imprint: Must be a non-empty string.")
            return

        current_time_monotonic = time.monotonic()

        # Episodic memory - always add
        self._episodic.append((key, value, current_time_monotonic))
        logger_braid.debug(f"Imprinted '{key}' to episodic memory.")

        # Semantic memory with TTL
        effective_ttl = (
            self.default_semantic_ttl_seconds if ttl_seconds == -1 else ttl_seconds
        )

        if (
            effective_ttl is not None and effective_ttl <= 0
        ):  # ttl of 0 or less (except None) means no semantic storage or immediate expiry
            if key in self._semantic:  # Remove if it was there with a previous TTL
                del self._semantic[key]
            logger_braid.debug(
                f"Key '{key}' not added/removed from semantic memory due to TTL <= 0."
            )
        else:
            expiry_time = (
                (current_time_monotonic + effective_ttl)
                if effective_ttl is not None
                else float("inf")
            )  # 'inf' for no expiry
            self._semantic[key] = (value, expiry_time)
            logger_braid.debug(
                f"Imprinted '{key}' to semantic memory. Expires at: {expiry_time if expiry_time != float('inf') else 'Never'}"
            )

    def handle_memory_operation(self, message: AsyncMessage):
        """Handle MEMORY_OPERATION messages for multi-tier memory processing"""
        try:
            content = message.content or {}
            key = content.get("key")
            value = content.get("value")
            ttl = content.get("ttl_seconds", -1)
            if key is not None and value is not None:
                self.imprint(key, value, ttl)
        except Exception as e:
            logger_braid.error(f"MemoryBraid failed to handle memory operation: {e}")

    def recall_semantic(
        self,
        key: str,
        extend_ttl_on_recall: bool = False,
        new_ttl_seconds: Optional[int] = -1,
    ) -> Optional[Any]:
        """
        Recalls a value from semantic memory by its key.
        Performs decay if auto_decay_on_recall is True.

        Args:
            key: The key to recall.
            extend_ttl_on_recall: If True, and the item is found, its TTL will be refreshed/extended.
            new_ttl_seconds: The new TTL to set if extend_ttl_on_recall is True.
                             - If None, makes the item persistent.
                             - If -1, uses default_semantic_ttl_seconds.

        Returns:
            The value if found and not expired, otherwise None.
        """
        if not isinstance(key, str):
            logger_braid.warning(
                f"Invalid key type for recall_semantic: {type(key)}. Expected str."
            )
            return None

        if self.auto_decay_on_recall:
            self.decay()

        entry = self._semantic.get(key)
        if entry:
            value, expiry_time = entry
            current_time_monotonic = time.monotonic()
            if current_time_monotonic < expiry_time:
                logger_braid.debug(f"Semantic recall hit for key '{key}'.")
                if extend_ttl_on_recall:
                    effective_new_ttl = (
                        self.default_semantic_ttl_seconds
                        if new_ttl_seconds == -1
                        else new_ttl_seconds
                    )
                    new_expiry = (
                        (current_time_monotonic + effective_new_ttl)
                        if effective_new_ttl is not None
                        else float("inf")
                    )
                    self._semantic[key] = (value, new_expiry)
                    logger_braid.debug(
                        f"Extended TTL for key '{key}'. New expiry: {new_expiry if new_expiry != float('inf') else 'Never'}"
                    )
                return value
            else:
                # Entry has expired, remove it (also done by decay, but good to be explicit)
                logger_braid.info(
                    f"Semantic recall miss for key '{key}': entry expired."
                )
                del self._semantic[key]
                return None
        logger_braid.debug(f"Semantic recall miss for key '{key}': not found.")
        return None

    def recall_episodic_recent(self, limit: int = 5) -> List[Tuple[str, Any, float]]:
        """
        Recalls the most recent 'limit' entries from episodic memory.

        Args:
            limit: The number of recent entries to return.

        Returns:
            A list of tuples (key, value, imprint_timestamp_monotonic).
        """
        if not isinstance(limit, int) or limit <= 0:
            limit = 5  # Default if invalid
        # Deque stores newest at the right, so to get recent, we iterate from right.
        return list(self._episodic)[-limit:]

    def recall_episodic_by_key(
        self, key: str, limit: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """Recalls all values associated with a key from episodic memory, with their imprint times."""
        if not isinstance(key, str):
            return []

        matches = [(val, ts) for k, val, ts in self._episodic if k == key]
        return matches[-limit:] if limit is not None else matches

    def decay(self) -> List[str]:
        """
        Removes expired entries from semantic memory based on their TTL.
        This should be called periodically by an external mechanism or via auto-decay flags.

        Returns:
            A list of keys that were expired and removed.
        """
        current_time_monotonic = time.monotonic()
        expired_keys: List[str] = []
        # Iterate over a copy of keys for safe deletion
        for key, (value, expiry_time) in list(self._semantic.items()):
            if current_time_monotonic >= expiry_time:
                expired_keys.append(key)
                del self._semantic[key]

        if expired_keys:
            logger_braid.info(
                f"Decayed and removed {len(expired_keys)} expired keys from semantic memory: {expired_keys}"
            )
        return expired_keys

    def forget_semantic(self, key: str) -> bool:
        """Explicitly removes an entry from semantic memory."""
        if not isinstance(key, str):
            return False
        if key in self._semantic:
            del self._semantic[key]
            logger_braid.info(f"Key '{key}' explicitly forgotten from semantic memory.")
            return True
        return False

    def get_semantic_memory_size(self) -> int:
        return len(self._semantic)

    def get_episodic_memory_size(self) -> int:
        return len(self._episodic)

    def clear_all_memory(self) -> None:
        """Clears both episodic and semantic memories."""
        self._episodic.clear()
        self._semantic.clear()
        logger_braid.info("MemoryBraid (episodic and semantic) has been cleared.")

    # Potential future feature: Persistence
    def save_semantic_memory(self, file_path: Union[str, Path]) -> bool:
        """Saves the current semantic memory (keys, values, and expiry times) to a file."""
        try:
            # Need to handle float('inf') for JSON serialization
            data_to_save = {
                k: (v, exp_time if exp_time != float("inf") else "inf")
                for k, (v, exp_time) in self._semantic.items()
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2)
            logger_braid.info(f"Semantic memory saved to {file_path}")
            return True
        except Exception as e:
            logger_braid.error(
                f"Failed to save semantic memory to {file_path}: {e}", exc_info=True
            )
            return False

    def load_semantic_memory(
        self, file_path: Union[str, Path], merge: bool = False
    ) -> bool:
        """Loads semantic memory from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict):
                logger_braid.error(
                    f"Data in {file_path} is not a valid semantic memory dict."
                )
                return False

            new_semantic_memory = {}
            for key, item in loaded_data.items():
                if (
                    isinstance(item, list) and len(item) == 2
                ):  # Expecting [value, expiry_str]
                    value, expiry_str = item
                    expiry_time = (
                        float(expiry_str) if expiry_str != "inf" else float("inf")
                    )
                    new_semantic_memory[key] = (value, expiry_time)
                else:
                    logger_braid.warning(
                        f"Skipping invalid entry in loaded semantic memory for key '{key}': {item}"
                    )

            if merge:
                self._semantic.update(new_semantic_memory)
            else:
                self._semantic = new_semantic_memory

            # It's important to run decay after loading to remove any entries that might have expired
            # while persisted, based on their absolute expiry times relative to current monotonic clock.
            # This assumes expiry_time stored was a future monotonic time.
            # If expiry_time was stored as a relative TTL, a different loading logic is needed.
            # For simplicity, we assume stored expiry_time is absolute monotonic.
            self.decay()
            logger_braid.info(
                f"Semantic memory loaded from {file_path}. Merged: {merge}. Current size: {len(self._semantic)}"
            )
            return True
        except Exception as e:
            logger_braid.error(
                f"Failed to load semantic memory from {file_path}: {e}", exc_info=True
            )
            return False
