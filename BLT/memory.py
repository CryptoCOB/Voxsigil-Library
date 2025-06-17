"""
BLT Memory Management Module
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MemoryManager:
    """Basic memory management for BLT system"""

    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.session_memory: Dict[str, Any] = {}

    def store(self, key: str, value: Any, session: bool = False):
        """Store a value in memory"""
        if session:
            self.session_memory[key] = value
        else:
            self.memory_store[key] = value

    def retrieve(self, key: str, session: bool = False) -> Any:
        """Retrieve a value from memory"""
        store = self.session_memory if session else self.memory_store
        return store.get(key)

    def clear_session(self):
        """Clear session memory"""
        self.session_memory.clear()

    def clear_all(self):
        """Clear all memory"""
        self.memory_store.clear()
        self.session_memory.clear()


# Default instance
default_memory_manager = MemoryManager()

__all__ = ["MemoryManager", "default_memory_manager"]
