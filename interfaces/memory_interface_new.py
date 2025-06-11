# voxsigil_supervisor/interfaces/memory_interface.py
"""
Memory Interface for VoxSigil Supervisor.
Now imports unified interface from Vanta instead of defining its own.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import json
import os
from pathlib import Path
from datetime import datetime

# Import unified interface from Vanta - replaces local definition
from Vanta.interfaces.base_interfaces import BaseMemoryInterface

# Setup the logger
logger_memory_interface = logging.getLogger("VoxSigilSupervisor.interfaces.memory")


class JsonFileMemoryInterface(BaseMemoryInterface):
    """
    Implementation of the memory interface using simple JSON files for storage.
    Not suitable for production, but sufficient for development and testing.
    """

    def __init__(self, memory_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the JsonFileMemoryInterface.

        Args:
            memory_dir: Directory to store memory files. If None, a default directory is used.
        """
        self.logger = logger_memory_interface

        if memory_dir is None:
            # Default to a 'memory' directory in the project root
            self.memory_dir = (
                Path(__file__).resolve().parent.parent.parent / "memory"
            )
        else:
            self.memory_dir = Path(memory_dir)

        # Ensure memory directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Files for different aspects of memory
        self.interactions_file = self.memory_dir / "interactions.jsonl"
        self.metadata_file = self.memory_dir / "metadata.json"

        self.logger.info(f"JsonFileMemoryInterface initialized with directory: {self.memory_dir}")

    def _load_interactions(self) -> List[Dict[str, Any]]:
        """Load all interactions from the interactions file."""
        interactions = []
        if self.interactions_file.exists():
            try:
                with open(self.interactions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            interactions.append(json.loads(line.strip()))
            except Exception as e:
                self.logger.error(f"Error loading interactions: {e}")
        return interactions

    def _append_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Append a new interaction to the interactions file."""
        try:
            with open(self.interactions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction_data) + "\n")
            return True
        except Exception as e:
            self.logger.error(f"Error appending interaction: {e}")
            return False

    def store_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Store a complete interaction."""
        if "timestamp" not in interaction_data:
            interaction_data["timestamp"] = datetime.now().isoformat()

        return self._append_interaction(interaction_data)

    def retrieve_similar_interactions(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve interactions with similar queries (simple string matching)."""
        interactions = self._load_interactions()
        similar = []

        # Simple keyword-based similarity
        query_words = set(query.lower().split())

        for interaction in interactions:
            interaction_query = interaction.get("query", "").lower()
            interaction_words = set(interaction_query.split())

            # Calculate simple similarity based on word overlap
            if query_words and interaction_words:
                similarity = len(query_words & interaction_words) / len(
                    query_words | interaction_words
                )
                if similarity > 0.1:  # Minimum similarity threshold
                    interaction["similarity"] = similarity
                    similar.append(interaction)

        # Sort by similarity and return top results
        similar.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return similar[:limit]

    def retrieve_interaction_by_id(
        self, interaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific interaction by ID."""
        interactions = self._load_interactions()
        for interaction in interactions:
            if interaction.get("id") == interaction_id:
                return interaction
        return None

    def get_interaction_history(
        self, query_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get interaction history, optionally filtered by query ID."""
        interactions = self._load_interactions()

        if query_id:
            # Filter by query ID if provided
            filtered = [
                interaction
                for interaction in interactions
                if interaction.get("query_id") == query_id
            ]
        else:
            filtered = interactions

        # Sort by timestamp (most recent first) and return top results
        filtered.sort(
            key=lambda x: x.get("timestamp", ""), reverse=True
        )
        return filtered[:limit]

    def update_interaction(self, interaction_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing interaction with new data."""
        interactions = self._load_interactions()
        updated = False

        for i, interaction in enumerate(interactions):
            if interaction.get("id") == interaction_id:
                interaction.update(updates)
                updated = True
                break

        if updated:
            # Rewrite the entire file with updated data
            try:
                with open(self.interactions_file, "w", encoding="utf-8") as f:
                    for interaction in interactions:
                        f.write(json.dumps(interaction) + "\n")
                return True
            except Exception as e:
                self.logger.error(f"Error updating interaction: {e}")
                return False

        return False

    def _generate_id(self, prefix: str = "int") -> str:
        """Generates a unique ID for a new entry."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"
