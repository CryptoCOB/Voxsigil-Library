# voxsigil_supervisor/interfaces/memory_interface.py
"""
Defines the interface for memory management in the VoxSigil Supervisor.
This includes storing past interactions, retrieval, and supporting
iterative improvement through memory of previous attempts.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import json
import os
from pathlib import Path
from datetime import datetime

# Setup the logger
logger_memory_interface = logging.getLogger("VoxSigilSupervisor.interfaces.memory")

class BaseMemoryInterface(ABC):
    """
    Abstract Base Class for a memory interface.
    Implementations will manage storage and retrieval of past interactions,
    including queries, responses, evaluations, and metadata.
    """

    @abstractmethod
    def store_interaction(self, 
                         interaction_data: Dict[str, Any]) -> bool:
        """
        Stores a complete interaction, including query, response, evaluation, etc.

        Args:
            interaction_data: Dictionary containing all data to be stored.
                Must contain at least 'query', 'response', and 'timestamp'.

        Returns:
            Boolean indicating success.
        """
        pass

    @abstractmethod
    def retrieve_similar_interactions(self, 
                                     query: str, 
                                     limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves interactions with similar queries.

        Args:
            query: The query to find similar interactions for.
            limit: Maximum number of interactions to return.

        Returns:
            List of interaction dictionaries.
        """
        pass

    @abstractmethod
    def retrieve_interaction_by_id(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific interaction by ID.

        Args:
            interaction_id: The ID of the interaction to retrieve.

        Returns:
            The interaction dictionary if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_interaction_history(self, 
                               query_id: Optional[str] = None,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the history of interactions, either for a specific query
        or the most recent interactions overall.

        Args:
            query_id: Optional ID to get history for a specific query chain.
            limit: Maximum number of interactions to return.

        Returns:
            List of interaction dictionaries.
        """
        pass

    @abstractmethod
    def update_interaction(self, 
                          interaction_id: str, 
                          updates: Dict[str, Any]) -> bool:
        """
        Updates an existing interaction with new data.

        Args:
            interaction_id: The ID of the interaction to update.
            updates: Dictionary containing the fields to update.

        Returns:
            Boolean indicating success.
        """
        pass


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
            self.memory_dir = Path(__file__).resolve().parent.parent.parent / "voxsigil_memory"
        else:
            self.memory_dir = Path(memory_dir)
        
        # Create the memory directory if it doesn't exist
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Create subdirectories for different types of data
        self.interactions_dir = self.memory_dir / "interactions"
        os.makedirs(self.interactions_dir, exist_ok=True)
        
        self.index_file = self.memory_dir / "index.json"
        self._ensure_index_file()
        
        self.logger.info(f"Initialized JsonFileMemoryInterface with directory: {self.memory_dir}")
    
    def _ensure_index_file(self):
        """Ensures the index file exists, creating it if necessary."""
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump({
                    "interactions": [],
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            self.logger.info(f"Created new index file at {self.index_file}")
    
    def _get_index(self) -> Dict[str, Any]:
        """Reads and returns the index file content."""
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _update_index(self, index_data: Dict[str, Any]):
        """Updates the index file with new data."""
        index_data["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _generate_id(self, prefix: str = "int") -> str:
        """Generates a unique ID for a new entry."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"
    
    def store_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Stores a complete interaction in a JSON file.
        """
        # Ensure required fields
        if not all(k in interaction_data for k in ['query', 'response']):
            self.logger.error("Missing required fields in interaction_data")
            return False
        
        # Add timestamp if not present
        if 'timestamp' not in interaction_data:
            interaction_data['timestamp'] = datetime.now().isoformat()
        
        # Generate an ID if not provided
        interaction_id = interaction_data.get('id', self._generate_id())
        interaction_data['id'] = interaction_id
        
        # Save the interaction to a file
        file_path = self.interactions_dir / f"{interaction_id}.json"
        with open(file_path, 'w') as f:
            json.dump(interaction_data, f, indent=2)
        
        # Update the index
        index = self._get_index()
        index["interactions"].append({
            "id": interaction_id,
            "query": interaction_data['query'],
            "timestamp": interaction_data['timestamp'],
            "success": interaction_data.get('evaluation', {}).get('success', None)
        })
        self._update_index(index)
        
        self.logger.info(f"Stored interaction with ID: {interaction_id}")
        return True
    
    def retrieve_similar_interactions(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves interactions with similar queries.
        This is a very basic implementation using simple string matching.
        In a real implementation, you would use embeddings and vector similarity.
        """
        # Load all interactions from the index
        index = self._get_index()
        interactions = index["interactions"]
        
        # Calculate a basic similarity score (number of shared words)
        query_words = set(query.lower().split())
        
        scored_interactions = []
        for interaction in interactions:
            interaction_query = interaction.get('query', '')
            interaction_words = set(interaction_query.lower().split())
            
            # Calculate Jaccard similarity: intersection / union
            intersection = len(query_words.intersection(interaction_words))
            union = len(query_words.union(interaction_words))
            similarity = intersection / union if union > 0 else 0
            
            scored_interactions.append((similarity, interaction["id"]))
        
        # Sort by similarity (highest first) and take the top 'limit'
        scored_interactions.sort(reverse=True)
        top_interactions = [item[1] for item in scored_interactions[:limit]]
        
        # Load the full interactions from files
        result = []
        for interaction_id in top_interactions:
            interaction = self.retrieve_interaction_by_id(interaction_id)
            if interaction:
                result.append(interaction)
        
        return result
    
    def retrieve_interaction_by_id(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific interaction by ID.
        """
        file_path = self.interactions_dir / f"{interaction_id}.json"
        if not file_path.exists():
            self.logger.warning(f"Interaction with ID {interaction_id} not found")
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_interaction_history(self, query_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the history of interactions.
        """
        index = self._get_index()
        interactions = index["interactions"]
        
        # Sort by timestamp (newest first)
        interactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Filter by query_id if provided
        if query_id:
            interactions = [i for i in interactions if i.get('query_id') == query_id]
        
        # Take only the specified limit
        interactions = interactions[:limit]
        
        # Load the full interactions from files
        result = []
        for interaction in interactions:
            full_interaction = self.retrieve_interaction_by_id(interaction["id"])
            if full_interaction:
                result.append(full_interaction)
        
        return result
    
    def update_interaction(self, interaction_id: str, updates: Dict[str, Any]) -> bool:
        """
        Updates an existing interaction with new data.
        """
        # Load the existing interaction
        interaction = self.retrieve_interaction_by_id(interaction_id)
        if not interaction:
            self.logger.error(f"Cannot update interaction {interaction_id}: not found")
            return False
        
        # Update the interaction
        interaction.update(updates)
        interaction['updated_at'] = datetime.now().isoformat()
        
        # Save the updated interaction
        file_path = self.interactions_dir / f"{interaction_id}.json"
        with open(file_path, 'w') as f:
            json.dump(interaction, f, indent=2)
        
        # If success status changed, update the index
        if 'evaluation' in updates and 'success' in updates.get('evaluation', {}):
            index = self._get_index()
            for idx, item in enumerate(index["interactions"]):
                if item["id"] == interaction_id:
                    item["success"] = updates['evaluation']['success']
                    break
            self._update_index(index)
        
        self.logger.info(f"Updated interaction with ID: {interaction_id}")
        return True