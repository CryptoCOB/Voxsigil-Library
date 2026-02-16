"""Retrieval layer: HNSW-based in-process vector retrieval."""

import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None

logger = logging.getLogger(__name__)


class HNSWRetriever:
    """In-process HNSW vector index for semantic retrieval."""

    def __init__(self, dim: int = 384, ef: int = 200, max_elements: int = 10000, m: int = 16):
        """
        Initialize HNSW index.
        
        Args:
            dim: Vector dimension (default 384 for sentence-transformers)
            ef: Search parameter (higher = more accurate, slower)
            max_elements: Maximum elements to store
            m: Number of bi-directional links per element
        """
        if hnswlib is None:
            msg = "hnswlib required for HNSWRetriever. Install with: pip install hnswlib"
            raise ImportError(msg)
        
        self.dim = dim
        self.ef = ef
        self.max_elements = max_elements
        self.m = m
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=m)
        self.index.ef = ef
        
        # Store ID mappings
        self._id_map: Dict[int, str] = {}
        self._reverse_map: Dict[str, int] = {}
        self._next_id = 0
        # Store HNSW index (use _hnsw_index to avoid method name conflict)
        self._hnsw_index = self.index
    
    def add_vectors(self, vectors: List[List[float]], ids: List[str]) -> None:
        """
        Index a batch of vectors.
        
        Args:
            vectors: List of vectors (each dimension = self.dim)
            ids: List of corresponding IDs
        
        Raises:
            ValueError: If lengths don't match or dimension mismatch
        """
        if len(vectors) != len(ids):
            raise ValueError(f"Vectors and IDs length mismatch: {len(vectors)} vs {len(ids)}")
        
        if not vectors:
            return  # No-op for empty input
        
        # Validate dimensions
        if len(vectors[0]) != self.dim:
            raise ValueError(f"Vector dimension mismatch: {len(vectors[0])} vs {self.dim}")
        
        # Add to index
        for vector, id_str in zip(vectors, ids):
            if id_str in self._reverse_map:
                # Update existing
                internal_id = self._reverse_map[id_str]
            else:
                # New entry
                internal_id = self._next_id
                self._id_map[internal_id] = id_str
                self._reverse_map[id_str] = internal_id
                self._next_id += 1
            
            # Add to HNSW with proper numpy array format
            vec_array = np.array(vector, dtype=np.float32)
            self._hnsw_index.add_items(data=vec_array, ids=np.array([internal_id], dtype=np.int32))
    
    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query_vector: Query vector (dimension = self.dim)
            k: Number of neighbors to return
        
        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (highest first)
        
        Raises:
            ValueError: If dimension mismatch or index empty
        """
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dimension mismatch: {len(query_vector)} vs {self.dim}")
        
        if self._next_id == 0:
            return []  # Empty index
        
        # Clamp k to actual index size
        k = min(k, self._next_id)
        
        # Search HNSW with proper numpy array format
        query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        internal_ids, distances = self._hnsw_index.knn_query(query_array, k=k)
        
        # Convert back to string IDs
        # Note: hnswlib returns distances as they are (cosine distance)
        # For cosine similarity, convert: similarity = 1 - distance
        results = []
        for internal_id, distance in zip(internal_ids[0], distances[0]):
            str_id = self._id_map.get(int(internal_id))
            if str_id:
                similarity = 1.0 - distance  # Convert distance to similarity
                results.append((str_id, similarity))
        
        return results
    
    def get_size(self) -> int:
        """Return number of indexed vectors."""
        return self._next_id
    
    def clear(self) -> None:
        """Clear all indexed vectors."""
        self._hnsw_index = hnswlib.Index(space="cosine", dim=self.dim)
        self._hnsw_index.init_index(max_elements=self.max_elements, ef_construction=200, M=self.m)
        self._hnsw_index.ef = self.ef
        self._id_map = {}
        self._reverse_map = {}
        self._next_id = 0


__all__ = ["HNSWRetriever"]
