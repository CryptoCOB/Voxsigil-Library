#!/usr/bin/env python
"""
Semantic Cache Manager for VoxSigil BLT middleware system.

This module provides semantic caching capabilities by storing query embeddings
and comparing new queries against the cache using vector similarity.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable

# Configure logging
logger = logging.getLogger("VoxSigilSemanticCache")

class SemanticCacheManager:
    """
    Manages a semantic cache of queries using embedding similarity.
    
    Instead of exact string matching, this cache uses vector similarity
    to identify semantically similar queries, which enables better reuse
    of retrieval results.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 ttl_seconds: int = 360,
                 embedding_function: Optional[Callable[[str], np.ndarray]] = None,
                 max_cache_size: int = 100):
        """
        Initialize the semantic cache manager.
        
        Args:
            similarity_threshold: Threshold to determine if cache entry is similar enough
            ttl_seconds: Time-to-live in seconds for cache entries
            embedding_function: Function to generate embeddings for queries
            max_cache_size: Maximum number of entries to store in cache
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embedding_function = embedding_function
        self.max_cache_size = max_cache_size
        
        # Cache structure: {cache_id: (query_text, query_embedding, data, timestamp)}
        self._cache: Dict[str, Tuple[str, np.ndarray, Any, float]] = {}
        
        logger.info(f"SemanticCacheManager initialized with similarity threshold {similarity_threshold}")

    def set_embedding_function(self, embedding_function: Callable[[str], np.ndarray]) -> None:
        """
        Set the function used to generate embeddings for cache lookup.
        
        Args:
            embedding_function: Function that takes a text string and returns an embedding vector
        """
        self.embedding_function = embedding_function
        
    def _generate_cache_id(self, query: str) -> str:
        """
        Generate a unique ID for the cache entry.
        
        Args:
            query: The query text
            
        Returns:
            A unique string ID
        """
        import hashlib
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Normalize the vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(embedding1_normalized, embedding2_normalized))
    
    def add(self, query: str, data: Any) -> None:
        """
        Add a query and its result data to the cache.
        
        Args:
            query: Query text
            data: Data to cache (e.g., RAG context and sigils)
        """
        if not self.embedding_function:
            logger.warning("No embedding function set, cannot add to semantic cache")
            return
            
        # Clean expired entries before adding new ones
        self.clean_expired()
        
        # If cache is at max capacity, remove oldest entry
        if len(self._cache) >= self.max_cache_size:
            oldest_id = None
            oldest_time = float('inf')
            
            for cache_id, (_, _, _, timestamp) in self._cache.items():
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_id = cache_id
                    
            if oldest_id:
                logger.debug(f"Removing oldest cache entry {oldest_id} to make room")
                del self._cache[oldest_id]
        
        # Generate embedding for the query
        try:
            query_embedding = self.embedding_function(query)
            
            # Add to cache
            cache_id = self._generate_cache_id(query)
            self._cache[cache_id] = (query, query_embedding, data, time.monotonic())
            
            logger.debug(f"Added query to semantic cache: '{query[:30]}...' [id: {cache_id}]")
        except Exception as e:
            logger.error(f"Error adding to semantic cache: {e}")
    
    def get(self, query: str) -> Optional[Tuple[Any, float]]:
        """
        Get cached data for a semantically similar query.
        
        Args:
            query: Query text to look up
            
        Returns:
            Tuple of (cached_data, similarity_score) or None if no match
        """
        if not self.embedding_function or not self._cache:
            return None
            
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function(query)
            
            best_match = None
            best_similarity = 0.0
            
            # Find the most similar cached query
            for cache_id, (cached_query, cached_embedding, data, timestamp) in list(self._cache.items()):
                # Skip expired entries
                if time.monotonic() - timestamp > self.ttl_seconds:
                    continue
                    
                similarity = self._calculate_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (data, similarity, cache_id, timestamp)
            
            # If we found a good match, refresh its timestamp and return
            if best_match and best_similarity >= self.similarity_threshold:
                data, similarity, cache_id, old_timestamp = best_match
                
                # Update timestamp to refresh TTL
                cached_query, cached_embedding, _, _ = self._cache[cache_id]
                self._cache[cache_id] = (cached_query, cached_embedding, data, time.monotonic())
                
                logger.info(f"Semantic cache HIT for query: '{query[:30]}...' [similarity: {best_similarity:.4f}]")
                return data, best_similarity
            
            if best_match:
                logger.debug(f"Semantic cache NEAR MISS for query: '{query[:30]}...' [similarity: {best_similarity:.4f}]")
            else:
                logger.debug(f"Semantic cache MISS for query: '{query[:30]}...'")
                
            return None
        except Exception as e:
            logger.error(f"Error retrieving from semantic cache: {e}")
            return None
    
    def clean_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.monotonic()
        expired_keys = [
            k for k, (_, _, _, ts) in self._cache.items() 
            if current_time - ts > self.ttl_seconds
        ]
        
        for k in expired_keys:
            del self._cache[k]
            
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired semantic cache entries")
            
        return len(expired_keys)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        logger.info("Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary of cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }
