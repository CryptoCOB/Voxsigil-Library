#!/usr/bin/env python
"""
blt_encoder_interface.py - Interface definition for BLT (Bidirectional Language Transformer) Encoder

This file defines the interface for the BLT encoder used by VantaCore.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger("VoxSigil.BLTEncoderInterface")


class BaseBLTEncoder(ABC):
    """
    Abstract base class defining the interface for BLT encoders.

    BLT encoders are responsible for converting text to vector embeddings
    that can be used for similarity comparison and other NLP tasks.
    """

    @abstractmethod
    def encode(self, text_content: str, task_type: str = "general") -> List[float]:
        """
        Encode text content into a vector embedding.

        Args:
            text_content: The text to encode
            task_type: The type of task (affects encoding parameters)

        Returns:
            List[float]: Vector embedding of the text
        """
        pass

    @abstractmethod
    def encode_batch(
        self, text_contents: List[str], task_type: str = "general"
    ) -> List[List[float]]:
        """
        Encode multiple text strings into vector embeddings.

        Args:
            text_contents: List of text strings to encode
            task_type: The type of task (affects encoding parameters)

        Returns:
            List[List[float]]: List of vector embeddings
        """
        pass

    @abstractmethod
    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Similarity score (typically between 0 and 1)
        """
        pass

    @abstractmethod
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: The query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top matches to return

        Returns:
            List[Dict[str, Any]]: Top matches with similarity scores
        """
        pass
