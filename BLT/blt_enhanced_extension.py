#!/usr/bin/env python
"""
Enhanced BLT Extension for VoxSigilRAG

This demonstrates how to create a comprehensive extension that integrates
BLT (Byte Latent Transformer) capabilities with the VoxSigilRAG system.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BLTEnhancedExtension:
    """
    Comprehensive BLT extension for VoxSigilRAG.

    This extension adds:
    - BLT encoding capabilities
    - Entropy-based routing
    - Enhanced semantic search
    - Patch-based processing
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        entropy_threshold: float = 0.5,
        blt_hybrid_weight: float = 0.7,
    ):
        """
        Initialize BLT extension.

        Args:
            embedding_dim: Dimension of BLT embeddings
            entropy_threshold: Threshold for BLT vs standard routing
            blt_hybrid_weight: Weight for BLT vs standard results
        """
        self.embedding_dim = embedding_dim
        self.entropy_threshold = entropy_threshold
        self.blt_hybrid_weight = blt_hybrid_weight

        self.blt_encoder = None
        self.patch_encoder = None
        self._initialized = False

        logger.info(f"BLT Enhanced Extension created with dim={embedding_dim}")

    def initialize(self, rag_instance) -> None:
        """Initialize BLT components."""
        try:
            # Try to import BLT components
            from .blt_encoder import ByteLatentTransformerEncoder, SigilPatchEncoder

            self.blt_encoder = ByteLatentTransformerEncoder(
                embedding_dim=self.embedding_dim
            )
            self.patch_encoder = SigilPatchEncoder()

            self._initialized = True
            logger.info("BLT Enhanced Extension initialized successfully")

        except ImportError as e:
            logger.warning(f"BLT components not available: {e}")
            self._initialized = False

    def extend_capabilities(self, rag_instance) -> Dict[str, Callable]:
        """Add BLT-enhanced methods to RAG instance."""
        if not self._initialized:
            return {}

        def blt_encode_text(text: str) -> np.ndarray:
            """Encode text using BLT."""
            if self.blt_encoder:
                return self.blt_encoder.encode(text)
            return np.zeros(self.embedding_dim)

        def calculate_entropy(text: str) -> float:
            """Calculate entropy of text for routing decisions."""
            if not text:
                return 0.0

            # Simple entropy calculation
            import math

            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1

            total_chars = len(text)
            entropy = 0.0
            for count in char_counts.values():
                probability = count / total_chars
                entropy -= probability * math.log2(probability)

            return entropy / 8.0  # Normalize to roughly 0-1 range

        def should_use_blt(query: str) -> bool:
            """Determine if BLT should be used for this query."""
            entropy = calculate_entropy(query)
            return entropy > self.entropy_threshold

        def blt_enhanced_search(query: str, **kwargs) -> Tuple[str, List[Dict]]:
            """Perform BLT-enhanced RAG search."""
            use_blt = should_use_blt(query)

            if use_blt and self.blt_encoder:
                # Use BLT encoding for better semantic understanding
                blt_embedding = self.encode(query)
                kwargs["blt_embedding"] = blt_embedding
                kwargs["use_blt_scoring"] = True
                logger.info(
                    f"Using BLT encoding for query (entropy > {self.entropy_threshold})"
                )
            else:
                logger.info("Using standard encoding for query")

            return rag_instance.create_rag_context(query=query, **kwargs)

        def hybrid_search(query: str, **kwargs) -> Tuple[str, List[Dict]]:
            """Perform hybrid BLT + standard search and combine results."""
            if not self._initialized:
                return rag_instance.create_rag_context(query=query, **kwargs)

            # Get both BLT and standard results
            blt_context, blt_sigils = blt_enhanced_search(query, **kwargs)
            std_context, std_sigils = rag_instance.create_rag_context(
                query=query, **kwargs
            )

            # Combine results based on hybrid weight
            combined_sigils = self._combine_search_results(blt_sigils, std_sigils)

            # Reformat combined context
            combined_context = rag_instance._format_sigils_to_context(
                combined_sigils, kwargs.get("detail_level", "standard")
            )

            return combined_context, combined_sigils

        def get_blt_stats() -> Dict[str, Any]:
            """Get BLT extension statistics."""
            return {
                "initialized": self._initialized,
                "embedding_dim": self.embedding_dim,
                "entropy_threshold": self.entropy_threshold,
                "hybrid_weight": self.blt_hybrid_weight,
                "encoder_available": self.blt_encoder is not None,
            }

        def store_metalearn_pattern(category: str, pattern: Dict[str, Any]) -> None:
            """Store a metalearn pattern for future use."""
            if not hasattr(rag_instance, "_metalearn_patterns"):
                rag_instance._metalearn_patterns = {}

            if category not in rag_instance._metalearn_patterns:
                rag_instance._metalearn_patterns[category] = []

            rag_instance._metalearn_patterns[category].append(
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "pattern": pattern,
                    "source": "blt_extension",
                }
            )

            logger.info(f"Stored metalearn pattern for category: {category}")

        def get_metalearn_patterns(
            category: str, limit: int = 10
        ) -> List[Dict[str, Any]]:
            """Retrieve metalearn patterns for a category."""
            if not hasattr(rag_instance, "_metalearn_patterns"):
                return []

            patterns = rag_instance._metalearn_patterns.get(category, [])
            return patterns[-limit:] if patterns else []

        return {
            "blt_encode_text": blt_encode_text,
            "calculate_entropy": calculate_entropy,
            "should_use_blt": should_use_blt,
            "blt_enhanced_search": blt_enhanced_search,
            "hybrid_search": hybrid_search,
            "get_blt_stats": get_blt_stats,
            "store_metalearn_pattern": store_metalearn_pattern,
            "get_metalearn_patterns": get_metalearn_patterns,
        }

    def _combine_search_results(
        self, blt_results: List[Dict], std_results: List[Dict]
    ) -> List[Dict]:
        """Combine BLT and standard search results."""
        # Simple combination strategy - could be more sophisticated
        combined = []
        seen_sigils = set()

        # Add BLT results with higher weight
        for sigil in blt_results:
            sigil_id = sigil.get("sigil", "")
            if sigil_id not in seen_sigils:
                sigil_copy = sigil.copy()
                if "_similarity_score" in sigil_copy:
                    sigil_copy["_similarity_score"] *= self.blt_hybrid_weight
                combined.append(sigil_copy)
                seen_sigils.add(sigil_id)

        # Add standard results with lower weight
        for sigil in std_results:
            sigil_id = sigil.get("sigil", "")
            if sigil_id not in seen_sigils:
                sigil_copy = sigil.copy()
                if "_similarity_score" in sigil_copy:
                    sigil_copy["_similarity_score"] *= 1.0 - self.blt_hybrid_weight
                combined.append(sigil_copy)
                seen_sigils.add(sigil_id)

        # Sort by combined scores
        combined.sort(key=lambda x: x.get("_similarity_score", 0), reverse=True)

        return combined

    def pre_query(self, query: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Process query before RAG execution."""
        if self._initialized and self.blt_encoder:
            # Add entropy-based routing information
            entropy = self._calculate_query_entropy(query)
            kwargs["_blt_entropy"] = entropy
            kwargs["_blt_routing"] = entropy > self.entropy_threshold

        return query, kwargs

    def post_query(
        self, query: str, result: Tuple[str, List[Dict]], **kwargs
    ) -> Tuple[str, List[Dict]]:
        """Process result after RAG execution."""
        if self._initialized:
            context, sigils = result

            # Add BLT metadata to results
            for sigil in sigils:
                if "_blt_entropy" in kwargs:
                    sigil["_blt_entropy"] = kwargs["_blt_entropy"]
                if "_blt_routing" in kwargs:
                    sigil["_blt_routing_used"] = kwargs["_blt_routing"]

        return result

    def _calculate_query_entropy(self, text: str) -> float:
        """Calculate text entropy for routing decisions."""
        if not text:
            return 0.0

        import math

        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy / 8.0  # Normalize

    def shutdown(self) -> None:
        """Clean up BLT resources."""
        if self.blt_encoder:
            # Cleanup if needed
            pass
        logger.info("BLT Enhanced Extension shutdown")


def create_blt_enhanced_rag(
    voxsigil_library_path: Optional[Path] = None,
    embedding_dim: int = 128,
    entropy_threshold: float = 0.5,
    blt_hybrid_weight: float = 0.7,
    **kwargs,
):
    """
    Factory function to create a BLT-enhanced VoxSigilRAG instance.

    Args:
        voxsigil_library_path: Path to VoxSigil library
        embedding_dim: BLT embedding dimension
        entropy_threshold: Entropy threshold for BLT routing
        blt_hybrid_weight: Weight for BLT vs standard results
        **kwargs: Additional VoxSigilRAG arguments

    Returns:
        VoxSigilRAG instance with BLT extension loaded
    """
    from .voxsigil_rag import VoxSigilRAG

    # Create base RAG instance
    rag = VoxSigilRAG(
        voxsigil_library_path=voxsigil_library_path,
        embedding_dim=embedding_dim,
        **kwargs,
    )

    # Create and register BLT extension
    blt_extension = BLTEnhancedExtension(
        embedding_dim=embedding_dim,
        entropy_threshold=entropy_threshold,
        blt_hybrid_weight=blt_hybrid_weight,
    )

    rag.register_extension("BLT_Enhanced", blt_extension)

    logger.info("Created BLT-enhanced VoxSigilRAG instance")
    return rag


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create BLT-enhanced RAG instance
    rag = create_blt_enhanced_rag(
        voxsigil_library_path=Path("/path/to/voxsigil/library"),
        embedding_dim=128,
        entropy_threshold=0.5,
        blt_hybrid_weight=0.7,
    )

    # Initialize BLT extension
    blt_extension = rag.get_extension("BLT_Enhanced")
    blt_extension.initialize(rag)

    # Test encoding and search
    query = "What is the capital of France?"
    context, sigils = blt_extension.blt_enhanced_search(query)

    print("Context:", context)
    print("Sigils:", sigils)
