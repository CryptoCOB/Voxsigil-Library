#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARTRAGBridge - A bridge between the ART module and the RAG (Retrieval Augmented Generation) system.

This module provides the ARTRAGBridge class that connects voxsigil.art.ARTManager
with the BLTEnhancedRAG system for intelligent pattern-aware retrieval and generation.
"""

import os
import sys
import importlib
import importlib.util
import logging
from typing import Any, Optional, Dict, List
from pathlib import Path
import time


# Import ART components
from .art_logger import get_art_logger
from .art_manager import ARTManager

# Dynamically add VoxSigilRag directory to sys.path
voxsigil_library_path = (
    Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    / "Voxsigil-Library-main"
)
if str(voxsigil_library_path) not in sys.path:
    sys.path.append(str(voxsigil_library_path))

# Global variables for RAG components (will be populated dynamically)
HAS_RAG = False
BLTEnhancedRAG = None
VoxSigilRAG = None


# Try to import RAG components dynamically
def load_rag_components():
    global HAS_RAG, BLTEnhancedRAG, VoxSigilRAG

    logger = get_art_logger("RAGLoader")

    try:  # Try to import BLTEnhancedRAG
        rag_spec = importlib.util.find_spec("VoxSigilRag.voxsigil_blt_rag")
        if not rag_spec:
            # Try with the full path for new structure
            rag_spec = importlib.util.find_spec(
                "Voxsigil_Library_main.VoxSigilRag.voxsigil_blt_rag"
            )

        if rag_spec and rag_spec.loader is not None:
            rag_module = importlib.util.module_from_spec(rag_spec)
            rag_spec.loader.exec_module(rag_module)
            BLTEnhancedRAG = getattr(rag_module, "BLTEnhancedRAG", None)
            logger.info("Successfully imported BLTEnhancedRAG")
        elif rag_spec:
            logger.warning(
                "rag_spec.loader is None, cannot exec_module for BLTEnhancedRAG"
            )

        # Try to import standard VoxSigilRAG as fallback
        std_rag_spec = importlib.util.find_spec("VoxSigilRag.voxsigil_rag")
        if not std_rag_spec:
            # Try with the full path for new structure
            std_rag_spec = importlib.util.find_spec(
                "Voxsigil_Library_main.VoxSigilRag.voxsigil_rag"
            )

        if std_rag_spec and std_rag_spec.loader is not None:
            std_rag_module = importlib.util.module_from_spec(std_rag_spec)
            std_rag_spec.loader.exec_module(std_rag_module)
            VoxSigilRAG = getattr(std_rag_module, "VoxSigilRAG", None)
            logger.info("Successfully imported VoxSigilRAG")
        elif std_rag_spec:
            logger.warning(
                "std_rag_spec.loader is None, cannot exec_module for VoxSigilRAG"
            )

        # Check if essential RAG components are available
        HAS_RAG = BLTEnhancedRAG is not None or VoxSigilRAG is not None

        if HAS_RAG:
            logger.info("RAG components are available")
            if BLTEnhancedRAG is None:
                logger.warning(
                    "BLTEnhancedRAG not available, will use standard VoxSigilRAG"
                )
        else:
            logger.warning("RAG components could not be loaded")

        return HAS_RAG

    except ImportError as e:
        logger.error(f"Failed to import RAG components: {e}")
        return False


# Try to load RAG components when module is imported
HAS_RAG = load_rag_components()


class ARTRAGBridge:
    """
    A bridge between the ART module and BLTEnhancedRAG system.

    This class integrates ART's pattern recognition capabilities with the
    BLTEnhancedRAG system for more intelligent and context-aware retrieval and generation.
    """

    def __init__(
        self,
        art_manager: Optional[ARTManager] = None,
        voxsigil_library_path: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        blt_hybrid_weight: float = 0.7,
        entropy_threshold: float = 0.4,
        category_relevance_boost: float = 0.2,
        novel_category_penalty: float = 0.1,
        context_feedback_enabled: bool = True,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ARTRAGBridge.

        Args:
            art_manager: Optional ARTManager instance. If None, a new one will be created.
            voxsigil_library_path: Path to the VoxSigil library for RAG.
            embedding_model: Name of the embedding model for RAG.
            blt_hybrid_weight: Weight for BLT in hybrid embeddings (0-1).
            entropy_threshold: Entropy threshold for processing.
            category_relevance_boost: Boost factor for ART category matches (0-1).
            novel_category_penalty: Penalty factor for novel categories (0-1).
            context_feedback_enabled: Whether to feed context back to ART.
            logger_instance: Optional logger instance. If None, a new one will be created."""
        self.logger = logger_instance or get_art_logger("ARTRAGBridge")

        # Initialize ARTManager
        self.art_manager = art_manager or ARTManager()

        # Configuration
        self.voxsigil_library_path = voxsigil_library_path
        self.blt_hybrid_weight = blt_hybrid_weight
        self.entropy_threshold = entropy_threshold
        self.category_relevance_boost = category_relevance_boost
        self.novel_category_penalty = novel_category_penalty
        self.context_feedback_enabled = context_feedback_enabled

        # Initialize RAG system if available
        self.rag_available = HAS_RAG
        self.rag = None

        if self.rag_available:
            try:
                if BLTEnhancedRAG is not None:
                    self.rag = BLTEnhancedRAG(
                        voxsigil_library_path=voxsigil_library_path,
                        cache_enabled=True,
                        embedding_model=embedding_model,
                        blt_hybrid_weight=blt_hybrid_weight,
                        entropy_threshold=entropy_threshold,
                    )
                    self.logger.info("ARTRAGBridge initialized with BLTEnhancedRAG")
                elif VoxSigilRAG is not None:
                    self.rag = VoxSigilRAG(
                        voxsigil_library_path=voxsigil_library_path,
                        cache_enabled=True,
                        embedding_model=embedding_model,
                    )
                    self.logger.info(
                        "ARTRAGBridge initialized with standard VoxSigilRAG (fallback)"
                    )
                else:
                    raise ImportError("No RAG implementation available")
            except Exception as e:
                self.rag_available = False
                self.logger.error(f"Failed to initialize RAG system: {e}")
                self.logger.warning(
                    "ARTRAGBridge will operate without RAG capabilities"
                )
        else:
            self.logger.warning(
                "RAG components not available. Bridge will use ART only."
            )

        # Memory for context association with ART categories
        self._category_context_cache = {}
        self._category_sigil_associations = {}

        # Statistics
        self.stats = {
            "total_queries_processed": 0,
            "art_enhanced_queries": 0,
            "rag_fallback_queries": 0,
            "context_feedback_count": 0,
            "category_boosts_applied": 0,
            "novel_penalties_applied": 0,
        }

    def create_context(
        self,
        query: str,
        num_sigils: int = 5,
        context_length: Optional[int] = None,
        art_analyze_query: bool = True,
    ) -> dict[str, Any]:
        """
        Create a RAG context enhanced with ART pattern recognition.

        Args:
            query: The query to create context for.
            num_sigils: Number of sigils to retrieve.
            context_length: Maximum context length in characters.
            art_analyze_query: Whether to analyze the query with ART.

        Returns:
            A dict containing the RAG context and metadata.
        """
        self.stats["total_queries_processed"] += 1
        start_time = time.time()
        result = {
            "query": query,
            "art_enhanced": False,
            "context": "",
            "sigils": [],
            "art_analysis": None,
            "processing_time": 0,
        }

        # Step 1: Analyze query with ART if enabled
        art_result = None
        if art_analyze_query:
            try:
                art_result = self.art_manager.analyze_input(query)
                result["art_analysis"] = art_result
                self.logger.info("ART analysis performed on query")
            except Exception as e:
                self.logger.error(f"Error in ART query analysis: {e}")

        # Step 2: Create RAG context
        if self.rag_available and self.rag:
            try:
                # Prepare sigil boosting/penalizing based on ART categories
                boosted_sigils = set()
                penalized_sigils = set()

                if art_result:
                    category = art_result.get("category", {})
                    category_id = category.get("id")
                    is_novel = art_result.get("is_novel_category", False)

                    # If we have a category, try to use associated sigils for boosting
                    if category_id and category_id in self._category_sigil_associations:
                        associated_sigils = self._category_sigil_associations[
                            category_id
                        ]
                        boosted_sigils.update(associated_sigils)
                        self.stats["category_boosts_applied"] += 1

                    # If this is a novel category, we might want to penalize common sigils
                    if is_novel:
                        # Find the most commonly used sigils across categories
                        common_sigils = self._find_common_sigils()
                        penalized_sigils.update(common_sigils)
                        self.stats["novel_penalties_applied"] += 1

                # Create RAG context with possible boosting/penalizing
                context_str, sigils = self.rag.create_rag_context(
                    query=query,
                    num_sigils=num_sigils,
                    max_context_chars=context_length
                    if context_length
                    else self.rag.default_max_context_chars,
                    boosted_sigils=boosted_sigils,
                    penalized_sigils=penalized_sigils,
                    boost_factor=self.category_relevance_boost,
                    penalty_factor=self.novel_category_penalty,
                )

                # Update result
                result["context"] = context_str
                result["sigils"] = sigils

                # Mark as ART-enhanced if we actually applied category-based modifications
                if boosted_sigils or penalized_sigils:
                    result["art_enhanced"] = True
                    self.stats["art_enhanced_queries"] += 1

                # Step 3: Optionally update ART category with context
                if self.context_feedback_enabled and art_result and sigils:
                    category = art_result.get("category", {})
                    category_id = category.get("id")

                    if category_id:
                        # Update context cache for this category
                        self._update_category_context(category_id, query, sigils)
                        self.stats["context_feedback_count"] += 1

                self.logger.info(f"Created RAG context with {len(sigils)} sigils")
            except Exception as e:
                self.logger.error(f"Error creating RAG context: {e}")
                result["error"] = str(e)
        else:
            self.logger.warning("RAG system not available, returning ART analysis only")
            self.stats["rag_fallback_queries"] += 1

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        return result

    def train_from_context(
        self,
        query: str,
        response: str,
        # Removed unused parameter "context"
        sigils: Optional[list[dict[str, Any]]] = None,
        art_analyze_all: bool = False,
    ) -> dict[str, Any]:
        """
        Train ART on the query, response, and optionally the context.

        Args:
            query: The query text.
            response: The response text.
            context: Optional context text used for generation.
            sigils: Optional list of sigils used in the context.
            art_analyze_all: Whether to perform ART analysis on each sigil.

        Returns:
            A dict containing training results and statistics.
        """
        # Start with query-response pair
        training_items = [(query, response)]

        # Optionally include context elements
        if art_analyze_all and sigils:
            for sigil in sigils:
                sigil_content = sigil.get("content", "")
                if sigil_content:
                    training_items.append(sigil_content)

        # Train on batch
        result = self.art_manager.train_on_batch(training_items)

        # If sigils were provided, update category-sigil associations
        if sigils:
            # First analyze the query to get the category
            try:
                art_result = self.art_manager.analyze_input(query)
                category = art_result.get("category", {})
                category_id = category.get("id")

                if category_id:
                    # Extract sigil IDs
                    sigil_ids = [s.get("id") for s in sigils if "id" in s]
                    # Update associations
                    self._update_category_sigil_associations(
                        category_id, [s for s in sigil_ids if isinstance(s, str)]
                    )
            except Exception as e:
                self.logger.error(f"Error updating category-sigil associations: {e}")

        return result

    def _update_category_context(
        self, category_id: str, query: str, sigils: list[dict[str, Any]]
    ) -> None:
        """Update the context cache for a category."""
        if category_id not in self._category_context_cache:
            self._category_context_cache[category_id] = []

        # Add current query and sigils to cache (limited to last 5)
        self._category_context_cache[category_id].append(
            {
                "query": query,
                "sigil_ids": [s.get("id") for s in sigils if "id" in s],
                "timestamp": time.time(),
            }
        )

        # Keep only most recent 5 entries
        if len(self._category_context_cache[category_id]) > 5:
            self._category_context_cache[category_id] = self._category_context_cache[
                category_id
            ][-5:]

    def _update_category_sigil_associations(
        self, category_id: str, sigil_ids: list[str]
    ) -> None:
        """Update the category-sigil associations."""
        if not sigil_ids:
            return

        if category_id not in self._category_sigil_associations:
            self._category_sigil_associations[category_id] = set()

        # Update with new sigils
        self._category_sigil_associations[category_id].update(sigil_ids)

        # Keep only top 20 most frequent sigils for this category
        if len(self._category_sigil_associations[category_id]) > 20:
            # This would ideally be based on frequency, but for simplicity, just keep 20
            self._category_sigil_associations[category_id] = set(
                list(self._category_sigil_associations[category_id])[:20]
            )

    def _find_common_sigils(self, top_n: int = 10) -> list[str]:
        """Find the most common sigils across all categories."""
        # Count sigil frequencies across all categories
        sigil_counts = {}
        for sigils in self._category_sigil_associations.values():
            for sigil_id in sigils:
                sigil_counts[sigil_id] = sigil_counts.get(sigil_id, 0) + 1

        # Sort by frequency and return top N
        common_sigils = sorted(sigil_counts.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in common_sigils[:top_n]]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ARTRAGBridge.

        Returns:
            A dict containing statistics
        """
        stats: Dict[str, Any] = self.stats.copy()

        # Add derived statistics
        if stats["total_queries_processed"] > 0:
            stats["art_enhanced_ratio"] = (
                float(stats["art_enhanced_queries"]) / stats["total_queries_processed"]
            )
            stats["rag_fallback_ratio"] = (
                float(stats["rag_fallback_queries"]) / stats["total_queries_processed"]
            )
            stats["context_feedback_ratio"] = (
                float(stats["context_feedback_count"])
                / stats["total_queries_processed"]
            )
        else:
            stats["art_enhanced_ratio"] = 0
            stats["rag_fallback_ratio"] = 0
            stats["context_feedback_ratio"] = 0

        # Add ART stats
        if hasattr(self.art_manager, "status") and callable(self.art_manager.status):
            art_stats = self.art_manager.status()
            stats["art_stats"] = art_stats if isinstance(art_stats, dict) else {}

        # Add RAG stats
        if self.rag_available and self.rag and hasattr(self.rag, "get_stats"):
            try:
                rag_stats = self.rag.get_stats()
                stats["rag_stats"] = rag_stats
            except Exception as e:
                self.logger.error(f"Error getting RAG stats: {e}")

        # Add category association stats
        stats["category_context_cache_size"] = len(self._category_context_cache)
        stats["category_sigil_associations_count"] = len(
            self._category_sigil_associations
        )

        return stats


if __name__ == "__main__":
    # Example usage
    logger = get_art_logger()

    # Check if RAG components are available and print status
    if HAS_RAG:
        if BLTEnhancedRAG:
            logger.info("BLTEnhancedRAG components are available.")
        else:
            logger.info("Standard VoxSigilRAG is available (fallback).")
        logger.info("Initializing ARTRAGBridge...")
    else:
        logger.info("RAG components are not available. ARTRAGBridge will use ART only.")

    # Create bridge
    lib_path = voxsigil_library_path if voxsigil_library_path.exists() else None
    bridge = ARTRAGBridge(
        voxsigil_library_path=lib_path, blt_hybrid_weight=0.7, entropy_threshold=0.4
    )

    # Example queries
    queries = [
        "What are the principles of quantum computing?",
        "How do neural networks process information?",
        "What is the meaning of life?",
    ]

    # Process each query
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i + 1}:")
        logger.info(f"'{query}'")

        # Create context
        result = bridge.create_context(query, num_sigils=3)

        # Print results
        if result["art_analysis"]:
            art_result = result["art_analysis"]
            category = art_result.get("category", {})
            logger.info(f"ART Category: {category.get('id', 'unknown')}")
            logger.info(f"Novel: {art_result.get('is_novel_category', False)}")

        if result["context"]:
            # Only show first 100 chars of context
            context_preview = (
                result["context"][:100] + "..."
                if len(result["context"]) > 100
                else result["context"]
            )
            logger.info(f"Context (preview): {context_preview}")
            logger.info(f"Sigils used: {len(result['sigils'])}")
            logger.info(f"ART Enhanced: {result['art_enhanced']}")
        else:
            logger.info("No RAG context created.")

        logger.info("-" * 50)

        # Mock response for training
        mock_response = f"This is a response to the query about {query.split()[0]}"
        train_result = bridge.train_from_context(
            query, mock_response, sigils=result.get("sigils"), art_analyze_all=False
        )
        logger.info(
            f"Training complete. Status: {train_result.get('status', 'unknown')}"
        )
        logger.info("-" * 50)

    # Show stats
    logger.info("\nBridge Statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):  # Skip nested dicts
            logger.info(f"{key}: {value}")
