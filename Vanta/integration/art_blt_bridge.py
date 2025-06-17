#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARTBLTBridge - A bridge between the ART module and BLT middleware.

This module provides the ARTBLTBridge class that connects voxsigil.art.ARTManager
with the BLT middleware for entropy-based selective pattern analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .art_controller import ARTManager
from .art_logger import get_art_logger

# Try to import BLT components - only importing what we need
try:
    # Only import the SigilPatchEncoder which is actually used in this module
    from VoxSigilRag.sigil_patch_encoder import SigilPatchEncoder

    # Set flag that BLT is available
    HAS_BLT = True
except ImportError as e:
    print(f"Failed to import BLT components: {e}")
    HAS_BLT = False


class ARTBLTBridge:
    """
    A bridge between the ART module and BLT middleware.

    This class uses BLT's entropy-based analysis to selectively process data with ART,
    optimizing computational resources by focusing ART's pattern recognition on data
    with high information content or entropy.
    """

    def __init__(
        self,
        art_manager: Optional[ARTManager] = None,
        entropy_threshold: float = 0.4,
        blt_patch_size: int = 8,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ARTBLTBridge.

        Args:
            art_manager: Optional ARTManager instance. If None, a new one will be created.
            entropy_threshold: Entropy threshold for triggering ART analysis.
                Higher values mean more selective analysis (only high entropy inputs).
            blt_patch_size: Size of BLT patches for entropy calculation.
            logger_instance: Optional logger instance. If None, a new one will be created.
        """
        self.logger = logger_instance or get_art_logger("ARTBLTBridge")
        self.entropy_threshold = entropy_threshold

        # Initialize ARTManager
        self.art_manager = art_manager or ARTManager(logger_instance=self.logger)

        # Initialize BLT components if available
        self.blt_available = HAS_BLT
        self.patch_encoder = None

        if self.blt_available:
            try:
                self.patch_encoder = SigilPatchEncoder(
                    entropy_threshold=entropy_threshold,
                    patch_size=blt_patch_size,
                )
                self.logger.info("ARTBLTBridge initialized with BLT components")
            except Exception as e:
                self.blt_available = False
                self.logger.error(f"Failed to initialize BLT components: {e}")
                self.logger.warning("ARTBLTBridge will operate without BLT entropy analysis")
        else:
            self.logger.warning(
                "BLT components not available. ARTBLTBridge will use fallback mode."
            )

        # Statistics
        self.stats = {
            "total_inputs_processed": 0,
            "art_processed_inputs": 0,
            "skipped_inputs": 0,
            "avg_entropy": 0.0,
            "high_entropy_count": 0,
        }

    def process_input(
        self,
        input_data: Union[str, Dict[str, Any], List],
        context: Optional[Dict[str, Any]] = None,
        force_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Process input data using the ART-BLT bridge.

        Args:
            input_data: The input data to process (text, dict, or list).
            context: Optional context information.
            force_analysis: Whether to force ART analysis regardless of entropy.

        Returns:
            A dict containing analysis results and metadata.
        """
        self.stats["total_inputs_processed"] += 1
        start_time = time.time()
        context = context or {}
        result = {
            "input_processed": True,
            "analysis_performed": False,
            "entropy_score": None,
            "processing_time": 0,
        }

        # Convert input_data to string for entropy calculation
        if isinstance(input_data, (dict, list)):
            # For structured data, convert to string representation
            input_str = str(input_data)
        else:
            input_str = str(input_data)

        # Calculate entropy using BLT if available
        should_analyze = force_analysis
        entropy_score = 0.0

        if self.blt_available and self.patch_encoder:
            try:
                # Calculate entropy using BLT's patch encoder
                entropy_score = self.patch_encoder.compute_average_entropy(input_str)
                should_analyze = entropy_score >= self.entropy_threshold or force_analysis

                result["entropy_score"] = entropy_score

                # Update stats
                self.stats["avg_entropy"] = (
                    self.stats["avg_entropy"] * (self.stats["total_inputs_processed"] - 1)
                    + entropy_score
                ) / self.stats["total_inputs_processed"]

                if entropy_score >= self.entropy_threshold:
                    self.stats["high_entropy_count"] += 1

                self.logger.debug(
                    f"Input entropy: {entropy_score:.4f}, threshold: {self.entropy_threshold}"
                )

                # If using BLT preprocessing, extract features for ART
                if should_analyze and "preprocess_input" in context and context["preprocess_input"]:
                    # Use BLT's patch-based processing to preprocess data
                    patches = self.patch_encoder.segment_into_patches(input_str)
                    if patches:
                        # Add patch analysis to context
                        context["blt_patches"] = patches
                        context["blt_entropy"] = entropy_score
            except Exception as e:
                self.logger.warning(f"Error in BLT entropy calculation: {e}")
                # Fallback to always analyzing if BLT fails
                should_analyze = True
        else:
            # Default to analyzing all inputs if BLT is not available
            should_analyze = True

        # Update stats based on whether we're analyzing
        if should_analyze:
            self.stats["art_processed_inputs"] += 1
        else:
            self.stats["skipped_inputs"] += 1

        # Analyze with ARTManager if entropy is high enough or forced
        art_result = None
        if should_analyze:
            try:
                art_result = self.art_manager.analyze_input(input_data, context)
                result["analysis_performed"] = True
                result["art_result"] = art_result

                self.logger.info(
                    f"ART analysis performed on input with entropy {entropy_score:.4f}"
                )

                # If this detected a novel category and BLT is available,
                # we could feed this back to BLT for future routing decisions
                if art_result.get("is_novel_category") and self.blt_available:
                    category = art_result.get("category", {})
                    category_id = category.get("id", "unknown")
                    result["novel_category_detected"] = {
                        "id": category_id,
                        "entropy": entropy_score,
                    }
            except Exception as e:
                self.logger.error(f"Error in ART analysis: {e}")
                result["error"] = str(e)
        else:
            self.logger.info(f"Skipped ART analysis due to low entropy: {entropy_score:.4f}")

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        return result

    def train_on_batch(
        self,
        batch: List[Union[str, Dict[str, Any], Tuple]],
        selective_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Train ARTManager on a batch of data, using BLT for selective training.

        Args:
            batch: A list of inputs to train on
            selective_training: If True, only train on high-entropy inputs

        Returns:
            A dict containing training results and statistics
        """
        if not batch:
            return {"status": "error", "message": "Empty batch provided"}

        # Filter batch if selective training is enabled
        if selective_training and self.blt_available and self.patch_encoder:
            high_entropy_batch = []
            for item in batch:
                # Extract text for entropy calculation
                if isinstance(item, tuple) and len(item) >= 1:
                    # For query-response pairs, calculate entropy on the query
                    text = str(item[0])
                else:
                    text = str(item)

                # Check if entropy exceeds threshold
                try:
                    entropy = self.patch_encoder.compute_average_entropy(text)
                    if entropy >= self.entropy_threshold:
                        high_entropy_batch.append(item)
                except Exception as e:
                    self.logger.warning(f"Error calculating entropy during batch filtering: {e}")
                    # Include items that caused errors to be safe
                    high_entropy_batch.append(item)

            # Use filtered batch if any items passed filter
            training_batch = high_entropy_batch if high_entropy_batch else batch
            self.logger.info(
                f"Filtered batch from {len(batch)} to {len(training_batch)} high-entropy items"
            )
        else:
            # Use full batch
            training_batch = batch

        # Train ARTManager on the selected batch
        result = self.art_manager.train_on_batch(training_batch)

        # Add BLT-specific information to result
        if self.blt_available:
            result["selective_training_used"] = selective_training
            result["entropy_threshold"] = self.entropy_threshold
            result["original_batch_size"] = len(batch)
            result["filtered_batch_size"] = len(training_batch)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ARTBLTBridge.

        Returns:
            A dict containing statistics
        """
        stats = self.stats.copy()

        # Add derived statistics
        if stats["total_inputs_processed"] > 0:
            stats["art_processing_ratio"] = (
                stats["art_processed_inputs"] / stats["total_inputs_processed"]
            )
        else:
            stats["art_processing_ratio"] = 0

        # Add ART stats
        if hasattr(self.art_manager, "controller") and hasattr(
            self.art_manager.controller, "status"
        ):
            art_stats = self.art_manager.status()
            stats["art_stats"] = art_stats

        return stats


if __name__ == "__main__":
    # Example usage
    logger = get_art_logger()

    # Check if BLT components are available and print status
    if HAS_BLT:
        logger.info("BLT components are available. Initializing ARTBLTBridge...")
    else:
        logger.info("BLT components are not available. ARTBLTBridge will use fallback mode.")

    # Create bridge
    bridge = ARTBLTBridge(entropy_threshold=0.4)

    # Example texts with varying entropy
    texts = [
        # High entropy (complex, diverse characters)
        "The quantum fluctuations in the early universe led to the formation of galaxies through gravitational collapse of dense regions.",
        # Medium entropy
        "Machine learning algorithms can recognize patterns in data that humans might miss.",
        # Low entropy (repetitive)
        "Simple simple simple simple simple simple simple simple simple.",
    ]

    # Process each text
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i + 1}:")
        logger.info(f"'{text}'")

        result = bridge.process_input(text)

        if result["entropy_score"] is not None:
            logger.info(f"Entropy: {result['entropy_score']:.4f}")

        if result["analysis_performed"] and "art_result" in result:
            art_result = result["art_result"]
            category = art_result.get("category", {})
            logger.info(f"ART Category: {category.get('id', 'unknown')}")
            logger.info(f"Novel: {art_result.get('is_novel_category', False)}")
        else:
            logger.info("ART analysis was not performed.")

        logger.info("-" * 50)

    # Show stats
    logger.info("\nBridge Statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):  # Skip nested dicts
            logger.info(f"{key}: {value}")
