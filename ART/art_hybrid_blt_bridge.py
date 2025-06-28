#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARTHybridBLTBridge - Enhanced bridge between the ART module and Hybrid BLT middleware.

This module provides the ARTHybridBLTBridge class that connects voxsigil.art.ARTManager
with the Hybrid BLT middleware for sophisticated entropy-based routing and pattern analysis.
"""

import importlib
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

from .art_logger import get_art_logger
from .art_manager import ARTManager

# Ensure Voxsigil_Library's parent directory is in sys.path
# This helps Python find Voxsigil_Library as a top-level package
voxsigil_library_path = Path(__file__).resolve().parents[3]  # Adjust depth as needed
if str(voxsigil_library_path) not in sys.path:
    sys.path.insert(0, str(voxsigil_library_path))

# Import ART components

# HOLO-1.5 Cognitive Mesh Integration
try:
    from ..agents.base import BaseAgent, CognitiveMeshRole, vanta_agent

    HOLO_AVAILABLE = True

    # Define VantaAgentCapability locally as it's not in a centralized location
    class VantaAgentCapability:
        ADAPTIVE_PROCESSING = "adaptive_processing"
        ENTROPY_ROUTING = "entropy_routing"
        HYBRID_COORDINATION = "hybrid_coordination"

except ImportError:
    # Fallback for non-HOLO environments
    def vanta_agent(
        role=None,
        name=None,
        cognitive_load=0,
        symbolic_depth=0,
        capabilities=None,
        **kwargs,
    ):
        def decorator(cls):
            cls._holo_role = role
            cls._vanta_name = name or cls.__name__
            cls._holo_cognitive_load = cognitive_load
            cls._holo_symbolic_depth = symbolic_depth
            cls._holo_capabilities = capabilities or []
            return cls

        return decorator

    class CognitiveMeshRole:
        PROCESSOR = "PROCESSOR"

    class VantaAgentCapability:
        ADAPTIVE_PROCESSING = "adaptive_processing"
        ENTROPY_ROUTING = "entropy_routing"
        HYBRID_COORDINATION = "hybrid_coordination"

    class BaseAgent:
        pass

    HOLO_AVAILABLE = False

# Global variables for BLT components (will be populated dynamically)
HAS_BLT = False
HybridMiddlewareConfig = None
EntropyRouter = None
HybridProcessor = None
ByteLatentTransformerEncoder = None
SigilPatchEncoder = None
BLTEnhancedRAG = None
patch_size = 8  # Default patch size for BLT, can be adjusted as needed


# Try to import BLT components dynamically
def load_blt_components():
    global HAS_BLT, HybridMiddlewareConfig, EntropyRouter, HybridProcessor
    global ByteLatentTransformerEncoder, SigilPatchEncoder, BLTEnhancedRAG

    logger = get_art_logger("BLTLoader")

    try:
        # Use relative import from VoxSigilRag
        hybrid_blt_module_name = "VoxSigilRag.hybrid_blt"
        hybrid_blt_spec = importlib.util.find_spec(hybrid_blt_module_name)

        if hybrid_blt_spec and hybrid_blt_spec.loader:
            hybrid_blt = importlib.util.module_from_spec(hybrid_blt_spec)
            hybrid_blt_spec.loader.exec_module(hybrid_blt)

            HybridMiddlewareConfig = getattr(hybrid_blt, "HybridMiddlewareConfig", None)
            EntropyRouter = getattr(hybrid_blt, "EntropyRouter", None)
            HybridProcessor = getattr(hybrid_blt, "HybridProcessor", None)

            if HybridMiddlewareConfig and EntropyRouter and HybridProcessor:
                logger.info(
                    f"Successfully imported core Hybrid BLT components from {hybrid_blt_module_name}."
                )
            else:
                logger.warning(
                    f"Core Hybrid BLT components (Config, Router, Processor) not found in {hybrid_blt_module_name} module."
                )

        else:
            logger.warning(
                f"Could not find hybrid_blt module: {hybrid_blt_module_name}"
            )  # Try to import BLT Encoder components from their specific modules
        try:
            # Use relative import from VoxSigilRag
            voxsigil_blt_module_name = "VoxSigilRag.voxsigil_blt"
            voxsigil_blt_spec = importlib.util.find_spec(voxsigil_blt_module_name)
            if voxsigil_blt_spec and voxsigil_blt_spec.loader:
                voxsigil_blt_module = importlib.util.module_from_spec(voxsigil_blt_spec)
                voxsigil_blt_spec.loader.exec_module(voxsigil_blt_module)
                ByteLatentTransformerEncoder = getattr(
                    voxsigil_blt_module, "ByteLatentTransformerEncoder", None
                )
            else:
                ByteLatentTransformerEncoder = (
                    None  # Ensure it's None if module not found
                )
                logger.warning(
                    f"Could not find module: {voxsigil_blt_module_name} for ByteLatentTransformerEncoder"
                )

            # Use relative import from VoxSigilRag
            sigil_patch_encoder_module_name = "VoxSigilRag.sigil_patch_encoder"
            sigil_patch_encoder_spec = importlib.util.find_spec(
                sigil_patch_encoder_module_name
            )
            if sigil_patch_encoder_spec and sigil_patch_encoder_spec.loader:
                sigil_patch_encoder_module = importlib.util.module_from_spec(
                    sigil_patch_encoder_spec
                )
                sigil_patch_encoder_spec.loader.exec_module(sigil_patch_encoder_module)
                SigilPatchEncoder = getattr(
                    sigil_patch_encoder_module, "SigilPatchEncoder", None
                )
            else:
                SigilPatchEncoder = None  # Ensure it's None if module not found
                logger.warning(
                    f"Could not find module: {sigil_patch_encoder_module_name} for SigilPatchEncoder"
                )

            if ByteLatentTransformerEncoder and SigilPatchEncoder:
                logger.info(
                    f"Successfully imported BLT Encoder components (ByteLatentTransformerEncoder from {voxsigil_blt_module_name}, SigilPatchEncoder from {sigil_patch_encoder_module_name})."
                )
            else:
                missing_encoders = []
                if not ByteLatentTransformerEncoder:
                    missing_encoders.append(
                        f"ByteLatentTransformerEncoder (from {voxsigil_blt_module_name})"
                    )
                if not SigilPatchEncoder:
                    missing_encoders.append(
                        f"SigilPatchEncoder (from {sigil_patch_encoder_module_name})"
                    )
                logger.warning(
                    f"Failed to import BLT Encoder components: {', '.join(missing_encoders)} missing."
                )
        except Exception as e_encoder:
            logger.error(f"Error importing encoder components: {e_encoder}")
            ByteLatentTransformerEncoder = None  # Ensure reset on error
            SigilPatchEncoder = None  # Ensure reset on error        # Try to import blt_rag module (for BLTEnhancedRAG)
        blt_rag_module_name = "VoxSigilRag.voxsigil_blt_rag"
        blt_rag_spec = importlib.util.find_spec(blt_rag_module_name)
        if blt_rag_spec and blt_rag_spec.loader:
            try:
                blt_rag = importlib.util.module_from_spec(blt_rag_spec)
                blt_rag_spec.loader.exec_module(blt_rag)
                BLTEnhancedRAG = getattr(blt_rag, "BLTEnhancedRAG", None)
                if BLTEnhancedRAG:
                    logger.info("Successfully imported BLTEnhancedRAG.")
                else:
                    logger.warning(
                        f"BLTEnhancedRAG not found in {blt_rag_module_name} module."
                    )
            except ImportError as e:
                logger.warning(
                    f"Could not import {blt_rag_module_name} due to missing dependencies: {e}"
                )
                BLTEnhancedRAG = None
            except Exception as e:
                logger.error(f"Error loading {blt_rag_module_name}: {e}")
                BLTEnhancedRAG = None
        else:
            logger.warning(f"Could not find {blt_rag_module_name} module.")
            BLTEnhancedRAG = None

        if (
            HybridMiddlewareConfig
            and EntropyRouter
            and HybridProcessor
            and ByteLatentTransformerEncoder
            and SigilPatchEncoder
            and BLTEnhancedRAG
        ):
            HAS_BLT = True
            logger.info("All BLT components loaded successfully.")
        else:
            logger.warning("One or more BLT components failed to load.")
            # Log which specific components are missing
            if not HybridMiddlewareConfig:
                logger.warning("Missing: HybridMiddlewareConfig")
            if not EntropyRouter:
                logger.warning("Missing: EntropyRouter")
            if not HybridProcessor:
                logger.warning("Missing: HybridProcessor")
            if not ByteLatentTransformerEncoder:
                logger.warning("Missing: ByteLatentTransformerEncoder")
            if not SigilPatchEncoder:
                logger.warning("Missing: SigilPatchEncoder")
            if not BLTEnhancedRAG:
                logger.warning("Missing: BLTEnhancedRAG")
            if not EntropyRouter:
                logger.warning("Missing: EntropyRouter")
            if not HybridProcessor:
                logger.warning("Missing: HybridProcessor")
            if not ByteLatentTransformerEncoder:
                logger.warning("Missing: ByteLatentTransformerEncoder")
            if not SigilPatchEncoder:
                logger.warning("Missing: SigilPatchEncoder")
            if not BLTEnhancedRAG:
                logger.warning("Missing: BLTEnhancedRAG")

    except ImportError as e:
        logger.error(f"Failed to import BLT components: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading BLT components: {e}")


def ensure_blt_loaded():
    """Lazy loader for BLT components - only load when actually needed."""
    global HAS_BLT
    if not HAS_BLT:
        logger.info("🔄 Loading BLT components on-demand...")
        HAS_BLT = load_blt_components()
        if HAS_BLT:
            logger.info("✅ BLT components loaded successfully!")
        else:
            logger.warning("⚠️ BLT components failed to load")
    return HAS_BLT


# Don't load BLT components at import time - make it lazy
HAS_BLT = False  # Will be set to True when actually needed


@vanta_agent(
    name="ARTHybridBLTBridge",
    subsystem="art_hybrid_processing",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    capabilities=[
        VantaAgentCapability.ADAPTIVE_PROCESSING,
        VantaAgentCapability.ENTROPY_ROUTING,
        VantaAgentCapability.HYBRID_COORDINATION,
        "entropy_based_routing",
        "hybrid_blt_processing",
        "adaptive_pattern_analysis",
    ],
    cognitive_load=3.5,
    symbolic_depth=4,
)
class ARTHybridBLTBridge(BaseAgent if HOLO_AVAILABLE else object):
    """
    Enhanced bridge between the ART module and Hybrid BLT middleware.

    This class uses the advanced Hybrid BLT middleware's entropy-based routing to
    selectively process data with ART, optimizing computational resources by focusing
    ART's pattern recognition on data with high information content or appropriate
    entropy characteristics.
      Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh:
    - Entropy-based intelligent routing with cognitive load monitoring
    - Hybrid BLT processing with symbolic depth tracking
    - Adaptive pattern analysis coordination across hybrid systems
    - Processing efficiency metrics for mesh optimization
    """

    def __init__(
        self,
        art_manager: Optional[ARTManager] = None,
        entropy_threshold: float = 0.4,
        blt_hybrid_weight: float = 0.7,
        patch_size: int = 8,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ARTHybridBLTBridge with HOLO-1.5 cognitive mesh integration.

        Args:
            art_manager: Optional ARTManager instance. If None, a new one will be created.
            entropy_threshold: Entropy threshold for triggering ART analysis.
                Higher values mean more selective analysis (only high entropy inputs).
            blt_hybrid_weight: Weight for BLT embeddings in hybrid mode (0-1).
            patch_size: Size of BLT patches for entropy calculation.
            logger_instance: Optional logger instance. If None, a new one will be created.
        """
        self.logger = logger_instance or get_art_logger("ARTHybridBLTBridge")
        self.entropy_threshold = entropy_threshold
        self.blt_hybrid_weight = blt_hybrid_weight

        # HOLO-1.5 Cognitive Mesh Initialization
        if HOLO_AVAILABLE:
            super().__init__()
            self.cognitive_metrics = {
                "processing_efficiency": 0.0,
                "routing_accuracy": 0.0,
                "hybrid_coordination": 1.0,
                "entropy_correlation": 0.0,
                "cognitive_load": 3.5,
                "symbolic_depth": 4,
            }
            # Schedule async initialization
            try:
                import asyncio

                asyncio.create_task(self.async_init())
            except RuntimeError:
                # If no event loop, defer initialization
                self._vanta_initialized = False
        else:
            self._vanta_initialized = False
            self.cognitive_metrics = {}

        # Initialize ARTManager
        self.art_manager = art_manager or ARTManager()

        # Initialize BLT components if available
        self.blt_available = HAS_BLT
        self.router = None
        self.hybrid_config = None
        self.hybrid_processor = None

        if (
            self.blt_available
            and HybridMiddlewareConfig
            and EntropyRouter
            and HybridProcessor
        ):
            try:
                # Create configuration
                self.hybrid_config = HybridMiddlewareConfig(
                    entropy_threshold=entropy_threshold,
                    blt_hybrid_weight=blt_hybrid_weight,
                    entropy_router_fallback="token_based",
                    cache_ttl_seconds=300,
                    log_level="INFO",
                )

                # Initialize router and processor
                self.router = EntropyRouter(self.hybrid_config)
                self.hybrid_processor = HybridProcessor(self.hybrid_config)

                self.logger.info(
                    f"ARTHybridBLTBridge initialized with: threshold={entropy_threshold}, "
                    f"weight={blt_hybrid_weight}"
                )
            except Exception as e:
                self.blt_available = False
                self.logger.error(f"Failed to initialize Hybrid BLT components: {e}")
                self.logger.warning(
                    "ARTHybridBLTBridge will operate without BLT entropy analysis"
                )
        else:
            self.logger.warning(
                "Hybrid BLT components not available. Bridge will use fallback mode."
            )

        # Statistics
        self.stats = {
            "total_inputs_processed": 0,
            "art_processed_inputs": 0,
            "skipped_inputs": 0,
            "avg_entropy": 0.0,
            "high_entropy_count": 0,
            "patch_based_route_count": 0,
            "token_based_route_count": 0,
        }

    def process_input(
        self,
        input_data: Union[str, dict[str, Any], list],
        context: Optional[dict[str, Any]] = None,
        force_analysis: bool = False,
    ) -> dict[str, Any]:
        """
        Process input data using the ART-Hybrid BLT bridge.

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
            "route_decision": None,
        }

        # Convert input_data to string for entropy calculation
        if isinstance(input_data, (dict, list)):
            # For structured data, convert to string representation
            input_str = str(input_data)
        else:
            input_str = str(input_data)

        # Calculate entropy and route using Hybrid BLT if available
        should_analyze = force_analysis
        entropy_scores = []
        route_decision = None
        patches = None

        if self.blt_available and self.router:
            try:
                # Use Hybrid BLT's EntropyRouter for routing decision
                route_decision, patches, entropy_scores = self.router.route(input_str)

                # Calculate average entropy
                avg_entropy = (
                    sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.0
                )

                # Determine if we should analyze based on entropy threshold
                should_analyze = avg_entropy >= self.entropy_threshold or force_analysis

                # Update result
                result["entropy_score"] = avg_entropy
                result["route_decision"] = route_decision

                # Update stats
                self.stats["avg_entropy"] = (
                    self.stats["avg_entropy"]
                    * (self.stats["total_inputs_processed"] - 1)
                    + avg_entropy
                ) / self.stats["total_inputs_processed"]

                if avg_entropy >= self.entropy_threshold:
                    self.stats["high_entropy_count"] += 1

                if route_decision == "patch_based":
                    self.stats["patch_based_route_count"] += 1
                else:  # token_based
                    self.stats["token_based_route_count"] += 1

                self.logger.debug(
                    f"Input entropy: {avg_entropy:.4f}, route: {route_decision}, "
                    f"threshold: {self.entropy_threshold}"
                )

                # If using BLT preprocessing, extract features for ART
                if (
                    should_analyze
                    and "preprocess_input" in context
                    and context["preprocess_input"]
                ):
                    if patches:
                        # Add patch analysis to context
                        context["blt_patches"] = patches
                        context["blt_entropy"] = avg_entropy
                        context["blt_route"] = route_decision
            except Exception as e:
                self.logger.warning(f"Error in Hybrid BLT entropy calculation: {e}")
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
                # Add routing info to context if available
                if route_decision:
                    context.setdefault("blt_info", {})
                    context["blt_info"]["route"] = route_decision
                    context["blt_info"]["entropy_scores"] = entropy_scores

                art_result = self.art_manager.analyze_input(
                    input_data, analysis_type=None
                )
                result["analysis_performed"] = True
                result["art_result"] = art_result

                entropy_str = f"entropy {result.get('entropy_score', 'N/A')}"
                route_str = f"route '{route_decision}'" if route_decision else ""
                self.logger.info(
                    f"ART analysis performed on input with {entropy_str} {route_str}"
                )

                # If this detected a novel category and BLT is available,
                # we could feed this back to BLT for future routing decisions
                if art_result.get("is_novel_category") and self.blt_available:
                    category = art_result.get("category", {})
                    category_id = category.get("id", "unknown")
                    result["novel_category_detected"] = {
                        "id": category_id,
                        "entropy": result.get("entropy_score"),
                        "route": route_decision,
                    }
            except Exception as e:
                self.logger.error(f"Error in ART analysis: {e}")
                result["error"] = str(e)
        else:
            self.logger.info(
                f"Skipped ART analysis due to low entropy: {result.get('entropy_score', 'N/A')}"
            )

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        return result

    def train_on_batch(
        self,
        batch: list[Union[str, dict[str, Any], tuple]],
        selective_training: bool = True,
    ) -> dict[str, Any]:
        """
        Train ARTManager on a batch of data, using Hybrid BLT for selective training.

        Args:
            batch: A list of inputs to train on
            selective_training: If True, only train on high-entropy inputs

        Returns:
            A dict containing training results and statistics
        """
        if not batch:
            return {"status": "error", "message": "Empty batch provided"}

        # Filter batch if selective training is enabled
        if selective_training and self.blt_available and self.router:
            high_entropy_batch = []
            for item in batch:
                # Extract text for entropy calculation
                if isinstance(item, tuple) and len(item) >= 1:
                    # For query-response pairs, calculate entropy on the query
                    text = str(item[0])
                else:
                    text = str(item)

                # Check if entropy exceeds threshold using router
                try:
                    _, _, entropy_scores = self.router.route(text)
                    avg_entropy = (
                        sum(entropy_scores) / len(entropy_scores)
                        if entropy_scores
                        else 0.0
                    )

                    if avg_entropy >= self.entropy_threshold:
                        high_entropy_batch.append(item)
                except Exception as e:
                    self.logger.warning(
                        f"Error calculating entropy during batch filtering: {e}"
                    )
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
            result["blt_hybrid_weight"] = self.blt_hybrid_weight
            result["original_batch_size"] = len(batch)
            result["filtered_batch_size"] = len(training_batch)

        return result

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the ARTHybridBLTBridge.

        Returns:
            A dict containing statistics
        """
        stats = self.stats.copy()

        # Add derived statistics
        if stats["total_inputs_processed"] > 0:
            stats["art_processing_ratio"] = (
                stats["art_processed_inputs"] / stats["total_inputs_processed"]
            )
            if (
                "patch_based_route_count" in stats
                and "token_based_route_count" in stats
            ):
                total_routed = (
                    stats["patch_based_route_count"] + stats["token_based_route_count"]
                )
                if total_routed > 0:
                    stats["patch_based_ratio"] = (
                        stats["patch_based_route_count"] / total_routed
                    )
        else:
            stats["art_processing_ratio"] = 0
            stats["patch_based_ratio"] = 0
        # Add ART stats
        if (
            hasattr(self.art_manager, "art_controller")
            and self.art_manager.art_controller
        ):
            art_stats = self.art_manager.status()
            stats["art_stats"] = art_stats

        return stats

    # HOLO-1.5 Async Initialization Methods
    async def async_init(self):
        """Async initialization for HOLO-1.5 cognitive mesh integration"""
        if not HOLO_AVAILABLE:
            return

        try:
            # Initialize cognitive mesh connection
            await self.initialize_vanta_core()

            # Register processing capabilities
            await self.register_processing_capabilities()

            # Start cognitive monitoring
            await self.start_cognitive_monitoring()

            self._vanta_initialized = True
            self.logger.info(
                "ARTHybridBLTBridge HOLO-1.5 cognitive mesh initialization complete"
            )

        except Exception as e:
            self.logger.warning(f"HOLO-1.5 initialization failed: {e}")
            self._vanta_initialized = False

    async def initialize_vanta_core(self):
        """Initialize VantaCore connection for cognitive mesh"""
        if hasattr(super(), "initialize_vanta_core"):
            await super().initialize_vanta_core()

    async def register_processing_capabilities(self):
        """Register ARTHybridBLTBridge processing capabilities with cognitive mesh"""
        capabilities = {
            "adaptive_processing": {
                "entropy_routing": self.blt_available,
                "hybrid_coordination": True,
                "blt_components_loaded": self.blt_available,
                "fallback_mode": not self.blt_available,
            },
            "entropy_routing": {
                "threshold": self.entropy_threshold,
                "hybrid_weight": self.blt_hybrid_weight,
                "router_available": self.router is not None,
                "selective_analysis": True,
            },
            "hybrid_coordination": {
                "art_manager_integration": self.art_manager is not None,
                "cognitive_load_balancing": True,
                "async_operations": True,
                "batch_processing": True,
            },
        }

        if hasattr(self, "vanta_core") and self.vanta_core:
            await self.vanta_core.register_capabilities(
                "art_hybrid_blt_bridge", capabilities
            )

    async def start_cognitive_monitoring(self):
        """Start cognitive load monitoring for hybrid processing"""
        # Begin cognitive load monitoring
        if hasattr(self, "vanta_core") and self.vanta_core:
            monitoring_config = {
                "processing_efficiency_target": 0.85,
                "routing_accuracy_target": 0.90,
                "hybrid_coordination_target": 0.95,
                "entropy_correlation_target": 0.80,
            }
            await self.vanta_core.start_monitoring(
                "art_hybrid_blt_bridge", monitoring_config
            )

    def _enhanced_process_input(
        self,
        input_data: Union[str, dict[str, Any], list],
        context: Optional[dict[str, Any]] = None,
        force_analysis: bool = False,
    ) -> dict[str, Any]:
        """Enhanced process_input with cognitive metrics tracking"""
        # Track cognitive load start
        cognitive_start_time = time.time()

        # Call original process_input
        result = self.process_input(input_data, context, force_analysis)

        # Update cognitive metrics
        cognitive_processing_time = time.time() - cognitive_start_time

        if HOLO_AVAILABLE and self._vanta_initialized:
            # Update processing efficiency metric
            entropy_score = result.get("entropy_score", 0.0)
            analysis_performed = result.get("analysis_performed", False)

            # Calculate processing efficiency (inverse of processing time, normalized)
            efficiency = min(1.0, 1.0 / max(cognitive_processing_time, 0.001))
            self.cognitive_metrics["processing_efficiency"] = (
                self.cognitive_metrics["processing_efficiency"] * 0.9 + efficiency * 0.1
            )

            # Calculate routing accuracy (how well entropy threshold worked)
            if entropy_score is not None:
                expected_analysis = entropy_score >= self.entropy_threshold
                routing_accuracy = (
                    1.0 if (expected_analysis == analysis_performed) else 0.0
                )
                self.cognitive_metrics["routing_accuracy"] = (
                    self.cognitive_metrics["routing_accuracy"] * 0.9
                    + routing_accuracy * 0.1
                )

                # Track entropy correlation
                self.cognitive_metrics["entropy_correlation"] = (
                    self.cognitive_metrics["entropy_correlation"] * 0.9
                    + entropy_score * 0.1
                )

            # Update hybrid coordination (measure of BLT-ART integration success)
            coordination_score = (
                1.0
                if result.get("analysis_performed") and not result.get("error")
                else 0.5
            )
            self.cognitive_metrics["hybrid_coordination"] = (
                self.cognitive_metrics["hybrid_coordination"] * 0.9
                + coordination_score * 0.1
            )

            # Generate cognitive trace
            result["cognitive_trace"] = self._generate_processing_trace(
                input_data, result, cognitive_processing_time
            )

        return result

    def _generate_processing_trace(self, input_data, result, processing_time):
        """Generate cognitive trace for mesh learning"""
        if not HOLO_AVAILABLE:
            return None

        trace = {
            "timestamp": time.time(),
            "processing_time": processing_time,
            "cognitive_metrics": self.cognitive_metrics.copy(),
            "entropy_score": result.get("entropy_score"),
            "route_decision": result.get("route_decision"),
            "analysis_performed": result.get("analysis_performed"),
            "hybrid_coordination_success": not result.get("error", False),
            "symbolic_depth": self.cognitive_metrics.get("symbolic_depth", 4),
            "mesh_role": "PROCESSOR",
        }

        return trace

    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on processing metrics"""
        if not HOLO_AVAILABLE:
            return 0.0

        base_load = 3.5  # Base cognitive load for PROCESSOR role

        # Adjust based on BLT availability and complexity
        if self.blt_available:
            load_adjustment = 0.0
            # Higher load if processing many high-entropy inputs
            if self.stats.get("high_entropy_count", 0) > 10:
                load_adjustment += 0.5
            # Lower load if routing is efficient
            if self.cognitive_metrics.get("routing_accuracy", 0.0) > 0.9:
                load_adjustment -= 0.3
        else:
            # Higher load when operating in fallback mode
            load_adjustment = 0.8

        return max(1.0, min(5.0, base_load + load_adjustment))

    def _calculate_symbolic_depth(self) -> int:
        """Calculate current symbolic processing depth"""
        if not HOLO_AVAILABLE:
            return 1

        base_depth = 4  # Base symbolic depth for hybrid processing

        # Adjust based on entropy correlation and coordination
        entropy_correlation = self.cognitive_metrics.get("entropy_correlation", 0.0)
        hybrid_coordination = self.cognitive_metrics.get("hybrid_coordination", 1.0)

        depth_adjustment = 0
        if entropy_correlation > 0.7:
            depth_adjustment += 1
        if hybrid_coordination > 0.9:
            depth_adjustment += 1
        if self.blt_available and self.router:
            depth_adjustment += 1

        return max(1, min(6, base_depth + depth_adjustment))

    def get_cognitive_status(self) -> dict:
        """Get current cognitive status for HOLO-1.5 mesh"""
        if not HOLO_AVAILABLE:
            return {}

        return {
            "cognitive_load": self._calculate_cognitive_load(),
            "symbolic_depth": self._calculate_symbolic_depth(),
            "processing_efficiency": self.cognitive_metrics.get(
                "processing_efficiency", 0.0
            ),
            "routing_accuracy": self.cognitive_metrics.get("routing_accuracy", 0.0),
            "hybrid_coordination": self.cognitive_metrics.get(
                "hybrid_coordination", 1.0
            ),
            "entropy_correlation": self.cognitive_metrics.get(
                "entropy_correlation", 0.0
            ),
            "blt_availability": self.blt_available,
            "total_processed": self.stats.get("total_inputs_processed", 0),
            "mesh_role": "PROCESSOR",
            "vanta_initialized": self._vanta_initialized,
        }


if __name__ == "__main__":
    # Example usage
    logger = get_art_logger()

    # Check if BLT components are available and print status
    if HAS_BLT:
        logger.info(
            "Hybrid BLT components are available. Initializing ARTHybridBLTBridge..."
        )
    else:
        logger.info(
            "Hybrid BLT components are not available. ARTHybridBLTBridge will use fallback mode."
        )

    # Create bridge
    bridge = ARTHybridBLTBridge(entropy_threshold=0.4, blt_hybrid_weight=0.7)

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

        if result.get("route_decision"):
            logger.info(f"Route: {result['route_decision']}")

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
