#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARTBLTBridge - Enhanced bridge between the ART module and BLT middleware.

This module provides the ARTBLTBridge class that connects voxsigil.art.ARTManager
with the BLT middleware for entropy-based selective pattern analysis.

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh pattern for advanced
BLT processing, entropy-based routing, and VantaCore integration.

Core functions:
1. Connects ARTManager to BLT middleware for pattern analysis
2. Provides entropy-based selective processing
3. Implements sigil patch encoding and decoding
4. Enables cognitive load balancing for BLT operations
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .art_logger import get_art_logger

# Import ART components
from .art_manager import ARTManager

# HOLO-1.5 Cognitive Mesh Integration
try:
    from ..agents.base import BaseAgent, CognitiveMeshRole, vanta_agent

    HOLO_AVAILABLE = True

    # Define VantaAgentCapability locally
    class VantaAgentCapability:
        ADAPTIVE_PROCESSING = "adaptive_processing"
        ENTROPY_ROUTING = "entropy_routing"
        SIGIL_ENCODING = "sigil_encoding"
        PATTERN_BRIDGING = "pattern_bridging"

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
        SIGIL_ENCODING = "sigil_encoding"
        PATTERN_BRIDGING = "pattern_bridging"

    class BaseAgent:
        pass

    HOLO_AVAILABLE = False

# Try to import BLT components - only importing what we need
try:
    # SigilPatchEncoder now lives in the BLT package __init__ to avoid
    # circular import issues. Import directly from BLT so the bridge can
    # locate the class whether BLT is installed as a package or in-tree.
    from BLT import SigilPatchEncoder

    # Set flag that BLT is available
    HAS_BLT = True
except ImportError as e:
    print(f"Failed to import BLT components: {e}")
    HAS_BLT = False


@vanta_agent(
    name="ARTBLTBridge",
    subsystem="art_blt_processing",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    capabilities=[
        VantaAgentCapability.ADAPTIVE_PROCESSING,
        VantaAgentCapability.ENTROPY_ROUTING,
        VantaAgentCapability.SIGIL_ENCODING,
        VantaAgentCapability.PATTERN_BRIDGING,
        "entropy_based_analysis",
        "sigil_patch_processing",
        "adaptive_pattern_bridging",
    ],
    cognitive_load=3.2,
    symbolic_depth=4,
)
class ARTBLTBridge(BaseAgent if HOLO_AVAILABLE else object):
    """
    Enhanced bridge between the ART module and BLT middleware.

    This class facilitates communication between the ART module's pattern recognition
    capabilities and the BLT middleware's symbolic reasoning engine with advanced
    entropy-based routing and cognitive mesh integration.

    Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh:
    - Entropy-based intelligent routing with cognitive load monitoring
    - Sigil patch processing with symbolic depth tracking
    - Adaptive pattern bridging coordination
    - Processing efficiency metrics for mesh optimization

    Features:
    - Entropy-threshold based selective ART analysis
    - Sigil patch encoding/decoding capabilities
    - Cognitive load balancing for BLT operations
    - Context-aware pattern bridging
    """

    def __init__(
        self,
        art_manager: Optional[ARTManager] = None,
        sigil_encoder: Optional[SigilPatchEncoder] = None,
        entropy_threshold: float = 0.4,
        blt_patch_size: int = 8,
        config: Optional[Dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ARTBLTBridge.

        Args:
            art_manager: An instance of ARTManager for pattern recognition
            sigil_encoder: An instance of SigilPatchEncoder for encoding patterns
            entropy_threshold: Entropy threshold for triggering ART analysis
            blt_patch_size: Size of BLT patches for entropy calculation
            config: Configuration settings for the bridge
            logger_instance: Optional logger instance
        """
        self.logger = logger_instance or get_art_logger("ARTBLTBridge")
        self.entropy_threshold = entropy_threshold
        self.blt_patch_size = blt_patch_size
        self.config = config or {}

        # HOLO-1.5 Cognitive Mesh Initialization
        if HOLO_AVAILABLE:
            super().__init__()
            self.cognitive_metrics = {
                "processing_efficiency": 0.0,
                "entropy_accuracy": 0.0,
                "encoding_success_rate": 0.0,
                "pattern_bridging_quality": 0.0,
                "blt_integration_health": 1.0,
                "symbolic_depth": 4.0,
            }
            self._vanta_initialized = False
            self.monitoring_task = None
        else:
            self._vanta_initialized = False
            self.cognitive_metrics = {}

        # Initialize ARTManager
        self.art_manager = art_manager or ARTManager()
        # Initialize BLT components if available
        self.blt_available = HAS_BLT
        if HAS_BLT:
            try:
                self.sigil_encoder = sigil_encoder or SigilPatchEncoder(
                    entropy_threshold=entropy_threshold,
                    patch_size=blt_patch_size,
                )
                self.logger.info("ARTBLTBridge initialized with BLT components")
            except ImportError as e:
                self.blt_available = False
                self.logger.error(f"Failed to import BLT components: {e}")
                self.logger.warning(
                    "ARTBLTBridge will operate without BLT entropy analysis"
                )
            except TypeError as e:
                self.blt_available = False
                self.logger.error(f"Invalid BLT component configuration: {e}")
                self.logger.warning(
                    "ARTBLTBridge will operate without BLT entropy analysis"
                )
            except Exception as e:
                self.blt_available = False
                self.logger.error(
                    f"Unexpected error initializing BLT components: {e}", exc_info=True
                )
                self.logger.warning(
                    "ARTBLTBridge will operate without BLT entropy analysis"
                )
        else:
            self.sigil_encoder = None
            self.logger.warning(
                "BLT components not available. Some functionality will be limited."
            )

        # Statistics
        self.stats = {
            "total_inputs_processed": 0,
            "art_processed_inputs": 0,
            "skipped_inputs": 0,
            "avg_entropy": 0.0,
            "high_entropy_count": 0,
            "encoding_operations": 0,
            "decoding_operations": 0,
            "pattern_bridges_created": 0,
        }

    async def async_init(self):
        """Initialize HOLO-1.5 cognitive mesh integration."""
        if not HOLO_AVAILABLE:
            self.logger.info(
                "HOLO-1.5 not available, skipping cognitive mesh initialization"
            )
            return

        try:
            # Initialize VantaCore connection
            await self.register_cognitive_capabilities()
            await self.start_cognitive_monitoring()
            self._vanta_initialized = True
            self.logger.info(
                "ARTBLTBridge HOLO-1.5 cognitive mesh initialization complete"
            )

        except AttributeError as e:
            self.logger.warning(
                f"HOLO-1.5 initialization failed - missing attribute: {e}"
            )
            self._vanta_initialized = False
        except TypeError as e:
            self.logger.warning(f"HOLO-1.5 initialization failed - type error: {e}")
            self._vanta_initialized = False
        except Exception as e:
            self.logger.error(
                f"HOLO-1.5 initialization failed - unexpected error: {e}", exc_info=True
            )
            self._vanta_initialized = False

    async def register_cognitive_capabilities(self):
        """Register cognitive capabilities with VantaCore."""
        if not HOLO_AVAILABLE or not hasattr(self, "vanta_core"):
            return

        capabilities = {
            "entropy_based_analysis": {
                "entropy_threshold": self.entropy_threshold,
                "blt_availability": self.blt_available,
                "adaptive_processing": True,
                "selective_analysis": True,
            },
            "sigil_patch_processing": {
                "encoding_capabilities": self.sigil_encoder is not None,
                "patch_size": self.blt_patch_size,
                "pattern_bridging": True,
                "symbolic_representation": True,
            },
            "adaptive_pattern_bridging": {
                "art_manager_integration": self.art_manager is not None,
                "cognitive_load_balancing": True,
                "context_awareness": True,
                "async_operations": True,
            },
        }

        if hasattr(self, "vanta_core") and self.vanta_core:
            await self.vanta_core.register_capabilities("art_blt_bridge", capabilities)

    async def start_cognitive_monitoring(self):
        """Start background cognitive monitoring."""
        if not HOLO_AVAILABLE:
            return

        monitoring_config = {
            "metrics": [
                "processing_efficiency",
                "entropy_accuracy",
                "encoding_success_rate",
                "pattern_bridging_quality",
            ],
            "adaptive_thresholds": True,
            "learning_rate": 0.1,
        }

        if hasattr(self, "vanta_core") and self.vanta_core:
            await self.vanta_core.start_monitoring("art_blt_bridge", monitoring_config)

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
            "encoding_performed": False,
            "entropy_score": None,
            "processing_time": 0,
        }

        # Convert input_data to string for entropy calculation
        if isinstance(input_data, (dict, list)):
            input_str = str(input_data)
        else:
            input_str = str(input_data)

        # Calculate entropy using BLT if available
        should_analyze = force_analysis
        entropy_score = 0.0

        if self.blt_available and self.sigil_encoder:
            try:
                # Calculate entropy using BLT's sigil encoder
                entropy_score = self.sigil_encoder.compute_average_entropy(input_str)
                should_analyze = (
                    entropy_score >= self.entropy_threshold or force_analysis
                )

                result["entropy_score"] = entropy_score

                # Update stats
                self.stats["avg_entropy"] = (
                    self.stats["avg_entropy"]
                    * (self.stats["total_inputs_processed"] - 1)
                    + entropy_score
                ) / self.stats["total_inputs_processed"]

                if entropy_score >= self.entropy_threshold:
                    self.stats["high_entropy_count"] += 1

                self.logger.debug(
                    f"Input entropy: {entropy_score:.4f}, threshold: {self.entropy_threshold}"
                )

                # If using BLT preprocessing, create sigil patches
                if (
                    should_analyze
                    and "preprocess_input" in context
                    and context["preprocess_input"]
                ):
                    patches = self.sigil_encoder.segment_into_patches(input_str)
                    if patches:
                        # Add patch analysis to context
                        context["blt_patches"] = patches
                        context["blt_entropy"] = entropy_score
                        result["encoding_performed"] = True
                        self.stats["encoding_operations"] += 1

            except ValueError as e:
                self.logger.warning(
                    f"Invalid data format for BLT entropy calculation: {e}"
                )
                # Fallback to always analyzing if BLT fails
                should_analyze = True
            except AttributeError as e:
                self.logger.warning(
                    f"Missing BLT component method for entropy calculation: {e}"
                )
                should_analyze = True
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in BLT entropy calculation: {e}", exc_info=True
                )
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
                # create a pattern bridge for future processing
                if art_result.get("is_novel_category") and self.blt_available:
                    category = art_result.get("category", {})
                    category_id = category.get("id", "unknown")
                    result["novel_category_detected"] = {
                        "id": category_id,
                        "entropy": entropy_score,
                    }
                    self.stats["pattern_bridges_created"] += 1

            except AttributeError as e:
                self.logger.error(f"Missing ART manager method: {e}")
                result["error"] = f"missing_method: {str(e)}"
            except ValueError as e:
                self.logger.error(f"Invalid input data for ART analysis: {e}")
                result["error"] = f"invalid_data: {str(e)}"
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in ART analysis: {e}", exc_info=True
                )
                result["error"] = f"unexpected_error: {str(e)}"
        else:
            self.logger.info(
                f"Skipped ART analysis due to low entropy: {entropy_score:.4f}"
            )

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        # Update cognitive metrics if HOLO available
        if HOLO_AVAILABLE and self._vanta_initialized:
            self._update_cognitive_metrics(result, start_time)

        return result

    async def process_input_async(
        self,
        input_data: Union[str, Dict[str, Any], List],
        context: Optional[Dict[str, Any]] = None,
        force_analysis: bool = False,
    ) -> Dict[str, Any]:
        """Async version of process_input with cognitive monitoring."""
        cognitive_start_time = time.time()

        # Run sync processing in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, self.process_input, input_data, context, force_analysis
        )

        # Update cognitive metrics
        cognitive_processing_time = time.time() - cognitive_start_time

        if HOLO_AVAILABLE and self._vanta_initialized:
            # Calculate processing efficiency (inverse of processing time, normalized)
            entropy_score = result.get("entropy_score", 0.0)
            analysis_performed = result.get("analysis_performed", False)

            efficiency = min(1.0, 1.0 / max(cognitive_processing_time, 0.001))
            self.cognitive_metrics["processing_efficiency"] = (
                self.cognitive_metrics["processing_efficiency"] * 0.9 + efficiency * 0.1
            )

            # Calculate entropy accuracy (how well entropy threshold worked)
            if entropy_score is not None:
                expected_analysis = entropy_score >= self.entropy_threshold
                entropy_accuracy = (
                    1.0 if (expected_analysis == analysis_performed) else 0.0
                )
                self.cognitive_metrics["entropy_accuracy"] = (
                    self.cognitive_metrics["entropy_accuracy"] * 0.9
                    + entropy_accuracy * 0.1
                )

            # Update encoding success rate
            encoding_performed = result.get("encoding_performed", False)
            encoding_success = (
                1.0 if encoding_performed and not result.get("error") else 0.5
            )
            self.cognitive_metrics["encoding_success_rate"] = (
                self.cognitive_metrics["encoding_success_rate"] * 0.9
                + encoding_success * 0.1
            )

            # Update pattern bridging quality
            bridge_quality = 1.0 if result.get("novel_category_detected") else 0.8
            self.cognitive_metrics["pattern_bridging_quality"] = (
                self.cognitive_metrics["pattern_bridging_quality"] * 0.9
                + bridge_quality * 0.1
            )

            # Generate cognitive trace
            result["cognitive_trace"] = self._generate_processing_trace(
                input_data, result, cognitive_processing_time
            )

        return result

    def encode_pattern(
        self,
        pattern_data: Union[str, Dict[str, Any]],
        encoding_type: str = "sigil_patch",
    ) -> Dict[str, Any]:
        """
        Encode pattern data using BLT sigil encoding.

        Args:
            pattern_data: The pattern data to encode
            encoding_type: Type of encoding to perform

        Returns:
            Dict containing encoded data and metadata
        """
        result = {
            "encoded": False,
            "encoding_type": encoding_type,
            "patches": None,
            "metadata": {},
        }

        if not self.blt_available or not self.sigil_encoder:
            result["error"] = "BLT components not available for encoding"
            return result

        try:
            # Convert pattern to string if needed
            if isinstance(pattern_data, dict):
                pattern_str = str(pattern_data)
            else:
                pattern_str = str(pattern_data)

            # Create sigil patches
            patches = self.sigil_encoder.segment_into_patches(pattern_str)

            if patches:
                result["encoded"] = True
                result["patches"] = patches
                result["metadata"] = {
                    "patch_count": len(patches),
                    "total_length": len(pattern_str),
                    "entropy": self.sigil_encoder.compute_average_entropy(pattern_str),
                }
                self.stats["encoding_operations"] += 1
                # Update cognitive metrics
                if HOLO_AVAILABLE and self._vanta_initialized:
                    success_rate = self.cognitive_metrics["encoding_success_rate"]
                    self.cognitive_metrics["encoding_success_rate"] = (
                        success_rate * 0.9 + 1.0 * 0.1
                    )

        except AttributeError as e:
            result["error"] = f"Encoding failed - missing encoder method: {e}"
            self.logger.error(f"Pattern encoding error - missing method: {e}")
        except ValueError as e:
            result["error"] = f"Encoding failed - invalid pattern data: {e}"
            self.logger.error(f"Pattern encoding error - invalid data: {e}")
        except Exception as e:
            result["error"] = f"Encoding failed - unexpected error: {e}"
            self.logger.error(
                f"Pattern encoding error - unexpected: {e}", exc_info=True
            )

        return result

    def decode_pattern(
        self, encoded_data: List[Any], decoding_type: str = "sigil_patch"
    ) -> Dict[str, Any]:
        """
        Decode BLT-encoded pattern data.

        Args:
            encoded_data: The encoded patch data
            decoding_type: Type of decoding to perform

        Returns:
            Dict containing decoded data and metadata
        """
        result = {
            "decoded": False,
            "decoding_type": decoding_type,
            "pattern": None,
            "metadata": {},
        }

        if not self.blt_available or not self.sigil_encoder:
            result["error"] = "BLT components not available for decoding"
            return result

        try:
            # Reconstruct pattern from patches
            if encoded_data and hasattr(self.sigil_encoder, "reconstruct_from_patches"):
                reconstructed = self.sigil_encoder.reconstruct_from_patches(
                    encoded_data
                )
                result["decoded"] = True
                result["pattern"] = reconstructed
                result["metadata"] = {
                    "patch_count": len(encoded_data),
                    "reconstructed_length": len(str(reconstructed))
                    if reconstructed
                    else 0,
                }
                self.stats["decoding_operations"] += 1
            else:
                result["error"] = "Invalid encoded data or decoder not available"

        except AttributeError as e:
            result["error"] = f"Decoding failed - missing decoder method: {e}"
            self.logger.error(f"Pattern decoding error - missing method: {e}")
        except ValueError as e:
            result["error"] = f"Decoding failed - invalid encoded data: {e}"
            self.logger.error(f"Pattern decoding error - invalid data: {e}")
        except Exception as e:
            result["error"] = f"Decoding failed - unexpected error: {e}"
            self.logger.error(
                f"Pattern decoding error - unexpected: {e}", exc_info=True
            )

        return result

    def train_on_batch(
        self,
        batch: List[Union[str, Dict[str, Any], Tuple]],
        selective_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Train ART on a batch with optional entropy-based filtering.

        Args:
            batch: List of training samples
            selective_training: Whether to filter by entropy threshold

        Returns:
            A dict containing training results and statistics
        """
        if not batch:
            return {"status": "error", "message": "Empty batch provided"}

        # Filter batch if selective training is enabled
        if selective_training and self.blt_available and self.sigil_encoder:
            high_entropy_batch = []
            for item in batch:
                # Extract text for entropy calculation
                if isinstance(item, tuple) and len(item) >= 1:
                    text = str(item[0])
                elif isinstance(item, dict):
                    text = str(item)
                else:
                    text = str(item)

                # Check if entropy exceeds threshold
                try:
                    entropy = self.sigil_encoder.compute_average_entropy(text)
                    if entropy >= self.entropy_threshold:
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
            result["original_batch_size"] = len(batch)
            result["filtered_batch_size"] = len(training_batch)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics."""
        stats = self.stats.copy()

        # Calculate derived statistics
        if self.stats["total_inputs_processed"] > 0:
            stats["art_processing_ratio"] = (
                self.stats["art_processed_inputs"]
                / self.stats["total_inputs_processed"]
            )
            stats["encoding_ratio"] = (
                self.stats["encoding_operations"] / self.stats["total_inputs_processed"]
            )
        else:
            stats["art_processing_ratio"] = 0
            stats["encoding_ratio"] = 0

        # Add ART stats
        if (
            hasattr(self.art_manager, "art_controller")
            and self.art_manager.art_controller
        ):
            art_stats = self.art_manager.status()
            stats["art_stats"] = art_stats

        # Add cognitive metrics if available
        if HOLO_AVAILABLE and self._vanta_initialized:
            stats["cognitive_metrics"] = self.cognitive_metrics.copy()

        return stats

    def _update_cognitive_metrics(self, result: Dict[str, Any], start_time: float):
        """Update cognitive metrics based on processing results."""
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return

        processing_time = time.time() - start_time
        entropy_score = result.get("entropy_score", 0.0)

        # Update processing efficiency based on timing
        # Faster processing (< 1 second) is more efficient
        efficiency = max(0.0, min(1.0, 2.0 - processing_time))
        self.cognitive_metrics["processing_efficiency"] = (
            self.cognitive_metrics["processing_efficiency"] * 0.9 + efficiency * 0.1
        )

        # Update entropy accuracy if we have a valid entropy score
        if entropy_score > 0:
            # Higher entropy scores indicate better pattern recognition
            entropy_accuracy = min(1.0, entropy_score / 10.0)  # Normalize to 0-1 range
            self.cognitive_metrics["entropy_accuracy"] = (
                self.cognitive_metrics["entropy_accuracy"] * 0.9
                + entropy_accuracy * 0.1
            )

        # Update BLT integration health
        blt_health = 1.0 if self.blt_available and not result.get("error") else 0.5
        self.cognitive_metrics["blt_integration_health"] = (
            self.cognitive_metrics["blt_integration_health"] * 0.95 + blt_health * 0.05
        )

    def _generate_processing_trace(
        self, input_data: Any, result: Dict[str, Any], processing_time: float
    ) -> Dict[str, Any]:
        """Generate cognitive trace for mesh learning."""
        return {
            "timestamp": time.time(),
            "processing_time": processing_time,
            "entropy_score": result.get("entropy_score"),
            "analysis_performed": result.get("analysis_performed"),
            "encoding_performed": result.get("encoding_performed"),
            "pattern_bridging_success": not result.get("error", False),
            "symbolic_depth": self.cognitive_metrics.get("symbolic_depth", 4),
            "mesh_role": "PROCESSOR",
        }

    def calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on processing metrics."""
        base_load = 3.2  # Base cognitive load

        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return base_load

        # Adjust based on processing efficiency and BLT health
        processing_efficiency = self.cognitive_metrics.get("processing_efficiency", 0.5)
        blt_health = self.cognitive_metrics.get("blt_integration_health", 1.0)

        load_adjustment = 0

        # Higher load if processing is inefficient
        if processing_efficiency < 0.5:
            load_adjustment += 0.5

        # Lower load if BLT integration is healthy
        if blt_health > 0.8:
            load_adjustment -= 0.2

        # Increase load based on operation counts
        if self.stats.get("encoding_operations", 0) > 50:
            load_adjustment += 0.3

        return max(1, min(6, base_load + load_adjustment))

    def calculate_symbolic_depth(self) -> int:
        """Calculate current symbolic depth based on pattern complexity."""
        base_depth = 4  # Base symbolic depth

        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return base_depth

        # Adjust based on encoding success and pattern bridging quality
        encoding_success = self.cognitive_metrics.get("encoding_success_rate", 0.0)
        bridging_quality = self.cognitive_metrics.get("pattern_bridging_quality", 0.0)

        depth_adjustment = 0
        if encoding_success > 0.8:
            depth_adjustment += 1
        if bridging_quality > 0.9:
            depth_adjustment += 1
        if self.blt_available and self.stats.get("pattern_bridges_created", 0) > 0:
            depth_adjustment += 1

        return max(1, min(6, base_depth + depth_adjustment))

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive status for mesh coordination."""
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return {
                "cognitive_load": 3.2,
                "symbolic_depth": 4,
                "mesh_role": "PROCESSOR",
                "vanta_initialized": False,
            }

        return {
            "cognitive_load": self.calculate_cognitive_load(),
            "symbolic_depth": self.calculate_symbolic_depth(),
            "processing_efficiency": self.cognitive_metrics.get(
                "processing_efficiency", 0.0
            ),
            "entropy_accuracy": self.cognitive_metrics.get("entropy_accuracy", 0.0),
            "encoding_success_rate": self.cognitive_metrics.get(
                "encoding_success_rate", 0.0
            ),
            "pattern_bridging_quality": self.cognitive_metrics.get(
                "pattern_bridging_quality", 0.0
            ),
            "blt_integration_health": self.cognitive_metrics.get(
                "blt_integration_health", 1.0
            ),
            "blt_availability": self.blt_available,
            "total_processed": self.stats.get("total_inputs_processed", 0),
            "mesh_role": "PROCESSOR",
            "vanta_initialized": self._vanta_initialized,
        }

    def shutdown(self):
        """Clean shutdown of the bridge and monitoring tasks."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()

        self.logger.info("ARTBLTBridge shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = get_art_logger("ARTBLTBridgeExample")

    # Check if BLT components are available and print status
    if HAS_BLT:
        logger.info("BLT components are available. Initializing ARTBLTBridge...")
    else:
        logger.info(
            "BLT components are not available. ARTBLTBridge will use fallback mode."
        )

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

    # Test encoding functionality
    logger.info("\nTesting pattern encoding:")
    pattern_data = "Complex pattern with symbolic meaning"
    encode_result = bridge.encode_pattern(pattern_data)
    logger.info(f"Encoding result: {encode_result.get('encoded', False)}")

    if encode_result.get("encoded"):
        # Test decoding
        patches = encode_result.get("patches")
        decode_result = bridge.decode_pattern(patches)
        logger.info(f"Decoding result: {decode_result.get('decoded', False)}")

    # Show stats
    logger.info("\nBridge Statistics:")
    stats = bridge.get_stats()
    for key, value in stats.items():
        if not isinstance(value, dict):  # Skip nested dicts
            logger.info(f"{key}: {value}")

    # Show cognitive status if HOLO available
    if HOLO_AVAILABLE:
        status = bridge.get_cognitive_status()
        logger.info(f"  Cognitive Load: {status.get('cognitive_load', 'N/A')}")
        logger.info(f"  Symbolic Depth: {status.get('symbolic_depth', 'N/A')}")

    # Shutdown
    bridge.shutdown()
    logger.info("ARTBLTBridge example completed")
