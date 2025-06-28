#!/usr/bin/env python
"""
BLT System Adapters Module

This module provides a unified interface to all BLT (Belief Learning Technology)
system components, acting as a facade for the various BLT encoders, middleware,
and extensions available in the VoxSigil system.

This resolves import issues in other modules that expect a centralized BLT system
interface while avoiding circular imports.
"""

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class BLTSystem:
    """
    Unified BLT System interface that provides access to all available BLT components.

    This class acts as a facade/registry for BLT components and handles graceful
    degradation when components are not available.
    """

    def __init__(self):
        """Initialize the BLT system with available components."""
        self.logger = logger
        self._components = {}
        self._initialized = False
        self._available_components = []

        # Initialize components
        self._discover_components()

    def _discover_components(self) -> None:
        """Discover and register available BLT components."""
        try:
            # Try to import core BLT components
            from . import BLTEncoder, ByteLatentTransformerEncoder, SigilPatchEncoder

            self._components["encoder"] = BLTEncoder
            self._components["patch_encoder"] = SigilPatchEncoder
            self._components["transformer_encoder"] = ByteLatentTransformerEncoder
            self._available_components.extend(
                ["encoder", "patch_encoder", "transformer_encoder"]
            )
            self.logger.debug("Registered core BLT encoders")
        except ImportError as e:
            self.logger.warning(f"Core BLT encoders not available: {e}")

        try:
            # Try to import hybrid middleware
            from .hybrid_blt import BLTEnhancedRAG, EntropyRouter, HybridMiddleware

            self._components["hybrid_middleware"] = HybridMiddleware
            self._components["enhanced_rag"] = BLTEnhancedRAG
            self._components["entropy_router"] = EntropyRouter
            self._available_components.extend(
                ["hybrid_middleware", "enhanced_rag", "entropy_router"]
            )
            self.logger.debug("Registered hybrid BLT components")
        except ImportError as e:
            self.logger.debug(f"Hybrid BLT components not available: {e}")

        try:
            # Try to import enhanced extension
            from .blt_enhanced_extension import BLTEnhancedExtension

            self._components["enhanced_extension"] = BLTEnhancedExtension
            self._available_components.append("enhanced_extension")
            self.logger.debug("Registered BLT enhanced extension")
        except ImportError as e:
            self.logger.debug(f"BLT enhanced extension not available: {e}")

        try:
            # Try to import supervisor integration
            from .blt_supervisor_integration import (
                BLTSupervisorRagInterface,
                TinyLlamaIntegration,
            )

            self._components["supervisor_rag"] = BLTSupervisorRagInterface
            self._components["tinyllama"] = TinyLlamaIntegration
            self._available_components.extend(["supervisor_rag", "tinyllama"])
            self.logger.debug("Registered BLT supervisor components")
        except ImportError as e:
            self.logger.debug(f"BLT supervisor components not available: {e}")

        try:
            # Try to import RAG compression
            from .blt_rag_compression import PatchAwareCompressor, RAGCompressionEngine

            self._components["rag_compression"] = RAGCompressionEngine
            self._components["patch_compressor"] = PatchAwareCompressor
            self._available_components.extend(["rag_compression", "patch_compressor"])
            self.logger.debug("Registered BLT compression components")
        except ImportError as e:
            self.logger.debug(f"BLT compression components not available: {e}")

        self._initialized = True
        self.logger.info(
            f"BLT system initialized with {len(self._available_components)} components: {self._available_components}"
        )

    @property
    def is_available(self) -> bool:
        """Check if the BLT system is available with at least some components."""
        return self._initialized and len(self._available_components) > 0

    @property
    def available_components(self) -> List[str]:
        """Get list of available component names."""
        return self._available_components.copy()

    def get_component(self, name: str) -> Optional[Type]:
        """Get a specific BLT component class by name."""
        return self._components.get(name)

    def has_component(self, name: str) -> bool:
        """Check if a specific component is available."""
        return name in self._components

    def create_encoder(self, encoder_type: str = "encoder", **kwargs) -> Optional[Any]:
        """Create a BLT encoder instance."""
        encoder_class = self.get_component(encoder_type)
        if encoder_class:
            try:
                return encoder_class(**kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create {encoder_type}: {e}")
                return None
        else:
            self.logger.warning(f"Encoder type '{encoder_type}' not available")
            return None

    def create_middleware(self, **kwargs) -> Optional[Any]:
        """Create a hybrid middleware instance if available."""
        if self.has_component("hybrid_middleware"):
            try:
                middleware_class = self.get_component("hybrid_middleware")
                return middleware_class(**kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create hybrid middleware: {e}")
                return None
        else:
            self.logger.warning("Hybrid middleware not available")
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the BLT system state."""
        return {
            "available": self.is_available,
            "initialized": self._initialized,
            "component_count": len(self._available_components),
            "components": self._available_components,
            "has_encoders": any(
                "encoder" in comp for comp in self._available_components
            ),
            "has_middleware": self.has_component("hybrid_middleware"),
            "has_rag": any("rag" in comp for comp in self._available_components),
        }


# Create a default instance for easy import
default_blt_system = BLTSystem()

# For backward compatibility and easy access
BLT_SYSTEM = default_blt_system
