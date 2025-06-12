#!/usr/bin/env python3
"""
RAG Integration Handler for VantaCore

This module provides the integration between RAG interfaces and VantaCore,
ensuring that the RAG functionality is properly registered with VantaCore
and available to other components.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from Vanta.core.UnifiedVantaCore import UnifiedVantaCore, get_vanta_core
from Vanta.interfaces.base_interfaces import BaseRagInterface
from Vanta.core.fallback_implementations import FallbackRagInterface
from Vanta.interfaces.rag_interface import (
    VOXSIGIL_RAG_AVAILABLE,
    SimpleRagInterface,
    SupervisorRagInterface,
)

logger = logging.getLogger("VantaCore.RAG")


class RagIntegrationHandler:
    """
    Handles integration of RAG interfaces with VantaCore.
    This ensures that RAG functionality is properly registered with VantaCore
    and available for use by other components.
    """

    def __init__(self, vanta_core: Optional[UnifiedVantaCore] = None):
        """Initialize with optional VantaCore instance."""
        self.vanta_core = vanta_core if vanta_core else get_vanta_core()
        self.rag_interface = None

    def initialize_rag_interface(
        self,
        interface_type: str = "supervisor",
        voxsigil_library_path: Optional[Path] = None,
        rag_processor: Any = None,
    ) -> BaseRagInterface:
        """
        Initialize the RAG interface based on the requested type.

        Args:
            interface_type: Type of RAG interface to initialize ('supervisor', 'simple', or 'mock')
            voxsigil_library_path: Optional path to the VoxSigil library (for SupervisorRagInterface)
            rag_processor: Optional RAG processor instance (for SimpleRagInterface)

        Returns:
            The initialized RAG interface
        """
        # First check if an interface is already registered
        existing_interface = self.vanta_core.get_component("rag_interface")
        if existing_interface:
            logger.info(
                f"Using existing RAG interface: {type(existing_interface).__name__}"
            )
            self.rag_interface = existing_interface
            return existing_interface

        # Create the appropriate interface based on type
        if interface_type == "supervisor":
            if VOXSIGIL_RAG_AVAILABLE:
                logger.info("Initializing SupervisorRagInterface")
                self.rag_interface = SupervisorRagInterface(
                    voxsigil_library_path=voxsigil_library_path
                )
            else:
                logger.warning(
                    "VoxSigilRAG not available, falling back to FallbackRagInterface"
                )
                self.rag_interface = FallbackRagInterface()
                interface_type = "mock"  # Update type for correct metadata

        elif interface_type == "simple":
            if rag_processor:
                logger.info("Initializing SimpleRagInterface")
                self.rag_interface = SimpleRagInterface(rag_processor)
            else:
                logger.warning(
                    "No RAG processor provided for SimpleRagInterface, falling back to FallbackRagInterface"
                )
                self.rag_interface = FallbackRagInterface()
                interface_type = "mock"  # Update type for correct metadata

        elif interface_type == "mock":
            logger.info("Initializing FallbackRagInterface")
            self.rag_interface = FallbackRagInterface()

        else:
            logger.warning(
                f"Unknown RAG interface type '{interface_type}', using FallbackRagInterface"
            )
            self.rag_interface = FallbackRagInterface()
            interface_type = "mock"  # Update type for correct metadata

        # Register the interface with VantaCore
        if self.rag_interface:
            self._register_rag_interface(interface_type)
            logger.info(
                f"Successfully initialized and registered {type(self.rag_interface).__name__}"
            )

        return self.rag_interface

    def _register_rag_interface(self, interface_type: str) -> None:
        """
        Register the RAG interface with VantaCore.

        Args:
            interface_type: Type of RAG interface being registered
        """
        if not self.rag_interface:
            logger.error("No RAG interface to register")
            return

        metadata = {
            "type": "rag_interface",
            "interface_type": interface_type,
            "capabilities": [
                "retrieve_sigils",
                "retrieve_context",
                "retrieve_scaffolds",
                "get_scaffold_definition",
                "get_sigil_by_id",
            ],
            "voxsigil_rag_available": VOXSIGIL_RAG_AVAILABLE,
        }

        # Register with VantaCore
        self.vanta_core.register_component(
            "rag_interface", self.rag_interface, metadata
        )
        logger.info("RAG interface registered with VantaCore as 'rag_interface'")


def initialize_rag_system(
    vanta_core: Optional[UnifiedVantaCore] = None,
    interface_type: str = "supervisor",
    voxsigil_library_path: Optional[Path] = None,
    rag_processor: Any = None,
) -> BaseRagInterface:
    """
    Initialize the RAG system with VantaCore.

    Args:
        vanta_core: Optional VantaCore instance
        interface_type: Type of RAG interface to initialize
        voxsigil_library_path: Optional path to the VoxSigil library
        rag_processor: Optional RAG processor instance

    Returns:
        The initialized RAG interface
    """
    handler = RagIntegrationHandler(vanta_core)
    return handler.initialize_rag_interface(
        interface_type=interface_type,
        voxsigil_library_path=voxsigil_library_path,
        rag_processor=rag_processor,
    )
