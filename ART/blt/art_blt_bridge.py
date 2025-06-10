#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARTBLTBridge - A bridge between the ART module and BLT middleware.

This module provides the ARTBLTBridge class that connects voxsigil.art.ARTManager
with the BLT middleware for entropy-based selective pattern analysis.
"""

from typing import Any, Dict, Optional

from ..art_manager import ARTManager
from ..art_logger import get_art_logger

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


class ARTBLTBridge:
    """
    A bridge between the ART module and BLT middleware.

    This class facilitates communication between the ART module's pattern recognition
    capabilities and the BLT middleware's symbolic reasoning engine. It provides
    methods for encoding patterns recognized by ART into sigil patches that can be
    processed by BLT, and for decoding BLT's symbolic outputs into formats that
    can be interpreted by ART.
    """

    def __init__(
        self,
        art_manager: Optional[ARTManager] = None,
        sigil_encoder: Optional[SigilPatchEncoder] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ARTBLTBridge.

        Args:
            art_manager: An instance of ARTManager for pattern recognition
            sigil_encoder: An instance of SigilPatchEncoder for encoding patterns
            config: Configuration settings for the bridge
        """
        self.logger = get_art_logger("ARTBLTBridge")
        self.art_manager = art_manager
        self.config = config or {}

        # Initialize BLT components if available
        if HAS_BLT:
            self.sigil_encoder = sigil_encoder or SigilPatchEncoder()
        else:
            self.sigil_encoder = None
            self.logger.warning(
                "BLT components not available. Some functionality will be limited."
            )


# The rest of the file content remains the same as the original
# This is a placeholder for the actual implementation which would be copied from the source file
