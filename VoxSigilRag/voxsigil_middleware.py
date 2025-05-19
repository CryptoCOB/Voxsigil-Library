#!/usr/bin/env python
"""
VoxSigil Middleware Module (Redirector)

This module is maintained for backward compatibility but now redirects to the
Hybrid BLT Middleware implementation which is the recommended middleware for all
VoxSigil integration scenarios.

For new code, please use HybridMiddleware from hybrid_blt.py directly.
"""

import logging
from .hybrid_blt import HybridMiddleware

# Set up logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Create an alias for backward compatibility
VoxSigilMiddleware = HybridMiddleware

# Emit a deprecation warning
logger.warning(
    "The standard VoxSigilMiddleware from voxsigil_middleware.py is deprecated. "
    "Using the HybridMiddleware from hybrid_blt.py instead. "
    "Please update your imports to use 'from VoxSigilRag.hybrid_blt import HybridMiddleware'."
)

# Re-export the standard symbols for compatibility
__all__ = ['VoxSigilMiddleware']