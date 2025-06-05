"""
ART (Adaptive Representation Transformer) System

This package provides the ART system components for adaptive representation learning:
- ART Controller: Main control interface for ART operations
- ART Trainer: Training functionality for ART models
- ART Adapter: Model adaptation capabilities
- Generative Art: Art generation functionality
- Integration bridges for BLT, RAG, and other systems

The ART system provides adaptive representation learning capabilities
for the VoxSigil framework.
"""

# Add project root to sys.path to make imports work properly
try:
    from .path_helper import setup_art_imports

    setup_art_imports()
except ImportError:
    import os
    import sys
    from pathlib import Path

    # Fallback if path_helper can't be imported
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

__all__ = [
    # Core ART components
    "ARTController",
    "ARTManager",  # New comprehensive manager class
    "ArtTrainer",
    "ArtAdapter",
    "GenerativeArt",
    # Bridge components
    "ArtEntropyBridge",
    "ARTHybridBLTBridge",
    "ARTRAGBridge",
    # Utilities
    "ARTLogger",
]

# Core ART components
try:
    from .art_controller import ARTController
except ImportError:
    ARTController = None

try:
    from .art_manager import ARTManager
except ImportError:
    ARTManager = None

try:
    from .art_trainer import ArtTrainer
except ImportError:
    ArtTrainer = None

try:
    from .art_adapter import ArtAdapter, create_art_adapter
except ImportError:
    ArtAdapter = None
    create_art_adapter = None

try:
    from .generative_art import GenerativeArt
except ImportError:
    GenerativeArt = None

# Bridge components
try:
    from .art_entropy_bridge import ArtEntropyBridge
except ImportError:
    ArtEntropyBridge = None

try:
    from .art_hybrid_blt_bridge import ARTHybridBLTBridge
except ImportError:
    ARTHybridBLTBridge = None

try:
    from .art_rag_bridge import ARTRAGBridge
except ImportError:
    ARTRAGBridge = None

# Utilities
try:
    from .art_logger import ARTLogger
except ImportError:
    ARTLogger = None
