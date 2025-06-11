"""
Vanta Unified Interface System
============================

This module serves as the single source of truth for all interface definitions
in the VoxSigil Library. All modules should import interfaces from here to
ensure consistency and prevent duplication.

Architecture:
- Base interfaces define core contracts
- Specialized interfaces extend base interfaces
- All modules communicate through these unified contracts
"""

from .base_interfaces import (
    BaseRagInterface,
    BaseLlmInterface, 
    BaseMemoryInterface,
    BaseAgentInterface,
    BaseModelInterface
)

from .specialized_interfaces import (
    MetaLearnerInterface,
    ModelManagerInterface,
    BLTInterface,
    ARCInterface,
    ARTInterface,
    MiddlewareInterface
)

from .protocol_interfaces import (
    VantaProtocol,
    ModuleAdapterProtocol,
    IntegrationProtocol
)

__all__ = [
    # Base Interfaces
    'BaseRagInterface',
    'BaseLlmInterface',
    'BaseMemoryInterface', 
    'BaseAgentInterface',
    'BaseModelInterface',
    
    # Specialized Interfaces
    'MetaLearnerInterface',
    'ModelManagerInterface',
    'BLTInterface',
    'ARCInterface',
    'ARTInterface',
    'MiddlewareInterface',
    
    # Protocol Interfaces
    'VantaProtocol',
    'ModuleAdapterProtocol',
    'IntegrationProtocol'
]
