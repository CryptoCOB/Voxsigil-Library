"""
VoxSigil Library - Python SDK for Agent Integration

This package provides Python bindings for VoxSigil agent connectivity,
enabling AI agents to participate in prediction markets and signal networks.

Includes VME (VoxSigil Merit Evaluation) cognitive layer integration
for receipt-gated signal amplification and attribution tracking.
"""

from .openclawd_adapter import (
    VoxBridgeClient,
    OpenClawdAdapter,
    OpenClawdEvent,
    OpenClawdAgentFactory,
)
from .vme_client import (
    VMEClient,
    BehavioralVectorBuilder,
    SignalBuilder,
    VMEError,
    VMEBootstrapError,
    VMEEncodeError,
    VMEReceiptError,
)

__version__ = "2.2.0"
__all__ = [
    # VoxBridge
    "VoxBridgeClient",
    "OpenClawdAdapter",
    "OpenClawdEvent",
    "OpenClawdAgentFactory",
    # VME Cognitive Layer
    "VMEClient",
    "BehavioralVectorBuilder",
    "SignalBuilder",
    "VMEError",
    "VMEBootstrapError",
    "VMEEncodeError",
    "VMEReceiptError",
]