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
from .schema_bridge import (
    ScaffoldType,
    CANONICAL_SCAFFOLDS,
    TAG_CLASSES,
    classify_schema_version,
    normalize_to_2_0_omega,
    validate_interconnected_schema,
)
from .schema_pipeline import (
    VoxSigilSchemaPipeline,
    run_default_pipeline,
)
from .rag import (
    SymbolicRAGMiddleware,
    QueryContext,
    FAISSRetriever,
    NumpyRetriever,
    SigilEmbedder,
    BLTBridge,
    ScoredSigil,
    LineageStore,
    PipelineResult,
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
    # Schema bridge
    "ScaffoldType",
    "CANONICAL_SCAFFOLDS",
    "TAG_CLASSES",
    "classify_schema_version",
    "normalize_to_2_0_omega",
    "validate_interconnected_schema",
    # Schema pipeline
    "VoxSigilSchemaPipeline",
    "run_default_pipeline",
    # Symbolic RAG Middleware
    "SymbolicRAGMiddleware",
    "QueryContext",
    "FAISSRetriever",
    "NumpyRetriever",
    "SigilEmbedder",
    "BLTBridge",
    "ScoredSigil",
    "LineageStore",
    "PipelineResult",
]