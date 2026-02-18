"""
VoxSigil Symbolic RAG Middleware

Public surface for the rag subpackage.

Quick start::

    from src.voxsigil_library.rag import (
        SymbolicRAGMiddleware,
        QueryContext,
        FAISSRetriever,
        NumpyRetriever,
        SigilEmbedder,
        BLTBridge,
        ScoredSigil,
        LineageStore,
        PipelineResult,
        SigilDeduplicator,
        DedupStatus,
        DedupResult,
        CognitiveCycleEngine,
        GeneratorConfig,
        WorldView,
    )

    # Build from a list of sigil dicts
    mw = SymbolicRAGMiddleware.build(sigil_list)

    # Query
    context = mw.retrieve_and_enrich(
        QueryContext(scaffold_type="flow", entropy_budget=0.7)
    )

    # Cognitive loop
    engine = CognitiveCycleEngine.create(
        generators=[GeneratorConfig(kind="ollama", model="llama3.2:latest")],
    )
    engine.run(target_corpus_size=1000)
"""

from .blt_bridge import BLTBridge, ScoredSigil
from .cognitive_loop import (
    CognitiveCycleEngine,
    GeneratorConfig,
    WorldView,
)
from .deduplicator import DedupResult, DedupStatus, SigilDeduplicator
from .embedder import SigilEmbedder
from .middleware import (
    FeedbackVerdict,
    LineageStore,
    PipelineResult,
    SymbolicRAGMiddleware,
)
from .retriever import FAISSRetriever, NumpyRetriever, QueryContext, SigilRetriever

__all__ = [
    # Core middleware
    "SymbolicRAGMiddleware",
    "QueryContext",
    "PipelineResult",
    "LineageStore",
    "FeedbackVerdict",
    # Retrieval backends
    "SigilRetriever",
    "FAISSRetriever",
    "NumpyRetriever",
    # Embedding
    "SigilEmbedder",
    # BLT layer
    "BLTBridge",
    "ScoredSigil",
    # 3-layer deduplicator
    "SigilDeduplicator",
    "DedupStatus",
    "DedupResult",
    # Cognitive loop
    "CognitiveCycleEngine",
    "GeneratorConfig",
    "WorldView",
]
