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
    )

    # Build from a list of sigil dicts
    mw = SymbolicRAGMiddleware.build(sigil_list)

    # Query
    context = mw.retrieve_and_enrich(
        QueryContext(scaffold_type="flow", entropy_budget=0.7)
    )

    # Full pipeline
    result = mw.run_pipeline(
        QueryContext(scaffold_type="identity", intent="oracle cognition"),
        generator_fn=lambda ctx: ctx["sigils"][0],
    )
"""

from .blt_bridge import BLTBridge, ScoredSigil
from .embedder import SigilEmbedder
from .middleware import LineageStore, PipelineResult, SymbolicRAGMiddleware
from .retriever import FAISSRetriever, NumpyRetriever, QueryContext, SigilRetriever

__all__ = [
    # Core middleware
    "SymbolicRAGMiddleware",
    "QueryContext",
    "PipelineResult",
    "LineageStore",
    # Retrieval backends
    "SigilRetriever",
    "FAISSRetriever",
    "NumpyRetriever",
    # Embedding
    "SigilEmbedder",
    # BLT layer
    "BLTBridge",
    "ScoredSigil",
]
