"""
SymbolicRAGMiddleware — The full retrieve → enrich → BLT_validate → generate loop

This module is the main entry point for the VoxSigil RAG layer.  It wires:

  VME orchestration intent
      ↓
  SigilRetriever  (FAISS / numpy)
      ↓
  BLTBridge  (compress / score / filter)
      ↓
  Enriched symbolic context payload
      ↓
  Generator / Model / Agent

Usage::

    from src.voxsigil_library.rag import SymbolicRAGMiddleware, QueryContext

    # Build once (or load pre-built index).
    mw = SymbolicRAGMiddleware.build(sigil_list)
    # or: mw = SymbolicRAGMiddleware.from_index("storage/rag_index.faiss")

    # Retrieve enriched context for a query.
    context = mw.retrieve_and_enrich(
        QueryContext(scaffold_type="flow", entropy_budget=0.7),
        top_k=10,
    )

    # context["sigils"]        — list of valid sigil dicts
    # context["avg_blt_score"] — quality summary
    # context["blt_filtered"]  — True

    # Full pipeline with a custom generator function.
    def my_generator(ctx): ...
    result = mw.run_pipeline(query, generator_fn=my_generator)

Lineage is stored automatically after each successful pipeline run so that
the retrieve → generate → store loop closes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from .blt_bridge import BLTBridge
from .embedder import SigilEmbedder
from .retriever import FAISSRetriever, NumpyRetriever, QueryContext, SigilRetriever


# ---------------------------------------------------------------------------
# LineageStore — lightweight in-process lineage log
# ---------------------------------------------------------------------------


class LineageStore:
    """
    Thin append-only lineage log.

    Can be swapped for a persistent store (Redis, SQLite, Qdrant, …) by
    subclassing and overriding ``append`` and ``recent``.
    """

    def __init__(self) -> None:
        self._log: List[Dict[str, Any]] = []
        self._feedback_by_artifact: Dict[str, float] = {}

    def append(self, entry: Dict[str, Any]) -> None:
        self._log.append(entry)

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return self._log[-n:]

    def add_feedback(self, artifact_id: str, delta: float) -> float:
        """Apply additive feedback delta and clamp to [-1.0, 1.0]."""
        current = self._feedback_by_artifact.get(artifact_id, 0.0)
        updated = max(-1.0, min(1.0, current + delta))
        self._feedback_by_artifact[artifact_id] = updated
        return updated

    def feedback_weight(self, artifact_id: str) -> float:
        return self._feedback_by_artifact.get(artifact_id, 0.0)

    def to_jsonl(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._log.append(json.loads(line))


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    query: QueryContext
    retrieved_count: int
    legal_count: int
    context: Dict[str, Any]
    generator_output: Optional[Any] = None
    compressed_output: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    lineage_entry: Optional[Dict[str, Any]] = None


class FeedbackVerdict(str, Enum):
    """User feedback signal applied after output compression."""

    APPROVE = "approve"
    REJECT = "reject"
    REFINE = "refine"
    REDIRECT = "redirect"


_FEEDBACK_DELTA: Dict[FeedbackVerdict, float] = {
    FeedbackVerdict.APPROVE: 0.30,
    FeedbackVerdict.REJECT: -0.40,
    FeedbackVerdict.REFINE: 0.10,
    FeedbackVerdict.REDIRECT: -0.10,
}


# ---------------------------------------------------------------------------
# SymbolicRAGMiddleware
# ---------------------------------------------------------------------------


class SymbolicRAGMiddleware:
    """
    Full VME → RAG → BLT → Generator pipeline.

    Responsibilities:
    - Route queries to the symbolic retriever (FAISS or numpy).
    - Pass results through BLT Bridge for compression, scoring, and
      legality filtering.
    - Assemble a canonical context payload for downstream generators.
    - Record lineage for every pipeline run.
    """

    def __init__(
        self,
        retriever: Optional[SigilRetriever] = None,
        blt_bridge: Optional[BLTBridge] = None,
        lineage_store: Optional[LineageStore] = None,
        default_top_k: int = 10,
    ) -> None:
        self.retriever: SigilRetriever = retriever or FAISSRetriever()
        self.blt: BLTBridge = blt_bridge or BLTBridge()
        self.lineage: LineageStore = lineage_store or LineageStore()
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        sigils: List[Dict[str, Any]],
        use_faiss: bool = True,
        **kwargs: Any,
    ) -> "SymbolicRAGMiddleware":
        """
        Build middleware from a list of sigil dicts (in-memory indexing).

        Args:
            sigils:    List of sigil objects to index.
            use_faiss: Prefer FAISS; falls back to numpy automatically.
            **kwargs:  Extra kwargs forwarded to the constructor.
        """
        retriever: SigilRetriever = FAISSRetriever() if use_faiss else NumpyRetriever()
        retriever.index(sigils)
        return cls(retriever=retriever, **kwargs)

    @classmethod
    def from_index(
        cls,
        index_path: str,
        **kwargs: Any,
    ) -> "SymbolicRAGMiddleware":
        """
        Load middleware from a pre-built FAISS index on disk.

        The index must have been saved via ``retriever.save(path)`` or
        ``middleware.save_index(path)``.
        """
        retriever = FAISSRetriever.from_index(index_path)
        return cls(retriever=retriever, **kwargs)

    # ------------------------------------------------------------------
    # Core: retrieve_and_enrich
    # ------------------------------------------------------------------

    def retrieve_and_enrich(
        self,
        query: QueryContext,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve sigils matching the query, score them through BLT, and
        return an enriched context dict ready for injection into a generator.

        Returns a dict with keys:
            sigils          — list of valid sigil dicts
            count           — number of sigils
            avg_blt_score   — mean quality score
            blt_filtered    — True
            scaffold_types  — scaffold types present
            query_meta      — query parameters used
        """
        k = top_k or self.default_top_k

        # Step 1: retrieve
        raw = self.retriever.retrieve(query, top_k=k * 2)  # over-fetch
        self._apply_feedback_weights(raw)

        # Step 2: BLT compress + score
        scored = self.blt.compress_and_score(raw)

        # Step 3: filter illegal / below-threshold
        legal = self.blt.filter_illegal(scored)[:k]

        # Step 4: emit canonical context
        context = self.blt.emit_canonical_context(legal)
        context["query_meta"] = {
            "scaffold_type": query.scaffold_type,
            "entropy_budget": query.entropy_budget,
            "lineage": query.lineage,
            "tags": query.tags,
            "intent": query.intent,
        }

        return context

    # ------------------------------------------------------------------
    # Full pipeline (retrieve → enrich → generate → store lineage)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        query: QueryContext,
        generator_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        top_k: Optional[int] = None,
        lineage_tag: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the full symbolic RAG pipeline:

          retrieve → enrich → BLT_validate → generate → store

        Args:
            query:         Structured query context.
            generator_fn:  Callable that accepts the context dict and returns
                           a generated sigil (or any output).  If None, only
                           retrieval + enrichment is performed.
            top_k:         Override the default top-k.
            lineage_tag:   Tag to attach to this pipeline run's lineage entry.

        Returns:
            PipelineResult with context, output, and timing.
        """
        t0 = time.perf_counter()
        k = top_k or self.default_top_k

        # --- Retrieve + enrich -----------------------------------------
        raw = self.retriever.retrieve(query, top_k=k * 2)
        self._apply_feedback_weights(raw)
        scored = self.blt.compress_and_score(raw)
        legal = self.blt.filter_illegal(scored)[:k]
        context = self.blt.emit_canonical_context(legal)
        context["query_meta"] = {
            "scaffold_type": query.scaffold_type,
            "entropy_budget": query.entropy_budget,
            "lineage": query.lineage,
            "tags": query.tags,
            "intent": query.intent,
        }

        # --- Generate (optional) --------------------------------------
        generator_output: Optional[Any] = None
        compressed_output: Optional[Dict[str, Any]] = None
        if generator_fn is not None:
            generator_output = generator_fn(context)
            compressed_output = self._compress_output(generator_output)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        # --- Store lineage --------------------------------------------
        lineage_entry: Dict[str, Any] = {
            "ts": time.time(),
            "query_scaffold": query.scaffold_type,
            "query_entropy_budget": query.entropy_budget,
            "retrieved": len(raw),
            "legal": len(legal),
            "avg_blt_score": context.get("avg_blt_score", 0.0),
            "output_accepted": (compressed_output or {}).get("count", 0),
            "duration_ms": round(duration_ms, 2),
        }
        if lineage_tag:
            lineage_entry["tag"] = lineage_tag
        if query.lineage:
            lineage_entry["lineage"] = query.lineage
        self.lineage.append(lineage_entry)

        return PipelineResult(
            query=query,
            retrieved_count=len(raw),
            legal_count=len(legal),
            context=context,
            generator_output=generator_output,
            compressed_output=compressed_output,
            duration_ms=duration_ms,
            lineage_entry=lineage_entry,
        )

    def apply_user_feedback(
        self,
        artifacts: Sequence[Any],
        verdict: FeedbackVerdict | str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply user feedback as a selection signal.

        Artifacts can be sigil dicts or artifact-id strings.
        Feedback updates persistent lineage weights and propagates those
        weights into currently indexed retriever items.
        """
        if isinstance(verdict, FeedbackVerdict):
            verdict_enum = verdict
        else:
            verdict_enum = FeedbackVerdict(str(verdict))
        delta = _FEEDBACK_DELTA[verdict_enum]

        touched: List[Dict[str, Any]] = []
        for artifact in artifacts:
            artifact_id = artifact if isinstance(artifact, str) else self._artifact_id(artifact)
            new_weight = self.lineage.add_feedback(artifact_id, delta)
            touched.append({"artifact_id": artifact_id, "weight": new_weight})

        self._propagate_feedback_to_retriever()

        entry = {
            "ts": time.time(),
            "feedback": verdict_enum.value,
            "delta": delta,
            "count": len(touched),
            "note": note or "",
            "artifacts": touched,
        }
        self.lineage.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Persist the retrieval index to disk."""
        self.retriever.save(path)

    def index_from_jsonl(self, *paths: str) -> int:
        """
        (Re-)index from one or more JSONL dataset files.
        Returns the number of sigils indexed.
        """
        sigils: List[Dict[str, Any]] = []
        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            sigils.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self.retriever.index(sigils)
        return len(sigils)

    # ------------------------------------------------------------------
    # Convenience: print summary
    # ------------------------------------------------------------------

    def describe(self) -> str:
        recent = self.lineage.recent(5)
        lines = [
            "SymbolicRAGMiddleware",
            f"  retriever : {type(self.retriever).__name__}",
            f"  blt_bridge: BLTBridge(min_score={self.blt.min_score_threshold})",
            f"  lineage   : {len(self.lineage._log)} entries",
        ]
        if recent:
            lines.append("  last runs :")
            for r in recent[-3:]:
                lines.append(
                    f"    {r.get('query_scaffold','?')} | "
                    f"retrieved={r.get('retrieved',0)} "
                    f"legal={r.get('legal',0)} "
                    f"score={r.get('avg_blt_score',0):.3f} "
                    f"({r.get('duration_ms',0):.1f}ms)"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _artifact_id(sigil: Dict[str, Any]) -> str:
        return SigilEmbedder.sigil_id(sigil)

    def _apply_feedback_weights(self, sigils: List[Dict[str, Any]]) -> None:
        for sigil in sigils:
            artifact_id = self._artifact_id(sigil)
            sigil["feedback_weight"] = self.lineage.feedback_weight(artifact_id)

    def _propagate_feedback_to_retriever(self) -> None:
        # Best-effort propagation for current in-memory retrievers.
        sigils = getattr(self.retriever, "_sigils", None)
        if not isinstance(sigils, list):
            return
        self._apply_feedback_weights(sigils)

    def _compress_output(self, output: Any) -> Optional[Dict[str, Any]]:
        """
        BLT output gate: compress and legality-filter model output *after* generation.
        Returns canonical context for output sigils, or None if output does not
        contain sigil-like dicts.
        """
        items: List[Dict[str, Any]] = []
        if isinstance(output, dict) and "sigil" in output:
            items = [output]
        elif isinstance(output, list):
            items = [x for x in output if isinstance(x, dict) and "sigil" in x]

        if not items:
            return None

        scored = self.blt.compress_and_score(items)
        legal = self.blt.filter_illegal(scored)
        return self.blt.emit_canonical_context(legal)
