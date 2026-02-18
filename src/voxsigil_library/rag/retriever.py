"""
VoxSigil Symbolic Retriever

Abstract SigilRetriever interface + two concrete backends:

  FAISSRetriever  — local, deterministic, zero-config (requires faiss-cpu)
  NumpyRetriever  — pure-Python fallback (always available, slower at scale)

Retrieval is by *structure*, not keywords:
  - scaffold_type
  - typed_tags (domain / function / polarity / temporal / epistemic / lifecycle)
  - intent text (embedded via SigilEmbedder)
  - entropy_budget  (filters out overly-entropic sigils if budget is low)
  - lineage         (prefers sigils with matching lineage chain)

Query interface::

    from src.voxsigil_library.rag.retriever import FAISSRetriever, QueryContext

    retriever = FAISSRetriever()
    retriever.index(list_of_sigil_dicts)
    retriever.save("storage/rag_index.faiss")

    results = retriever.retrieve(
        QueryContext(scaffold_type="flow", entropy_budget=0.7),
        top_k=10,
    )
"""

from __future__ import annotations

import json
import math
import os
import pickle
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embedder import SigilEmbedder


# ---------------------------------------------------------------------------
# Query context
# ---------------------------------------------------------------------------


@dataclass
class QueryContext:
    """
    Structured query for symbolic sigil retrieval.

    All fields are optional; any combination is valid.
    """

    scaffold_type: Optional[str] = None          # "flow" | "identity" | "assembly" | "mutation"
    tags: Optional[Dict[str, List[str]]] = None  # {"domain": ["cognition"], ...}
    intent: Optional[str] = None                  # free-text intent description
    entropy_budget: float = 1.0                   # 0.0 – 1.0 upper bound on entropy fields
    lineage: Optional[str] = None                 # preferred lineage chain identifier

    def to_pseudo_sigil(self) -> Dict[str, Any]:
        """
        Construct a minimal sigil-shaped dict from this query so it can be
        embedded using the same SigilEmbedder as real sigils.
        """
        doc: Dict[str, Any] = {}
        if self.scaffold_type:
            doc["scaffold"] = {"scaffold_type": self.scaffold_type}
        if self.tags:
            doc["typed_tags"] = self.tags
        if self.intent:
            doc["sigil"] = self.intent[:64]  # use intent as pseudo-glyph text
        if self.lineage:
            doc["lineage"] = self.lineage
        return doc


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SigilRetriever(ABC):
    """
    Contract for all symbolic sigil retrieval backends.

    Implementors MUST guarantee:
    - Only valid sigils are returned (illegal sigils are not indexed or are
      filtered at retrieval time).
    - Lifecycle / lineage fields are preserved on returned sigil dicts.
    - No prose is ever returned — only sigil objects.
    """

    @abstractmethod
    def index(self, sigils: List[Dict[str, Any]]) -> None:
        """Build or replace the retrieval index from a list of sigil dicts."""

    @abstractmethod
    def retrieve(
        self,
        query: QueryContext,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `top_k` sigil dicts ranked by structural similarity to
        the query context.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the index from disk."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_entropy_budget(sigil: Dict[str, Any], budget: float) -> bool:
        """True if the sigil's entropy is within the caller's budget."""
        # Check lifecycle tags — 'deprecated' / 'sealed' sigils are always out.
        lifecycle = []
        typed_tags = sigil.get("typed_tags", {})
        if isinstance(typed_tags, dict):
            lifecycle = typed_tags.get("lifecycle", [])

        if isinstance(lifecycle, list):
            if "deprecated" in lifecycle or "sealed" in lifecycle:
                return False

        # Check explicit entropy field (0.0 – 1.0) if present.
        entropy_val = sigil.get("entropy", None)
        if entropy_val is not None:
            try:
                if float(entropy_val) > budget:
                    return False
            except (TypeError, ValueError):
                pass

        return True

    @staticmethod
    def _lineage_boost(sigil: Dict[str, Any], lineage: Optional[str]) -> float:
        """Extra similarity bonus for lineage-matching sigils."""
        if lineage is None:
            return 0.0
        sigil_lineage = sigil.get("lineage", "") or sigil.get("chain", "")
        if isinstance(sigil_lineage, str) and lineage in sigil_lineage:
            return 0.05
        return 0.0


# ---------------------------------------------------------------------------
# Pure-numpy fallback backend
# ---------------------------------------------------------------------------


class NumpyRetriever(SigilRetriever):
    """
    Linear-scan cosine-similarity retriever using only stdlib + basic math.

    Correct for any dataset size; becomes slow above ~100k sigils but is
    always available as a fallback.
    """

    def __init__(self) -> None:
        self._embedder = SigilEmbedder()
        self._sigils: List[Dict[str, Any]] = []
        self._vectors: List[List[float]] = []

    # ------------------------------------------------------------------

    def index(self, sigils: List[Dict[str, Any]]) -> None:
        """Embed all sigils and store in a flat list."""
        self._sigils = []
        self._vectors = []
        for s in sigils:
            if not isinstance(s, dict):
                continue
            # Skip deprecated / sealed at index time.
            if not self._passes_entropy_budget(s, budget=2.0):  # generous at index
                continue
            vec = self._embedder.embed(s)
            self._sigils.append(s)
            self._vectors.append(vec)

    def retrieve(
        self,
        query: QueryContext,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self._sigils:
            return []

        query_vec = self._embedder.embed(query.to_pseudo_sigil())
        scores: List[Tuple[float, int]] = []

        for i, (vec, sigil) in enumerate(zip(self._vectors, self._sigils)):
            if not self._passes_entropy_budget(sigil, query.entropy_budget):
                continue
            # Cosine similarity (vecs are unit-normalised → dot product).
            score = sum(a * b for a, b in zip(query_vec, vec))
            score += self._lineage_boost(sigil, query.lineage)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self._sigils[i] for _, i in scores[:top_k]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"sigils": self._sigils, "vectors": self._vectors}, f)

    def load(self, path: str) -> None:
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)
        self._sigils = data["sigils"]
        self._vectors = data["vectors"]


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------


class FAISSRetriever(SigilRetriever):
    """
    FAISS-backed retriever for large sigil libraries.

    Falls back to NumpyRetriever automatically if faiss-cpu is not installed.
    The user-facing API is identical either way.

    Install:  pip install faiss-cpu
    """

    def __init__(self) -> None:
        self._embedder = SigilEmbedder()
        self._sigils: List[Dict[str, Any]] = []
        self._index: Any = None          # faiss.IndexFlatIP or None
        self._fallback: Optional[NumpyRetriever] = None
        self._faiss_available = self._try_import_faiss()

    # ------------------------------------------------------------------

    @staticmethod
    def _try_import_faiss() -> bool:
        try:
            import faiss  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_faiss(self):
        import faiss
        return faiss

    # ------------------------------------------------------------------

    def index(self, sigils: List[Dict[str, Any]]) -> None:
        if not self._faiss_available:
            self._fallback = NumpyRetriever()
            self._fallback.index(sigils)
            self._sigils = self._fallback._sigils
            return

        faiss = self._get_faiss()
        dim = SigilEmbedder.embed_dim

        valid_sigils = [
            s for s in sigils
            if isinstance(s, dict) and self._passes_entropy_budget(s, budget=2.0)
        ]
        embeddings = self._embedder.embed_batch(valid_sigils)

        # Build flat inner-product index (unit vecs → cosine similarity).
        idx = faiss.IndexFlatIP(dim)
        import array as arr

        # Convert to a flat array of float32 values.
        flat: List[float] = []
        for vec in embeddings:
            flat.extend(vec)

        # Use struct to pack as float32 bytes, then feed to faiss.
        n = len(embeddings)
        if n == 0:
            self._sigils = []
            self._index = idx
            return

        buf = struct.pack(f"{n * dim}f", *flat)
        # faiss expects numpy array — use array module as a lightweight substitute.
        try:
            import numpy as np
            matrix = np.array(flat, dtype=np.float32).reshape(n, dim)
            idx.add(matrix)
        except ImportError:
            # Without numpy, fall back to the pure-Python path.
            self._faiss_available = False
            self._fallback = NumpyRetriever()
            self._fallback.index(sigils)
            self._sigils = self._fallback._sigils
            return

        self._sigils = valid_sigils
        self._index = idx

    def retrieve(
        self,
        query: QueryContext,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if self._fallback is not None:
            # Apply entropy filter on fallback results too.
            results = self._fallback.retrieve(query, top_k=top_k * 2)
            return [r for r in results if self._passes_entropy_budget(r, query.entropy_budget)][:top_k]

        if self._index is None or not self._sigils:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        query_vec = self._embedder.embed(query.to_pseudo_sigil())
        q = np.array(query_vec, dtype=np.float32).reshape(1, -1)

        k = min(top_k * 3, len(self._sigils))  # over-fetch to allow filtering
        distances, indices = self._index.search(q, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._sigils):
                continue
            sigil = self._sigils[idx]
            if not self._passes_entropy_budget(sigil, query.entropy_budget):
                continue
            score = float(dist) + self._lineage_boost(sigil, query.lineage)
            results.append((score, sigil))

        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results[:top_k]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self._fallback is not None:
            self._fallback.save(path)
            return
        if self._index is None:
            return
        try:
            import faiss
            import numpy as np
            faiss.write_index(self._index, path + ".faiss")
            with open(path + ".meta.pkl", "wb") as f:
                pickle.dump(self._sigils, f)
        except Exception as exc:
            raise RuntimeError(f"FAISSRetriever.save failed: {exc}") from exc

    def load(self, path: str) -> None:
        # Try numpy / faiss path first.
        faiss_path = path + ".faiss"
        meta_path = path + ".meta.pkl"
        pkl_path = path + ".pkl"

        if os.path.exists(pkl_path):
            self._fallback = NumpyRetriever()
            self._fallback.load(path)
            self._sigils = self._fallback._sigils
            return

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"No index found at {path}[.faiss|.meta.pkl|.pkl]")

        try:
            import faiss
            self._index = faiss.read_index(faiss_path)
        except ImportError:
            raise RuntimeError(
                "faiss-cpu is required to load a FAISS index. "
                "Install it with: pip install faiss-cpu"
            )
        with open(meta_path, "rb") as f:
            self._sigils = pickle.load(f)
        self._faiss_available = True
        self._fallback = None

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_index(cls, path: str) -> "FAISSRetriever":
        """Load a pre-built index from disk."""
        r = cls()
        r.load(path)
        return r
