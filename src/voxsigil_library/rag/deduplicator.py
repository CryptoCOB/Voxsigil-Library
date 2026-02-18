"""
VoxSigil 3-Layer Deduplicator

Ensures every sigil entering the corpus is unique at three levels:

  Layer 1 — Hard uniqueness (SHA-256 hash)
    Hash over: normalized glyph sequence + scaffold_type + canonicalized principle.
    Reject immediately if hash already exists.

  Layer 2 — Semantic collision (SigilEmbedder cosine similarity)
    Embed the incoming sigil, compare to all existing embeddings.
    > 0.97  →  reject (semantic duplicate)
    0.90–0.97 → accept as DERIVATIVE only (lineage required)
    < 0.90  →  accept as novel

  Layer 3 — Structural redundancy
    Reject if all three of the following match an existing sigil:
      scaffold_type, tag vector fingerprint, ecosystem_role

This ensures BLT learns discrimination, not memorization.

Usage::

    dedup = SigilDeduplicator()
    result = dedup.check(sigil)

    if result.status == "novel":
        corpus.append(sigil)
    elif result.status == "derivative":
        sigil["parent_refs"] = result.nearest_ids
        corpus.append(sigil)
    else:  # "duplicate" or "redundant"
        lineage_log.append(result.as_rejection_entry())
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .embedder import SigilEmbedder


# ---------------------------------------------------------------------------
# DedupStatus
# ---------------------------------------------------------------------------


class DedupStatus(str, Enum):
    NOVEL = "novel"               # Layer 1+2+3 all clear
    DERIVATIVE = "derivative"     # Layer 2: 0.90–0.97 semantic similarity
    DUPLICATE = "duplicate"       # Layer 1 (hash collision) or Layer 2 (>0.97)
    REDUNDANT = "redundant"       # Layer 3: structural fingerprint match


@dataclass
class DedupResult:
    status: DedupStatus
    sigil_hash: str
    nearest_similarity: float = 0.0
    nearest_ids: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.status in (DedupStatus.NOVEL, DedupStatus.DERIVATIVE)

    def as_rejection_entry(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "hash": self.sigil_hash,
            "nearest_similarity": round(self.nearest_similarity, 4),
            "nearest_ids": self.nearest_ids,
            "reason": self.rejection_reason,
        }


# ---------------------------------------------------------------------------
# SigilDeduplicator
# ---------------------------------------------------------------------------


class SigilDeduplicator:
    """
    3-layer deduplication gate for VoxSigil corpus construction.

    Thread-note: NOT thread-safe. Wrap with a lock if calling from multiple
    generator threads (CognitiveCycleEngine handles this).
    """

    SEMANTIC_REJECT_THRESHOLD: float = 0.97
    SEMANTIC_DERIVATIVE_THRESHOLD: float = 0.90

    def __init__(
        self,
        semantic_reject: float = SEMANTIC_REJECT_THRESHOLD,
        semantic_derivative: float = SEMANTIC_DERIVATIVE_THRESHOLD,
    ) -> None:
        self._embedder = SigilEmbedder()
        self.semantic_reject = semantic_reject
        self.semantic_derivative = semantic_derivative

        # Layer 1: hash set
        self._hashes: dict[str, str] = {}  # hash → sigil_id

        # Layer 2: embedding store for cosine comparisons
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []

        # Layer 3: structural fingerprint set
        self._struct_fingerprints: dict[str, str] = {}  # fp → sigil_id

        self.stats = {
            "checked": 0,
            "novel": 0,
            "derivative": 0,
            "duplicate_hash": 0,
            "duplicate_semantic": 0,
            "redundant": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, sigil: Dict[str, Any]) -> DedupResult:
        """
        Run all three deduplication layers and return a DedupResult.
        Does NOT mutate the corpus; call ``register()`` to commit a sigil.
        """
        self.stats["checked"] += 1
        sig_hash = _compute_hash(sigil)
        sig_id = SigilEmbedder.sigil_id(sigil)

        # --- Layer 1: hard hash ----------------------------------------
        if sig_hash in self._hashes:
            self.stats["duplicate_hash"] += 1
            return DedupResult(
                status=DedupStatus.DUPLICATE,
                sigil_hash=sig_hash,
                nearest_ids=[self._hashes[sig_hash]],
                rejection_reason="hash_collision",
            )

        # --- Layer 2: semantic embedding similarity --------------------
        if self._vectors:
            vec = self._embedder.embed(sigil)
            sim, nearest_id = self._max_cosine(vec)

            if sim >= self.semantic_reject:
                self.stats["duplicate_semantic"] += 1
                return DedupResult(
                    status=DedupStatus.DUPLICATE,
                    sigil_hash=sig_hash,
                    nearest_similarity=sim,
                    nearest_ids=[nearest_id],
                    rejection_reason=f"semantic_similarity={sim:.4f}>={self.semantic_reject}",
                )

            if sim >= self.semantic_derivative:
                self.stats["derivative"] += 1
                return DedupResult(
                    status=DedupStatus.DERIVATIVE,
                    sigil_hash=sig_hash,
                    nearest_similarity=sim,
                    nearest_ids=[nearest_id],
                    rejection_reason=None,
                )
        else:
            sim = 0.0

        # --- Layer 3: structural redundancy ----------------------------
        struct_fp = _structural_fingerprint(sigil)
        if struct_fp and struct_fp in self._struct_fingerprints:
            self.stats["redundant"] += 1
            return DedupResult(
                status=DedupStatus.REDUNDANT,
                sigil_hash=sig_hash,
                nearest_similarity=sim,
                nearest_ids=[self._struct_fingerprints[struct_fp]],
                rejection_reason="structural_fingerprint_match",
            )

        self.stats["novel"] += 1
        return DedupResult(
            status=DedupStatus.NOVEL,
            sigil_hash=sig_hash,
            nearest_similarity=sim,
        )

    def register(self, sigil: Dict[str, Any], result: Optional[DedupResult] = None) -> str:
        """
        Commit a sigil to the deduplication corpus (after check() returned accepted).
        Returns the sigil's ID.
        If result is not provided, check() is called internally.
        """
        if result is None:
            result = self.check(sigil)
        if not result.accepted:
            raise ValueError(
                f"Cannot register a {result.status.value} sigil: {result.rejection_reason}"
            )
        sig_id = SigilEmbedder.sigil_id(sigil)
        sig_hash = result.sigil_hash

        self._hashes[sig_hash] = sig_id
        vec = self._embedder.embed(sigil)
        self._ids.append(sig_id)
        self._vectors.append(vec)

        struct_fp = _structural_fingerprint(sigil)
        if struct_fp:
            self._struct_fingerprints[struct_fp] = sig_id

        return sig_id

    def check_and_register(self, sigil: Dict[str, Any]) -> DedupResult:
        """
        Convenience: check and immediately register if accepted.
        """
        result = self.check(sigil)
        if result.accepted:
            self.register(sigil, result)
        return result

    def corpus_size(self) -> int:
        return len(self._ids)

    def summary(self) -> str:
        s = self.stats
        total = s["checked"] or 1
        return (
            f"Deduplicator | {self.corpus_size()} accepted | "
            f"checked={s['checked']}  "
            f"novel={s['novel']}({100*s['novel']//total}%)  "
            f"derivative={s['derivative']}  "
            f"dup_hash={s['duplicate_hash']}  "
            f"dup_semantic={s['duplicate_semantic']}  "
            f"redundant={s['redundant']}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _max_cosine(self, vec: List[float]) -> Tuple[float, str]:
        """Return (max_cosine_similarity, corresponding_sigil_id)."""
        best_sim = -1.0
        best_id = ""
        for sid, v in zip(self._ids, self._vectors):
            sim = sum(a * b for a, b in zip(vec, v))  # dot = cosine (unit vecs)
            if sim > best_sim:
                best_sim = sim
                best_id = sid
        return best_sim, best_id


# ---------------------------------------------------------------------------
# Hash computation (Layer 1)
# ---------------------------------------------------------------------------


def _compute_hash(sigil: Dict[str, Any]) -> str:
    """
    SHA-256 hash over:
      - normalized glyph sequence
      - scaffold_type
      - canonicalized principle (lowercased, stripped)

    This guarantees that two sigils with the same structure but different
    names will still collide if their essence is the same.
    """
    # Glyph sequence
    pglyph = sigil.get("pglyph") or sigil.get("glyph_sequence") or []
    if isinstance(pglyph, list):
        glyph_str = "|".join(str(g) for g in pglyph)
    else:
        glyph_str = str(sigil.get("sigil", ""))

    # Scaffold type
    scaffold = sigil.get("scaffold", {})
    if isinstance(scaffold, dict):
        scaffold_type = scaffold.get("scaffold_type", "") or ""
    else:
        scaffold_type = str(scaffold)

    # Principle
    principle = str(sigil.get("principle", "")).lower().strip()

    key = f"{glyph_str}\x00{scaffold_type}\x00{principle}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Structural fingerprint (Layer 3)
# ---------------------------------------------------------------------------


def _tag_vector_fingerprint(typed_tags: Dict[str, Any]) -> str:
    """
    A short fingerprint of the typed_tags by sorting each class's values.
    """
    if not isinstance(typed_tags, dict):
        return ""
    parts = []
    for cls in sorted(typed_tags.keys()):
        vals = typed_tags[cls]
        if isinstance(vals, list):
            parts.append(f"{cls}:{','.join(sorted(str(v) for v in vals))}")
        else:
            parts.append(f"{cls}:{vals}")
    return ";".join(parts)


def _structural_fingerprint(sigil: Dict[str, Any]) -> str:
    """
    Layer-3 fingerprint: scaffold_type + tag_vector + ecosystem_role.
    Returns "" if any component is missing (partial fingerprints are not
    used for structural rejection).
    """
    scaffold = sigil.get("scaffold", {})
    if isinstance(scaffold, dict):
        scaffold_type = scaffold.get("scaffold_type", "") or ""
    else:
        scaffold_type = str(scaffold)

    typed_tags = sigil.get("typed_tags") or {}
    tag_fp = _tag_vector_fingerprint(typed_tags)

    ecosystem_role = str(
        sigil.get("ecosystem_role", "")
        or (sigil.get("biological_identity") or {}).get("ecosystem_role", "")
    ).strip()

    if not scaffold_type or not tag_fp or not ecosystem_role:
        return ""

    key = f"{scaffold_type}\x00{tag_fp}\x00{ecosystem_role}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
