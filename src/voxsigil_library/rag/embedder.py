"""
VoxSigil Sigil Embedding Pipeline

Produces deterministic 768D unit-normalized embeddings for sigil objects.
No external models required — all projections are hash-based so they are
stable across environments and consistent with BLT's 768D space.

Embedding layout (3 × 256D sub-encodings):
  [0:256]   — Glyph sequence  (ordered glyph characters → sha256 projection)
  [256:512] — Category distribution  (8 canonical categories, count-normalized)
  [512:768] — Scaffold + tags  (scaffold_type + flattened typed_tag values → sha256 projection)

L2-normalized at the end so cosine similarity == dot product.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Optional

import math

# Canonical glyph categories in a fixed order for the category sub-encoding.
_CATEGORIES: List[str] = [
    "PROTO",
    "ASTRAL",
    "LOGIC",
    "PHYSICS",
    "NOETIC",
    "STRUCTURAL",
    "ENTROPY",
    "EMERGENCE",
]
_CAT_INDEX: Dict[str, int] = {c: i for i, c in enumerate(_CATEGORIES)}

_EMBED_DIM = 768
_SUB_DIM = 256  # each of the 3 sub-encodings


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sha256_to_floats(text: str, n: int) -> List[float]:
    """
    Project an arbitrary string to an `n`-dimensional float vector via
    repeated SHA-256 hashing (deterministic, uniform-ish).

    Each 4-byte word of the digest becomes one float in [-1.0, 1.0].
    We hash iteratively until we have enough words.
    """
    blob = text.encode("utf-8")
    values: List[float] = []
    seed = blob
    while len(values) < n:
        digest = hashlib.sha256(seed).digest()
        seed = digest  # chain: next hash uses previous digest as input
        for i in range(0, len(digest) - 3, 4):
            word = struct.unpack_from(">i", digest, i)[0]  # signed int32
            values.append(word / 2_147_483_648.0)  # normalise to (-1, 1)
    return values[:n]


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return vec
    inv = 1.0 / norm
    return [v * inv for v in vec]


def _add_vecs(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


# ---------------------------------------------------------------------------
# Glyph sub-encoding  [0:256]
# ---------------------------------------------------------------------------


def _encode_glyphs(glyphs: List[str]) -> List[float]:
    """
    Encode an ordered glyph sequence to a 256D vector.

    The full sequence is serialized as a pipe-separated string and then
    SHA-256-projected.  Order is preserved.
    """
    sequence_text = "|".join(g for g in glyphs if isinstance(g, str))
    if not sequence_text:
        sequence_text = "__empty__"
    return _sha256_to_floats(sequence_text, _SUB_DIM)


# ---------------------------------------------------------------------------
# Category distribution sub-encoding  [256:512]
# ---------------------------------------------------------------------------


def _encode_categories(glyphs: List[str], sigil_doc: Dict[str, Any]) -> List[float]:
    """
    Encode the category distribution of a sigil as a 256D vector.

    Strategy:
    1. Count how many glyphs fall into each of the 8 canonical categories
       (looked up via 'glyph_categories' field if present, else inferred
       from sigil metadata).
    2. Normalise counts to [0, 1].
    3. Pad/repeat to 256D via linear interpolation of the 8 values.
    """
    counts = [0.0] * len(_CATEGORIES)

    # Try to read per-glyph category annotations if stored on the sigil.
    glyph_cats: Dict[str, str] = {}
    if isinstance(sigil_doc.get("glyph_categories"), dict):
        glyph_cats = sigil_doc["glyph_categories"]

    # Also check scaffold-level allowed/forbidden categories for bias.
    scaffold = sigil_doc.get("scaffold", {})
    if isinstance(scaffold, dict):
        for cat in scaffold.get("allowed_categories", []):
            idx = _CAT_INDEX.get(cat.upper())
            if idx is not None:
                counts[idx] += 0.5

    # Per-glyph counts.
    for g in glyphs:
        cat = glyph_cats.get(g, "").upper()
        idx = _CAT_INDEX.get(cat)
        if idx is not None:
            counts[idx] += 1.0

    # Normalise category vector.
    total = sum(counts)
    if total > 0:
        counts = [c / total for c in counts]

    # Expand 8-element cat vector → 256D via periodic tiling and modulation.
    n = len(_CATEGORIES)  # 8
    expanded: List[float] = []
    for i in range(_SUB_DIM):
        base_idx = i % n
        # Introduce gentle frequency modulation so all 256 dims are distinct.
        phase = math.pi * 2.0 * i / _SUB_DIM
        val = counts[base_idx] * math.cos(phase) + counts[(base_idx + 1) % n] * math.sin(phase)
        expanded.append(val)

    return expanded


# ---------------------------------------------------------------------------
# Scaffold + tags sub-encoding  [512:768]
# ---------------------------------------------------------------------------


def _encode_scaffold_tags(sigil_doc: Dict[str, Any]) -> List[float]:
    """
    Encode scaffold type and typed_tags into a 256D vector.

    Serialisation:
      "<scaffold_type>;<tag_class>:<value>,<value>;..."
    Then SHA-256-projected.
    """
    scaffold = sigil_doc.get("scaffold", {})
    scaffold_type = ""
    if isinstance(scaffold, dict):
        scaffold_type = scaffold.get("scaffold_type", "") or ""
    elif isinstance(scaffold, str):
        scaffold_type = scaffold

    parts = [f"scaffold:{scaffold_type}"]

    typed_tags = sigil_doc.get("typed_tags", {})
    if isinstance(typed_tags, dict):
        for cls in sorted(typed_tags.keys()):
            vals = typed_tags[cls]
            if isinstance(vals, list):
                parts.append(f"{cls}:{','.join(sorted(str(v) for v in vals))}")
            else:
                parts.append(f"{cls}:{vals}")

    # Also fold in top-level tags list if present.
    top_tags = sigil_doc.get("tags", [])
    if isinstance(top_tags, list) and top_tags:
        parts.append("tags:" + ",".join(sorted(str(t) for t in top_tags)))

    text = ";".join(parts) or "__no_scaffold__"
    return _sha256_to_floats(text, _SUB_DIM)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SigilEmbedder:
    """
    Deterministic 768D embedder for VoxSigil sigil objects.

    Usage::

        embedder = SigilEmbedder()
        vec = embedder.embed(sigil_dict)         # List[float], len=768
        vecs = embedder.embed_batch(sigil_list)  # List[List[float]]
    """

    embed_dim: int = _EMBED_DIM

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def embed(self, sigil: Dict[str, Any]) -> List[float]:
        """
        Embed a single sigil dict into a 768D unit vector.

        Args:
            sigil: Any sigil dict (schema 1.4 through 2.0-omega).

        Returns:
            768D float list, L2-normalised.
        """
        glyphs = self._extract_glyphs(sigil)

        glyph_vec = _encode_glyphs(glyphs)
        cat_vec = _encode_categories(glyphs, sigil)
        scaffold_vec = _encode_scaffold_tags(sigil)

        combined = glyph_vec + cat_vec + scaffold_vec  # 768D
        return _l2_normalize(combined)

    def embed_batch(self, sigils: List[Dict[str, Any]]) -> List[List[float]]:
        """Embed a list of sigils. Returns embeddings in the same order."""
        return [self.embed(s) for s in sigils]

    # ------------------------------------------------------------------
    # Sigil ID helper (stable, hashable)
    # ------------------------------------------------------------------

    @staticmethod
    def sigil_id(sigil: Dict[str, Any]) -> str:
        """Return a stable 16-char hex ID for a sigil (SHA-256 of key fields)."""
        key = "|".join([
            str(sigil.get("sigil", "")),
            str(sigil.get("name", "")),
            str(sigil.get("principle", "")),
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_glyphs(sigil: Dict[str, Any]) -> List[str]:
        """Pull glyph characters from a sigil in schema-agnostic way."""
        # Try explicit pglyph field (list of chars).
        pglyph = sigil.get("pglyph") or sigil.get("glyph_sequence") or []
        if isinstance(pglyph, list) and pglyph:
            return [str(g) for g in pglyph]

        # Try the 'sigil' field (a string of glyph chars).
        sigil_str = sigil.get("sigil", "")
        if isinstance(sigil_str, str) and sigil_str:
            return list(sigil_str)

        # Fallback: concatenate all string values up to 11 chars.
        chars: List[str] = []
        for val in sigil.values():
            if isinstance(val, str):
                chars.extend(list(val))
            if len(chars) >= 11:
                break
        return chars[:11]
