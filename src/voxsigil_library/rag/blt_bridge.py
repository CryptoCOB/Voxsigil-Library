"""
BLT Bridge — Compress, Score, and Filter Retrieved Sigils

The BLT Bridge sits between the retriever and the generator:

  [ Retrieved Sigil Set ]
          ↓
  [ BLTBridge.compress_and_score() ]   ← rank by symbolic quality
          ↓
  [ BLTBridge.filter_illegal() ]        ← hard-remove schema violations
          ↓
  [ BLTBridge.emit_canonical_context() ] ← produce injection payload

The bridge does NOT call an external BLT model at this stage.  It uses the
same deterministic schema rules that BLT was trained on, so scoring and
legality decisions are schema-grounded and reproducible regardless of whether
a trained checkpoint is present.

When a trained BLT checkpoint *is* available, drop it in by subclassing
BLTBridge and overriding `_blt_score()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Import schema validation from the existing bridge so BLT and RAG agree on
# what "valid" means.
try:
    from ..schema_bridge import (
        CANONICAL_SCAFFOLDS,
        ScaffoldType,
        validate_interconnected_schema,
    )
    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# ScoredSigil — carries a sigil plus its BLT quality score
# ---------------------------------------------------------------------------


@dataclass
class ScoredSigil:
    sigil: Dict[str, Any]
    score: float                          # 0.0 (worst) – 1.0 (best)
    violations: List[str] = field(default_factory=list)
    is_legal: bool = True

    def __lt__(self, other: "ScoredSigil") -> bool:
        return self.score < other.score


# ---------------------------------------------------------------------------
# BLT Bridge
# ---------------------------------------------------------------------------


class BLTBridge:
    """
    Deterministic BLT proxy.

    Scores sigils using schema constraints (category limits, glyph counts,
    scaffold rules, tag completeness) and filters out illegal ones before
    they reach the generator.

    Override ``_blt_score()`` to plug in a real BLT model embedding.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.30,
        penalise_missing_tags: bool = True,
        penalise_entropy_excess: bool = True,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.penalise_missing_tags = penalise_missing_tags
        self.penalise_entropy_excess = penalise_entropy_excess

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_and_score(
        self, sigils: List[Dict[str, Any]]
    ) -> List[ScoredSigil]:
        """
        Score each sigil.  Does NOT drop any; call ``filter_illegal`` after.
        Returns the list sorted best-first.
        """
        scored = [self._evaluate(s) for s in sigils]
        scored.sort(reverse=True)
        return scored

    def filter_illegal(
        self, scored: List[ScoredSigil]
    ) -> List[ScoredSigil]:
        """
        Remove sigils that fail schema validation or score below threshold.
        """
        return [
            s for s in scored
            if s.is_legal and s.score >= self.min_score_threshold
        ]

    def emit_canonical_context(
        self, scored: List[ScoredSigil]
    ) -> Dict[str, Any]:
        """
        Package the scored sigil set into a generator-injectable context dict.

        The output preserves all lineage and lifecycle tags and includes a
        BLT-level quality summary.
        """
        canonical = [s.sigil for s in scored]
        total = len(canonical)
        avg_score = (
            sum(s.score for s in scored) / total if total else 0.0
        )

        return {
            "sigils": canonical,
            "count": total,
            "avg_blt_score": round(avg_score, 4),
            "scaffold_types": list(
                {
                    _get_scaffold(s)
                    for s in canonical
                    if _get_scaffold(s)
                }
            ),
            "blt_filtered": True,
        }

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _evaluate(self, sigil: Dict[str, Any]) -> ScoredSigil:
        violations: List[str] = []
        score = 1.0

        # ---- Glyph count check ----------------------------------------
        glyphs = _extract_glyphs(sigil)
        n_glyphs = len(glyphs)
        if n_glyphs < 1:
            violations.append("no_glyphs")
            score -= 0.30
        elif n_glyphs > 11:
            violations.append(f"too_many_glyphs:{n_glyphs}")
            score -= 0.20

        # ---- Scaffold rule check --------------------------------------
        scaffold_type = _get_scaffold(sigil)
        if scaffold_type and _SCHEMA_AVAILABLE:
            rule = CANONICAL_SCAFFOLDS.get(scaffold_type)
            if rule:
                score_delta, new_violations = _check_scaffold_rule(sigil, rule)
                score += score_delta
                violations.extend(new_violations)
        elif not scaffold_type:
            violations.append("missing_scaffold_type")
            score -= 0.15

        # ---- Tag completeness -----------------------------------------
        if self.penalise_missing_tags:
            missing = _check_required_tags(sigil)
            if missing:
                violations.extend([f"missing_tag:{t}" for t in missing])
                score -= 0.05 * len(missing)

        # ---- Deprecated / sealed lifecycle ----------------------------
        lifecycle = _get_lifecycle(sigil)
        if "deprecated" in lifecycle or "sealed" in lifecycle:
            violations.append(f"lifecycle:{lifecycle}")
            score -= 0.50

        # ---- Schema-level validation (full check) ----------------------
        if _SCHEMA_AVAILABLE:
            try:
                ok, errs = validate_interconnected_schema(sigil)
                if not ok:
                    violations.extend(errs[:5])  # cap error list
                    score -= 0.10 * min(len(errs), 5)
            except Exception:
                pass  # validation errors should never crash the pipeline

        # ---- Optional model score override ----------------------------
        model_bonus = self._blt_score(sigil)
        score += model_bonus

        score = max(0.0, min(1.0, score))
        is_legal = len([v for v in violations if _is_fatal(v)]) == 0

        return ScoredSigil(
            sigil=sigil,
            score=score,
            violations=violations,
            is_legal=is_legal,
        )

    def _blt_score(self, sigil: Dict[str, Any]) -> float:
        """
        Hook for subclasses to add a trained-model quality score.
        Returns a delta in [-0.2, +0.2].  Default: 0.0.
        """
        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_scaffold(sigil: Dict[str, Any]) -> str:
    scaffold = sigil.get("scaffold", {})
    if isinstance(scaffold, dict):
        return scaffold.get("scaffold_type", "") or ""
    if isinstance(scaffold, str):
        return scaffold
    return ""


def _get_lifecycle(sigil: Dict[str, Any]) -> List[str]:
    typed_tags = sigil.get("typed_tags", {})
    if isinstance(typed_tags, dict):
        lc = typed_tags.get("lifecycle", [])
        if isinstance(lc, list):
            return lc
    return []


def _extract_glyphs(sigil: Dict[str, Any]) -> List[str]:
    pglyph = sigil.get("pglyph") or sigil.get("glyph_sequence") or []
    if isinstance(pglyph, list) and pglyph:
        return [str(g) for g in pglyph]
    s = sigil.get("sigil", "")
    if isinstance(s, str) and s:
        return list(s)
    return []


_REQUIRED_TAG_CLASSES = ["domain", "function", "lifecycle"]


def _check_required_tags(sigil: Dict[str, Any]) -> List[str]:
    typed_tags = sigil.get("typed_tags", {})
    if not isinstance(typed_tags, dict):
        return _REQUIRED_TAG_CLASSES[:]
    missing = []
    for cls in _REQUIRED_TAG_CLASSES:
        val = typed_tags.get(cls)
        if not val:
            missing.append(cls)
    return missing


def _check_scaffold_rule(
    sigil: Dict[str, Any], rule: Any
) -> Tuple[float, List[str]]:
    """Return (score_delta, violations)."""
    glyphs = _extract_glyphs(sigil)
    violations: List[str] = []
    delta = 0.0

    constraints = rule.constraints if _SCHEMA_AVAILABLE else {}

    glyph_min = constraints.get("glyph_min", 1)
    glyph_max = constraints.get("glyph_max", 11)

    if len(glyphs) < glyph_min:
        violations.append(f"glyph_below_min:{glyph_min}")
        delta -= 0.10
    if len(glyphs) > glyph_max:
        violations.append(f"glyph_above_max:{glyph_max}")
        delta -= 0.10

    # Assembly must not act directly.
    if getattr(rule, "scaffold_type", None) and str(rule.scaffold_type) == "assembly":
        if sigil.get("acts_directly"):
            violations.append("assembly_acts_directly")
            delta -= 0.25

    # Flow must be ordered.
    if getattr(rule, "scaffold_type", None) and str(rule.scaffold_type) == "flow":
        flow_def = sigil.get("flow_definition", {})
        if isinstance(flow_def, dict) and not flow_def.get("ordered", True):
            violations.append("flow_unordered")
            delta -= 0.15

    return delta, violations


_FATAL_PREFIXES = (
    "no_glyphs",
    "assembly_acts_directly",
    "lifecycle:deprecated",
    "lifecycle:sealed",
)


def _is_fatal(violation: str) -> bool:
    return any(violation.startswith(p) for p in _FATAL_PREFIXES)
