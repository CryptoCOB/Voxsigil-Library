"""
Tests for SymbolicRAGMiddleware — specifically the negative / adversarial paths.

Covers:
  - All-illegal sigil set: retrieve_and_enrich() returns empty but valid context
  - All-illegal sigil set: run_pipeline() does not crash
  - All-illegal sigil set: lineage still records the attempt
  - Mixed legal/illegal: only legal sigils pass through
  - Entropy budget: sigils above budget are excluded at retrieval
  - Deprecated/sealed lifecycle: always filtered regardless of score
"""

from __future__ import annotations

import sys
import os

# Ensure src is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


from src.voxsigil_library.rag import (
    BLTBridge,
    QueryContext,
    SigilEmbedder,
    SymbolicRAGMiddleware,
)
from src.voxsigil_library.rag.blt_bridge import ScoredSigil
from src.voxsigil_library.rag.retriever import NumpyRetriever


# ---------------------------------------------------------------------------
# Fixtures — canonical illegal sigil catalogue
# ---------------------------------------------------------------------------

def _illegal_sigils():
    """
    A collection of sigils that should ALL fail BLT legality checks.
    """
    return [
        # 1. Zero glyphs
        {
            "sigil": "",
            "name": "empty-sigil",
            "scaffold": {"scaffold_type": "flow"},
            "typed_tags": {
                "domain": ["cognition"], "function": ["evaluate"],
                "lifecycle": ["active"],
            },
            "principle": "nothing",
        },
        # 2. Deprecated lifecycle
        {
            "sigil": "∀∃",
            "name": "deprecated-logic",
            "scaffold": {"scaffold_type": "identity"},
            "typed_tags": {
                "domain": ["governance"], "function": ["bind"],
                "lifecycle": ["deprecated"],
            },
            "principle": "should be rejected",
        },
        # 3. Sealed lifecycle
        {
            "sigil": "ᚠᚢ",
            "name": "sealed-proto",
            "scaffold": {"scaffold_type": "identity"},
            "typed_tags": {
                "domain": ["identity"], "function": ["observe"],
                "lifecycle": ["sealed"],
            },
            "principle": "sealed — should never be retrieved",
        },
        # 4. Assembly that acts directly (scaffold violation)
        {
            "sigil": "∑Δ",
            "name": "acting-assembly",
            "scaffold": {"scaffold_type": "assembly"},
            "acts_directly": True,
            "typed_tags": {
                "domain": ["market"], "function": ["orchestrate"],
                "lifecycle": ["active"],
            },
            "principle": "assembly cannot act directly",
        },
    ]


def _legal_sigils():
    return [
        {
            "sigil": "∀∃",
            "name": "logic-binder",
            "scaffold": {"scaffold_type": "identity", "allowed_categories": ["LOGIC"]},
            "typed_tags": {
                "domain": ["governance"], "function": ["bind"],
                "temporal": ["static"], "lifecycle": ["active"],
                "epistemic": ["verified"], "polarity": ["neutral"],
            },
            "principle": "logical binding construct",
            "entropy": 0.1,
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mw(sigils):
    """Build SymbolicRAGMiddleware from a sigil list (numpy backend, no faiss needed)."""
    retriever = NumpyRetriever()
    retriever.index(sigils)
    bridge = BLTBridge(min_score_threshold=0.30)
    return SymbolicRAGMiddleware(retriever=retriever, blt_bridge=bridge)


# ---------------------------------------------------------------------------
# Negative tests — all sigils are illegal
# ---------------------------------------------------------------------------

def test_all_illegal_retrieve_and_enrich_returns_empty_valid_context():
    """
    When every indexed sigil is illegal, retrieve_and_enrich() must return a
    context dict that is structurally valid (not empty dict, not an exception)
    but contains zero sigils.
    """
    mw = _build_mw(_illegal_sigils())
    ctx = mw.retrieve_and_enrich(
        QueryContext(scaffold_type="flow", entropy_budget=1.0),
        top_k=10,
    )

    # Context must always be a dict with mandatory keys
    assert isinstance(ctx, dict), "Context must be a dict"
    assert "sigils" in ctx, "Context must have 'sigils' key"
    assert "count" in ctx, "Context must have 'count' key"
    assert "avg_blt_score" in ctx, "Context must have 'avg_blt_score' key"
    assert "blt_filtered" in ctx, "Context must have 'blt_filtered' key"

    # No illegal sigils pass through
    assert ctx["count"] == 0, f"Expected 0 legal sigils, got {ctx['count']}"
    assert ctx["sigils"] == [], "Expected empty sigil list"
    assert ctx["blt_filtered"] is True


def test_all_illegal_run_pipeline_does_not_crash():
    """
    run_pipeline() must complete without raising when all sigils are illegal.
    """
    mw = _build_mw(_illegal_sigils())

    generator_called_with = []

    def capture_generator(ctx):
        generator_called_with.append(ctx)
        return None

    result = mw.run_pipeline(
        QueryContext(scaffold_type="identity", entropy_budget=1.0),
        generator_fn=capture_generator,
    )

    # Must not crash and must return a PipelineResult
    assert result is not None
    assert result.legal_count == 0, f"Expected 0 legal, got {result.legal_count}"
    assert result.generator_output is None
    # Generator was still called (with empty context)
    assert len(generator_called_with) == 1


def test_all_illegal_lineage_records_the_attempt():
    """
    Even when zero sigils are legal, the pipeline run MUST be logged in the
    LineageStore.  This is mandatory for BLT to learn what not to compress.
    """
    mw = _build_mw(_illegal_sigils())

    initial_count = len(mw.lineage._log)
    mw.run_pipeline(
        QueryContext(scaffold_type="mutation", entropy_budget=1.0),
        lineage_tag="negative-test",
    )

    assert len(mw.lineage._log) == initial_count + 1, "Lineage must record the attempt"
    entry = mw.lineage._log[-1]
    assert entry["legal"] == 0, f"Expected legal=0 in lineage, got {entry}"
    assert entry.get("tag") == "negative-test"


# ---------------------------------------------------------------------------
# Negative tests — mixed legal / illegal
# ---------------------------------------------------------------------------

def test_mixed_only_legal_pass_through():
    """
    In a mixed pool, only legal sigils reach the generator context.
    """
    mixed = _illegal_sigils() + _legal_sigils()
    mw = _build_mw(mixed)

    ctx = mw.retrieve_and_enrich(
        QueryContext(entropy_budget=1.0),
        top_k=20,
    )

    for s in ctx["sigils"]:
        lc = (s.get("typed_tags") or {}).get("lifecycle", [])
        assert "deprecated" not in lc, "Deprecated sigil passed filter"
        assert "sealed" not in lc, "Sealed sigil passed filter"
        assert s.get("sigil", "") != "", "Empty-sigil passed filter"


# ---------------------------------------------------------------------------
# Entropy budget gate
# ---------------------------------------------------------------------------

def test_entropy_budget_excludes_high_entropy_sigils():
    """
    Sigils with entropy > budget must not be returned.
    """
    sigils = [
        {
            "sigil": "∞σ",
            "name": "high-entropy",
            "scaffold": {"scaffold_type": "mutation"},
            "typed_tags": {
                "domain": ["cognition"], "function": ["evaluate"],
                "lifecycle": ["active"],
            },
            "principle": "high entropy mutation",
            "entropy": 0.95,
        },
        {
            "sigil": "∀",
            "name": "low-entropy",
            "scaffold": {"scaffold_type": "identity"},
            "typed_tags": {
                "domain": ["governance"], "function": ["bind"],
                "lifecycle": ["active"],
            },
            "principle": "low entropy identity",
            "entropy": 0.1,
        },
    ]
    mw = _build_mw(sigils)

    ctx = mw.retrieve_and_enrich(
        QueryContext(entropy_budget=0.5),
        top_k=10,
    )

    names = [s.get("name") for s in ctx["sigils"]]
    assert "high-entropy" not in names, "High-entropy sigil passed budget filter"


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------

def test_context_structure_invariants():
    """
    Context dict must always have the same shape regardless of result count.
    """
    for sigil_set in [[], _illegal_sigils(), _legal_sigils()]:
        mw = _build_mw(sigil_set)
        ctx = mw.retrieve_and_enrich(QueryContext())
        required_keys = {"sigils", "count", "avg_blt_score", "blt_filtered", "query_meta"}
        missing = required_keys - ctx.keys()
        assert not missing, f"Missing context keys {missing} for set size {len(sigil_set)}"
        assert isinstance(ctx["sigils"], list)
        assert isinstance(ctx["count"], int)
        assert ctx["count"] == len(ctx["sigils"])
        assert ctx["blt_filtered"] is True


def test_pipeline_with_no_generator_fn_still_succeeds():
    """
    run_pipeline() without a generator_fn must not crash and must return None
    for generator_output.
    """
    mw = _build_mw(_legal_sigils())
    result = mw.run_pipeline(QueryContext(scaffold_type="identity"))
    assert result.generator_output is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_all_illegal_retrieve_and_enrich_returns_empty_valid_context,
        test_all_illegal_run_pipeline_does_not_crash,
        test_all_illegal_lineage_records_the_attempt,
        test_mixed_only_legal_pass_through,
        test_entropy_budget_excludes_high_entropy_sigils,
        test_context_structure_invariants,
        test_pipeline_with_no_generator_fn_still_succeeds,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed", "✓" if failed == 0 else "✗")
    raise SystemExit(0 if failed == 0 else 1)
