from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.voxsigil_library.rag import (
    BLTBridge,
    FeedbackVerdict,
    QueryContext,
    SymbolicRAGMiddleware,
)
from src.voxsigil_library.rag.retriever import NumpyRetriever


def _sigil(name: str, sigil: str, scaffold_type: str = "identity", entropy: float = 0.2):
    return {
        "name": name,
        "sigil": sigil,
        "scaffold": {"scaffold_type": scaffold_type},
        "typed_tags": {
            "domain": ["cognition"],
            "function": ["evaluate"],
            "polarity": ["neutral"],
            "temporal": ["evolving"],
            "epistemic": ["probabilistic"],
            "lifecycle": ["active"],
        },
        "principle": f"principle for {name}",
        "entropy": entropy,
    }


def _mw(sigils):
    r = NumpyRetriever()
    r.index(sigils)
    return SymbolicRAGMiddleware(retriever=r, blt_bridge=BLTBridge(min_score_threshold=0.30))


def test_feedback_signal_updates_weights_and_lineage():
    s1 = _sigil("alpha", "∀∃")
    s2 = _sigil("beta", "ᚠᚢ")
    mw = _mw([s1, s2])

    # Apply explicit feedback to the first artifact.
    entry = mw.apply_user_feedback([s1], FeedbackVerdict.APPROVE, note="good outcome")

    assert entry["feedback"] == "approve"
    assert entry["count"] == 1
    assert len(mw.lineage.recent(1)) == 1
    assert entry["artifacts"][0]["weight"] > 0.0


def test_output_gate_compresses_generated_payload():
    good = _sigil("good", "∀∃")
    bad = _sigil("bad", "", scaffold_type="flow")  # illegal: empty sigil
    mw = _mw([good])

    result = mw.run_pipeline(
        QueryContext(scaffold_type="identity"),
        generator_fn=lambda _ctx: [good, bad],
    )

    assert result.compressed_output is not None
    assert result.compressed_output["count"] == 1
    assert result.compressed_output["sigils"][0]["name"] == "good"


def test_feedback_affects_retrieval_score_ordering():
    # Two near-equivalent candidates.
    a = _sigil("favored", "∀∃")
    b = _sigil("other", "∑Δ")

    mw = _mw([a, b])

    # Baseline retrieval
    ctx0 = mw.retrieve_and_enrich(QueryContext(intent="evaluate cognition"), top_k=2)
    names0 = [x.get("name") for x in ctx0["sigils"]]

    # Strongly down-vote current first, up-vote second.
    if names0 and names0[0] == "favored":
        mw.apply_user_feedback([a], FeedbackVerdict.REJECT)
        mw.apply_user_feedback([b], FeedbackVerdict.APPROVE)
    else:
        mw.apply_user_feedback([b], FeedbackVerdict.REJECT)
        mw.apply_user_feedback([a], FeedbackVerdict.APPROVE)

    ctx1 = mw.retrieve_and_enrich(QueryContext(intent="evaluate cognition"), top_k=2)
    by_name = {x.get("name"): x for x in ctx1["sigils"]}

    assert "favored" in by_name
    assert "other" in by_name

    favored_w = by_name["favored"].get("feedback_weight", 0.0)
    other_w = by_name["other"].get("feedback_weight", 0.0)
    assert favored_w != other_w
