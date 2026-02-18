"""
VoxSigil Cognitive Loop Engine

A closed-loop symbolic evolution engine where:

  Generators (Ollama, LM Studio) act under PARTIAL perception
  RAG is the sensory cortex (retrieves the statistical silhouette)
  BLT is the metabolic + immune system (scores / filters)
  LineageStore is memory + heredity (records every proposal)
  Schema is DNA constraints (hard rules never change)

The key principle is STIGMERGY, not conversation:
  — generators never talk to each other
  — generators never see the full corpus
  — generators see only a summarized cognitive niche (QueryContext)
  — the environment (RAG index + BLT scores) guides next actions

Architecture::

    CognitiveCycleEngine
      ├── RAG (perception)
      ├── GeneratorScheduler
      │     ├── OllamaGenerator   (bulk, low-entropy, temp=0.3)
      │     └── LMStudioGenerator (refinement, high-structure, temp=0.5)
      ├── BLTBridge (survival gate)
      ├── SigilDeduplicator (3-layer uniqueness)
      ├── LineageStore (heredity — stores ALL proposals, even rejected)
      └── WorldView (scaffold saturation, entropy drift, novelty yield)

Usage::

    from src.voxsigil_library.rag.cognitive_loop import (
        CognitiveCycleEngine, GeneratorConfig,
    )

    engine = CognitiveCycleEngine.create(
        generators=[
            GeneratorConfig(kind="ollama", model="llama3.2:latest", role="bulk"),
            GeneratorConfig(kind="lmstudio", model="llama-3.2-3b", role="refine"),
        ],
        update_worldview_every=25,
    )
    engine.run(target_corpus_size=1000, output_path="training/datasets/cognitive.jsonl")
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests

from .blt_bridge import BLTBridge
from .deduplicator import DedupStatus, SigilDeduplicator
from .embedder import SigilEmbedder
from .middleware import LineageStore, SymbolicRAGMiddleware
from .retriever import QueryContext


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for one generator (Ollama or LM Studio)."""

    kind: str                          # "ollama" | "lmstudio"
    model: str                         # model name
    role: str = "bulk"                 # "bulk" | "refine"
    base_url: str = ""                 # auto-set from kind if empty
    temperature: float = 0.0           # auto-set from role if 0
    max_tokens: int = 1024
    timeout: int = 60

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = (
                "http://localhost:11434"
                if self.kind == "ollama"
                else "http://localhost:1234"
            )
        if self.temperature == 0.0:
            self.temperature = 0.3 if self.role == "bulk" else 0.5


@dataclass
class ProposalRecord:
    """Single generator proposal — stored in lineage regardless of outcome."""

    proposal_id: str
    generator_kind: str
    generator_model: str
    role: str
    scaffold_type: str
    accepted: bool
    dedup_status: str
    blt_score: float
    duration_ms: float
    cycle: int
    timestamp: float = field(default_factory=time.time)
    parent_refs: List[str] = field(default_factory=list)
    context_hash: str = ""
    rejection_reason: Optional[str] = None


@dataclass
class WorldView:
    """
    Statistical snapshot of the evolving sigil population.

    Updated every N cycles.  This is what shapes the next QueryContext —
    generators never receive this directly.
    """

    scaffold_counts: Dict[str, int] = field(default_factory=dict)
    scaffold_saturated: Dict[str, bool] = field(default_factory=dict)
    entropy_mean: float = 0.5
    entropy_std: float = 0.2
    novelty_yield: float = 1.0           # recent novel / recent checked
    rejection_rate: Dict[str, float] = field(default_factory=dict)  # per generator
    cycle: int = 0

    def scaffold_gap(self, targets: Dict[str, int]) -> List[str]:
        """Scaffold types that still have room for new entries."""
        gaps = []
        for stype, target in targets.items():
            current = self.scaffold_counts.get(stype, 0)
            if current < target:
                gaps.append(stype)
        return gaps

    def entropy_target(self) -> float:
        """
        Recommend an entropy budget for the next generator query.
        If corpus is mostly low-entropy, request higher entropy to diversify.
        """
        if self.entropy_mean < 0.3:
            return min(0.9, self.entropy_mean + 3 * self.entropy_std)
        if self.entropy_mean > 0.7:
            return max(0.2, self.entropy_mean - 2 * self.entropy_std)
        return self.entropy_mean + self.entropy_std


# ---------------------------------------------------------------------------
# Generator protocol (internal)
# ---------------------------------------------------------------------------


class BaseGenerator:
    """Thin adapter around an LLM endpoint."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Send the prompt to the model, parse the first valid JSON object.
        Returns None on timeout, connection error, or parse failure.
        """
        try:
            if self.config.kind == "ollama":
                return self._ollama(prompt)
            if self.config.kind == "lmstudio":
                return self._lmstudio(prompt)
        except Exception:
            return None
        return None

    def _ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        resp = requests.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        return _extract_json(text)

    def _lmstudio(self, prompt: str) -> Optional[Dict[str, Any]]:
        resp = requests.post(
            f"{self.config.base_url}/v1/chat/completions",
            json={
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        if not choices:
            return None
        text = choices[0].get("message", {}).get("content", "")
        return _extract_json(text)

    def is_reachable(self) -> bool:
        """Quick health check — does not throw."""
        try:
            if self.config.kind == "ollama":
                r = requests.get(f"{self.config.base_url}/api/tags", timeout=3)
                return r.status_code == 200
            if self.config.kind == "lmstudio":
                r = requests.get(f"{self.config.base_url}/v1/models", timeout=3)
                return r.status_code == 200
        except Exception:
            return False
        return False


# ---------------------------------------------------------------------------
# GeneratorScheduler
# ---------------------------------------------------------------------------


class GeneratorScheduler:
    """
    Selects the next generator for a cycle.

    Strategy:
      - Round-robin between reachable generators by default.
      - If novelty yield drops below 0.20, prefer the higher-quality refine role.
      - If a generator's rejection rate exceeds 0.80, back off (skip every other).
    """

    def __init__(self, generators: List[BaseGenerator]) -> None:
        self._gens = generators
        self._idx = 0
        self._backoff_counts: Dict[int, int] = {}

    def next(self, worldview: WorldView) -> Optional[BaseGenerator]:
        if not self._gens:
            return None

        # Prefer refine role when novelty yield is low and it's available.
        if worldview.novelty_yield < 0.20:
            refiners = [g for g in self._gens if g.config.role == "refine"]
            if refiners:
                return random.choice(refiners)

        # Round-robin with backoff for high-rejection generators.
        attempts = 0
        while attempts < len(self._gens):
            i = self._idx % len(self._gens)
            self._idx += 1
            gen = self._gens[i]

            rej_rate = worldview.rejection_rate.get(gen.config.kind, 0.0)
            backoff = self._backoff_counts.get(i, 0)
            if rej_rate > 0.80 and backoff % 2 == 1:
                self._backoff_counts[i] = backoff + 1
                attempts += 1
                continue

            self._backoff_counts[i] = backoff + 1
            return gen
            attempts += 1

        return self._gens[0]  # fallback


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SCAFFOLD_PROMPTS: Dict[str, str] = {
    "identity": (
        "Generate a VoxSigil IDENTITY organism. "
        "It must define a bound entity: noetic, logical, or astral in character. "
        "It must NOT include STRUCTURAL glyphs."
    ),
    "flow": (
        "Generate a VoxSigil FLOW organism. "
        "It must describe a causal, ordered, non-emergent process. "
        "It must NOT include EMERGENCE glyphs. "
        "Include a flow_definition with ordered=true."
    ),
    "assembly": (
        "Generate a VoxSigil ASSEMBLY organism. "
        "It MUST NOT act directly (acts_directly must be false or absent). "
        "It coordinates multiple parts without becoming an agent itself."
    ),
    "mutation": (
        "Generate a VoxSigil MUTATION organism. "
        "It MUST include exactly one ENTROPY glyph. "
        "Mutation is ENTROPY-mediated change only."
    ),
}


def build_prompt(
    scaffold_type: str,
    forbidden_zones: Optional[List[str]] = None,
    entropy_target: float = 0.5,
    context_sigils: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build the generator prompt for one sigil.

    Generators see:
      - Scaffold gap (what to build)
      - Forbidden zones (what not to build)
      - Entropy target (how exotic)
      - A statistical silhouette of 3 existing sigils' names + principles
        (NOT the full inventory)
    """
    scaffold_hint = _SCAFFOLD_PROMPTS.get(scaffold_type, "Generate a VoxSigil organism.")

    forbidden_block = ""
    if forbidden_zones:
        forbidden_block = (
            "FORBIDDEN GLYPHS / NAMES (do not reproduce):\n"
            + "\n".join(f"  - {z}" for z in forbidden_zones[:10])
            + "\n"
        )

    # Statistical silhouette — 3 examples, name + principle only
    silhouette_block = ""
    if context_sigils:
        sample = context_sigils[:3]
        lines = []
        for s in sample:
            name = s.get("name", "")
            principle = s.get("principle", "")[:80]
            lines.append(f"  [{name}]: {principle}")
        silhouette_block = "EXISTING NICHE (reference only, do not copy):\n" + "\n".join(lines) + "\n"

    return f"""You are a VoxSigil organism generator. Output ONLY valid JSON for one sigil.

TASK: {scaffold_hint}

ENTROPY TARGET: {entropy_target:.2f} (0.0=static, 1.0=chaotic)

{forbidden_block}{silhouette_block}
REQUIRED FIELDS:
  sigil          (string, 1–11 unicode glyph characters)
  name           (unique symbolic name)
  scaffold       {{scaffold_type: "{scaffold_type}"}}
  typed_tags     {{domain, function, polarity, temporal, epistemic, lifecycle}}
  principle      (one sentence, what this organism embodies)
  biological_identity {{ecosystem_role, neural_pattern, metabolic_cost, lifecycle_stage}}
  entropy        (float 0.0–1.0)
  generation_source  "{scaffold_type}"

OUTPUT ONLY THE JSON OBJECT. No prose. No markdown. No explanation.
"""


# ---------------------------------------------------------------------------
# CognitiveCycleEngine
# ---------------------------------------------------------------------------


class CognitiveCycleEngine:
    """
    The closed-loop symbolic evolution engine.

    See module docstring for the full architecture.
    """

    def __init__(
        self,
        generators: List[BaseGenerator],
        middleware: SymbolicRAGMiddleware,
        deduplicator: SigilDeduplicator,
        scaffold_targets: Optional[Dict[str, int]] = None,
        update_worldview_every: int = 25,
    ) -> None:
        self.generators = generators
        self.scheduler = GeneratorScheduler(generators)
        self.middleware = middleware
        self.dedup = deduplicator
        self.scaffold_targets = scaffold_targets or {
            "identity": 200,
            "flow": 200,
            "assembly": 200,
            "mutation": 200,
        }
        self.update_worldview_every = update_worldview_every
        self.worldview = WorldView()
        self._corpus: List[Dict[str, Any]] = []
        self._proposals: List[ProposalRecord] = []
        self._lock = threading.Lock()
        self._cycle = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        generators: Optional[List[GeneratorConfig]] = None,
        scaffold_targets: Optional[Dict[str, int]] = None,
        update_worldview_every: int = 25,
        **kwargs: Any,
    ) -> "CognitiveCycleEngine":
        if generators is None:
            generators = [GeneratorConfig(kind="ollama", model="llama3.2:latest", role="bulk")]

        gen_adapters = [BaseGenerator(cfg) for cfg in generators]

        # Only keep reachable generators.
        reachable = [g for g in gen_adapters if g.is_reachable()]
        if not reachable:
            # If nothing is reachable, keep all (useful for dry-run / testing).
            reachable = gen_adapters

        mw = SymbolicRAGMiddleware(blt_bridge=BLTBridge())
        dedup = SigilDeduplicator()
        return cls(
            generators=reachable,
            middleware=mw,
            deduplicator=dedup,
            scaffold_targets=scaffold_targets,
            update_worldview_every=update_worldview_every,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        target_corpus_size: int = 1000,
        max_cycles: int = 10_000,
        output_path: str = "training/datasets/cognitive.jsonl",
        lineage_path: str = "training/datasets/cognitive_lineage.jsonl",
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the cognitive loop until target_corpus_size is reached or
        max_cycles is exceeded.

        Returns the accepted corpus.
        """
        output_path = str(Path(output_path))
        lineage_path = str(Path(lineage_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        out_file = open(output_path, "w", encoding="utf-8")
        lineage_file = open(lineage_path, "w", encoding="utf-8")

        try:
            while self.dedup.corpus_size() < target_corpus_size and self._cycle < max_cycles:
                self._cycle += 1
                record = self._one_cycle()

                # Write to lineage unconditionally (even rejected proposals).
                lineage_file.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

                if record.accepted:
                    sigil = self._corpus[-1]
                    out_file.write(json.dumps(sigil, ensure_ascii=False) + "\n")

                if verbose and self._cycle % 25 == 0:
                    print(
                        f"Cycle {self._cycle:>5} | "
                        f"corpus={self.dedup.corpus_size()}/{target_corpus_size} | "
                        + self.dedup.summary()
                    )

                if self._cycle % self.update_worldview_every == 0:
                    self._update_worldview()
        finally:
            out_file.close()
            lineage_file.close()

        if verbose:
            print(f"\n✓ Done. {self.dedup.corpus_size()} sigils written to {output_path}")
            print(self.dedup.summary())

        return self._corpus

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def _one_cycle(self) -> ProposalRecord:
        t0 = time.perf_counter()

        # 1. Sense — form a QueryContext from the current worldview.
        scaffold_type = self._choose_scaffold()
        query = QueryContext(
            scaffold_type=scaffold_type,
            entropy_budget=self.worldview.entropy_target(),
            intent=f"generate {scaffold_type} organism",
        )

        # 2. Retrieve context silhouette (not the full corpus).
        context = self.middleware.retrieve_and_enrich(query, top_k=5)
        context_sigils = context.get("sigils", [])
        context_hash = hashlib.sha256(
            json.dumps([s.get("sigil", "") for s in context_sigils], sort_keys=True).encode()
        ).hexdigest()[:12]

        # 3. Build forbidden zones (recently rejected sigil names / glyphs).
        forbidden_zones = self._recent_forbidden_zones()

        # 4. Build prompt — generators see silhouette, not corpus.
        prompt = build_prompt(
            scaffold_type=scaffold_type,
            forbidden_zones=forbidden_zones,
            entropy_target=self.worldview.entropy_target(),
            context_sigils=context_sigils,
        )

        # 5. Select generator.
        gen = self.scheduler.next(self.worldview)
        if gen is None:
            return self._make_record(
                proposal_id=f"no-gen-{self._cycle}",
                generator_kind="none",
                generator_model="none",
                role="none",
                scaffold_type=scaffold_type,
                accepted=False,
                dedup_status="no_generator",
                blt_score=0.0,
                duration_ms=0.0,
                context_hash=context_hash,
                rejection_reason="no_reachable_generator",
            )

        # 6. Generate proposal.
        proposed = gen.generate(prompt)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        proposal_id = f"prop-{self._cycle}-{gen.config.kind}"

        if proposed is None:
            return self._make_record(
                proposal_id=proposal_id,
                generator_kind=gen.config.kind,
                generator_model=gen.config.model,
                role=gen.config.role,
                scaffold_type=scaffold_type,
                accepted=False,
                dedup_status="generation_failed",
                blt_score=0.0,
                duration_ms=duration_ms,
                context_hash=context_hash,
                rejection_reason="generator_returned_none",
            )

        # Stamp metadata before evaluation.
        proposed["generation_source"] = gen.config.kind
        proposed.setdefault("scaffold", {"scaffold_type": scaffold_type})
        proposed.setdefault("entropy", self.worldview.entropy_target())

        # 7. BLT gate.
        scored = self.middleware.blt.compress_and_score([proposed])
        legal = self.middleware.blt.filter_illegal(scored)
        if not legal:
            return self._make_record(
                proposal_id=proposal_id,
                generator_kind=gen.config.kind,
                generator_model=gen.config.model,
                role=gen.config.role,
                scaffold_type=scaffold_type,
                accepted=False,
                dedup_status="blt_rejected",
                blt_score=scored[0].score if scored else 0.0,
                duration_ms=duration_ms,
                context_hash=context_hash,
                rejection_reason="; ".join(scored[0].violations[:3]) if scored else "unknown",
            )

        blt_score = legal[0].score

        # 8. Deduplication gate.
        dedup_result = self.dedup.check(proposed)
        if not dedup_result.accepted:
            return self._make_record(
                proposal_id=proposal_id,
                generator_kind=gen.config.kind,
                generator_model=gen.config.model,
                role=gen.config.role,
                scaffold_type=scaffold_type,
                accepted=False,
                dedup_status=dedup_result.status.value,
                blt_score=blt_score,
                duration_ms=duration_ms,
                context_hash=context_hash,
                rejection_reason=dedup_result.rejection_reason,
            )

        # 9. Add lineage refs if derivative.
        if dedup_result.status == DedupStatus.DERIVATIVE:
            proposed["parent_refs"] = dedup_result.nearest_ids

        # 10. Commit.
        proposed["hash"] = dedup_result.sigil_hash
        proposed["proposal_id"] = proposal_id
        with self._lock:
            self.dedup.register(proposed, dedup_result)
            self._corpus.append(proposed)

            # Re-index middleware with updated corpus every 50 sigils.
            if len(self._corpus) % 50 == 0:
                self.middleware.retriever.index(self._corpus)

        return self._make_record(
            proposal_id=proposal_id,
            generator_kind=gen.config.kind,
            generator_model=gen.config.model,
            role=gen.config.role,
            scaffold_type=scaffold_type,
            accepted=True,
            dedup_status=dedup_result.status.value,
            blt_score=blt_score,
            duration_ms=duration_ms,
            context_hash=context_hash,
            parent_refs=proposed.get("parent_refs", []),
        )

    # ------------------------------------------------------------------
    # WorldView update
    # ------------------------------------------------------------------

    def _update_worldview(self) -> None:
        """Recompute the statistical snapshot of the corpus."""
        sc: Dict[str, int] = {}
        entropies: List[float] = []

        for s in self._corpus:
            scaffold = s.get("scaffold", {})
            stype = (
                scaffold.get("scaffold_type", "") if isinstance(scaffold, dict) else ""
            )
            if stype:
                sc[stype] = sc.get(stype, 0) + 1
            e = s.get("entropy")
            if isinstance(e, (int, float)):
                entropies.append(float(e))

        saturated = {
            stype: sc.get(stype, 0) >= self.scaffold_targets.get(stype, 9999)
            for stype in self.scaffold_targets
        }

        mean_e = sum(entropies) / len(entropies) if entropies else 0.5
        var_e = (
            sum((e - mean_e) ** 2 for e in entropies) / len(entropies)
            if entropies else 0.04
        )
        std_e = math.sqrt(var_e)

        # Novelty yield: last update_worldview_every proposals
        recent = self._proposals[-self.update_worldview_every:]
        novel_count = sum(1 for p in recent if p.accepted and p.dedup_status == "novel")
        novelty_yield = novel_count / len(recent) if recent else 1.0

        # Per-generator rejection rate
        rej_rate: Dict[str, float] = {}
        for kind in {g.config.kind for g in self.generators}:
            kind_proposals = [p for p in recent if p.generator_kind == kind]
            if kind_proposals:
                rejected = sum(1 for p in kind_proposals if not p.accepted)
                rej_rate[kind] = rejected / len(kind_proposals)

        self.worldview = WorldView(
            scaffold_counts=sc,
            scaffold_saturated=saturated,
            entropy_mean=mean_e,
            entropy_std=std_e,
            novelty_yield=novelty_yield,
            rejection_rate=rej_rate,
            cycle=self._cycle,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _choose_scaffold(self) -> str:
        """Pick scaffold type with most room, biased toward variety."""
        gaps = self.worldview.scaffold_gap(self.scaffold_targets)
        if not gaps:
            gaps = list(self.scaffold_targets.keys())
        return random.choice(gaps)

    def _recent_forbidden_zones(self, n: int = 20) -> List[str]:
        """Return sigil names + glyph chars from recent rejections."""
        zones: List[str] = []
        for p in self._proposals[-n:]:
            if not p.accepted and p.proposal_id:
                zones.append(p.proposal_id)
        # Also add all currently committed sigil names.
        for s in self._corpus[-10:]:
            name = s.get("name", "")
            if name:
                zones.append(name)
        return list(set(zones))[:15]

    def _make_record(
        self,
        proposal_id: str,
        generator_kind: str,
        generator_model: str,
        role: str,
        scaffold_type: str,
        accepted: bool,
        dedup_status: str,
        blt_score: float,
        duration_ms: float,
        context_hash: str = "",
        rejection_reason: Optional[str] = None,
        parent_refs: Optional[List[str]] = None,
    ) -> ProposalRecord:
        record = ProposalRecord(
            proposal_id=proposal_id,
            generator_kind=generator_kind,
            generator_model=generator_model,
            role=role,
            scaffold_type=scaffold_type,
            accepted=accepted,
            dedup_status=dedup_status,
            blt_score=blt_score,
            duration_ms=duration_ms,
            cycle=self._cycle,
            context_hash=context_hash,
            rejection_reason=rejection_reason,
            parent_refs=parent_refs or [],
        )
        self._proposals.append(record)
        return record


# ---------------------------------------------------------------------------
# Utility: extract first JSON object from model response
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Find and parse the first {...} JSON object in an LLM response."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
