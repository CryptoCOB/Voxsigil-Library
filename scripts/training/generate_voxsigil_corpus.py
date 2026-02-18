"""Generate large-scale VoxSigil corpus via Ollama with strict validation.

This script uses a Llama/Ollama model to generate VoxSigil records, validates
them against the 2.0-omega schema + hard constraints, and writes JSONL outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests

from voxsigil_library.schema_bridge import (
    CANONICAL_SCAFFOLDS,
    ScaffoldType,
    normalize_to_2_0_omega,
    validate_interconnected_schema,
)
from voxsigil_library.sigil_generation import RUNES

try:
    from src.voxsigil_library.rag import (
        SigilDeduplicator, DedupStatus,
        CognitiveCycleEngine, GeneratorConfig,
    )
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False


OLLAMA_BASE_URL = "http://localhost:11434"
LMSTUDIO_BASE_URL = "http://localhost:1234"


SIGIL_DIRECTIVE = """Objective:
Generate a large-scale, legally valid VoxSigil dataset for training BLT compression and VML symbolic grammar.

Generation Targets

Total sigils: 50k–100k

Distribution by type:

Primitive / Atomic: ~5k

Standard Organism: ~30k

Flow (ordered): ~15k

Assembly / Meta: ~7k

Mutation Variants: ~15k

Mandatory Constraints

Every sigil MUST declare:

scaffold_type

tags (domain, function, polarity, temporal, epistemic, lifecycle)

Glyph count per sigil:

Minimum: 1

Maximum: 11

Category limits enforced:

NOETIC ≤ 2

PHYSICS ≤ 2

LOGIC ≤ 2

ASTRAL ≤ 1

STRUCTURAL ≤ 2 (paired)

ENTROPY ≤ 1 (≤2 only for omega)

EMERGENCE (𐑒) ≤ 1

Flow scaffolds MUST be ordered and non-emergent.

Assembly scaffolds MUST NOT act directly.

Mutation variants MUST differ by ENTROPY-mediated change only.

All sigils MUST pass validation or be discarded.

Diversity Rules

70% canonical

20% edge-case

10% near-boundary legal

Output Format

Canonical JSON

Deterministic ordering

Hashable

One sigil per record

Do NOT generate prose explanations.
Do NOT invent new glyph categories.
Do NOT violate scaffold-tag compatibility.
"""


GLYPH_LIMITS = {
    "NOETIC": 2,
    "PHYSICS": 2,
    "LOGIC": 2,
    "ASTRAL": 1,
    "STRUCTURAL": 2,
    "ENTROPY": 1,
    "EMERGENCE": 1,
    "PROTO": 4,
}


SCAFFOLD_TYPE_ORDER = [
    ScaffoldType.IDENTITY.value,
    ScaffoldType.FLOW.value,
    ScaffoldType.ASSEMBLY.value,
    ScaffoldType.MUTATION.value,
]


@dataclass(frozen=True)
class GenerationSpec:
    """Specification for a single sigil generation request."""

    scaffold_type: str
    profile: str
    sigil_kind: str


@dataclass
class GenerationResult:
    """Summary of corpus generation outcomes."""

    requested: int
    generated: int
    accepted: int
    rejected: int
    elapsed_sec: float
    output_path: str
    issues_path: str


def _canonicalize_tags(tags: Dict[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for key in ["domain", "function", "polarity", "temporal", "epistemic", "lifecycle"]:
        value = tags.get(key)
        if isinstance(value, list) and value:
            result[key] = [str(v) for v in value]
        elif isinstance(value, str) and value.strip():
            result[key] = [value.strip()]
        else:
            result[key] = []
    return result


def _normalize_glyphs(glyphs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for entry in glyphs:
        if not isinstance(entry, dict):
            continue
        category = str(entry.get("category", "")).strip().upper()
        glyph = str(entry.get("glyph", "")).strip()
        if not category or not glyph:
            continue
        cleaned.append({"category": category, "glyph": glyph})
    return cleaned


def _glyph_counts(glyphs: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in glyphs:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    return counts


def _validate_glyphs(
    glyphs: List[Dict[str, str]],
    scaffold_type: str,
    is_omega: bool,
) -> List[str]:
    issues: List[str] = []
    if not glyphs:
        issues.append("Missing glyphs")
        return issues

    if len(glyphs) < 1 or len(glyphs) > 11:
        issues.append("Glyph count out of bounds (1-11)")

    counts = _glyph_counts(glyphs)
    for category, count in counts.items():
        if category not in RUNES:
            issues.append(f"Unknown glyph category: {category}")
            continue
        limit = GLYPH_LIMITS.get(category)
        if limit is not None:
            if category == "ENTROPY" and is_omega:
                limit = 2
            if count > limit:
                issues.append(f"{category} glyph count exceeds limit ({count}>{limit})")
        for entry in glyphs:
            if entry["category"] == category and entry["glyph"] not in RUNES[category]:
                issues.append(f"Invalid glyph {entry['glyph']} for category {category}")

    if scaffold_type == ScaffoldType.FLOW.value:
        if "EMERGENCE" in counts:
            issues.append("Flow scaffolds cannot include EMERGENCE glyphs")
    return issues


def _validate_scaffold_rules(
    scaffold_type: str,
    glyphs: List[Dict[str, str]],
) -> List[str]:
    issues: List[str] = []
    rule = CANONICAL_SCAFFOLDS.get(scaffold_type)
    if not rule:
        return [f"Unknown scaffold_type: {scaffold_type}"]

    categories = {entry["category"] for entry in glyphs}
    for forbidden in rule.forbidden_categories:
        if forbidden in categories:
            issues.append(f"Forbidden category for scaffold {scaffold_type}: {forbidden}")
    if scaffold_type == ScaffoldType.ASSEMBLY.value:
        scaffold_meta = rule.constraints
        if scaffold_meta.get("acts_directly", False):
            issues.append("Assembly scaffolds must not act directly")
    if scaffold_type == ScaffoldType.MUTATION.value and "ENTROPY" not in categories:
        issues.append("Mutation scaffolds require ENTROPY glyph")
    return issues


def _lmstudio_generate(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    base_url: str = LMSTUDIO_BASE_URL,
) -> Optional[str]:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    choices = response.json().get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content", "").strip()


def _ollama_generate(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Optional[str]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def _build_prompt(spec: GenerationSpec, seed: int) -> str:
    return (
        f"{SIGIL_DIRECTIVE}\n\n"
        "Return JSON only.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"sigil\": \"<glyphic_name>\",\n"
        "  \"name\": \"<human_alias>\",\n"
        "  \"principle\": \"<short principle>\",\n"
        "  \"usage\": {\"description\": \"<short usage>\"},\n"
        "  \"scaffold\": {\n"
        "     \"scaffold_type\": \"<identity|flow|assembly|mutation>\",\n"
        "     \"ordered\": true|false\n"
        "  },\n"
        "  \"typed_tags\": {\n"
        "     \"domain\": [],\n"
        "     \"function\": [],\n"
        "     \"polarity\": [],\n"
        "     \"temporal\": [],\n"
        "     \"epistemic\": [],\n"
        "     \"lifecycle\": []\n"
        "  },\n"
        "  \"glyphs\": [\n"
        "     {\"category\": \"NOETIC\", \"glyph\": \"𓂀\"}\n"
        "  ],\n"
        "  \"profile\": \"canonical|edge|near-boundary\",\n"
        "  \"sigil_kind\": \"primitive|standard|flow|assembly|mutation\"\n"
        "}\n\n"
        f"Use scaffold_type='{spec.scaffold_type}'.\n"
        f"Use sigil_kind='{spec.sigil_kind}'.\n"
        f"Use profile='{spec.profile}'.\n"
        f"Seed: {seed}.\n"
    )


def _parse_generated(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_record(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    scaffold = doc.get("scaffold") if isinstance(doc.get("scaffold"), dict) else {}
    scaffold_type = str(scaffold.get("scaffold_type", "")).strip()
    if scaffold_type not in CANONICAL_SCAFFOLDS:
        issues.append("Missing or invalid scaffold_type")

    typed_tags = _canonicalize_tags(
        doc.get("typed_tags") if isinstance(doc.get("typed_tags"), dict) else {}
    )
    if any(len(values) == 0 for values in typed_tags.values()):
        issues.append("typed_tags missing required classes")

    glyphs = _normalize_glyphs(doc.get("glyphs") or [])
    is_omega = str(doc.get("schema_version", "")).startswith("2.0")
    issues.extend(_validate_glyphs(glyphs, scaffold_type, is_omega))
    issues.extend(_validate_scaffold_rules(scaffold_type, glyphs))

    doc["typed_tags"] = typed_tags
    doc["glyphs"] = glyphs
    doc["scaffold"] = {
        "scaffold_type": scaffold_type,
        "ordered": bool(scaffold.get("ordered", False)),
    }

    normalized = normalize_to_2_0_omega(doc)
    ok, schema_issues = validate_interconnected_schema(normalized)
    if not ok:
        issues.extend(schema_issues)

    return len(issues) == 0, issues


def _generate_single(
    model: str,
    spec: GenerationSpec,
    seed: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    backend: str = "ollama",
    lmstudio_model: Optional[str] = None,
    dedup: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    for _ in range(retries + 1):
        prompt = _build_prompt(spec, seed)
        try:
            # Ollama is the primary bulk generator.
            if backend == "ollama":
                response = _ollama_generate(model, prompt, temperature, max_tokens, timeout)
            else:
                response = _lmstudio_generate(model, prompt, temperature, max_tokens, timeout)
        except (requests.RequestException, ValueError) as exc:
            issues.append(f"{backend} error: {exc}")
            continue

        if response is None:
            issues.append("No response")
            continue

        payload = _parse_generated(response)
        if not payload:
            issues.append("JSON parse failed")
            continue

        is_valid, validation_issues = _validate_record(payload)
        if not is_valid:
            issues.extend(validation_issues)
            continue

        # 3-layer deduplication check (if dedup is provided).
        if dedup is not None:
            dedup_result = dedup.check(payload)
            if not dedup_result.accepted:
                issues.append(f"dedup:{dedup_result.status.value}")
                continue  # try again with a different seed
            dedup.register(payload, dedup_result)
            if dedup_result.status.value == "derivative":
                payload["parent_refs"] = dedup_result.nearest_ids
            payload["hash"] = dedup_result.sigil_hash

        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["generator_model"] = model
        payload["generation_source"] = backend
        return payload, []

    return None, issues


def _build_specs(counts: Dict[str, int]) -> List[GenerationSpec]:
    """Expand type counts into per-sigil generation specs."""
    specs: List[GenerationSpec] = []
    profiles = [("canonical", 0.7), ("edge", 0.2), ("near-boundary", 0.1)]
    for sigil_kind, target in counts.items():
        scaffold_type = {
            "primitive": ScaffoldType.IDENTITY.value,
            "standard": ScaffoldType.IDENTITY.value,
            "flow": ScaffoldType.FLOW.value,
            "assembly": ScaffoldType.ASSEMBLY.value,
            "mutation": ScaffoldType.MUTATION.value,
        }[sigil_kind]
        for profile, ratio in profiles:
            bucket = int(round(target * ratio))
            for _ in range(bucket):
                specs.append(
                    GenerationSpec(
                        scaffold_type=scaffold_type,
                        profile=profile,
                        sigil_kind=sigil_kind,
                    )
                )
    return specs


def generate_corpus(
    model: str,
    counts: Dict[str, int],
    output_dir: Path,
    seed: int,
    workers: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    lmstudio_model: Optional[str] = None,
    enable_dedup: bool = True,
) -> GenerationResult:
    """
    Generate a full corpus and return summary stats.

    If lmstudio_model is set, LM Studio is used for the top 30% entropy +
    bottom 20% entropy sigils (refinement role).  Ollama handles the rest
    (bulk role).  Both generators never see each other's output directly;
    they operate through the shared deduplicator.

    enable_dedup (default True): enforce 3-layer deduplication.  Disable
    only for quick debugging runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(counts)
    rng = random.Random(seed)
    rng.shuffle(specs)

    # Build deduplicator (shared across threads — we lock on each check).
    dedup_lock = threading.Lock()
    dedup = SigilDeduplicator() if (enable_dedup and _RAG_AVAILABLE) else None

    # Assign backends per spec.
    # LM Studio handles ~30% of the load (refinement role) if available.
    lmstudio_available = False
    if lmstudio_model:
        try:
            r = requests.get("http://localhost:1234/v1/models", timeout=3)
            lmstudio_available = r.status_code == 200
        except Exception:
            pass

    start = time.time()
    output_path = output_dir / "voxsigil_generated.jsonl"
    issues_path = output_dir / "voxsigil_generation_issues.json"
    issues: Dict[str, List[str]] = {}

    accepted = 0
    rejected = 0
    generated = 0

    def _task(spec: GenerationSpec, index: int) -> Tuple[int, Optional[Dict[str, Any]], List[str]]:
        local_seed = seed + index
        # Assign LM Studio to refine-role specs (every ~3rd spec) if available.
        use_lmstudio = lmstudio_available and (index % 3 == 0)
        backend = "lmstudio" if use_lmstudio else "ollama"
        backend_model = lmstudio_model if use_lmstudio else model
        temp = (temperature + 0.1) if use_lmstudio else temperature  # slightly richer for refine

        # Grab a thread-local snapshot of dedup.
        with dedup_lock:
            local_dedup = dedup  # shared, but check+register is serialised via lock

        record, record_issues = _generate_single(
            model=backend_model,
            spec=spec,
            seed=local_seed,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            backend=backend,
            lmstudio_model=lmstudio_model,
            dedup=local_dedup,
        )
        return index, record, record_issues

    with output_path.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_task, spec, idx): idx
                for idx, spec in enumerate(specs)
            }
            for future in concurrent.futures.as_completed(futures):
                generated += 1
                idx, record, record_issues = future.result()
                if record:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    accepted += 1
                else:
                    rejected += 1
                    issues[str(idx)] = record_issues

    issues_path.write_text(json.dumps(issues, indent=2), encoding="utf-8")
    elapsed = time.time() - start
    return GenerationResult(
        requested=len(specs),
        generated=generated,
        accepted=accepted,
        rejected=rejected,
        elapsed_sec=elapsed,
        output_path=str(output_path),
        issues_path=str(issues_path),
    )


def main() -> None:
    """CLI entrypoint for corpus generation."""
    parser = argparse.ArgumentParser(
        description="Generate VoxSigil corpus via Ollama (+ optional LM Studio refinement)."
    )
    parser.add_argument("--model", default="llama3.2:latest",
                        help="Ollama model for bulk generation")
    parser.add_argument("--lmstudio-model", default="",
                        help="LM Studio model for refinement role (empty = skip LM Studio)")
    parser.add_argument("--output-dir", default="generated_sigils")
    parser.add_argument("--seed", type=int, default=20260217)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable 3-layer deduplication (debug only)")

    # Cognitive loop mode (replaces direct generation)
    parser.add_argument("--cognitive-loop", action="store_true",
                        help="Use stigmergic CognitiveCycleEngine instead of direct generation")
    parser.add_argument("--target-size", type=int, default=1000,
                        help="Target corpus size for cognitive loop mode")

    parser.add_argument("--primitive", type=int, default=5000)
    parser.add_argument("--standard", type=int, default=30000)
    parser.add_argument("--flow", type=int, default=15000)
    parser.add_argument("--assembly", type=int, default=7000)
    parser.add_argument("--mutation", type=int, default=15000)

    args = parser.parse_args()

    # --- Cognitive loop mode -------------------------------------------
    if args.cognitive_loop:
        if not _RAG_AVAILABLE:
            print("ERROR: RAG package not available. Run from repo root with PYTHONPATH set.")
            raise SystemExit(1)
        gen_configs = [GeneratorConfig(kind="ollama", model=args.model, role="bulk")]
        if args.lmstudio_model:
            gen_configs.append(
                GeneratorConfig(kind="lmstudio", model=args.lmstudio_model, role="refine")
            )
        engine = CognitiveCycleEngine.create(
            generators=gen_configs,
            scaffold_targets={
                "identity": args.primitive + args.standard,
                "flow": args.flow,
                "assembly": args.assembly,
                "mutation": args.mutation,
            },
        )
        output_path = str(Path(args.output_dir) / "cognitive_corpus.jsonl")
        engine.run(
            target_corpus_size=args.target_size,
            output_path=output_path,
            lineage_path=str(Path(args.output_dir) / "cognitive_lineage.jsonl"),
        )
        return

    # --- Direct generation mode ----------------------------------------
    counts = {
        "primitive": args.primitive,
        "standard": args.standard,
        "flow": args.flow,
        "assembly": args.assembly,
        "mutation": args.mutation,
    }

    result = generate_corpus(
        model=args.model,
        counts=counts,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        lmstudio_model=args.lmstudio_model or None,
        enable_dedup=not args.no_dedup,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
