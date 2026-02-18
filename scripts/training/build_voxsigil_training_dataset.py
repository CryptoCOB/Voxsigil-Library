"""Build deterministic VoxSigil training datasets from schema corpus.

This utility compiles entries from:
- sigils/
- tags/
- scaffolds/
- normalized_voxsigils/ (optional)

and expands them deterministically to a requested target size (default 36,000)
for reproducible BLT/VME training data preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class SourceRecord:
    """Canonical source record extracted from a .voxsigil file."""

    source_path: str
    source_bucket: str
    sigil: str
    schema_version: str
    document: Dict[str, Any]


def _read_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parsed = yaml.safe_load(raw)
    return parsed if isinstance(parsed, dict) else {}


def _digest_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _infer_schema_version(doc: Dict[str, Any]) -> str:
    if isinstance(doc.get("schema_version"), str) and doc["schema_version"].strip():
        return doc["schema_version"].strip()

    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        value = str(metadata.get("voxsigil_schema_version", "")).strip()
        if value:
            return value

    meta = doc.get("meta")
    if isinstance(meta, dict):
        value = str(meta.get("schema_version", "")).strip()
        if value:
            return value

    return "legacy"


def _infer_sigil(doc: Dict[str, Any], fallback_name: str) -> str:
    for key in ("sigil", "name"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    meta = doc.get("meta")
    if isinstance(meta, dict):
        for key in ("sigil", "alias", "name"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return fallback_name


def collect_source_records(repo_root: Path, include_normalized: bool) -> List[SourceRecord]:
    """Load source VoxSigil documents and deduplicate by normalized content hash."""

    roots: List[Tuple[str, Path]] = [
        ("sigils", repo_root / "sigils"),
        ("tags", repo_root / "tags"),
        ("scaffolds", repo_root / "scaffolds"),
    ]
    if include_normalized:
        roots.extend(
            [
                ("normalized_sigils", repo_root / "normalized_voxsigils" / "sigils"),
                ("normalized_tags", repo_root / "normalized_voxsigils" / "tags"),
                (
                    "normalized_scaffolds",
                    repo_root / "normalized_voxsigils" / "scaffolds",
                ),
            ]
        )

    seen: Dict[str, SourceRecord] = {}

    for bucket, root in roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.voxsigil")):
            doc = _read_yaml(path)
            if not doc:
                continue

            rel = str(path.relative_to(repo_root)).replace("\\", "/")
            fallback = path.stem
            record = SourceRecord(
                source_path=rel,
                source_bucket=bucket,
                sigil=_infer_sigil(doc, fallback_name=fallback),
                schema_version=_infer_schema_version(doc),
                document=doc,
            )

            fingerprint = _digest_payload(
                {
                    "sigil": record.sigil,
                    "schema_version": record.schema_version,
                    "document": record.document,
                }
            )

            if fingerprint not in seen:
                seen[fingerprint] = record

    return list(seen.values())


def _split_for_index(index: int) -> str:
    # 80/10/10 deterministic split by modulo bucket.
    bucket = index % 10
    if bucket <= 7:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def _build_training_text(doc: Dict[str, Any], sigil: str, variant: int, rng: random.Random) -> str:
    principle = ""
    if isinstance(doc.get("principle"), str):
        principle = doc["principle"]
    elif isinstance(doc.get("cognitive"), dict):
        principle = str(doc["cognitive"].get("principle", ""))

    usage = doc.get("usage")
    usage_text = ""
    if isinstance(usage, dict):
        usage_text = " ".join(str(v) for v in usage.values() if isinstance(v, str))
    elif isinstance(usage, str):
        usage_text = usage

    typed_tags = doc.get("typed_tags") if isinstance(doc.get("typed_tags"), dict) else {}
    tag_tokens: List[str] = []
    for key in ("domain", "function", "temporal", "lifecycle"):
        values = typed_tags.get(key)
        if isinstance(values, list):
            tag_tokens.extend(str(v) for v in values)

    if not tag_tokens and isinstance(doc.get("tags"), list):
        tag_tokens = [str(v) for v in doc["tags"]]

    if tag_tokens:
        rng.shuffle(tag_tokens)

    prompt_styles = [
        "schema_alignment",
        "behavioral_embedding",
        "cross_version_normalization",
        "vme_enrichment",
    ]
    style = prompt_styles[variant % len(prompt_styles)]

    return (
        f"[style={style}] "
        f"Sigil={sigil}. "
        f"Principle={principle or 'unspecified'}. "
        f"Usage={usage_text or 'unspecified'}. "
        f"Tags={','.join(tag_tokens) if tag_tokens else 'none'}."
    )


def expand_records(
    records: List[SourceRecord],
    target_size: int,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    """Create deterministic expanded dataset rows from source records."""

    if not records:
        return []

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    for index in range(target_size):
        record = records[index % len(records)]
        variant = index // len(records)
        local_rng = random.Random(f"{seed}:{record.source_path}:{variant}")

        text = _build_training_text(record.document, record.sigil, variant, local_rng)
        row = {
            "id": f"voxsigil-{index:06d}",
            "split": _split_for_index(index),
            "sigil": record.sigil,
            "schema_version": record.schema_version,
            "source_bucket": record.source_bucket,
            "source_path": record.source_path,
            "variant": variant,
            "seed": seed,
            "text": text,
            "labels": {
                "target_schema": "2.0-omega",
                "task": "schema_integrated_embedding",
                "is_flow": bool(record.document.get("is_flow")),
            },
            "document": record.document,
        }
        rows.append(row)

    rng.shuffle(rows)
    return rows


def write_sharded_jsonl(
    output_dir: Path,
    stem: str,
    rows: List[Dict[str, Any]],
    shard_size: int,
) -> List[str]:
    """Write rows into multiple JSONL shards and return shard paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[str] = []

    if shard_size <= 0:
        shard_size = len(rows)

    total = len(rows)
    shard_count = max(1, (total + shard_size - 1) // shard_size)

    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        shard_rows = rows[start:end]
        shard_name = f"{stem}.part-{shard_idx + 1:05d}.jsonl"
        shard_path = output_dir / shard_name

        with shard_path.open("w", encoding="utf-8") as handle:
            for row in shard_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        shard_paths.append(str(shard_path))

    return shard_paths


def build_dataset(
    repo_root: Path,
    output_dir: Path,
    target_size: int,
    include_normalized: bool,
    seed: int,
    shard_size: int,
) -> Dict[str, Any]:
    """Build dataset + manifest files and return the manifest payload."""

    records = collect_source_records(repo_root, include_normalized=include_normalized)
    rows = list(expand_records(records, target_size=target_size, seed=seed))

    dataset_stem = f"voxsigil_training_dataset_{target_size}"
    manifest_path = output_dir / "voxsigil_training_manifest.json"

    shard_paths = write_sharded_jsonl(
        output_dir=output_dir,
        stem=dataset_stem,
        rows=rows,
        shard_size=shard_size,
    )
    written = len(rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "target_size": target_size,
        "rows_written": written,
        "shard_size": shard_size,
        "shard_count": len(shard_paths),
        "seed": seed,
        "include_normalized": include_normalized,
        "unique_source_records": len(records),
        "source_buckets": sorted({r.source_bucket for r in records}),
        "output_files": {
            "dataset_shards": shard_paths,
            "manifest_json": str(manifest_path),
        },
        "notes": [
            "Deterministic expansion from available VoxSigil corpus.",
            "Includes full source document payload for downstream BLT/VME training.",
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Parse CLI args and generate the training dataset artifacts."""

    parser = argparse.ArgumentParser(
        description="Build deterministic VoxSigil training JSONL dataset.",
    )
    parser.add_argument("--repo-root", default="c:/UBLT")
    parser.add_argument("--output-dir", default="training/datasets")
    parser.add_argument("--target-size", type=int, default=36000)
    parser.add_argument("--shard-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260217)
    parser.add_argument("--no-normalized", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    manifest = build_dataset(
        repo_root=repo_root,
        output_dir=output_dir,
        target_size=args.target_size,
        include_normalized=not args.no_normalized,
        seed=args.seed,
        shard_size=args.shard_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
