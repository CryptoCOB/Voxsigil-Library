"""
VoxSigil Enhancement Pipeline (Schema 2.0-omega)

- Loads existing .voxsigil files
- Enhances missing/weak sections using Ollama
- Enforces 64-char DNA
- Writes enhanced outputs to library_sigil_enhanced

Usage:
  python voxsigil_enhance_pipeline.py --model llama3.2:latest
  python voxsigil_enhance_pipeline.py --model llama3.2:latest --include-legacy
"""

import argparse
import hashlib
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import yaml

LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")
OUTPUT_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil_enhanced")

REQUIRED_TOP_LEVEL = ["meta", "biological_identity", "principle", "usage", "tags"]
REQUIRED_META = ["sigil", "name", "tag", "category", "schema_version", "created"]

# Category-specific requirements
REQUIRED_FOR_FLOW = ["is_flow", "flow_definition", "mental_model_of_execution", "flow_personality"]
REQUIRED_FOR_PGLYPH = ["intellectual_ancestry", "social_bonds"]


def generate_dna(seed: str) -> str:
    """Generate unique 64-char DNA sequence (ATGC only)."""
    content = f"{seed}|{time.time()}|{random.random()}"
    hash1 = hashlib.sha256(content.encode()).digest()
    hash2 = hashlib.sha256(hash1).digest()
    combined = hash1 + hash2
    dna = ""
    for byte in combined:
        dna += "ATGC"[byte % 4]
    return dna[:64]


def ensure_dna(vox: Dict) -> None:
    """Ensure biological_identity.dna_sequence is 64 chars."""
    bio = vox.get("biological_identity")
    if not isinstance(bio, dict):
        vox["biological_identity"] = {
            "dna_sequence": generate_dna(vox.get("meta", {}).get("name", "seed"))
        }
        return
    dna = bio.get("dna_sequence")
    if not isinstance(dna, str) or len(dna) != 64:
        bio["dna_sequence"] = generate_dna(vox.get("meta", {}).get("name", "seed"))


def ensure_meta(vox: Dict, category: str) -> None:
    """Ensure meta fields exist and are aligned to schema 2.0-omega."""
    meta = vox.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        vox["meta"] = meta
    meta.setdefault("category", category)
    meta.setdefault("schema_version", "2.0-omega")
    meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))


def ollama_enhance(model: str, category: str, yaml_text: str) -> str:
    """Use Ollama CLI to enhance YAML. Returns enhanced YAML string."""
    prompt = f"""
You are a VoxSigil curator. Enhance the following VoxSigil YAML to conform strictly to Schema 2.0-omega.

Rules:
- Preserve existing identity and meaning; do not rename.
- Keep all existing fields and fill missing ones.
- Ensure biological_identity.dna_sequence is exactly 64 chars of ATGC.
- Ensure meta includes sigil, name, tag, category, schema_version=2.0-omega, created (YYYY-MM-DD).
- For pglyph: ensure intellectual_ancestry and social_bonds exist and are rich.
- For flows: include flow_definition, mental_model_of_execution, flow_personality.
- Keep YAML valid and only output YAML (no extra text).

Category: {category}

YAML to enhance:
{yaml_text}
"""

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=120,
        check=False,
    )
    try:
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except UnicodeDecodeError:
        return ""


def parse_yaml(text: str) -> Dict:
    """Parse YAML with best-effort trimming to the first meta block."""
    start = text.find("meta:")
    if start != -1:
        text = text[start:]
    return yaml.safe_load(text)


def needs_enhancement(vox: Dict, category: str) -> List[str]:
    """Return list of missing required fields for enhancement."""
    missing = []
    for k in REQUIRED_TOP_LEVEL:
        if k not in vox:
            missing.append(k)
    meta = vox.get("meta")
    if not isinstance(meta, dict):
        missing.append("meta")
    else:
        for k in REQUIRED_META:
            if k not in meta:
                missing.append(f"meta.{k}")
    if category == "pglyph":
        for k in REQUIRED_FOR_PGLYPH:
            if k not in vox:
                missing.append(k)
    if category == "flows":
        for k in REQUIRED_FOR_FLOW:
            if k not in vox:
                missing.append(k)
    return missing


def enhance_file(path: Path, model: str, include_legacy: bool) -> Tuple[bool, str]:
    """Enhance a single VoxSigil file and write to enhanced output."""
    category = path.parent.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        return False, f"parse_error: {e}"

    if not isinstance(data, dict):
        return False, "invalid_yaml"

    schema_version = data.get("schema_version") or data.get("meta", {}).get("schema_version")
    if (
        not include_legacy
        and schema_version
        and isinstance(schema_version, str)
        and schema_version.startswith("1.")
    ):
        return False, "skipped_legacy"

    ensure_meta(data, category)
    ensure_dna(data)

    missing = needs_enhancement(data, category)
    if not missing:
        enhanced = data
    else:
        raw_yaml = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        enhanced_text = ollama_enhance(model, category, raw_yaml)
        try:
            enhanced = parse_yaml(enhanced_text)
            if not isinstance(enhanced, dict):
                enhanced = data
        except (yaml.YAMLError, TypeError, ValueError):
            enhanced = data

    ensure_meta(enhanced, category)
    ensure_dna(enhanced)

    out_path = OUTPUT_BASE / category / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_yaml = yaml.dump(
        enhanced,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    out_path.write_text(out_yaml, encoding="utf-8")
    return True, "enhanced"


def main():
    """Run the enhancement pipeline across the library."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:latest")
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args()

    files = list(LIBRARY_BASE.rglob("*.voxsigil"))
    enhanced = 0
    skipped_legacy = 0
    errors = 0

    for p in files:
        ok, reason = enhance_file(p, args.model, args.include_legacy)
        if ok:
            enhanced += 1
        elif reason == "skipped_legacy":
            skipped_legacy += 1
        else:
            errors += 1

    print("\n=== Enhancement Summary ===")
    print(f"Total input files: {len(files)}")
    print(f"Enhanced outputs: {enhanced}")
    print(f"Skipped legacy: {skipped_legacy}")
    print(f"Errors: {errors}")
    print(f"Output folder: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
