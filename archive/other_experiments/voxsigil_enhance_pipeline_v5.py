"""
VoxSigil Enhancement Pipeline v5 - Bulletproof Error Handling

- Wraps EVERY operation in try/except
- Skips any file that causes ANY error
- Reports final stats
- Fast and simple
"""

import sys
from pathlib import Path
from datetime import datetime
import hashlib
import random
import time

import yaml

LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")
OUTPUT_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil_enhanced")


def generate_dna(seed: str) -> str:
    """Generate unique 64-char DNA sequence (ATGC only)."""
    try:
        content = f"{seed}|{time.time()}|{random.random()}"
        h = hashlib.sha256(content.encode()).digest()
        dna = ""
        for byte in h:
            dna += "ATGC"[byte % 4]
        return (dna + dna)[:64]
    except Exception:
        return "A" * 64


def enhance_one_file(in_path: Path) -> bool:
    """Process and enhance a single file. Skip on ANY error."""
    try:
        cat = in_path.parent.name
        if cat == "schema":
            return True

        # Read - try multiple encodings
        text = None
        for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                text = in_path.read_text(encoding=encoding, errors="ignore")
                break
            except Exception:
                continue

        if not text:
            return False

        # Parse
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            data = {}

        # Ensure meta
        if "meta" not in data:
            data["meta"] = {}
        meta = data["meta"]
        meta.setdefault("category", cat)
        meta.setdefault("schema_version", "2.0-omega")
        meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))
        meta.setdefault("tag", cat)
        if "name" not in meta:
            meta["name"] = f"{cat}_{random.randint(10000, 99999)}"

        # Ensure DNA
        if "biological_identity" not in data:
            data["biological_identity"] = {}
        if not isinstance(data["biological_identity"], dict):
            data["biological_identity"] = {}

        dna = data["biological_identity"].get("dna_sequence")
        if not dna or len(str(dna)) != 64:
            data["biological_identity"]["dna_sequence"] = generate_dna(
                meta.get("name", "seed")
            )

        # Ensure principle
        if "principle" not in data:
            data["principle"] = f"{meta['name']} operates with purpose."

        # Ensure usage
        if "usage" not in data:
            data["usage"] = {
                "description": f"Use {meta['name']}",
                "example": "invoke()",
            }

        # Ensure tags
        if "tags" not in data:
            data["tags"] = [cat]

        # Write output
        out_path = OUTPUT_BASE / cat / in_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        out_path.write_text(yaml_str, encoding="utf-8")
        return True

    except Exception:
        return False


def main():
    """Main loop."""
    all_files = sorted(LIBRARY_BASE.rglob("*.voxsigil"))
    total = len(all_files)

    successes = 0
    failures = 0

    print(f"[*] Processing {total} files...")
    sys.stdout.flush()

    for idx, fpath in enumerate(all_files):
        if enhance_one_file(fpath):
            successes += 1
        else:
            failures += 1

        if (idx + 1) % 2000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"[*] {idx + 1}/{total} ({pct:.1f}%)")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print(f"TOTAL FILES:    {total}")
    print(f"ENHANCED:       {successes}")
    print(f"SKIPPED/ERROR:  {failures}")
    print(f"SUCCESS RATE:   {100*successes/total:.1f}%")
    print(f"OUTPUT FOLDER:  {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
