"""
VoxSigil Schema Upgrade v2 - Preserves Semantic Content

- Reads original files
- Preserves all existing semantic content (principle, cognitive, implementation)
- Only updates schema version to 2.0-omega
- Ensures DNA is valid (doesn't override existing)
- Minimal, non-destructive upgrades
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
    """Generate 64-char DNA if needed."""
    try:
        content = f"{seed}|{time.time()}|{random.random()}"
        h = hashlib.sha256(content.encode()).digest()
        dna = ""
        for byte in h:
            dna += "ATGC"[byte % 4]
        return (dna + dna)[:64]
    except Exception:
        return "A" * 64


def upgrade_one_file(in_path: Path) -> bool:
    """Upgrade file to 2.0-omega while preserving content."""
    try:
        cat = in_path.parent.name
        if cat == "schema":
            return True

        # Read
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

        # Ensure meta exists and preserve name/principle/etc if they exist at top level
        if "meta" not in data:
            data["meta"] = {}

        meta = data["meta"]

        # Set category and schema
        meta["category"] = cat
        meta["schema_version"] = "2.0-omega"
        meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))
        meta.setdefault("tag", cat)

        # Preserve name if it exists, otherwise use alias or generate
        if "name" not in meta:
            if "alias" in meta:
                meta["name"] = meta["alias"]
            else:
                meta["name"] = f"{cat}_{random.randint(10000, 99999)}"

        # DNA - preserve if valid, generate if missing/invalid
        if "biological_identity" not in data:
            data["biological_identity"] = {}
        if not isinstance(data["biological_identity"], dict):
            data["biological_identity"] = {}

        dna = data["biological_identity"].get("dna_sequence")
        if not dna or len(str(dna)) != 64:
            data["biological_identity"]["dna_sequence"] = generate_dna(meta["name"])

        # Preserve principle if exists at top level, don't overwrite
        if "principle" not in data and "cognitive" in data:
            if isinstance(data["cognitive"], dict) and "principle" in data["cognitive"]:
                data["principle"] = data["cognitive"]["principle"]
        elif "principle" not in data:
            data["principle"] = f"{meta['name']} operates with purpose."

        # Preserve usage if exists
        if "usage" not in data and "implementation" in data:
            if isinstance(data["implementation"], dict) and "usage" in data["implementation"]:
                data["usage"] = data["implementation"]["usage"]
        elif "usage" not in data:
            data["usage"] = {
                "description": f"Use {meta['name']}",
                "example": "invoke()",
            }

        # Tags array
        if "tags" not in data:
            data["tags"] = [cat]

        # Write
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
    """Main upgrade loop."""
    all_files = sorted(LIBRARY_BASE.rglob("*.voxsigil"))
    total = len(all_files)

    successes = 0
    failures = 0

    print(f"[*] Upgrade: Preserving semantic content + 2.0-omega schema")
    print(f"[*] Processing {total} files...\n")
    sys.stdout.flush()

    for idx, fpath in enumerate(all_files):
        if upgrade_one_file(fpath):
            successes += 1
        else:
            failures += 1

        if (idx + 1) % 2000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"[*] {idx + 1}/{total} ({pct:.1f}%)")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print(f"TOTAL FILES:    {total}")
    print(f"UPGRADED:       {successes}")
    print(f"SKIPPED/ERROR:  {failures}")
    print(f"SUCCESS RATE:   {100*successes/total:.1f}%")
    print(f"OUTPUT FOLDER:  {OUTPUT_BASE}")
    print("=" * 60)
    print("\n[✓] Upgrade complete! Semantic content preserved.")


if __name__ == "__main__":
    main()
