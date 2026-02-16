"""
VoxSigil Enhancement Pipeline v4 - With problematic file detection

Uses subprocess timeout to avoid hanging on specific files
"""

import sys
import subprocess
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
    content = f"{seed}|{time.time()}|{random.random()}"
    h = hashlib.sha256(content.encode()).digest()
    dna = ""
    for byte in h:
        dna += "ATGC"[byte % 4]
    return (dna + dna)[:64]


def enhance_one_file(in_path: Path) -> bool:
    """Process and enhance a single file, write to output."""
    cat = in_path.parent.name
    if cat == "schema":
        return True

    # Read
    try:
        # Use subprocess with timeout to avoid hanging on problematic files
        code = f"""
import sys
data = open(r'{in_path}', 'rb').read()
print(data.decode('utf-8', errors='ignore')[:500000])
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=2,
            text=False
        )
        if result.returncode != 0:
            return False
        text = result.stdout.decode("utf-8", errors="ignore")
    except (subprocess.TimeoutExpired, Exception):
        return False

    # Parse
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False

    if not isinstance(data, dict):
        data = {}

    # Enhance: ensure key fields
    if "meta" not in data or not isinstance(data["meta"], dict):
        data["meta"] = {}
    
    meta = data["meta"]
    meta.setdefault("category", cat)
    meta.setdefault("schema_version", "2.0-omega")
    meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))
    meta.setdefault("tag", cat)
    if "name" not in meta:
        meta["name"] = f"{cat}_{random.randint(10000, 99999)}"
    if "sigil" not in meta:
        meta["sigil"] = {
            "pglyph": "🌟", "scaffolds": "🏗️", "sigils": "⚡",
            "flows": "🌊", "tags": "🏷️"
        }.get(cat, "📌")

    # DNA
    if "biological_identity" not in data:
        data["biological_identity"] = {}
    if not isinstance(data["biological_identity"], dict):
        data["biological_identity"] = {}

    dna = data["biological_identity"].get("dna_sequence")
    if not isinstance(dna, str) or len(dna) != 64:
        data["biological_identity"]["dna_sequence"] = generate_dna(meta["name"])

    # Principle
    if "principle" not in data:
        data["principle"] = f"{meta['name']} operates with purpose."

    # Usage
    if "usage" not in data:
        data["usage"] = {"description": f"Use {meta['name']}", "example": "invoke()"}

    # Tags
    if "tags" not in data:
        data["tags"] = [cat, meta["name"].lower()]

    # Write
    out_path = OUTPUT_BASE / cat / in_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        yaml_out = yaml.dump(data, default_flow_style=False, allow_unicode=True,
                           sort_keys=False)
        out_path.write_text(yaml_out, encoding="utf-8")
        return True
    except (OSError, UnicodeEncodeError):
        return False


def main():
    """Main enhancement loop."""
    all_files = sorted(LIBRARY_BASE.rglob("*.voxsigil"))
    total = len(all_files)

    stats = {"success": 0, "fail": 0}

    print(f"[*] Processing {total} files...")
    sys.stdout.flush()

    for idx, fpath in enumerate(all_files):
        if enhance_one_file(fpath):
            stats["success"] += 1
        else:
            stats["fail"] += 1

        if (idx + 1) % 2000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"[*] {idx + 1}/{total} ({pct:.1f}%)")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print(f"TOTAL PROCESSED: {stats['success']}")
    print(f"ERRORS:          {stats['fail']}")
    print(f"OUTPUT:          {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
