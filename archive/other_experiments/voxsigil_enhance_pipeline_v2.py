"""
VoxSigil Enhancement Pipeline v2 - Fast Schema Validation

- Loads existing .voxsigil files from all categories
- Enforces 64-char DNA sequences
- Ensures Schema 2.0-omega meta fields
- Handles ALL categories (pglyph, scaffolds, sigils, flows, tags)
- No Ollama dependency (just structural validation)
- Writes enhanced outputs to library_sigil_enhanced

Usage:
  python voxsigil_enhance_pipeline_v2.py
"""

import hashlib
import random
import time
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from collections import defaultdict

import yaml


LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")
OUTPUT_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil_enhanced")

REQUIRED_TOP_LEVEL = ["meta", "principle", "usage", "tags"]


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


def ensure_dna(vox: Dict, name: str = "seed") -> None:
    """Ensure biological_identity.dna_sequence is 64 chars."""
    bio = vox.get("biological_identity")
    if not isinstance(bio, dict):
        vox["biological_identity"] = {
            "dna_sequence": generate_dna(name)
        }
        return
    
    dna = bio.get("dna_sequence")
    if not isinstance(dna, str) or len(dna) != 64:
        bio["dna_sequence"] = generate_dna(name)


def ensure_meta(vox: Dict, category: str) -> None:
    """Ensure meta fields exist and are aligned to schema 2.0-omega."""
    meta = vox.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        vox["meta"] = meta
    
    # Set defaults
    meta.setdefault("category", category)
    meta.setdefault("schema_version", "2.0-omega")
    meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))
    
    # Ensure name exists (use from meta.name or fallback)
    if "name" not in meta:
        meta["name"] = f"{category}_auto_{random.randint(1000, 9999)}"
    
    # Ensure tag exists
    if "tag" not in meta:
        meta["tag"] = category
    
    # Ensure sigil exists (cosmetic emoji)
    if "sigil" not in meta:
        sigils = {"pglyph": "🌟", "scaffolds": "🏗️", "sigils": "⚡", "flows": "🌊", "tags": "🏷️"}
        meta["sigil"] = sigils.get(category, "📌")


def parse_yaml(text: str) -> Dict:
    """Parse YAML with best-effort trimming to the first meta block."""
    start = text.find("meta:")
    if start != -1:
        text = text[start:]
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return {}


def enhance_file(path: Path) -> Tuple[bool, str]:
    """Enhance a single VoxSigil file and write to enhanced output."""
    category = path.parent.name
    
    # Skip schema directory (not a voxsigil category)
    if category == "schema":
        return False, "skipped_schema_dir"
    
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = path.read_text(encoding="latin-1")
        
        data = yaml.safe_load(text)
    except Exception as e:
        print(f"[E] {path.name}: {str(e)[:60]}")
        return False, "parse_error"

    if not isinstance(data, dict):
        data = {}

    # Ensure required fields
    ensure_meta(data, category)
    ensure_dna(data, data.get("meta", {}).get("name", "seed"))

    # Ensure principle exists (required top-level)
    if "principle" not in data:
        data["principle"] = f"{data['meta']['name']} operates with purpose and integrity."
    
    # Ensure usage exists
    if "usage" not in data:
        data["usage"] = {
            "description": f"Use {data['meta']['name']} for {category} operations.",
            "example": f"{data['meta']['name']}.invoke()"
        }
    
    # Ensure tags exists
    if "tags" not in data:
        data["tags"] = [category, data['meta']['name'].lower()]

    # Write enhanced output
    out_path = OUTPUT_BASE / category / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_yaml = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    out_path.write_text(out_yaml, encoding="utf-8")
    return True, "enhanced"


def main():
    """Run the enhancement pipeline across the library."""

    files = list(LIBRARY_BASE.rglob("*.voxsigil"))
    
    stats = defaultdict(lambda: {"enhanced": 0, "skipped": 0, "errors": 0})
    overall_enhanced = 0
    overall_errors = 0

    print(f"[*] Processing {len(files)} files...")
    
    for idx, p in enumerate(files):
        category = p.parent.name
        ok, reason = enhance_file(p)
        
        if ok:
            stats[category]["enhanced"] += 1
            overall_enhanced += 1
        elif reason == "skipped_schema_dir":
            stats[category]["skipped"] += 1
        else:
            stats[category]["errors"] += 1
            overall_errors += 1
        
        if (idx + 1) % 5000 == 0:
            print(f"[*] Progress: {idx + 1}/{len(files)} ({100*(idx+1)/len(files):.1f}%)")

    print("\n" + "="*60)
    print("ENHANCEMENT SUMMARY")
    print("="*60)
    
    for category in sorted(stats.keys()):
        s = stats[category]
        enhanced_count = s["enhanced"]
        errors_count = s["errors"]
        skipped_count = s["skipped"]
        cat_upper = category.upper()
        print(f"\n{cat_upper:15} | Enhanced: {enhanced_count:5} | "
              f"Errors: {errors_count:3} | Skipped: {skipped_count:3}")
    
    print("\n" + "="*60)
    print(f"TOTAL FILES:      {len(files)}")
    print(f"TOTAL ENHANCED:   {overall_enhanced}")
    print(f"TOTAL ERRORS:     {overall_errors}")
    print(f"OUTPUT FOLDER:    {OUTPUT_BASE}")
    print("="*60)
    
    print("\n[✓] Enhancement complete!")



if __name__ == "__main__":
    main()

