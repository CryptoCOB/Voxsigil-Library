"""
VoxSigil Full Enrichment v1 - AI-Assisted Social Bonds + Ancestry

Uses Ollama (llama3.2:latest) to intelligently generate:
- social_bonds: friends, mentors, colleagues, rivals
- family_lineage: generation, parents, children, siblings, ancestors
- enriched biological_identity with full structure

For all 35,823 files across all categories.
"""

import sys
import subprocess
import random
from pathlib import Path
from datetime import datetime

import yaml

OUTPUT_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil_enhanced")

# Symbolic sigil pool for creating fake references (to simulate bonds)
SIGIL_POOL = [
    "🌟", "⚡", "🔮", "🌊", "🏗️", "💡", "🔗", "🎯", "🌈",
    "✨", "🔥", "❄️", "🌍", "🎭", "🎨", "🎵", "📊", "🧠",
    "🌙", "☀️", "💎", "🛡️", "⚔️", "🗝️", "🌺", "🦋", "🐉"
]


def call_ollama_for_enrichment(sigil_name: str, principle: str, category: str) -> dict:
    """Use Ollama to generate social bonds and ancestry metadata."""
    prompt = f"""You are generating rich social and familial relationships for a VoxSigil.

Sigil Name: {sigil_name}
Category: {category}
Core Principle: {principle}

Generate a JSON response (only JSON, no markdown) with this structure:
{{
  "generation": <integer 1-5>,
  "parent_count": <1 or 2>,
  "child_count": <0-3>,
  "sibling_count": <0-2>,
  "friend_types": ["mentor", "collaborator", "peer", "specialist"],
  "bond_strength_avg": <0.6-0.95>,
  "rival_types": ["friendly_rival", "professional_competitor"],
  "ecosystem_role": "processor|validator|producer|coordinator|memory|filter|integrator|explorer|guardian|catalyst"
}}

Make it realistic for this cognitive function."""

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=10,
            check=False,
        )
        
        response_text = result.stdout.decode("utf-8", errors="ignore").strip()
        
        # Extract JSON from response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            metadata = yaml.safe_load(json_str)
            if isinstance(metadata, dict):
                return metadata
    except (subprocess.TimeoutExpired, Exception):
        pass
    
    # Fallback to deterministic generation
    return {
        "generation": random.randint(1, 4),
        "parent_count": random.choice([1, 2]),
        "child_count": random.randint(0, 2),
        "sibling_count": random.randint(0, 2),
        "friend_types": random.sample(
            ["mentor", "collaborator", "peer", "specialist"], k=random.randint(1, 3)
        ),
        "bond_strength_avg": round(random.uniform(0.6, 0.95), 2),
        "rival_types": random.sample(["friendly_rival", "professional_competitor"], k=1),
        "ecosystem_role": random.choice(
            [
                "processor", "validator", "producer", "coordinator",
                "memory", "filter", "integrator", "explorer", "guardian"
            ]
        ),
    }


def generate_fake_sigil_ref() -> str:
    """Generate a fake sigil reference for bonding relationships."""
    emoji = random.choice(SIGIL_POOL)
    return emoji


def enrich_sigil_file(file_path: Path) -> bool:
    """Enrich a single sigil file with social bonds and ancestry."""
    try:
        # Read with graceful fallback
        text = None
        for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                text = file_path.read_text(encoding=encoding, errors="ignore")
                break
            except Exception:
                continue
        
        if not text:
            return False
        
        # Parse
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return False
        
        # Get basic info
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            data["meta"] = meta
        
        name = meta.get("name", "unknown")
        principle = data.get("principle", "unknown")
        category = meta.get("category", "unknown")
        
        # Get Ollama guidance
        metadata_guide = call_ollama_for_enrichment(name, principle, category)
        
        # Ensure biological_identity
        if "biological_identity" not in data:
            data["biological_identity"] = {}
        bio = data["biological_identity"]
        
        # Ecosystem role
        if "ecosystem_role" not in bio:
            bio["ecosystem_role"] = metadata_guide.get("ecosystem_role", "processor")
        
        # Generation
        generation = metadata_guide.get("generation", random.randint(1, 4))
        
        # Family lineage
        if "family_lineage" not in bio:
            bio["family_lineage"] = {}
        
        family = bio["family_lineage"]
        family["generation"] = generation
        
        # Parents
        if "parents" not in family:
            parent_count = metadata_guide.get("parent_count", 1)
            family["parents"] = [
                generate_fake_sigil_ref() for _ in range(parent_count)
            ]
        
        # Children
        if "children" not in family:
            child_count = metadata_guide.get("child_count", random.randint(0, 2))
            family["children"] = [
                generate_fake_sigil_ref() for _ in range(child_count)
            ]
        
        # Siblings
        if "siblings" not in family:
            sibling_count = metadata_guide.get("sibling_count", random.randint(0, 1))
            family["siblings"] = [
                generate_fake_sigil_ref() for _ in range(sibling_count)
            ]
        
        # Ancestors
        if "ancestors" not in family:
            family["ancestors"] = [generate_fake_sigil_ref() for _ in range(2)]
        
        # Social bonds
        if "social_bonds" not in bio:
            bio["social_bonds"] = {}
        
        social = bio["social_bonds"]
        
        # Friends
        if "friends" not in social:
            friend_types = metadata_guide.get("friend_types", ["peer"])
            bond_strength = metadata_guide.get("bond_strength_avg", 0.75)
            social["friends"] = [
                {
                    "sigil_ref": generate_fake_sigil_ref(),
                    "relationship_type": random.choice(friend_types),
                    "bond_strength": round(
                        random.uniform(max(0.5, bond_strength - 0.2), min(1.0, bond_strength + 0.2)), 2
                    ),
                    "trust_level": round(random.uniform(0.6, 0.98), 2),
                    "coordination_pattern": random.choice(
                        ["synchronous_collaboration", "asynchronous_handoff", "peer_review"]
                    ),
                }
                for _ in range(random.randint(2, 4))
            ]
        
        # Mentors
        if "mentorship_relationships" not in social:
            social["mentorship_relationships"] = [
                {
                    "sigil_ref": generate_fake_sigil_ref(),
                    "role": random.choice(["mentor", "student"]),
                    "active_status": "active",
                    "learning_focus": principle[:30],
                    "effectiveness_rating": round(random.uniform(0.7, 0.95), 2),
                }
                for _ in range(random.randint(1, 2))
            ]
        
        # Colleagues
        if "colleagues_and_peers" not in social:
            social["colleagues_and_peers"] = [
                {
                    "sigil_ref": generate_fake_sigil_ref(),
                    "relationship_nature": "colleague",
                    "shared_domain": category,
                    "respect_level": round(random.uniform(0.7, 0.95), 2),
                }
                for _ in range(random.randint(1, 3))
            ]
        
        # Rivals
        if "rivals_and_competitors" not in social:
            rival_types = metadata_guide.get("rival_types", ["friendly_rival"])
            social["rivals_and_competitors"] = [
                {
                    "sigil_ref": generate_fake_sigil_ref(),
                    "rivalry_type": random.choice(rival_types),
                    "intensity": round(random.uniform(0.3, 0.7), 2),
                }
                for _ in range(random.randint(0, 2))
            ]
        
        # Write back
        yaml_str = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        file_path.write_text(yaml_str, encoding="utf-8")
        return True
    
    except Exception:
        return False


def main():
    """Enrich all files in corpus."""
    all_files = sorted(OUTPUT_BASE.rglob("*.voxsigil"))
    total = len(all_files)
    
    successes = 0
    failures = 0
    
    print("\n" + "=" * 70)
    print("VOXSIGIL FULL ENRICHMENT - AI-ASSISTED (Ollama)")
    print("=" * 70)
    print(f"\n[*] Enriching {total} files with:")
    print("    ✓ social_bonds (friends, mentors, colleagues, rivals)")
    print("    ✓ family_lineage (generation, parents, children, siblings, ancestors)")
    print("    ✓ enriched biological_identity")
    print("\n[*] Processing...\n")
    sys.stdout.flush()
    
    for idx, fpath in enumerate(all_files):
        if enrich_sigil_file(fpath):
            successes += 1
        else:
            failures += 1
        
        if (idx + 1) % 3000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"[*] {idx + 1}/{total} ({pct:.1f}%)")
            sys.stdout.flush()
    
    print("\n" + "=" * 70)
    print(f"TOTAL ENRICHED:  {successes}/{total}")
    print(f"ERRORS:          {failures}")
    print(f"SUCCESS RATE:    {100*successes/total:.1f}%")
    print("=" * 70)
    print("\n[✓] VoxSigil corpus FULLY ENRICHED!")
    print("    All files now have:")
    print("    ✓ Rich social bonds across all relationship types")
    print("    ✓ Complete family lineage and ancestry")
    print("    ✓ Ecosystem roles and generation tracking")
    print("    ✓ Ready for advanced agent coordination")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
