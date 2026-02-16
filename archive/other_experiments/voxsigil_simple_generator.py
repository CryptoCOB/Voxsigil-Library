"""
Simple VoxSigil Generator using Ollama CLI

Uses subprocess to call ollama directly - more reliable than API.
Generates VoxSigils conforming to Schema 2.0-omega.
"""

import os
import sys
import yaml
import time
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Set
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")

CATEGORIES = ["pglyph", "scaffolds", "sigils", "tags", "flows"]

GENERATION_PROMPTS = {
    "pglyph": """Create a unique AI personality glyph following VoxSigil Schema 2.0-omega.

Generate ONLY valid YAML with these sections:
- meta: sigil (emoji), name, alias, tag
- biological_identity: 64-char DNA (ATGC only), ecosystem_role, identity_anchor
- social_bonds: friends with relationships
- intellectual_ancestry: human creators, research papers, elder wisdom
- principle: core philosophy
- usage: how to activate this personality

Make it creative and unique. Use international diversity.

RESPOND WITH YAML ONLY (no explanations):""",
    
    "scaffolds": """Create a meta-cognitive scaffold following VoxSigil Schema 2.0-omega.

Generate ONLY valid YAML with these sections:
- meta: sigil, name, alias, tag
- biological_identity: DNA, ecosystem_role (coordinator/integrator), identity_anchor
- principle: what this scaffold organizes
- usage: how to use this framework
- tags: ["scaffold", "cognition", etc]

Think: attention framework, memory hierarchy, perception pipeline.

RESPOND WITH YAML ONLY:""",
    
    "sigils": """Create a cognitive primitive sigil following VoxSigil Schema 2.0-omega.

Generate ONLY valid YAML with these sections:
- meta: sigil, name, alias, tag
- biological_identity: DNA, ecosystem_role, identity_anchor
- principle: what operation this performs
- usage: when and how to invoke
- tags: ["primitive", "operation", etc]

Think: compare, filter, route, aggregate, validate.

RESPOND WITH YAML ONLY:""",
    
    "tags": """Create a classification tag system following VoxSigil Schema 2.0-omega.

Generate ONLY valid YAML with these sections:
- meta: sigil, name, alias, tag
- biological_identity: DNA, ecosystem_role (filter/classifier), identity_anchor
- principle: what this tag categorizes
- usage: how to apply this tag
- tags: ["classification", etc]

RESPOND WITH YAML ONLY:""",
    
    "flows": """Create a procedural flow following VoxSigil Schema 2.0-omega.

Generate ONLY valid YAML with these sections:
- meta: sigil, name, alias, tag
- biological_identity: DNA, ecosystem_role (validator/guardian), identity_anchor
- is_flow: true
- flow_definition: ordering_constraints, gates, failure_modes
- mental_model_of_execution: risk surface, cognitive rehearsal
- flow_personality: persona, motto
- principle: what this flow prevents/ensures
- usage: when to enforce this flow

Think: "test before document", "validate before deploy", "measure before optimize".

RESPOND WITH YAML ONLY:"""
}


def generate_dna(name: str, principle: str) -> str:
    """Generate 64-char DNA sequence."""
    content = f"{name}|{principle}|{time.time()}"
    hash_bytes = hashlib.sha256(content.encode()).digest()
    dna = ""
    for byte in hash_bytes:
        dna += "ATGC"[byte % 4]
    return dna[:64]


def call_ollama(prompt: str, model: str = "llama3.2:latest") -> str:
    """Call Ollama via command line."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except Exception as e:
        logger.error(f"Ollama failed: {e}")
        return ""


def parse_yaml_from_response(response: str) -> Optional[Dict]:
    """Extract YAML from response."""
    # Find where YAML starts
    yaml_start = response.find("meta:")
    if yaml_start == -1:
        return None
    
    yaml_str = response[yaml_start:].strip()
    
    # Remove markdown or trailing content
    lines = yaml_str.split('\n')
    clean_lines = []
    for line in lines:
        if line.strip().startswith('```'):
            continue
        clean_lines.append(line)
    
    yaml_str = '\n'.join(clean_lines)
    
    try:
        data = yaml.safe_load(yaml_str)
        return data if isinstance(data, dict) else None
    except:
        return None


def enhance_voxsigil(voxsigil: Dict, category: str) -> Dict:
    """Add missing required fields."""
    # Ensure meta
    if 'meta' not in voxsigil:
        voxsigil['meta'] = {}
    voxsigil['meta']['category'] = category
    voxsigil['meta']['schema_version'] = '2.0-omega'
    voxsigil['meta']['created'] = datetime.now().strftime('%Y-%m-%d')
    
    if 'name' not in voxsigil['meta']:
        voxsigil['meta']['name'] = f"Generated_{category}_{int(time.time())}"
    
    # Ensure biological_identity
    if 'biological_identity' not in voxsigil:
        voxsigil['biological_identity'] = {}
    
    bio = voxsigil['biological_identity']
    if 'dna_sequence' not in bio or len(bio.get('dna_sequence', '')) != 64:
        bio['dna_sequence'] = generate_dna(
            voxsigil['meta']['name'],
            voxsigil.get('principle', '')
        )
    
    if 'ecosystem_role' not in bio:
        bio['ecosystem_role'] = 'processor'
    
    if 'identity_anchor' not in bio:
        bio['identity_anchor'] = f"I am {voxsigil['meta']['name']}"
    
    return voxsigil


def save_voxsigil(voxsigil: Dict, category: str) -> bool:
    """Save VoxSigil to file."""
    category_path = LIBRARY_BASE / category
    category_path.mkdir(parents=True, exist_ok=True)
    
    name = voxsigil['meta']['name'].lower().replace(' ', '_').replace('-', '_')
    name = ''.join(c for c in name if c.isalnum() or c == '_')
    
    filepath = category_path / f"{name}.voxsigil"
    
    # Check if exists
    counter = 1
    while filepath.exists():
        filepath = category_path / f"{name}_{counter}.voxsigil"
        counter += 1
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(voxsigil, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"✅ Saved: {filepath.name}")
        return True
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False


def generate_one(category: str, model: str = "llama3.2:latest") -> bool:
    """Generate one VoxSigil."""
    logger.info(f"Generating {category}...")
    
    prompt = GENERATION_PROMPTS[category]
    response = call_ollama(prompt, model)
    
    if not response:
        logger.error("Empty response")
        return False
    
    voxsigil = parse_yaml_from_response(response)
    if not voxsigil:
        logger.error("Failed to parse YAML")
        logger.debug(f"Response preview: {response[:500]}")
        return False
    
    voxsigil = enhance_voxsigil(voxsigil, category)
    return save_voxsigil(voxsigil, category)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--model", type=str, default="llama3.2:latest")
    args = parser.parse_args()
    
    # Test ollama
    try:
        subprocess.run(["ollama", "list"], check=True, capture_output=True)
        logger.info("✅ Ollama is available")
    except:
        logger.error("❌ Ollama not found. Install with: curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)
    
    categories = CATEGORIES if args.category == "all" else [args.category]
    per_category = max(args.count // len(categories), 1)
    
    total_success = 0
    total_attempts = 0
    
    for cat in categories:
        logger.info(f"\n{'='*80}")
        logger.info(f"Category: {cat} (target: {per_category})")
        logger.info(f"{'='*80}")
        
        cat_success = 0
        cat_attempts = 0
        
        while cat_success < per_category and cat_attempts < per_category * 3:
            cat_attempts += 1
            total_attempts += 1
            
            if generate_one(cat, args.model):
                cat_success += 1
                total_success += 1
                logger.info(f"Progress: {cat_success}/{per_category}")
            
            time.sleep(2)  # Rate limit
        
        logger.info(f"{cat} complete: {cat_success}/{cat_attempts} success")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE: {total_success} created from {total_attempts} attempts")
    logger.info(f"Success rate: {total_success/total_attempts*100:.1f}%")


if __name__ == "__main__":
    main()
