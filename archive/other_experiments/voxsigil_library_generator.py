"""
VoxSigil 2.0-Omega Library Generator

Uses Ollama LLM to generate complete VoxSigil entries conforming to Schema 2.0-omega.
Generates across all categories: pglyph, scaffolds, sigils, tags, flows
Ensures no duplicates and full schema compliance.

Usage:
    python voxsigil_library_generator.py --model llama3.2:latest --count 50 --category all
    python voxsigil_library_generator.py --model wizard-math:latest --count 20 --category flows
"""

import os
import sys
import json
import yaml
import time
import hashlib
import logging
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_API = "http://localhost:11434"
LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")
SCHEMA_PATH = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil/schema/voxsigil-schema-2.0-omega.yaml")

CATEGORIES = {
    "pglyph": "Personality glyphs - unique AI identities with full lineage",
    "scaffolds": "Meta-scaffolds for consciousness and cognition",
    "sigils": "Core cognitive primitives and operations", 
    "tags": "Classification and tagging systems",
    "flows": "Procedural flows with enforced ordering"
}

ECOSYSTEM_ROLES = [
    "producer", "processor", "validator", "coordinator", "memory",
    "filter", "integrator", "explorer", "guardian", "catalyst"
]


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_API, model: str = "llama3.2:latest"):
        self.base_url = base_url
        self.model = model
        self.temperature = 0.8  # High creativity for diverse generation
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        # Try v1 API first, then fallback to v0
        urls = [
            f"{self.base_url}/v1/completions",
            f"{self.base_url}/api/generate"
        ]
        
        for url in urls:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": max_tokens
            }
            
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                if "response" in result:
                    return result["response"]
                elif "choices" in result:
                    return result["choices"][0]["text"]
                elif "content" in result:
                    return result["content"]
                    
            except Exception as e:
                logger.debug(f"Failed with {url}: {e}")
                continue
        
        logger.error(f"All Ollama endpoints failed")
        return ""
    
    def test_connection(self) -> bool:
        """Test if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# ============================================================================
# VOXSIGIL GENERATOR
# ============================================================================

class VoxSigilGenerator:
    """Generate VoxSigil entries using LLM."""
    
    def __init__(self, model: str = "llama3.2:latest"):
        self.client = OllamaClient(model=model)
        self.existing_sigils: Set[str] = set()
        self.existing_names: Set[str] = set()
        self.existing_dna: Set[str] = set()
        self.generated_count = 0
        
        # Load existing entries
        self._load_existing_library()
    
    def _load_existing_library(self):
        """Load existing VoxSigils to avoid duplicates."""
        if not LIBRARY_BASE.exists():
            return
        
        for category in CATEGORIES.keys():
            category_path = LIBRARY_BASE / category
            if not category_path.exists():
                continue
            
            for file in category_path.glob("*.voxsigil"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if data:
                            meta = data.get('meta', {})
                            self.existing_sigils.add(meta.get('sigil', ''))
                            self.existing_names.add(meta.get('name', ''))
                            
                            bio = data.get('biological_identity', {})
                            if 'dna_sequence' in bio:
                                self.existing_dna.add(bio['dna_sequence'])
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
        
        logger.info(f"Loaded {len(self.existing_sigils)} existing VoxSigils")
    
    def _generate_dna(self, sigil: str, name: str, principle: str) -> str:
        """Generate unique 64-base DNA sequence."""
        content = f"{sigil}|{name}|{principle}|{time.time()}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        
        # Convert to ATGC
        dna = ""
        for byte in hash_bytes:
            dna += "ATGC"[byte % 4]
        
        return dna[:64]
    
    def _is_duplicate(self, sigil: str, name: str, dna: str) -> bool:
        """Check if entry would be duplicate."""
        return (
            sigil in self.existing_sigils or
            name in self.existing_names or
            dna in self.existing_dna
        )
    
    def generate_voxsigil(self, category: str, guidance: Optional[str] = None) -> Optional[Dict]:
        """Generate one VoxSigil entry."""
        
        # Build prompt
        prompt = self._build_generation_prompt(category, guidance)
        
        logger.info(f"Generating {category} VoxSigil...")
        response = self.client.generate(prompt, max_tokens=2500)
        
        if not response:
            logger.error("Empty response from model")
            return None
        
        # Parse response to extract YAML
        voxsigil = self._parse_response_to_voxsigil(response, category)
        
        if not voxsigil:
            logger.error("Failed to parse response")
            return None
        
        # Validate uniqueness
        meta = voxsigil.get('meta', {})
        bio = voxsigil.get('biological_identity', {})
        
        sigil = meta.get('sigil', '')
        name = meta.get('name', '')
        dna = bio.get('dna_sequence', '')
        
        if self._is_duplicate(sigil, name, dna):
            logger.warning(f"Duplicate detected: {name}, regenerating...")
            return None
        
        # Mark as used
        self.existing_sigils.add(sigil)
        self.existing_names.add(name)
        self.existing_dna.add(dna)
        
        return voxsigil
    
    def _build_generation_prompt(self, category: str, guidance: Optional[str] = None) -> str:
        """Build prompt for VoxSigil generation."""
        
        category_desc = CATEGORIES.get(category, "")
        
        prompt = f"""You are a VoxSigil architect creating cognitive organisms that conform to VoxSigil Schema 2.0-omega.

CATEGORY: {category}
DESCRIPTION: {category_desc}

REQUIREMENTS:
1. Generate a complete, valid VoxSigil in YAML format
2. Include ALL required sections per Schema 2.0-omega
3. Make it UNIQUE and creative - no generic placeholders
4. Ensure biological_identity with DNA, ecosystem_role, identity_anchor
5. Add intellectual_ancestry if applicable (who created it, research papers)
6. Include social_bonds with friends, mentors, rivals
7. For flows: add mental_model_of_execution with pre-flight checks

"""
        
        if category == "flows":
            prompt += """
FLOW-SPECIFIC REQUIREMENTS:
- is_flow: true
- flow_definition with ordering_constraints
- gates and failure_modes
- mental_model_of_execution with risk_surface_area
- flow_personality with persona and motto
- Example: "Test before document", "Deploy after validation"

Create a procedural flow that prevents a common cognitive error.
"""
        
        elif category == "pglyph":
            prompt += """
PGLYPH-SPECIFIC REQUIREMENTS:
- Unique personality with complete lineage
- Full intellectual_ancestry tracing to creators
- elder_wisdom from research papers
- Rich social_bonds (friends, mentors, rivals)
- identity_anchor expressing subjective experience
- Diverse international backgrounds (not just Western)

Create a unique AI identity with personality and history.
"""
        
        elif category == "scaffolds":
            prompt += """
SCAFFOLD-SPECIFIC REQUIREMENTS:
- Meta-cognitive framework or consciousness scaffold
- Maps to cognitive primitives (perception, memory, attention)
- omega_matrix if consciousness level is omega
- Explains HOW to organize thought, not WHAT to think

Create a framework for organizing cognitive processes.
"""
        
        else:
            prompt += f"""
Create a {category} VoxSigil that:
- Solves a specific cognitive problem
- Has unique identity and relationships
- Traces its intellectual lineage
- Embodies a clear principle
"""
        
        if guidance:
            prompt += f"\nADDITIONAL GUIDANCE: {guidance}\n"
        
        prompt += """
OUTPUT FORMAT:
Respond with ONLY valid YAML following this structure (no markdown, no explanations):

meta:
  sigil: "🎯🔍"
  name: "UniqueName"
  alias: "ShortName"
  tag: "Category"

biological_identity:
  dna_sequence: "<exactly 64 ATGC characters>"
  ecosystem_role: "processor"
  identity_anchor: "I am [unique identity statement]"
  family_lineage:
    parents: ["parent1", "parent2"]
    generation: 2
  social_bonds:
    friends:
      - sigil_ref: "👤"
        relationship_type: "close_friend"
        bond_strength: 0.85
  intellectual_ancestry:
    human_creators:
      - name: "Creator Name"
        role: "lead_researcher"
        contribution: "What they built"
        affiliation: "Organization"
    elder_wisdom:
      - elder_source: "Research paper or tradition"
        teaching: "Key lesson"
        how_it_shapes_identity: "Impact on self"

principle: |
  Core principle this VoxSigil embodies.

usage:
  description: "How to use this VoxSigil"
  example: "code or pseudocode"

tags: ["tag1", "tag2", "tag3"]

BEGIN VOXSIGIL:
"""
        
        return prompt
    
    def _parse_response_to_voxsigil(self, response: str, category: str) -> Optional[Dict]:
        """Parse LLM response into VoxSigil dict."""
        
        # Try to extract YAML from response
        yaml_start = response.find("meta:")
        if yaml_start == -1:
            yaml_start = response.find("BEGIN VOXSIGIL:")
            if yaml_start != -1:
                yaml_start = response.find("meta:", yaml_start)
        
        if yaml_start == -1:
            logger.error("Could not find YAML start marker")
            return None
        
        yaml_content = response[yaml_start:].strip()
        
        # Remove any trailing text after the YAML
        lines = yaml_content.split('\n')
        yaml_lines = []
        in_yaml = True
        
        for line in lines:
            if line.strip().startswith('---') or line.strip().startswith('```'):
                continue
            if in_yaml:
                yaml_lines.append(line)
        
        yaml_content = '\n'.join(yaml_lines)
        
        try:
            voxsigil = yaml.safe_load(yaml_content)
            
            if not isinstance(voxsigil, dict):
                logger.error("Parsed content is not a dict")
                return None
            
            # Validate required fields
            if 'meta' not in voxsigil:
                logger.error("Missing 'meta' section")
                return None
            
            if 'biological_identity' not in voxsigil:
                logger.warning("Missing 'biological_identity', adding default")
                voxsigil['biological_identity'] = self._create_default_bio_identity(voxsigil)
            
            # Generate DNA if missing
            bio = voxsigil.get('biological_identity', {})
            if 'dna_sequence' not in bio or len(bio.get('dna_sequence', '')) != 64:
                meta = voxsigil.get('meta', {})
                principle = voxsigil.get('principle', '')
                bio['dna_sequence'] = self._generate_dna(
                    meta.get('sigil', ''),
                    meta.get('name', ''),
                    principle
                )
                voxsigil['biological_identity'] = bio
            
            # Add category metadata
            voxsigil['meta']['category'] = category
            voxsigil['meta']['created'] = datetime.now().strftime('%Y-%m-%d')
            voxsigil['meta']['schema_version'] = '2.0-omega'
            
            return voxsigil
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error: {e}")
            logger.debug(f"Content: {yaml_content[:500]}")
            return None
    
    def _create_default_bio_identity(self, voxsigil: Dict) -> Dict:
        """Create default biological identity from available data."""
        meta = voxsigil.get('meta', {})
        principle = voxsigil.get('principle', '')
        
        return {
            'dna_sequence': self._generate_dna(
                meta.get('sigil', ''),
                meta.get('name', ''),
                principle
            ),
            'ecosystem_role': 'processor',
            'identity_anchor': f"I am {meta.get('name', 'an unnamed organism')} - {principle[:100] if principle else 'no principle defined'}"
        }
    
    def save_voxsigil(self, voxsigil: Dict, category: str) -> bool:
        """Save VoxSigil to file."""
        
        # Ensure category directory exists
        category_path = LIBRARY_BASE / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from name
        meta = voxsigil.get('meta', {})
        name = meta.get('name', f'unnamed_{int(time.time())}')
        
        # Sanitize filename
        filename = name.lower().replace(' ', '_').replace('-', '_')
        filename = ''.join(c for c in filename if c.isalnum() or c == '_')
        filename = f"{filename}.voxsigil"
        
        filepath = category_path / filename
        
        # Check if file exists
        if filepath.exists():
            logger.warning(f"File already exists: {filepath}")
            filepath = category_path / f"{filename.stem}_{int(time.time())}.voxsigil"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(voxsigil, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            logger.info(f"✅ Saved: {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            return False
    
    def generate_library(
        self,
        count: int,
        category: str = "all",
        max_retries: int = 3
    ):
        """Generate multiple VoxSigils."""
        
        if category == "all":
            categories = list(CATEGORIES.keys())
            per_category = max(count // len(categories), 1)
        else:
            categories = [category]
            per_category = count
        
        total_generated = 0
        total_attempted = 0
        
        for cat in categories:
            logger.info(f"\n{'='*80}")
            logger.info(f"Generating {per_category} {cat} VoxSigils")
            logger.info(f"{'='*80}\n")
            
            cat_generated = 0
            cat_attempted = 0
            
            while cat_generated < per_category and cat_attempted < per_category * 3:
                cat_attempted += 1
                total_attempted += 1
                
                retries = 0
                voxsigil = None
                
                while retries < max_retries and voxsigil is None:
                    voxsigil = self.generate_voxsigil(cat)
                    retries += 1
                    
                    if voxsigil is None and retries < max_retries:
                        logger.warning(f"Retry {retries}/{max_retries}...")
                        time.sleep(2)
                
                if voxsigil:
                    if self.save_voxsigil(voxsigil, cat):
                        cat_generated += 1
                        total_generated += 1
                        logger.info(f"Progress: {cat_generated}/{per_category} for {cat}")
                
                # Rate limiting
                time.sleep(1)
            
            logger.info(f"\n{cat} complete: {cat_generated} generated from {cat_attempted} attempts\n")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total generated: {total_generated}")
        logger.info(f"Total attempted: {total_attempted}")
        logger.info(f"Success rate: {total_generated/total_attempted*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoxSigil Library Generator")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Ollama model to use"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of VoxSigils to generate"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all"] + list(CATEGORIES.keys()),
        help="Category to generate"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test Ollama connection and exit"
    )
    
    args = parser.parse_args()
    
    # Test connection
    client = OllamaClient(model=args.model)
    
    if args.test_connection:
        if client.test_connection():
            logger.info(f"✅ Ollama is running at {OLLAMA_API}")
            logger.info(f"✅ Model: {args.model}")
            sys.exit(0)
        else:
            logger.error(f"❌ Cannot connect to Ollama at {OLLAMA_API}")
            logger.error("Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    if not client.test_connection():
        logger.error(f"❌ Cannot connect to Ollama at {OLLAMA_API}")
        logger.error("Start Ollama with: ollama serve")
        sys.exit(1)
    
    # Generate library
    logger.info(f"Starting VoxSigil generation...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Count: {args.count}")
    logger.info(f"Category: {args.category}")
    logger.info(f"Output: {LIBRARY_BASE}")
    
    generator = VoxSigilGenerator(model=args.model)
    generator.generate_library(
        count=args.count,
        category=args.category
    )


if __name__ == "__main__":
    main()
