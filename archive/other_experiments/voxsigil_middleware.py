"""
VoxSigil Context Middleware
Loads all 177 VoxSigils to provide context for model-based sigil generation
Enables few-shot learning so models understand sigil structure
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SigilContext:
    """Extracted context from existing VoxSigils"""
    total_sigils: int
    example_sigils: List[Dict[str, Any]]
    common_patterns: Dict[str, Any]
    structure_template: str

class VoxSigilMiddleware:
    """Middleware to load and contextualize VoxSigils for BLT"""
    
    def __init__(self, sigil_dir: str = "c:\\nebula-social-crypto-core\\voxsigil_library\\library_sigil\\sigils"):
        self.sigil_dir = Path(sigil_dir)
        self.sigils = []
        self.context = None
        
    def load_all_sigils(self) -> None:
        """Load all .voxsigil files from directory"""
        print(f"📚 Loading VoxSigils from {self.sigil_dir}...")
        
        sigil_files = list(self.sigil_dir.glob("*.voxsigil"))
        print(f"   Found {len(sigil_files)} sigil files")
        
        for sigil_file in sigil_files:
            try:
                with open(sigil_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Try YAML first
                    try:
                        data = yaml.safe_load(content)
                        self.sigils.append({
                            "name": sigil_file.stem,
                            "data": data,
                            "raw": content[:500]  # First 500 chars
                        })
                    except:
                        # Store as raw text if not YAML
                        self.sigils.append({
                            "name": sigil_file.stem,
                            "data": None,
                            "raw": content[:500]
                        })
            except Exception as e:
                print(f"   ⚠️  Could not load {sigil_file.name}: {e}")
        
        print(f"✅ Loaded {len(self.sigils)} VoxSigils successfully")
    
    def extract_patterns(self) -> Dict[str, Any]:
        """Extract common patterns from loaded sigils"""
        print("\n🔍 Analyzing sigil patterns...")
        
        patterns = {
            "common_fields": {},
            "behavioral_dimensions": set(),
            "numeric_ranges": [],
            "structure_types": set()
        }
        
        for sigil in self.sigils:
            if sigil["data"]:
                # Extract common top-level fields
                for key in sigil["data"].keys():
                    if key in patterns["common_fields"]:
                        patterns["common_fields"][key] += 1
                    else:
                        patterns["common_fields"][key] = 1
                
                # Look for behavioral/cognitive fields
                if "cognitive" in sigil["data"]:
                    cog = sigil["data"]["cognitive"]
                    if isinstance(cog, dict) and "tags" in cog:
                        for tag in cog["tags"]:
                            patterns["behavioral_dimensions"].add(tag)
        
        patterns["behavioral_dimensions"] = list(patterns["behavioral_dimensions"])
        
        print(f"   Common fields: {list(patterns['common_fields'].keys())[:5]}...")
        print(f"   Behavioral dimensions found: {len(patterns['behavioral_dimensions'])}")
        
        return patterns
    
    def create_few_shot_examples(self, count: int = 3) -> List[str]:
        """Create few-shot examples from real sigils"""
        examples = []
        
        # Select diverse examples
        selected_sigils = self.sigils[:count] if len(self.sigils) >= count else self.sigils
        
        for sigil in selected_sigils:
            example = f"""
Example VoxSigil: {sigil['name']}
---
{sigil['raw']}
---
"""
            examples.append(example.strip())
        
        return examples
    
    def build_context(self) -> SigilContext:
        """Build complete context for model prompting"""
        print("\n🏗️  Building context for BLT sigil generation...")
        
        patterns = self.extract_patterns()
        examples = self.create_few_shot_examples(3)
        
        structure_template = """
A VoxSigil typically contains:
- schema_version: Version identifier
- meta: Metadata (alias, tag, sigil symbol)
- cognitive: Core behavioral/cognitive patterns
  - principle: Guiding principle
  - tags: Behavioral dimensions
  - structure: Component breakdown
- holo_mesh: Integration capabilities
"""
        
        context = SigilContext(
            total_sigils=len(self.sigils),
            example_sigils=examples,
            common_patterns=patterns,
            structure_template=structure_template
        )
        
        self.context = context
        print(f"✅ Context ready: {context.total_sigils} sigils analyzed")
        return context
    
    def create_contextualized_prompt(self, base_prompt: str) -> str:
        """Inject VoxSigil context into a base prompt"""
        if not self.context:
            self.load_all_sigils()
            self.build_context()
        
        contextualized = f"""You are generating a VoxSigil - a structured behavioral/cognitive profile.

CONTEXT: We have {self.context.total_sigils} existing VoxSigils in our library.

{self.context.structure_template}

Here are 3 REAL examples from our library:

{self.context.example_sigils[0]}

{self.context.example_sigils[1]}

{self.context.example_sigils[2]}

Common behavioral dimensions in our sigils:
{', '.join(self.context.common_patterns['behavioral_dimensions'][:10])}

NOW GENERATE:
{base_prompt}

FORMAT: Follow the VoxSigil structure shown in examples above. Include:
- Clear behavioral metrics (0-1 scales where applicable)
- Specific cognitive patterns
- Structured components
"""
        
        return contextualized
    
    def save_context_summary(self, output_path: str = "c:\\UBLT\\voxsigil_context_summary.json") -> None:
        """Save context summary for review"""
        if not self.context:
            self.build_context()
        
        summary = {
            "total_sigils_loaded": self.context.total_sigils,
            "common_patterns": {
                "fields": self.context.common_patterns["common_fields"],
                "behavioral_dimensions": self.context.common_patterns["behavioral_dimensions"]
            },
            "structure_template": self.context.structure_template,
            "examples_count": len(self.context.example_sigils)
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Context summary saved: {output_path}")

def main():
    """Test the middleware"""
    print("\n" + "="*70)
    print("🧬 VOXSIGIL CONTEXT MIDDLEWARE")
    print("="*70)
    
    middleware = VoxSigilMiddleware()
    middleware.load_all_sigils()
    middleware.build_context()
    middleware.save_context_summary()
    
    # Show example contextualized prompt
    print("\n" + "="*70)
    print("📝 EXAMPLE CONTEXTUALIZED PROMPT:")
    print("="*70)
    
    base_prompt = "Create a behavioral sigil for an analytical engineer"
    contextualized = middleware.create_contextualized_prompt(base_prompt)
    print(contextualized[:800] + "...\n")
    
    print("✅ Middleware ready for BLT training data generation")

if __name__ == "__main__":
    main()
