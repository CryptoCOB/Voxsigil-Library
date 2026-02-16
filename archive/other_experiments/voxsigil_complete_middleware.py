"""
Complete VoxSigil Middleware - Loads ALL 177 Sigils
Tracks existing sigils, generates similar variations, validates with BLT
"""

import json
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SigilInventory:
    """Complete inventory of all existing sigils"""
    total_sigils: int
    sigils_by_category: Dict[str, List[Dict]]
    unique_names: Set[str]
    sigil_hashes: Set[str]
    behavioral_dimensions: Set[str]
    common_fields: Dict[str, int]

class VoxSigilCompleteMiddleware:
    """Complete middleware tracking all 177 VoxSigils"""
    
    def __init__(self, library_base: str = "c:\\nebula-social-crypto-core\\voxsigil_library"):
        self.library_base = Path(library_base)
        self.sigils = []
        self.inventory = None
        
    def load_all_sigils(self) -> SigilInventory:
        """Load ALL 177 VoxSigils from all subdirectories"""
        print(f"\n{'='*70}")
        print(f"📚 LOADING COMPLETE VOXSIGIL LIBRARY")
        print(f"{'='*70}")
        
        base_path = self.library_base / "library_sigil"
        categories = ["sigils", "scaffolds", "tags", "pglyph", "flows"]
        
        sigils_by_category = defaultdict(list)
        unique_names = set()
        sigil_hashes = set()
        
        for category in categories:
            category_path = base_path / category
            if not category_path.exists():
                print(f"   ⚠️  {category}/ not found")
                continue
                
            sigil_files = list(category_path.glob("*.voxsigil"))
            print(f"   📁 {category}/: {len(sigil_files)} files")
            
            for sigil_file in sigil_files:
                try:
                    with open(sigil_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Calculate hash for deduplication
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # Skip duplicates
                    if content_hash in sigil_hashes:
                        continue
                    
                    sigil_hashes.add(content_hash)
                    name = sigil_file.stem
                    unique_names.add(name)
                    
                    # Try parsing as YAML
                    try:
                        data = yaml.safe_load(content)
                    except:
                        data = None
                    
                    sigil_record = {
                        "name": name,
                        "category": category,
                        "path": str(sigil_file),
                        "data": data,
                        "raw_content": content,
                        "hash": content_hash,
                        "size": len(content)
                    }
                    
                    sigils_by_category[category].append(sigil_record)
                    self.sigils.append(sigil_record)
                    
                except Exception as e:
                    print(f"      ❌ Error loading {sigil_file.name}: {e}")
        
        # Analyze patterns
        behavioral_dimensions = set()
        common_fields = defaultdict(int)
        
        for sigil in self.sigils:
            if sigil["data"] and isinstance(sigil["data"], dict):
                for key in sigil["data"].keys():
                    common_fields[key] += 1
                
                # Extract behavioral tags
                if "cognitive" in sigil["data"]:
                    if isinstance(sigil["data"]["cognitive"], dict):
                        for k in sigil["data"]["cognitive"].keys():
                            behavioral_dimensions.add(k)
        
        self.inventory = SigilInventory(
            total_sigils=len(self.sigils),
            sigils_by_category=dict(sigils_by_category),
            unique_names=unique_names,
            sigil_hashes=sigil_hashes,
            behavioral_dimensions=behavioral_dimensions,
            common_fields=dict(common_fields)
        )
        
        print(f"\n{'='*70}")
        print(f"✅ LOADED {self.inventory.total_sigils} UNIQUE VOXSIGILS")
        print(f"{'='*70}")
        print(f"   Categories:")
        for cat, sigils in sigils_by_category.items():
            print(f"      {cat}: {len(sigils)}")
        print(f"   Unique names: {len(self.inventory.unique_names)}")
        print(f"   Behavioral dimensions: {len(self.inventory.behavioral_dimensions)}")
        print(f"   Common fields: {list(self.inventory.common_fields.keys())[:10]}")
        print(f"{'='*70}\n")
        
        return self.inventory
    
    def get_random_sigil_for_variation(self, category: str = None) -> Dict[str, Any]:
        """Get a random existing sigil to create a variation from"""
        import random
        
        if category:
            candidates = [s for s in self.sigils if s["category"] == category]
        else:
            candidates = self.sigils
        
        return random.choice(candidates) if candidates else None
    
    def is_duplicate(self, content: str) -> bool:
        """Check if generated content is a duplicate"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return content_hash in self.inventory.sigil_hashes
    
    def is_name_taken(self, name: str) -> bool:
        """Check if sigil name already exists"""
        return name in self.inventory.unique_names
    
    def create_variation_prompt(self, base_sigil: Dict[str, Any], variation_type: str = "similar") -> str:
        """Create prompt for generating similar variation of an existing sigil"""
        name = base_sigil["name"]
        category = base_sigil["category"]
        
        variation_instructions = {
            "similar": "Create a similar sigil with minor variations in parameters and slightly different focus",
            "evolved": "Create an evolved version with enhanced capabilities and additional features",
            "inverted": "Create an inverted version with opposite behavioral characteristics",
            "hybrid": "Create a hybrid combining aspects of this sigil with new characteristics"
        }
        
        instruction = variation_instructions.get(variation_type, variation_instructions["similar"])
        
        # Include the full sigil as reference
        reference_content = base_sigil["raw_content"][:1500]  # Truncate if too long
        
        prompt = f"""You are creating a NEW VoxSigil that is a {variation_type} variation of an existing one.

BASE SIGIL REFERENCE:
Name: {name}
Category: {category}

{reference_content}

TASK: {instruction}

REQUIREMENTS:
1. Must follow VoxSigil YAML format (schema_version, meta, cognitive, holo_mesh, implementation)
2. Must have DIFFERENT name than '{name}'
3. Must be structurally distinct (not a copy)
4. Must maintain valid VoxSigil structure
5. Must be {variation_type} to the reference but NOT identical

Generate the complete new VoxSigil YAML:"""
        
        return prompt
    
    def save_inventory_report(self, output_path: str = "c:\\UBLT\\voxsigil_inventory.json") -> None:
        """Save complete inventory report"""
        report = {
            "summary": {
                "total_sigils": self.inventory.total_sigils,
                "unique_names": len(self.inventory.unique_names),
                "behavioral_dimensions": len(self.inventory.behavioral_dimensions),
                "categories": {k: len(v) for k, v in self.inventory.sigils_by_category.items()}
            },
            "common_fields": self.inventory.common_fields,
            "behavioral_dimensions": sorted(list(self.inventory.behavioral_dimensions)),
            "all_sigil_names": sorted(list(self.inventory.unique_names)),
            "sigils_by_category": {
                cat: [s["name"] for s in sigils] 
                for cat, sigils in self.inventory.sigils_by_category.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Inventory saved: {output_path}")

def main():
    """Load and analyze complete VoxSigil library"""
    middleware = VoxSigilCompleteMiddleware()
    inventory = middleware.load_all_sigils()
    middleware.save_inventory_report()
    
    print("\n📊 Sample sigils by category:")
    for category in ["sigils", "scaffolds", "tags", "pglyph"]:
        if category in inventory.sigils_by_category:
            samples = inventory.sigils_by_category[category][:5]
            names = [s["name"] for s in samples]
            print(f"   {category}: {', '.join(names)}...")

if __name__ == "__main__":
    main()
