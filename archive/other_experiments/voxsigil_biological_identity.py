"""
VoxSigil Biological Identity System
Each sigil has DNA, family relationships, and ecosystem role
Makes AI EMBODY a specific identity rather than sampling from everything
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass

@dataclass
class BiologicalIdentity:
    """Biological identity encoding for a VoxSigil"""
    name: str
    dna_sequence: str  # Unique genetic code
    family_lineage: Dict  # Parents, siblings, children
    ecosystem_role: str  # Niche in cognitive ecosystem
    embodiment_traits: List[str]  # Grounded characteristics
    coordination_affinities: List[str]  # Who they work well with
    identity_anchor: str  # Core "I am..." statement

class VoxSigilBiologicalFramework:
    """Framework for biological identity in VoxSigils"""
    
    # Biological roles in cognitive ecosystem
    ECOSYSTEM_ROLES = {
        "producer": "Creates new cognitive patterns from raw input",
        "processor": "Transforms and refines existing patterns",
        "validator": "Checks and verifies cognitive outputs",
        "coordinator": "Orchestrates multiple cognitive functions",
        "memory": "Stores and recalls patterns",
        "filter": "Selectively attends to relevant information",
        "integrator": "Combines multiple streams into coherent whole",
        "explorer": "Seeks novel patterns and possibilities",
        "guardian": "Protects against errors and inconsistencies",
        "catalyst": "Triggers and accelerates cognitive processes"
    }
    
    # Family relationship types (biological metaphor)
    FAMILY_RELATIONSHIPS = {
        "parent": "Foundational prerequisite - must exist first",
        "child": "Derived extension - builds on this",
        "sibling": "Peer function - complementary role",
        "cousin": "Related function - similar domain",
        "symbiotic_partner": "Mutual dependency - work together",
        "ancestor": "Deep lineage - evolutionary origin"
    }
    
    def generate_dna_sequence(self, sigil_content: str, traits: List[str]) -> str:
        """Generate unique DNA-like sequence encoding identity"""
        # Hash content + traits to create stable genetic code
        combined = sigil_content + "".join(sorted(traits))
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        
        # Convert to DNA-like base pairs (A, T, G, C)
        dna_map = {'0': 'A', '1': 'T', '2': 'G', '3': 'C', 
                   '4': 'A', '5': 'T', '6': 'G', '7': 'C',
                   '8': 'A', '9': 'T', 'a': 'G', 'b': 'C',
                   'c': 'A', 'd': 'T', 'e': 'G', 'f': 'C'}
        
        dna = ''.join(dna_map[c] for c in hash_val[:32])
        # Format as base pairs
        return '-'.join([dna[i:i+4] for i in range(0, len(dna), 4)])
    
    def create_biological_identity_prompt(self,
                                         sigil_name: str,
                                         category: str,
                                         existing_identities: List[Dict]) -> str:
        """Create prompt that defines biological identity"""
        
        prompt = f"""You are defining the BIOLOGICAL IDENTITY for a VoxSigil cognitive entity.

This is NOT an abstract pattern - this is a LIVING COGNITIVE ORGANISM with:
- DNA (unique genetic identity)
- Family (relationships to other sigils)
- Ecosystem role (niche in cognitive ecology)
- Embodied traits (grounded characteristics)

SIGIL TO DEFINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {sigil_name}
Category: {category}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BIOLOGICAL FRAMEWORK:

1. ECOSYSTEM ROLES (choose one):
{chr(10).join(f"   - {role}: {desc}" for role, desc in self.ECOSYSTEM_ROLES.items())}

2. FAMILY RELATIONSHIPS (define for this sigil):
{chr(10).join(f"   - {rel}: {desc}" for rel, desc in self.FAMILY_RELATIONSHIPS.items())}

3. EMBODIMENT TRAITS (physical/grounded characteristics):
   - What this sigil "feels like" experientially
   - Sensory/somatic metaphors
   - Energy patterns (fast/slow, hot/cold, dense/light)

4. IDENTITY ANCHOR (complete this sentence):
   "I am the cognitive function that..."

TASK: Define the COMPLETE biological identity for {sigil_name}.

Generate in this format:
```yaml
biological_identity:
  name: {sigil_name}
  
  # Core identity statement
  identity_anchor: "I am..."
  
  # Ecosystem role
  ecosystem_role: [choose from list]
  niche_description: "In the cognitive ecosystem, I..."
  
  # Family relationships (define each)
  family:
    parents: 
      - sigil_name: [prerequisite sigil]
        relationship: "They provide [what foundation]"
    
    siblings:
      - sigil_name: [peer sigil]
        relationship: "We complement each other by..."
    
    children:
      - sigil_name: [derived sigil]
        relationship: "They extend me by..."
  
  # Embodied characteristics (grounded metaphors)
  embodiment:
    energy_pattern: [fast/slow, hot/cold, etc]
    sensory_metaphor: "Feels like [physical sensation]"
    somatic_quality: "Moves like [physical movement]"
    materiality: "Has the texture of [physical material]"
  
  # Coordination affinities
  works_best_with:
    - [other sigil]: "because [reason]"
  
  # Identity constraints (what this is NOT)
  not_this:
    - "I am NOT [other function]"
```

Make this sigil feel ALIVE and EMBODIED - give it identity it can coordinate around.
"""
        return prompt
    
    def create_family_tree_visualization(self, identities: List[BiologicalIdentity]) -> str:
        """Create family tree showing relationships"""
        tree = "VOXSIGIL COGNITIVE FAMILY TREE\n"
        tree += "="*70 + "\n\n"
        
        # Group by ecosystem role
        by_role = {}
        for identity in identities:
            role = identity.ecosystem_role
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(identity)
        
        for role, sigils in by_role.items():
            tree += f"\n🧬 {role.upper()} LINEAGE:\n"
            tree += "-" * 70 + "\n"
            for sigil in sigils:
                tree += f"  {sigil.name}\n"
                tree += f"    DNA: {sigil.dna_sequence[:20]}...\n"
                tree += f"    Identity: {sigil.identity_anchor}\n"
                if sigil.family_lineage.get('parents'):
                    parents = ", ".join(sigil.family_lineage['parents'])
                    tree += f"    Parents: {parents}\n"
                if sigil.family_lineage.get('siblings'):
                    siblings = ", ".join(sigil.family_lineage['siblings'])
                    tree += f"    Siblings: {siblings}\n"
                tree += "\n"
        
        return tree
    
    def generate_biological_profile(self, sigil_data: Dict) -> BiologicalIdentity:
        """Generate biological identity for a sigil"""
        
        name = sigil_data.get("name", "unknown")
        category = sigil_data.get("category", "unknown")
        content = sigil_data.get("raw_content", "")
        
        # Extract traits from sigil
        traits = []
        if sigil_data.get("data"):
            data = sigil_data["data"]
            if isinstance(data, dict):
                if "cognitive" in data:
                    traits.extend(data["cognitive"].get("tags", []))
        
        # Generate DNA
        dna = self.generate_dna_sequence(content, traits)
        
        # Placeholder family (would be filled by analyzing relationships)
        family = {
            "parents": [],
            "siblings": [],
            "children": []
        }
        
        # Determine ecosystem role based on category
        role_mapping = {
            "pglyph": "coordinator",
            "tags": "filter",
            "scaffolds": "integrator",
            "sigils": "processor",
            "flows": "catalyst"
        }
        role = role_mapping.get(category, "processor")
        
        # Generate identity anchor
        identity = f"I am the cognitive function that {self.ECOSYSTEM_ROLES[role]}"
        
        return BiologicalIdentity(
            name=name,
            dna_sequence=dna,
            family_lineage=family,
            ecosystem_role=role,
            embodiment_traits=traits[:5] if traits else [],
            coordination_affinities=[],
            identity_anchor=identity
        )

def create_biological_identity_schema():
    """Schema for biological identity in VoxSigils"""
    return {
        "biological_identity": {
            "dna_sequence": "string - unique genetic code",
            "identity_anchor": "string - core 'I am...' statement",
            "ecosystem_role": "string - role in cognitive ecosystem",
            "family_lineage": {
                "parents": ["list of prerequisite sigils"],
                "siblings": ["list of peer/complementary sigils"],
                "children": ["list of derived/extended sigils"],
                "ancestors": ["list of deep evolutionary origins"]
            },
            "embodiment": {
                "energy_pattern": "string - fast/slow, hot/cold, dense/light",
                "sensory_metaphor": "string - what it feels like physically",
                "somatic_quality": "string - how it moves/flows",
                "materiality": "string - texture/substance metaphor"
            },
            "coordination_affinities": {
                "synergistic_with": ["sigils that amplify each other"],
                "inhibited_by": ["sigils that conflict"],
                "requires": ["sigils that must be present"],
                "enables": ["sigils that this activates"]
            },
            "identity_constraints": {
                "not_this": ["what this identity is NOT"],
                "boundaries": ["where identity ends"]
            }
        }
    }

def main():
    """Demo biological identity system"""
    framework = VoxSigilBiologicalFramework()
    
    print("="*70)
    print("🧬 VOXSIGIL BIOLOGICAL IDENTITY FRAMEWORK")
    print("="*70)
    print("\nGiving AI embodied identity through biological metaphor")
    print("\nEach VoxSigil is a COGNITIVE ORGANISM with:")
    print("  • DNA - unique genetic identity code")
    print("  • Family - relational connections (parents, siblings, children)")
    print("  • Ecosystem role - specific niche and purpose")
    print("  • Embodiment - grounded physical metaphors")
    print("  • Identity anchor - 'I am...' statement")
    print("\n" + "="*70)
    
    # Show ecosystem roles
    print("\n🌍 COGNITIVE ECOSYSTEM ROLES:")
    print("-"*70)
    for role, desc in framework.ECOSYSTEM_ROLES.items():
        print(f"  {role:15} - {desc}")
    
    # Show family relationships
    print("\n👨‍👩‍👧‍👦 FAMILY RELATIONSHIP TYPES:")
    print("-"*70)
    for rel, desc in framework.FAMILY_RELATIONSHIPS.items():
        print(f"  {rel:20} - {desc}")
    
    # Show biological identity schema
    print("\n📋 BIOLOGICAL IDENTITY SCHEMA:")
    print("-"*70)
    schema = create_biological_identity_schema()
    print(json.dumps(schema, indent=2))
    
    print("\n" + "="*70)
    print("✅ Framework ready to give VoxSigils embodied biological identity")
    print("="*70)

if __name__ == "__main__":
    main()
