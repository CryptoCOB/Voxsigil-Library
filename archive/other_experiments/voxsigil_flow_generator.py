"""
VoxSigil Flow Generator - Category-Specific Procedural Sequences
Generates flows for each category (pglyph, tags, scaffolds, sigils, flows)
Shows how categories interconnect and coordinate
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from voxsigil_complete_middleware import VoxSigilCompleteMiddleware
from datetime import datetime

FASTEST_MODEL = "wizard-math:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

class CategoryFlowPatterns:
    """Flow patterns specific to each VoxSigil category"""
    
    PGLYPH_FLOWS = [
        # Identity and system initialization flows
        "identity_initialization_sequence",
        "recursive_anchor_binding_flow",
        "system_identity_propagation",
        "echo_inheritance_chain",
        "core_orchestration_startup"
    ]
    
    TAG_FLOWS = [
        # Atomic cognitive primitive sequencing
        "attention_focus_switching_flow",
        "cognitive_primitive_composition",
        "tag_activation_cascade",
        "selective_attention_routing",
        "atomic_operation_pipeline"
    ]
    
    SCAFFOLD_FLOWS = [
        # Framework assembly and coordination
        "curriculum_progression_flow",
        "learning_module_sequencing",
        "framework_component_assembly",
        "scaffold_initialization_protocol",
        "hierarchical_structure_building"
    ]
    
    SIGIL_FLOWS = [
        # Operational component workflow
        "anomaly_detection_investigation_flow",
        "pattern_recognition_refinement",
        "component_activation_sequence",
        "operational_mode_transition",
        "sigil_coordination_protocol"
    ]
    
    META_FLOWS = [
        # Cross-category orchestration
        "pglyph_to_scaffold_initialization",
        "tag_activated_sigil_flow",
        "scaffold_guided_tag_composition",
        "full_system_bootstrap_sequence",
        "category_coordination_protocol"
    ]

class VoxSigilFlowGenerator:
    """Generate category-specific flows showing interconnection"""
    
    def __init__(self, model: str = FASTEST_MODEL):
        self.model = model
        self.middleware = VoxSigilCompleteMiddleware()
        self.inventory = None
        self.generated_flows = []
        
    def initialize(self):
        print(f"\n{'='*70}")
        print(f"🔄 VOXSIGIL FLOW GENERATOR - CATEGORY-SPECIFIC SEQUENCES")
        print(f"{'='*70}")
        self.inventory = self.middleware.load_all_sigils()
        
    def create_category_flow_prompt(self,
                                     base_sigil: Dict[str, Any],
                                     target_category: str,
                                     flow_pattern: str) -> str:
        """Create prompt for category-specific flow"""
        
        name = base_sigil["name"]
        category = base_sigil["category"]
        reference = base_sigil["raw_content"][:1000]
        
        prompt = f"""You are creating a VoxSigil FLOW - a procedural sequence with ordering constraints.

This flow is specific to the {target_category.upper()} category.

BASE REFERENCE SIGIL:
Name: {name}
Category: {category}

{reference}

FLOW SPECIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Category: {target_category}
Flow Pattern: {flow_pattern}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VOXSIGIL CATEGORY DEFINITIONS:
1. pglyph - System core identity & recursive orchestration
2. tags - Atomic cognitive primitives (focus, attention)
3. scaffolds - Composite frameworks (curriculum, learning)
4. sigils - Operational components (anomaly detection)
5. flows - Procedural sequences (ordering constraints)

TASK: Create a NEW VoxSigil in the FLOWS category that encodes: {flow_pattern}

CRITICAL REQUIREMENTS:
1. Category must be "flows" in the meta section
2. Must include "ordering_constraints" section with step-by-step sequence
3. Must include "dependency_graph" showing prerequisites
4. Must include "gates" that enforce ordering
5. Must include "failure_modes" showing what happens if order violated
6. Show how this flow relates to {target_category} category operations
7. Use DIFFERENT name than '{name}'

FLOW STRUCTURE (required sections):
- schema_version: 1.5-holo-alpha
- meta: (with category: flows, tag: ProceduralFlow)
- holo_mesh: (temporal_pattern: sequential_enforced_ordering)
- cognitive:
  - principle: (what this flow ensures)
  - ordering_constraints: (step 1, step 2, etc. with prerequisites)
  - dependency_graph: (what requires what)
  - failure_modes: (violations and consequences)
  - gates: (locks preventing premature execution)
- implementation: (how gates work)
- connectivity: (inputs/outputs)

Generate complete VoxSigil FLOW YAML for {flow_pattern} in {target_category} context:"""
        
        return prompt
    
    def generate_flow_for_category(self,
                                   target_category: str,
                                   flow_pattern: str) -> Dict[str, Any]:
        """Generate single flow for specific category"""
        
        print(f"\n🔄 [{target_category}] {flow_pattern}", end=" ... ", flush=True)
        
        # Pick a base sigil from the target category if possible
        category_sigils = [s for s in self.middleware.sigils if s["category"] == target_category]
        if category_sigils:
            base_sigil = category_sigils[0]  # Use first one as reference
        else:
            base_sigil = self.middleware.get_random_sigil_for_variation()
        
        prompt = self.create_category_flow_prompt(base_sigil, target_category, flow_pattern)
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            data = response.json()
            content = data.get("response", "")
            tokens = data.get("eval_count", 0)
            tps = tokens / elapsed if elapsed > 0 else 0
            
            is_dup = self.middleware.is_duplicate(content)
            is_valid = self._validate_flow_structure(content)
            
            result = {
                "target_category": target_category,
                "flow_pattern": flow_pattern,
                "base_sigil": base_sigil["name"],
                "generated_content": content,
                "is_duplicate": is_dup,
                "is_valid_flow": is_valid,
                "tokens": tokens,
                "time_sec": elapsed,
                "tokens_per_sec": tps,
                "timestamp": datetime.now().isoformat()
            }
            
            self.generated_flows.append(result)
            
            status = "✅" if is_valid and not is_dup else ("⚠️ DUP" if is_dup else "❌")
            print(f"{status} {tokens}tok {elapsed:.1f}s ({tps:.1f}tok/s)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _validate_flow_structure(self, content: str) -> bool:
        """Validate flow has required structure"""
        required = ["ordering_constraints", "dependency_graph", "gates", "failure_modes"]
        return sum(1 for req in required if req in content.lower()) >= 3
    
    def generate_category_flow_set(self, flows_per_category: int = 3) -> None:
        """Generate flows for each category"""
        import random
        
        print(f"\n{'='*70}")
        print(f"🎯 GENERATING FLOWS FOR EACH CATEGORY")
        print(f"{'='*70}")
        print(f"Creating {flows_per_category} flows per category\n")
        
        categories_and_patterns = [
            ("pglyph", CategoryFlowPatterns.PGLYPH_FLOWS),
            ("tags", CategoryFlowPatterns.TAG_FLOWS),
            ("scaffolds", CategoryFlowPatterns.SCAFFOLD_FLOWS),
            ("sigils", CategoryFlowPatterns.SIGIL_FLOWS),
            ("flows", CategoryFlowPatterns.META_FLOWS)
        ]
        
        for category, patterns in categories_and_patterns:
            print(f"\n{'─'*70}")
            print(f"📂 CATEGORY: {category.upper()}")
            print(f"{'─'*70}")
            
            for i in range(flows_per_category):
                flow_pattern = random.choice(patterns)
                self.generate_flow_for_category(category, flow_pattern)
                time.sleep(0.5)
        
        self.save_flow_set()
    
    def save_flow_set(self) -> None:
        """Save generated category flows"""
        output_dir = Path("c:\\UBLT\\blt_category_flows")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"category_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        valid_count = sum(1 for f in self.generated_flows if f and f["is_valid_flow"])
        duplicate_count = sum(1 for f in self.generated_flows if f and f["is_duplicate"])
        
        flow_set = {
            "metadata": {
                "purpose": "Category-specific flows showing procedural sequences and inter-category coordination",
                "model_used": self.model,
                "categories_covered": ["pglyph", "tags", "scaffolds", "sigils", "flows"],
                "total_generated": len(self.generated_flows),
                "valid_count": valid_count,
                "duplicate_count": duplicate_count,
                "timestamp": datetime.now().isoformat()
            },
            "category_patterns": {
                "pglyph_flows": CategoryFlowPatterns.PGLYPH_FLOWS,
                "tag_flows": CategoryFlowPatterns.TAG_FLOWS,
                "scaffold_flows": CategoryFlowPatterns.SCAFFOLD_FLOWS,
                "sigil_flows": CategoryFlowPatterns.SIGIL_FLOWS,
                "meta_flows": CategoryFlowPatterns.META_FLOWS
            },
            "generated_flows": self.generated_flows
        }
        
        with open(filename, 'w') as f:
            json.dump(flow_set, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Category flows saved: {filename}")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Total flows: {len(self.generated_flows)}")
        print(f"   Valid: {valid_count}")
        print(f"   By category:")
        for cat in ["pglyph", "tags", "scaffolds", "sigils", "flows"]:
            count = sum(1 for f in self.generated_flows if f and f["target_category"] == cat)
            print(f"      {cat}: {count} flows")
        print(f"{'='*70}\n")

def main():
    """Generate category-specific flows showing interconnection"""
    generator = VoxSigilFlowGenerator()
    generator.initialize()
    
    print("\n📋 GENERATING FLOWS FOR 5 CATEGORIES:")
    print("   1. pglyph flows - Identity/orchestration sequences")
    print("   2. tag flows - Atomic operation chains")
    print("   3. scaffold flows - Framework assembly procedures")
    print("   4. sigil flows - Component workflows")
    print("   5. meta flows - Cross-category coordination\n")
    
    generator.generate_category_flow_set(flows_per_category=3)
    
    print("\n✅ Category flow generation complete!")
    print("   Created flows showing how categories interconnect")
    print("   Ready for BLT training on procedural ordering")

if __name__ == "__main__":
    main()
