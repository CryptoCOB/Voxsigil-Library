# Apply Encapsulated Registration to All Agents
"""
Automation script to apply @vanta_agent decorator pattern to all agent files
in the agents/ folder, using the existing HOLO-1.5 infrastructure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def get_agent_mesh_role(agent_name: str) -> str:
    """Auto-detect HOLO-1.5 mesh role based on agent characteristics."""
    
    # Agent name to mesh role mapping based on AGENTS.md analysis
    role_mapping = {
        # PLANNER role - architects, strategists, planners
        'phi': 'PLANNER',
        'sam': 'PLANNER', 
        'astra': 'PLANNER',
        'orion': 'PLANNER',
        
        # GENERATOR role - creators, synthesizers, builders
        'echo': 'GENERATOR',
        'codeweaver': 'GENERATOR',
        'gizmo': 'GENERATOR',
        'andy': 'GENERATOR',
        'carla': 'GENERATOR',
        'dreamer': 'GENERATOR',
        'voxka': 'GENERATOR',
        'bridgeflesh': 'GENERATOR',
        'pulsesmith': 'GENERATOR',
        
        # CRITIC role - validators, guardians, checkers
        'dave': 'CRITIC',
        'warden': 'CRITIC',
        'mirrorwarden': 'CRITIC',
        'wendy': 'CRITIC',
        
        # EVALUATOR role - analyzers, predictors, assessors
        'oracle': 'EVALUATOR',
        'nebula': 'EVALUATOR',
        'evo': 'EVALUATOR',
        'echolore': 'EVALUATOR',
        'entropybard': 'EVALUATOR',
        'socraticengine': 'EVALUATOR',
        'orionapprentice': 'EVALUATOR',
        'nix': 'EVALUATOR',
        'voxagent': 'EVALUATOR',
        'sdkcontext': 'EVALUATOR',
        'sleep_time_compute_agent': 'EVALUATOR',
        'holo_mesh': 'EVALUATOR'
    }
    
    return role_mapping.get(agent_name.lower(), 'GENERATOR')  # Default to GENERATOR

def get_agent_subsystem(agent_name: str) -> str:
    """Get subsystem mapping for agent."""
    
    # Based on AGENT_SUBSYSTEM_MAP in agents/base.py
    subsystem_mapping = {
        'phi': 'architectonic_frame',
        'voxka': 'dual_cognition_core', 
        'gizmo': 'forge_subsystem',
        'nix': 'chaos_subsystem',
        'echo': 'echo_memory',
        'oracle': 'temporal_foresight',
        'astra': 'navigation',
        'warden': 'integrity_monitor',
        'nebula': 'adaptive_core',
        'orion': 'trust_chain',
        'evo': 'evolution_engine',
        'orionapprentice': 'learning_shard',
        'socraticengine': 'reasoning_module',
        'dreamer': 'dream_state_core',
        'entropybard': 'rag_subsystem',
        'codeweaver': 'meta_learner',
        'echolore': 'historical_archive',
        'mirrorwarden': 'meta_learner',
        'pulsesmith': 'gridformer_connector',
        'bridgeflesh': 'vmb_integration',
        'sam': 'planner_subsystem',
        'dave': 'validator_subsystem',
        'carla': 'speech_style_layer',
        'andy': 'output_composer',
        'wendy': 'tone_audit',
        'voxagent': 'system_interface',
        'sdkcontext': 'module_registry',
        'sleep_time_compute_agent': 'sleep_scheduler',
        'holo_mesh': 'llm_mesh'
    }
    
    return subsystem_mapping.get(agent_name.lower(), 'default_subsystem')

def get_all_agent_files() -> List[Path]:
    """Get all agent files in the agents/ directory."""
    agents_dir = Path('agents')
    if not agents_dir.exists():
        print("❌ agents/ directory not found!")
        return []
    
    agent_files = []
    for file in agents_dir.glob('*.py'):
        if file.name not in ['__init__.py', 'base.py', 'vanta_registration.py', 'enhanced_vanta_registration.py']:
            agent_files.append(file)
    
    return agent_files

def check_if_already_decorated(file_path: Path) -> bool:
    """Check if file already has @vanta_agent decorator."""
    try:
        content = file_path.read_text(encoding='utf-8')
        return '@vanta_agent' in content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def apply_vanta_agent_decorator(file_path: Path) -> bool:
    """Apply @vanta_agent decorator to an agent file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Extract agent name from filename
        agent_name = file_path.stem
        
        # Get mesh role and subsystem
        mesh_role = get_agent_mesh_role(agent_name)
        subsystem = get_agent_subsystem(agent_name)
        
        # Extract class name from file content
        class_match = re.search(r'class (\w+)\(BaseAgent\):', content)
        if not class_match:
            print(f"⚠️  Could not find class definition in {file_path}")
            return False
        
        class_name = class_match.group(1)
        
        # Check if already has the import
        if 'from .base import BaseAgent, vanta_agent, CognitiveMeshRole' not in content:
            # Update the import line
            old_import = 'from .base import BaseAgent'
            new_import = 'from .base import BaseAgent, vanta_agent, CognitiveMeshRole'
            content = content.replace(old_import, new_import)
        
        # Add decorator before class definition
        decorator = f'@vanta_agent(name="{class_name}", subsystem="{subsystem}", mesh_role=CognitiveMeshRole.{mesh_role})'
        
        # Find class definition and add decorator
        class_line_pattern = f'class {class_name}\\(BaseAgent\\):'
        if re.search(class_line_pattern, content):
            content = re.sub(
                class_line_pattern,
                f'{decorator}\nclass {class_name}(BaseAgent):',
                content
            )
        else:
            print(f"⚠️  Could not find class definition pattern in {file_path}")
            return False
        
        # Write updated content
        file_path.write_text(content, encoding='utf-8')
        
        print(f"✅ Applied @vanta_agent decorator to {class_name} ({mesh_role} role)")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Apply encapsulated registration pattern to all agents."""
    print("🤖 Applying Encapsulated Registration Pattern to All Agents")
    print("=" * 60)
    
    # Get all agent files
    agent_files = get_all_agent_files()
    if not agent_files:
        print("❌ No agent files found!")
        return
    
    print(f"📁 Found {len(agent_files)} agent files in agents/ folder")
    
    # Process each file
    processed = 0
    skipped = 0
    failed = 0
    
    for file_path in agent_files:
        print(f"\n📋 Processing: {file_path.name}")
        
        # Check if already decorated
        if check_if_already_decorated(file_path):
            print(f"⏭️  Already decorated: {file_path.name}")
            skipped += 1
            continue
        
        # Apply decorator
        if apply_vanta_agent_decorator(file_path):
            processed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ENCAPSULATED REGISTRATION APPLICATION COMPLETE")
    print(f"✅ Processed: {processed} agents")
    print(f"⏭️  Skipped: {skipped} agents (already decorated)")
    print(f"❌ Failed: {failed} agents")
    print(f"📊 Total: {len(agent_files)} agent files")
    
    if processed > 0:
        print("\n🔥 All agents now have self-registration capabilities with HOLO-1.5 mesh!")
        print("🔗 They will auto-register when instantiated by VantaCore")
        print("🧠 HOLO-1.5 cognitive mesh collaboration ready!")

if __name__ == '__main__':
    main()
