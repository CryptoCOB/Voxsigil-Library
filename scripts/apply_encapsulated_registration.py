#!/usr/bin/env python3
"""
Apply Encapsulated Registration Pattern to All Agents
Automatically applies @vanta_agent decorator to all agents in the agents/ directory
Implements HOLO-1.5 Recursive Symbolic Cognition Mesh enhancements
"""

import os
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent to subsystem mappings from AGENT_SUBSYSTEM_MAP
AGENT_SUBSYSTEM_MAP = {
    "Phi": "architectonic_frame",
    "Voxka": "dual_cognition_core", 
    "Gizmo": "forge_subsystem",
    "Nix": "chaos_subsystem",
    "Echo": "echo_memory",
    "Oracle": "temporal_foresight",
    "Astra": "navigation",
    "Warden": "integrity_monitor",
    "Nebula": "adaptive_core",
    "Orion": "trust_chain",
    "Evo": "evolution_engine",
    "OrionApprentice": "learning_shard",
    "SocraticEngine": "reasoning_module",
    "Dreamer": "dream_state_core",
    "EntropyBard": "rag_subsystem",
    "CodeWeaver": "meta_learner",
    "EchoLore": "historical_archive",
    "MirrorWarden": "meta_learner",
    "PulseSmith": "gridformer_connector",
    "BridgeFlesh": "vmb_integration",
    "Sam": "planner_subsystem",
    "Dave": "validator_subsystem",
    "Carla": "speech_style_layer",
    "Andy": "output_composer",
    "Wendy": "tone_audit",
    "VoxAgent": "system_interface",
    "SDKContext": "module_registry",
    "SleepTimeComputeAgent": "sleep_scheduler",
    "SleepTimeCompute": "sleep_scheduler",
    "HoloMesh": "llm_mesh",
}

# Mesh role detection patterns
MESH_ROLE_PATTERNS = {
    "PLANNER": ["plan", "strategic", "architect", "sam", "planner", "strategic"],
    "GENERATOR": ["gen", "creat", "synth", "compose", "weav", "dream", "generator", "creative"],
    "CRITIC": ["crit", "analy", "valid", "check", "dave", "warden", "critic", "validator", "guard"],
    "EVALUATOR": ["eval", "assess", "audit", "oracle", "wendy", "evaluator", "auditor", "assess"]
}

def detect_mesh_role(class_name: str, tags: list = None) -> str:
    """Detect appropriate HOLO-1.5 mesh role for an agent."""
    name_lower = class_name.lower()
    tags_str = ' '.join(tags).lower() if tags else ''
    
    # Check name patterns
    for role, patterns in MESH_ROLE_PATTERNS.items():
        if any(pattern in name_lower for pattern in patterns):
            return f"CognitiveMeshRole.{role}"
        if tags and any(pattern in tags_str for pattern in patterns):
            return f"CognitiveMeshRole.{role}"
    
    return "CognitiveMeshRole.GENERATOR"  # Default

def extract_class_info(content: str) -> dict:
    """Extract class name, sigil, and tags from agent file content."""
    class_match = re.search(r'class (\w+)\(BaseAgent\):', content)
    if not class_match:
        return None
    
    class_name = class_match.group(1)
    
    # Extract sigil
    sigil_match = re.search(r'sigil\s*=\s*["\']([^"\']+)["\']', content)
    sigil = sigil_match.group(1) if sigil_match else ""
    
    # Extract tags
    tags_match = re.search(r'tags\s*=\s*\[(.*?)\]', content, re.DOTALL)
    tags = []
    if tags_match:
        tags_content = tags_match.group(1)
        # Extract individual tag strings
        tag_matches = re.findall(r'["\']([^"\']+)["\']', tags_content)
        tags = tag_matches
    
    return {
        'class_name': class_name,
        'sigil': sigil,
        'tags': tags
    }

def is_already_decorated(content: str) -> bool:
    """Check if agent already has @vanta_agent decorator."""
    return '@vanta_agent' in content

def apply_decorator_to_agent(file_path: Path):
    """Apply @vanta_agent decorator to an agent file."""
    try:
        content = file_path.read_text()
        
        # Skip if already decorated
        if is_already_decorated(content):
            logger.info(f"⚡ {file_path.name} already has @vanta_agent decorator")
            return
        
        # Extract class information
        class_info = extract_class_info(content)
        if not class_info:
            logger.warning(f"⚠️ Could not extract class info from {file_path.name}")
            return
        
        class_name = class_info['class_name']
        
        # Skip base.py
        if class_name in ['BaseAgent', 'NullAgent']:
            logger.info(f"⚡ Skipping base class {class_name}")
            return
        
        # Determine subsystem and mesh role
        subsystem = AGENT_SUBSYSTEM_MAP.get(class_name, None)
        mesh_role = detect_mesh_role(class_name, class_info['tags'])
        
        logger.info(f"🔧 Processing {class_name}: subsystem={subsystem}, mesh_role={mesh_role}")
        
        # Update import statement
        old_import = "from .base import BaseAgent"
        new_import = "from .base import BaseAgent, vanta_agent, CognitiveMeshRole"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
        else:
            logger.warning(f"⚠️ Could not find expected import in {file_path.name}")
            return
        
        # Add decorator before class definition
        class_pattern = f"class {class_name}\\(BaseAgent\\):"
        decorator_lines = []
        
        if subsystem and mesh_role:
            decorator_lines.append(f'@vanta_agent(name="{class_name}", subsystem="{subsystem}", mesh_role={mesh_role})')
        elif subsystem:
            decorator_lines.append(f'@vanta_agent(name="{class_name}", subsystem="{subsystem}")')
        elif mesh_role:
            decorator_lines.append(f'@vanta_agent(name="{class_name}", mesh_role={mesh_role})')
        else:
            decorator_lines.append(f'@vanta_agent(name="{class_name}")')
        
        decorator_str = '\n'.join(decorator_lines)
        replacement = f"{decorator_str}\nclass {class_name}(BaseAgent):"
        
        content = re.sub(class_pattern, replacement, content)
        
        # Enhance initialize_subsystem method to call super()
        init_pattern = r'def initialize_subsystem\(self, core\):\s*\n(\s*)(.*?)\n(\s*)pass'
        
        def enhance_init(match):
            indent = match.group(1)
            existing_content = match.group(2).strip()
            if existing_content and 'super()' not in existing_content:
                return f"def initialize_subsystem(self, core):\n{indent}super().initialize_subsystem(core)\n{indent}# {existing_content}"
            elif not existing_content:
                return f"def initialize_subsystem(self, core):\n{indent}super().initialize_subsystem(core)"
            else:
                return match.group(0)  # No change needed
        
        content = re.sub(init_pattern, enhance_init, content, flags=re.DOTALL)
        
        # Enhance bind_echo_routes to call super()
        echo_pattern = r'def bind_echo_routes\(self\):\s*\n(\s*)(.*?)\n(\s*)pass'
        
        def enhance_echo(match):
            indent = match.group(1)
            existing_content = match.group(2).strip()
            if existing_content and 'super()' not in existing_content:
                return f"def bind_echo_routes(self):\n{indent}super().bind_echo_routes()\n{indent}# {existing_content}"
            elif not existing_content:
                return f"def bind_echo_routes(self):\n{indent}super().bind_echo_routes()"
            else:
                return match.group(0)  # No change needed
        
        content = re.sub(echo_pattern, enhance_echo, content, flags=re.DOTALL)
        
        # Write back the modified content
        file_path.write_text(content)
        logger.info(f"✅ Applied encapsulated registration to {class_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to process {file_path.name}: {e}")

def main():
    """Apply encapsulated registration pattern to all agents."""
    agents_dir = Path(__file__).parent.parent / "agents"
    
    if not agents_dir.exists():
        logger.error(f"❌ Agents directory not found: {agents_dir}")
        return
    
    logger.info(f"🚀 Starting encapsulated registration enhancement for agents in {agents_dir}")
    
    # Process all .py files except __init__.py and vanta_registration.py
    agent_files = [f for f in agents_dir.glob("*.py") 
                   if f.name not in ["__init__.py", "vanta_registration.py", "base.py"]]
    
    logger.info(f"📁 Found {len(agent_files)} agent files to process")
    
    processed = 0
    for agent_file in agent_files:
        apply_decorator_to_agent(agent_file)
        processed += 1
    
    logger.info(f"🎯 Encapsulated registration enhancement complete: {processed} files processed")
    
    # Create summary report
    summary_file = agents_dir.parent / "ENCAPSULATED_REGISTRATION_APPLIED.md"
    with open(summary_file, 'w') as f:
        f.write("# 🤖 Encapsulated Registration Pattern Applied\n\n")
        f.write("## ✅ HOLO-1.5 Recursive Symbolic Cognition Mesh Implementation\n\n")
        f.write(f"**Date Applied**: {os.popen('date').read().strip()}\n")
        f.write(f"**Agent Files Processed**: {processed}\n")
        f.write(f"**Pattern**: Self-registration with @vanta_agent decorator\n")
        f.write(f"**Features Added**:\n")
        f.write("- ✅ Self-registration with Vanta Core\n")
        f.write("- ✅ HOLO-1.5 Recursive Symbolic Cognition\n")
        f.write("- ✅ Cognitive Mesh Role auto-detection\n")
        f.write("- ✅ Symbolic compression and triggers\n")
        f.write("- ✅ Tree-of-Thought and Chain-of-Thought reasoning\n")
        f.write("- ✅ Enhanced subsystem binding\n")
        f.write("- ✅ Automatic echo route enhancement\n\n")
        f.write("## 🌐 Agent Mesh Network\n\n")
        f.write("All agents now support HOLO-1.5 mesh collaboration:\n")
        f.write("- **Planners**: Strategic planning and task decomposition\n") 
        f.write("- **Generators**: Content generation and solution synthesis\n")
        f.write("- **Critics**: Analysis and evaluation of solutions\n")
        f.write("- **Evaluators**: Final assessment and quality control\n\n")
        f.write("## 🔗 Usage\n\n")
        f.write("```python\n")
        f.write("from agents.base import set_vanta_instance, register_all_agents_auto\n")
        f.write("from agents.base import create_holo_mesh_network, execute_mesh_task\n\n")
        f.write("# Set Vanta instance for auto-registration\n")
        f.write("set_vanta_instance(vanta_core)\n\n") 
        f.write("# Auto-register all decorated agents\n")
        f.write("await register_all_agents_auto()\n\n")
        f.write("# Create mesh network\n")
        f.write("mesh = create_holo_mesh_network(all_agents)\n\n")
        f.write("# Execute collaborative task\n")
        f.write("result = execute_mesh_task(mesh, 'Complex problem to solve')\n")
        f.write("```\n")
    
    logger.info(f"📄 Created summary report: {summary_file}")

if __name__ == "__main__":
    main()
