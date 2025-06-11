# agents/vanta_registration.py
"""
Complete Agent System Registration with Vanta
Registers all 31 agents in the agents/ directory as individual modules
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Vanta.AgentsRegistration")


class AgentModuleAdapter:
    """Adapter for registering individual agents as Vanta modules."""
    
    def __init__(self, module_id: str, agent_class, description: str):
        self.module_id = module_id
        self.agent_class = agent_class
        self.description = description
        self.agent_instance = None
        self.capabilities = []
        
        # Extract agent metadata
        if hasattr(agent_class, 'sigil'):
            self.sigil = agent_class.sigil
        if hasattr(agent_class, 'tags'):
            self.tags = agent_class.tags
        if hasattr(agent_class, 'invocations'):
            self.invocations = agent_class.invocations
            
    async def initialize(self, vanta_core):
        """Initialize the agent instance with vanta core."""
        try:
            self.agent_instance = self.agent_class(vanta_core=vanta_core)
            logger.info(f"Agent {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the agent."""
        if not self.agent_instance:
            return {"error": f"Agent {self.module_id} not initialized"}
            
        try:
            # Route request to agent's handle_message method if available
            if hasattr(self.agent_instance, 'handle_message'):
                result = await self.agent_instance.handle_message(request)
                return {"agent": self.module_id, "result": result}
            else:
                return {"agent": self.module_id, "message": "Agent processed request"}
        except Exception as e:
            logger.error(f"Error processing request in agent {self.module_id}: {e}")
            return {"error": str(e)}
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata for Vanta registration."""
        metadata = {
            "type": "agent",
            "description": self.description,
            "capabilities": ["conversation", "reasoning", "task_execution"],
            "agent_class": self.agent_class.__name__,
        }
        
        if hasattr(self, 'sigil'):
            metadata["sigil"] = self.sigil
        if hasattr(self, 'tags'):
            metadata["tags"] = self.tags
        if hasattr(self, 'invocations'):
            metadata["invocations"] = self.invocations
            
        return metadata


def import_agent_class(agent_name: str):
    """Dynamically import an agent class."""
    try:
        module = __import__(f'agents.{agent_name}', fromlist=[agent_name])
        # Get the class name (capitalize first letter)
        class_name = agent_name.capitalize()
        
        # Handle special cases
        if agent_name == 'andy':
            class_name = 'Andy'
        elif agent_name == 'holo_mesh':
            class_name = 'HoloMesh'
        elif agent_name == 'sleep_time_compute_agent':
            class_name = 'SleepTimeComputeAgent'
        elif agent_name == 'socraticengine':
            class_name = 'SocraticEngine'
        elif agent_name == 'mirrorwarden':
            class_name = 'MirrorWarden'
        elif agent_name == 'orionapprentice':
            class_name = 'OrionApprentice'
        elif agent_name == 'entropybard':
            class_name = 'EntropyBard'
        elif agent_name == 'bridgeflesh':
            class_name = 'BridgeFlesh'
        elif agent_name == 'codeweaver':
            class_name = 'CodeWeaver'
        elif agent_name == 'pulsesmith':
            class_name = 'PulseSmith'
        elif agent_name == 'sdkcontext':
            class_name = 'SDKContext'
        elif agent_name == 'voxagent':
            class_name = 'VoxAgent'
            
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import agent {agent_name}: {e}")
        return None


async def register_all_agents():
    """Register all agents in the agents/ directory."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    # Complete list of all agents
    agent_modules = [
        ('andy', 'Andy Agent - AI assistant with natural conversation'),
        ('astra', 'Astra Agent - Cosmic reasoning and exploration'),
        ('base', 'Base Agent - Foundation agent class'),
        ('bridgeflesh', 'Bridgeflesh Agent - Connection specialist'),
        ('carla', 'Carla Agent - Speech and communication'),
        ('codeweaver', 'Codeweaver Agent - Code generation specialist'),
        ('dave', 'Dave Agent - Practical problem solver'),
        ('dreamer', 'Dreamer Agent - Creative and imaginative reasoning'),
        ('echo', 'Echo Agent - Memory and reflection specialist'),
        ('echolore', 'Echolore Agent - Knowledge curator'),
        ('entropybard', 'Entropybard Agent - Entropy and chaos analysis'),
        ('evo', 'Evo Agent - Evolutionary optimization'),
        ('gizmo', 'Gizmo Agent - Technical gadget specialist'),
        ('holo_mesh', 'Holo Mesh Agent - Distributed agent network'),
        ('mirrorwarden', 'Mirror Warden Agent - Self-reflection guardian'),
        ('nebula', 'Nebula Agent - Cosmic data processing'),
        ('nix', 'Nix Agent - System administration'),
        ('oracle', 'Oracle Agent - Prediction and forecasting'),
        ('orion', 'Orion Agent - Navigation and guidance'),
        ('orionapprentice', 'Orion Apprentice - Learning navigator'),
        ('phi', 'Phi Agent - Mathematical reasoning'),
        ('pulsesmith', 'Pulsesmith Agent - Rhythm and pattern analysis'),
        ('sam', 'Sam Agent - Strategic analysis'),
        ('sdkcontext', 'SDK Context Agent - Development context'),
        ('sleep_time_compute_agent', 'Sleep Compute Agent - Resource management'),
        ('socraticengine', 'Socratic Engine Agent - Question-based learning'),
        ('voxagent', 'Vox Agent - Voice processing specialist'),
        ('voxka', 'Voxka Agent - Multi-modal interaction'),
        ('warden', 'Warden Agent - Security and monitoring'),
        ('wendy', 'Wendy Agent - User experience specialist'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🤖 Starting registration of {len(agent_modules)} agents...")
    
    for agent_name, description in agent_modules:
        try:
            # Import the agent class
            agent_class = import_agent_class(agent_name)
            if agent_class is None:
                logger.warning(f"Skipping agent {agent_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = AgentModuleAdapter(
                module_id=f'agent_{agent_name}',
                agent_class=agent_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'agent_{agent_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register agent {agent_name}: {e}")
            failed_count += 1
    
    logger.info(f"🎉 Agent registration complete: {registered_count} successful, {failed_count} failed")
    return registered_count, failed_count


async def register_base_agent():
    """Register the base agent class for other modules to use."""
    from Vanta import get_vanta_core_instance
    from .base import BaseAgent
    
    vanta = get_vanta_core_instance()
    
    adapter = AgentModuleAdapter(
        module_id='base_agent',
        agent_class=BaseAgent,
        description='Base Agent - Foundation class for all agents'
    )
    
    await vanta.register_module('base_agent', adapter)
    logger.info("✅ Registered base agent foundation class")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        await register_base_agent()
        await register_all_agents()
    
    asyncio.run(main())
