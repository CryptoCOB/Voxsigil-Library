"""
Enhanced Vanta Registration System with HOLO-1.5 Recursive Symbolic Cognition Mesh
Integrates self-registration pattern with existing Vanta infrastructure
"""

import logging
import asyncio
from typing import Dict, Any
from agents.base import (
    set_vanta_instance, 
    get_registered_agents, 
    register_all_agents_auto,
    create_holo_mesh_network,
    execute_mesh_task
)

logger = logging.getLogger(__name__)


class EnhancedVantaRegistration:
    """Enhanced registration system with HOLO-1.5 mesh capabilities."""
    
    def __init__(self, vanta_core):
        self.vanta_core = vanta_core
        self.mesh_network = None
        self.registered_agents = {}
        
        # Set global Vanta instance for auto-registration
        set_vanta_instance(vanta_core)
        
    async def initialize_enhanced_registration(self):
        """Initialize the enhanced registration system."""
        logger.info("🚀 Initializing Enhanced Vanta Registration with HOLO-1.5")
        
        try:
            # Auto-register all decorated agents
            result = await register_all_agents_auto()
            if result:
                logger.info(f"✅ Auto-registration: {result['registered']} success, {result['failed']} failed")
            
            # Get all registered agents from the decorator registry
            decorator_agents = get_registered_agents()
            logger.info(f"🤖 Found {len(decorator_agents)} agents in decorator registry")
            
            # Import and instantiate agents
            self.registered_agents = await self._import_all_agents()
            
            # Create HOLO-1.5 mesh network
            if self.registered_agents:
                agent_instances = list(self.registered_agents.values())
                self.mesh_network = create_holo_mesh_network(agent_instances)
                logger.info("🌐 HOLO-1.5 mesh network created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced registration: {e}")
            return False
    
    async def _import_all_agents(self) -> Dict[str, Any]:
        """Import and instantiate all agent classes."""
        agents = {}
        
        # Import all agent modules
        agent_modules = [
            ('phi', 'Phi'),
            ('echo', 'Echo'), 
            ('dave', 'Dave'),
            ('oracle', 'Oracle'),
            ('sam', 'Sam'),
            ('codeweaver', 'CodeWeaver'),
            ('andy', 'Andy'),
            ('astra', 'Astra'),
            ('carla', 'Carla'),
            ('wendy', 'Wendy'),
            ('nebula', 'Nebula'),
            ('orion', 'Orion'),
            ('evo', 'Evo'),
            ('dreamer', 'Dreamer'),
            ('entropybard', 'EntropyBard'),
            ('echolore', 'EchoLore'),
            ('mirrorwarden', 'MirrorWarden'),
            ('pulsesmith', 'PulseSmith'),
            ('bridgeflesh', 'BridgeFlesh'),
            ('gizmo', 'Gizmo'),
            ('nix', 'Nix'),
            ('warden', 'Warden'),
            ('orionapprentice', 'OrionApprentice'),
            ('socraticengine', 'SocraticEngine'),
            ('voxagent', 'VoxAgent'),
            ('sdkcontext', 'SDKContext'),
            ('holo_mesh', 'HoloMesh'),
            ('sleep_time_compute_agent', 'SleepTimeComputeAgent'),
            ('voxka', 'Voxka'),
        ]
        
        for module_name, class_name in agent_modules:
            try:
                # Import the module
                module = __import__(f'agents.{module_name}', fromlist=[class_name])
                agent_class = getattr(module, class_name)
                
                # Instantiate with Vanta core (this will trigger auto-registration)
                agent_instance = agent_class(self.vanta_core)
                agents[class_name] = agent_instance
                
                logger.debug(f"✅ Imported and instantiated {class_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to import {class_name}: {e}")
        
        logger.info(f"📦 Successfully imported {len(agents)} agent instances")
        return agents
    
    async def execute_mesh_collaboration(self, task: str) -> Dict[str, Any]:
        """Execute a task through the HOLO-1.5 mesh network."""
        if not self.mesh_network:
            logger.warning("⚠️ No mesh network available for collaboration")
            return {"error": "No mesh network available"}
        
        logger.info(f"🧠 Executing mesh collaboration: {task}")
        return execute_mesh_task(self.mesh_network, task)
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get status of the HOLO-1.5 mesh network."""
        if not self.mesh_network:
            return {"status": "not_initialized"}
        
        status = {
            "status": "active",
            "roles": {},
            "total_agents": 0
        }
        
        for role, agents in self.mesh_network.items():
            status["roles"][role] = [agent.__class__.__name__ for agent in agents]
            status["total_agents"] += len(agents)
        
        return status
    
    async def demonstrate_mesh_capabilities(self):
        """Demonstrate HOLO-1.5 mesh capabilities."""
        logger.info("🎭 Demonstrating HOLO-1.5 Recursive Symbolic Cognition Mesh")
        
        # Test basic symbolic compression
        if self.registered_agents and 'Phi' in self.registered_agents:
            phi_agent = self.registered_agents['Phi']
            
            # Test symbolic compression
            test_data = {"complex": "data", "with": ["multiple", "values"]}
            symbol_ref = phi_agent.compress_to_symbol(test_data, "test_data")
            expanded = phi_agent.expand_symbol(symbol_ref)
            
            logger.info(f"🔄 Symbolic compression test:")
            logger.info(f"  Original: {test_data}")
            logger.info(f"  Compressed: {symbol_ref}")
            logger.info(f"  Expanded: {expanded}")
            
            # Test cognitive chain
            chain_id = phi_agent.create_cognitive_chain("Test HOLO-1.5 reasoning", "chain_of_thought")
            phi_agent.add_reasoning_step(chain_id, "Initialize mesh network", "Setting up cognitive mesh")
            phi_agent.add_reasoning_step(chain_id, "Execute symbolic processing", "Compressing data symbols")
            
            logger.info(f"🧠 Created cognitive chain: {chain_id}")
            logger.info(f"  Chains: {len(phi_agent._cognitive_chains)}")
            
        # Test mesh collaboration
        if self.mesh_network:
            result = await self.execute_mesh_collaboration("Analyze system performance and suggest improvements")
            logger.info(f"🤝 Mesh collaboration result: {result}")
        
        logger.info("✅ HOLO-1.5 demonstration complete")


# Factory function for easy integration
async def create_enhanced_vanta_registration(vanta_core):
    """Create and initialize enhanced Vanta registration system."""
    registration = EnhancedVantaRegistration(vanta_core)
    success = await registration.initialize_enhanced_registration()
    
    if success:
        logger.info("🎯 Enhanced Vanta Registration System ready")
        return registration
    else:
        logger.error("❌ Failed to create enhanced registration system")
        return None


# Integration with existing registration
async def enhance_existing_registration():
    """Enhance existing Vanta registration with HOLO-1.5 capabilities."""
    try:
        from Vanta import get_vanta_core_instance
        vanta_core = get_vanta_core_instance()
        
        if vanta_core:
            enhanced_registration = await create_enhanced_vanta_registration(vanta_core)
            if enhanced_registration:
                # Demonstrate capabilities
                await enhanced_registration.demonstrate_mesh_capabilities()
                return enhanced_registration
        else:
            logger.warning("⚠️ No Vanta core instance available")
            
    except Exception as e:
        logger.error(f"❌ Failed to enhance existing registration: {e}")
    
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(enhance_existing_registration())
