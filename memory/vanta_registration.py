# memory/vanta_registration.py
"""
Memory Subsystems Registration with Vanta
==========================================

Registers memory subsystems including echo memory, external echo layers,
and braided memory architectures with the Vanta orchestrator.
"""

import logging
from typing import Dict, Any, List, Type

logger = logging.getLogger("Vanta.MemoryRegistration")


class MemoryModuleAdapter:
    """Adapter for registering memory subsystems as Vanta modules."""
    
    def __init__(self, module_id: str, memory_class: Type, description: str):
        self.module_id = module_id
        self.memory_class = memory_class
        self.description = description
        self.memory_instance = None
        self.capabilities = []
        
    async def initialize(self, vanta_core):
        """Initialize the memory instance with vanta core."""
        try:
            # Try to initialize with vanta_core parameter first
            if hasattr(self.memory_class, '__init__'):
                import inspect
                sig = inspect.signature(self.memory_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params or 'core' in params:
                    self.memory_instance = self.memory_class(vanta_core=vanta_core)
                elif 'config' in params:
                    # For memory systems that take config
                    config = {}  # Basic config
                    self.memory_instance = self.memory_class(config=config)
                else:
                    self.memory_instance = self.memory_class()
            else:
                self.memory_instance = self.memory_class()
                
            # If the memory system has an initialize method, call it
            if hasattr(self.memory_instance, 'initialize'):
                await self.memory_instance.initialize()
            elif hasattr(self.memory_instance, 'setup'):
                self.memory_instance.setup()
                
            logger.info(f"Memory system {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory system {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the memory system."""
        if not self.memory_instance:
            return {"error": f"Memory system {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate memory method
            if hasattr(self.memory_instance, 'store'):
                result = await self.memory_instance.store(request)
            elif hasattr(self.memory_instance, 'retrieve'):
                result = await self.memory_instance.retrieve(request)
            elif hasattr(self.memory_instance, 'process'):
                result = await self.memory_instance.process(request)
            elif hasattr(self.memory_instance, 'handle_request'):
                result = await self.memory_instance.handle_request(request)
            else:
                result = {"message": f"Memory system {self.module_id} processed request"}
                
            return {"memory_system": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in memory system {self.module_id}: {e}")
            return {"error": str(e)}
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get memory system metadata for Vanta registration."""
        metadata = {
            "type": "memory_subsystem",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "memory_class": self.memory_class.__name__,
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on memory type."""
        module_name = self.module_id.lower()
        capabilities = ["memory"]
        
        if 'echo' in module_name:
            capabilities.extend(['echo_memory', 'cognitive_traces', 'reflection'])
        elif 'braid' in module_name:
            capabilities.extend(['braided_memory', 'multi_stream', 'memory_weaving'])
        elif 'external' in module_name:
            capabilities.extend(['external_memory', 'layer_processing', 'echo_processing'])
        
        # Add common memory capabilities
        capabilities.extend(['storage', 'retrieval', 'memory_management'])
        
        return capabilities


def import_memory_class(module_name: str):
    """Dynamically import a memory class."""
    try:
        if module_name == 'echo_memory':
            from memory.echo_memory import EchoMemory
            return EchoMemory
        elif module_name == 'external_echo_layer':
            from memory.external_echo_layer import ExternalEchoLayer
            return ExternalEchoLayer
        elif module_name == 'memory_braid':
            from memory.memory_braid import MemoryBraid
            return MemoryBraid
        else:
            logger.warning(f"Unknown memory module: {module_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import memory class {module_name}: {e}")
        return None


async def register_memory_modules():
    """Register all memory subsystems."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    memory_modules = [
        ('echo_memory', 'Echo memory system for cognitive traces'),
        ('external_echo_layer', 'External echo processing layer'),
        ('memory_braid', 'Braided memory architecture'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🧠 Starting registration of {len(memory_modules)} memory modules...")
    
    for module_name, description in memory_modules:
        try:
            # Import the memory module class
            memory_class = import_memory_class(module_name)
            if memory_class is None:
                logger.warning(f"Skipping memory module {module_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = MemoryModuleAdapter(
                module_id=f'memory_{module_name}',
                memory_class=memory_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'memory_{module_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered memory module: {module_name}")
            
        except Exception as e:
            logger.error(f"Failed to register memory module {module_name}: {str(e)}")
            failed_count += 1
    
    logger.info(f"🎉 Memory module registration complete: {registered_count}/{len(memory_modules)} successful")
    
    return {
        'total_modules': len(memory_modules),
        'registered': registered_count,
        'failed': failed_count,
        'success_rate': f"{(registered_count/len(memory_modules))*100:.1f}%"
    }


async def register_single_memory_module(module_name: str, description: str = None):
    """Register a single memory module."""
    try:
        from Vanta import get_vanta_core_instance
        
        vanta = get_vanta_core_instance()
        
        # Import the memory class
        memory_class = import_memory_class(module_name)
        if memory_class is None:
            raise ValueError(f"Failed to import memory class: {module_name}")
        
        # Create adapter
        adapter = MemoryModuleAdapter(
            module_id=f'memory_{module_name}',
            memory_class=memory_class,
            description=description or f"Memory module: {module_name}"
        )
        
        # Register with Vanta
        await vanta.register_module(f'memory_{module_name}', adapter)
        
        logger.info(f"✅ Successfully registered memory module: {module_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register memory module {module_name}: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("Starting memory modules registration...")
        results = await register_memory_modules()
        
        print("\n" + "="*50)
        print("🧠 MEMORY MODULES REGISTRATION RESULTS")
        print("="*50)
        print(f"✅ Success Rate: {results['success_rate']}")
        print(f"📊 Modules Registered: {results['registered']}/{results['total_modules']}")
        if results['failed'] > 0:
            print(f"⚠️ Failed Modules: {results['failed']}")
        print("="*50)
        
    asyncio.run(main())
