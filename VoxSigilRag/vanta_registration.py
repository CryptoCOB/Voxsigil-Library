# VoxSigilRag/vanta_registration.py
"""
VoxSigil RAG System Registration with Vanta
===========================================

Registers all VoxSigil RAG components and processors with the Vanta orchestrator
for centralized management and coordination.
"""

import logging
from typing import Dict, Any, Type

logger = logging.getLogger("Vanta.RAGRegistration")


class RAGModuleAdapter:
    """Adapter for registering RAG system components as Vanta modules."""
    
    def __init__(self, module_id: str, rag_class: Type, description: str):
        self.module_id = module_id
        self.rag_class = rag_class
        self.description = description
        self.rag_instance = None
        self.capabilities = ['retrieval', 'generation', 'augmentation']
        
    async def initialize(self, vanta_core):
        """Initialize the RAG component with vanta core."""
        try:
            # Try to initialize with vanta_core parameter first
            if hasattr(self.rag_class, '__init__'):
                import inspect
                sig = inspect.signature(self.rag_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params or 'core' in params:
                    self.rag_instance = self.rag_class(vanta_core=vanta_core)
                elif 'config' in params:
                    # For RAG components that take config
                    config = {
                        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                        'chunk_size': 512,
                        'chunk_overlap': 50,
                        'max_context_length': 4096
                    }
                    self.rag_instance = self.rag_class(config=config)
                else:
                    self.rag_instance = self.rag_class()
            else:
                self.rag_instance = self.rag_class()
                
            # If the RAG component has an initialize method, call it
            if hasattr(self.rag_instance, 'initialize'):
                await self.rag_instance.initialize()
            elif hasattr(self.rag_instance, 'setup'):
                self.rag_instance.setup()
                
            logger.info(f"RAG component {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG component {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a RAG request through the component."""
        if not self.rag_instance:
            return {"error": f"RAG component {self.module_id} not initialized"}
            
        try:
            # Route request to component's process method if available
            if hasattr(self.rag_instance, 'process'):
                result = await self.rag_instance.process(request)
                return {"rag_component": self.module_id, "result": result}
            elif hasattr(self.rag_instance, 'query'):
                query = request.get('query', '')
                result = await self.rag_instance.query(query)
                return {"rag_component": self.module_id, "result": result}
            elif hasattr(self.rag_instance, 'retrieve'):
                query = request.get('query', '')
                result = self.rag_instance.retrieve(query)
                return {"rag_component": self.module_id, "result": result}
            else:
                return {"error": f"RAG component {self.module_id} has no processing method"}
        except Exception as e:
            logger.error(f"Error processing request in RAG component {self.module_id}: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG component."""
        return {
            "module_id": self.module_id,
            "initialized": self.rag_instance is not None,
            "capabilities": self.capabilities,
            "description": self.description
        }


def import_rag_class(component_name: str):
    """Import a RAG component class by name."""
    try:
        module_name = f"VoxSigilRag.{component_name}"
        module = __import__(module_name, fromlist=[component_name])
        
        # Try common class naming patterns
        class_names = [
            # Exact name
            component_name,
            # CamelCase variations
            ''.join(word.capitalize() for word in component_name.split('_')),
            # With common suffixes
            f"{component_name.title().replace('_', '')}",
            f"{component_name.title().replace('_', '')}Processor",
            f"{component_name.title().replace('_', '')}RAG",
            f"{component_name.title().replace('_', '')}System",
        ]
        
        for class_name in class_names:
            if hasattr(module, class_name):
                return getattr(module, class_name)
                
        # If no class found, look for any class in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                not attr_name.startswith('_') and
                attr_name not in ['logging', 'Dict', 'Any', 'List', 'Optional']):
                logger.info(f"Using class {attr_name} for component {component_name}")
                return attr
                
        logger.warning(f"No suitable class found in {module_name}")
        return None
        
    except ImportError as e:
        logger.warning(f"Could not import RAG component {component_name}: {e}")
        return None


async def register_voxsigil_rag():
    """Register all VoxSigil RAG components with Vanta."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    rag_components = [
        ('voxsigil_rag', 'Main VoxSigil RAG processor'),
        ('voxsigil_blt', 'BLT-enhanced RAG system'),
        ('voxsigil_blt_rag', 'BLT RAG integration layer'),
        ('voxsigil_evaluator', 'RAG response evaluation system'),
        ('voxsigil_mesh', 'RAG mesh networking system'),
        ('voxsigil_processors', 'RAG data processors'),
        ('hybrid_blt', 'Hybrid BLT middleware'),
        ('sigil_patch_encoder', 'Sigil patch encoding system'),
        ('voxsigil_semantic_cache', 'Semantic caching system'),
        ('voxsigil_rag_compression', 'RAG compression engine'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🔄 Starting registration of {len(rag_components)} RAG components...")
    
    for component_name, description in rag_components:
        try:
            # Import the RAG component class
            rag_class = import_rag_class(component_name)
            if rag_class is None:
                logger.warning(f"Skipping RAG component {component_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = RAGModuleAdapter(
                module_id=f'rag_{component_name}',
                rag_class=rag_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'rag_{component_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered RAG component: {component_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register RAG component {component_name}: {e}")
            failed_count += 1
    
    logger.info(f"🎉 RAG component registration complete: {registered_count} successful, {failed_count} failed")
    return registered_count, failed_count


if __name__ == "__main__":
    import asyncio
    asyncio.run(register_voxsigil_rag())
