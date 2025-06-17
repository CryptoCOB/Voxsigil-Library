# engines/vanta_registration.py
"""
Processing Engines Registration with Vanta
Registers all processing engines as individual modules
"""

import logging
from typing import Dict, Any, List, Type

logger = logging.getLogger("Vanta.EnginesRegistration")


class EngineModuleAdapter:
    """Adapter for registering processing engines as Vanta modules."""
    
    def __init__(self, module_id: str, engine_class: Type, description: str):
        self.module_id = module_id
        self.engine_class = engine_class
        self.description = description
        self.engine_instance = None
        self.capabilities = []
        
    async def initialize(self, vanta_core):
        """Initialize the engine instance with vanta core."""
        try:
            # Try to initialize with vanta_core parameter first
            if hasattr(self.engine_class, '__init__'):
                import inspect
                sig = inspect.signature(self.engine_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params or 'core' in params:
                    self.engine_instance = self.engine_class(vanta_core=vanta_core)
                elif 'config' in params:
                    # For engines that take config, create a minimal config
                    config = getattr(self.engine_class, 'Config', None)
                    if config:
                        self.engine_instance = self.engine_class(config=config())
                    else:
                        self.engine_instance = self.engine_class()
                else:
                    self.engine_instance = self.engine_class()
            else:
                self.engine_instance = self.engine_class()
                
            # If the engine has an initialize method, call it
            if hasattr(self.engine_instance, 'initialize'):
                await self.engine_instance.initialize()
            elif hasattr(self.engine_instance, 'initialize_subsystem'):
                self.engine_instance.initialize_subsystem(vanta_core)
                
            logger.info(f"Engine {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize engine {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the engine."""
        if not self.engine_instance:
            return {"error": f"Engine {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate engine method
            if hasattr(self.engine_instance, 'process'):
                result = await self.engine_instance.process(request)
            elif hasattr(self.engine_instance, 'handle_request'):
                result = await self.engine_instance.handle_request(request)
            elif hasattr(self.engine_instance, 'execute'):
                result = await self.engine_instance.execute(request)
            else:
                result = {"message": f"Engine {self.module_id} processed request"}
                
            return {"engine": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in engine {self.module_id}: {e}")
            return {"error": str(e)}
            
    async def shutdown(self):
        """Shutdown the engine gracefully."""
        if self.engine_instance and hasattr(self.engine_instance, 'shutdown'):
            await self.engine_instance.shutdown()
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get engine metadata for Vanta registration."""
        metadata = {
            "type": "processing_engine",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "engine_class": self.engine_class.__name__,
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on engine type."""
        engine_name = self.module_id.lower()
        
        if 'async' in engine_name:
            capabilities = ['asynchronous_processing']
        else:
            capabilities = ['processing']
            
        if 'stt' in engine_name:
            capabilities.append('speech_to_text')
        elif 'tts' in engine_name:
            capabilities.append('text_to_speech')
        elif 'training' in engine_name:
            capabilities.append('model_training')
        elif 'rag' in engine_name:
            capabilities.append('rag_processing')
        elif 'compression' in engine_name:
            capabilities.append('data_compression')
        elif 'cat' in engine_name:
            capabilities.extend(['cognitive_architecture', 'reasoning'])
        elif 'hybrid' in engine_name:
            capabilities.extend(['multi_modal', 'hybrid_processing'])
        elif 'tot' in engine_name:
            capabilities.extend(['tree_of_thoughts', 'reasoning'])
            
        return capabilities


def import_engine_class(engine_name: str):
    """Dynamically import an engine class."""
    try:
        module = __import__(f'engines.{engine_name}', fromlist=[engine_name])
        
        # Get the main class from the engine module
        class_mapping = {
            'async_processing_engine': 'AsyncProcessor',
            'async_stt_engine': 'AsyncSTTEngine',
            'async_tts_engine': 'AsyncTTSEngine',
            'async_training_engine': 'AsyncTrainingEngine',
            'cat_engine': 'CATEngine',
            'hybrid_cognition_engine': 'HybridCognitionEngine',
            'rag_compression_engine': 'RAGCompressionEngine',
            'tot_engine': 'ToTEngine',
        }
        
        class_name = class_mapping.get(engine_name)
        if not class_name:
            # Fallback: try to find the main class in the module
            classes = [name for name in dir(module) if name[0].isupper() and not name.startswith('_')]
            if classes:
                class_name = classes[0]  # Take the first class found
            else:
                logger.error(f"No suitable class found in engine {engine_name}")
                return None
                
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import engine {engine_name}: {e}")
        return None


async def register_engines():
    """Register all processing engines."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    engines = [
        ('async_processing_engine', 'Asynchronous task processing engine'),
        ('async_stt_engine', 'Asynchronous speech-to-text processing'),
        ('async_tts_engine', 'Text-to-speech synthesis engine'),
        ('async_training_engine', 'Asynchronous training pipeline'),
        ('cat_engine', 'Cognitive Architecture Toolkit engine'),
        ('hybrid_cognition_engine', 'Multi-modal cognitive processing'),
        ('rag_compression_engine', 'RAG compression and optimization'),
        ('tot_engine', 'Tree of Thoughts reasoning engine'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"⚙️ Starting registration of {len(engines)} processing engines...")
    
    for engine_name, description in engines:
        try:
            # Import the engine class
            engine_class = import_engine_class(engine_name)
            if engine_class is None:
                logger.warning(f"Skipping engine {engine_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = EngineModuleAdapter(
                module_id=f'engine_{engine_name}',
                engine_class=engine_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'engine_{engine_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered engine: {engine_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register engine {engine_name}: {e}")
            failed_count += 1
    
    logger.info(f"🎉 Engine registration complete: {registered_count} successful, {failed_count} failed")
    return registered_count, failed_count


if __name__ == "__main__":
    import asyncio
    asyncio.run(register_engines())
