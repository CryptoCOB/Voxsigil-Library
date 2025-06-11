# handlers/vanta_registration.py
"""
Integration Handlers Registration with Vanta
============================================

Registers integration handlers for ARC LLM, RAG, speech processing,
and VMB integration with the Vanta orchestrator.
"""

import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

logger = logging.getLogger("Vanta.HandlersRegistration")


class HandlerModuleAdapter:
    """Adapter for registering integration handlers as Vanta modules."""
    
    def __init__(self, module_id: str, handler_class: Type, description: str):
        self.module_id = module_id
        self.handler_class = handler_class
        self.description = description
        self.handler_instance = None
        self.capabilities = []
        
    async def initialize(self, vanta_core):
        """Initialize the handler instance with vanta core."""
        try:
            # Try to initialize with vanta_core parameter first
            if hasattr(self.handler_class, '__init__'):
                import inspect
                sig = inspect.signature(self.handler_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params:
                    self.handler_instance = self.handler_class(vanta_core=vanta_core)
                elif 'core' in params:
                    self.handler_instance = self.handler_class(core=vanta_core)
                elif 'config' in params:
                    # For handlers that take config
                    config = {}  # Basic config
                    self.handler_instance = self.handler_class(config=config)
                else:
                    self.handler_instance = self.handler_class()
            else:
                self.handler_instance = self.handler_class()
                
            # If the handler has an initialize method, call it
            if hasattr(self.handler_instance, 'initialize'):
                await self.handler_instance.initialize()
            elif hasattr(self.handler_instance, 'setup'):
                self.handler_instance.setup()
                
            logger.info(f"Handler {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize handler {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the handler."""
        if not self.handler_instance:
            return {"error": f"Handler {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate handler method
            if hasattr(self.handler_instance, 'handle'):
                result = await self.handler_instance.handle(request)
            elif hasattr(self.handler_instance, 'process'):
                result = await self.handler_instance.process(request)
            elif hasattr(self.handler_instance, 'handle_request'):
                result = await self.handler_instance.handle_request(request)
            elif hasattr(self.handler_instance, 'integrate'):
                result = await self.handler_instance.integrate(request)
            else:
                result = {"message": f"Handler {self.module_id} processed request"}
                
            return {"handler": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in handler {self.module_id}: {e}")
            return {"error": str(e)}
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get handler metadata for Vanta registration."""
        metadata = {
            "type": "integration_handler",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "handler_class": self.handler_class.__name__,
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on handler type."""
        module_name = self.module_id.lower()
        capabilities = ["integration", "handler"]
        
        if 'arc' in module_name:
            capabilities.extend(['arc_integration', 'llm_bridge', 'cognitive_architecture'])
        elif 'rag' in module_name:
            capabilities.extend(['rag_integration', 'document_retrieval', 'context_generation'])
        elif 'speech' in module_name:
            capabilities.extend(['speech_processing', 'audio_integration', 'tts_stt'])
        elif 'vmb' in module_name:
            capabilities.extend(['vmb_integration', 'model_building', 'production_execution'])
        
        # Add common handler capabilities
        capabilities.extend(['request_handling', 'integration_management'])
        
        return capabilities


def import_handler_class(handler_name: str):
    """Dynamically import a handler class."""
    try:
        if handler_name == 'arc_llm_handler':
            from handlers.arc_llm_handler import ARCLLMHandler
            return ARCLLMHandler
        elif handler_name == 'rag_integration_handler':
            from handlers.rag_integration_handler import RagIntegrationHandler
            return RagIntegrationHandler
        elif handler_name == 'speech_integration_handler':
            from handlers.speech_integration_handler import SpeechIntegrationHandler
            return SpeechIntegrationHandler
        elif handler_name == 'vmb_integration_handler':
            from handlers.vmb_integration_handler import VMBIntegrationHandler
            return VMBIntegrationHandler
        else:
            logger.warning(f"Unknown handler: {handler_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import handler class {handler_name}: {e}")
        return None


async def register_handlers():
    """Register integration handlers."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    handlers = [
        ('arc_llm_handler', 'ARC LLM integration handler'),
        ('rag_integration_handler', 'RAG integration handler'),
        ('speech_integration_handler', 'Speech integration handler'),
        ('vmb_integration_handler', 'VMB integration handler'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🔗 Starting registration of {len(handlers)} integration handlers...")
    
    for handler_name, description in handlers:
        try:
            # Import the handler class
            handler_class = import_handler_class(handler_name)
            if handler_class is None:
                logger.warning(f"Skipping handler {handler_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = HandlerModuleAdapter(
                module_id=f'handler_{handler_name}',
                handler_class=handler_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'handler_{handler_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered handler: {handler_name}")
            
        except Exception as e:
            logger.error(f"Failed to register handler {handler_name}: {str(e)}")
            failed_count += 1
    
    logger.info(f"🎉 Handler registration complete: {registered_count}/{len(handlers)} successful")
    
    return {
        'total_handlers': len(handlers),
        'registered': registered_count,
        'failed': failed_count,
        'success_rate': f"{(registered_count/len(handlers))*100:.1f}%"
    }


async def register_single_handler(handler_name: str, description: str = None):
    """Register a single integration handler."""
    try:
        from Vanta import get_vanta_core_instance
        
        vanta = get_vanta_core_instance()
        
        # Import the handler class
        handler_class = import_handler_class(handler_name)
        if handler_class is None:
            raise ValueError(f"Failed to import handler class: {handler_name}")
        
        # Create adapter
        adapter = HandlerModuleAdapter(
            module_id=f'handler_{handler_name}',
            handler_class=handler_class,
            description=description or f"Handler: {handler_name}"
        )
        
        # Register with Vanta
        await vanta.register_module(f'handler_{handler_name}', adapter)
        
        logger.info(f"✅ Successfully registered handler: {handler_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register handler {handler_name}: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("Starting integration handlers registration...")
        results = await register_handlers()
        
        print("\n" + "="*50)
        print("🔗 INTEGRATION HANDLERS REGISTRATION RESULTS")
        print("="*50)
        print(f"✅ Success Rate: {results['success_rate']}")
        print(f"📊 Handlers Registered: {results['registered']}/{results['total_handlers']}")
        if results['failed'] > 0:
            print(f"⚠️ Failed Handlers: {results['failed']}")
        print("="*50)
        
    asyncio.run(main())
