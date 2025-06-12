# middleware/vanta_registration.py
"""
Middleware Components Registration with Vanta
=============================================

Registers middleware components for communication, compression, BLT operations,
and hybrid processing with the Vanta orchestrator.

Components:
- HybridMiddleware: RAG + LLM hybrid processing
- VoxSigilMiddleware: Core VoxSigil middleware (redirector)
- VoxSigilBLTMiddleware: BLT-specific middleware operations
- BLTCompressionMiddleware: Compression optimization for BLT
- BLTMiddlewareLoader: Dynamic middleware loading system
"""

import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

logger = logging.getLogger("Vanta.MiddlewareRegistration")


class MiddlewareModuleAdapter:
    """Adapter for registering middleware components as Vanta modules."""
    
    def __init__(self, module_id: str, middleware_class: Type, description: str):
        self.module_id = module_id
        self.middleware_class = middleware_class
        self.description = description
        self.middleware_instance = None
        self.capabilities = []
        
    async def initialize(self, vanta_core):
        """Initialize the middleware instance with vanta core."""
        try:
            # Try to initialize with appropriate parameters
            if hasattr(self.middleware_class, '__init__'):
                import inspect
                sig = inspect.signature(self.middleware_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params:
                    self.middleware_instance = self.middleware_class(vanta_core=vanta_core)
                elif 'core' in params:
                    self.middleware_instance = self.middleware_class(core=vanta_core)
                elif 'config' in params:
                    # For middleware that takes config
                    config = self._get_middleware_config()
                    self.middleware_instance = self.middleware_class(config=config)
                else:
                    self.middleware_instance = self.middleware_class()
            else:
                self.middleware_instance = self.middleware_class()
                
            # If the middleware has an initialize method, call it
            if hasattr(self.middleware_instance, 'initialize'):
                await self.middleware_instance.initialize()
            elif hasattr(self.middleware_instance, 'setup'):
                self.middleware_instance.setup()
                
            logger.info(f"Middleware {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize middleware {self.module_id}: {e}")
            return False
            
    def _get_middleware_config(self) -> Dict[str, Any]:
        """Get middleware-specific configuration."""
        module_name = self.module_id.lower()
        config = {
            "middleware_type": module_name,
            "enable_logging": True,
            "enable_metrics": True
        }
        
        if 'hybrid' in module_name:
            config.update({
                "enable_rag": True,
                "enable_llm": True,
                "hybrid_mode": "balanced",
                "cache_size": 1000
            })
        elif 'blt' in module_name:
            config.update({
                "blt_enabled": True,
                "optimization_level": 2,
                "compression_enabled": 'compression' in module_name
            })
        elif 'compression' in module_name:
            config.update({
                "compression_level": 6,
                "enable_caching": True,
                "max_cache_size": "100MB"
            })
        elif 'loader' in module_name:
            config.update({
                "auto_discovery": True,
                "load_timeout": 30,
                "fallback_enabled": True
            })
            
        return config
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the middleware."""
        if not self.middleware_instance:
            return {"error": f"Middleware {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate middleware method
            if hasattr(self.middleware_instance, 'process'):
                result = await self.middleware_instance.process(request)
            elif hasattr(self.middleware_instance, 'handle'):
                result = await self.middleware_instance.handle(request)
            elif hasattr(self.middleware_instance, 'execute'):
                result = await self.middleware_instance.execute(request)
            elif hasattr(self.middleware_instance, 'middleware_process'):
                result = await self.middleware_instance.middleware_process(request)
            else:
                # For simple middleware, just pass through
                result = {"message": f"Middleware {self.module_id} processed request"}
                
            return {"middleware": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in middleware {self.module_id}: {e}")
            return {"error": str(e)}
            
    async def check_health(self) -> Dict[str, Any]:
        """Check middleware health status."""
        try:
            status = {
                "middleware_id": self.module_id,
                "status": "healthy" if self.middleware_instance else "not_initialized",
                "capabilities": self._extract_capabilities(),
                "initialized": self.middleware_instance is not None
            }
            
            # Check if middleware has health check method
            if self.middleware_instance and hasattr(self.middleware_instance, 'check_health'):
                health_info = await self.middleware_instance.check_health()
                status.update({"health_details": health_info})
                
            return status
        except Exception as e:
            logger.error(f"Health check failed for middleware {self.module_id}: {e}")
            return {"middleware_id": self.module_id, "status": "error", "error": str(e)}
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get middleware metadata for Vanta registration."""
        metadata = {
            "type": "middleware",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "middleware_class": self.middleware_class.__name__,
            "processing_type": self._get_processing_type()
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on middleware type."""
        module_name = self.module_id.lower()
        capabilities = ["middleware", "processing"]
        
        if 'hybrid' in module_name:
            capabilities.extend(['hybrid_processing', 'rag_middleware', 'llm_middleware', 'multi_modal'])
        elif 'blt' in module_name:
            capabilities.extend(['blt_processing', 'language_optimization', 'encoding'])
        elif 'compression' in module_name:
            capabilities.extend(['compression', 'data_optimization', 'bandwidth_reduction'])
        elif 'loader' in module_name:
            capabilities.extend(['dynamic_loading', 'module_discovery', 'plugin_management'])
        elif 'voxsigil' in module_name:
            capabilities.extend(['voxsigil_processing', 'audio_middleware', 'voice_processing'])
            
        # Add common middleware capabilities
        capabilities.extend(['request_routing', 'data_transformation', 'pipeline_processing'])
        
        return capabilities
        
    def _get_processing_type(self) -> str:
        """Get the primary processing type for this middleware."""
        module_name = self.module_id.lower()
        
        if 'hybrid' in module_name:
            return "hybrid_processing"
        elif 'blt' in module_name:
            return "blt_processing"
        elif 'compression' in module_name:
            return "compression_processing"
        elif 'loader' in module_name:
            return "loader_management"
        else:
            return "general_middleware"


def import_middleware_class(middleware_name: str):
    """Dynamically import a middleware class."""
    try:
        if middleware_name == 'hybrid_middleware':
            from middleware.hybrid_middleware import HybridMiddleware
            return HybridMiddleware
        elif middleware_name == 'voxsigil_middleware':
            from middleware.voxsigil_middleware import VoxSigilMiddleware
            return VoxSigilMiddleware
        elif middleware_name == 'voxsigil_blt_middleware':
            from middleware.voxsigil_blt_middleware import VoxSigilBLTMiddleware
            return VoxSigilBLTMiddleware
        elif middleware_name == 'blt_compression_middleware':
            from middleware.blt_compression_middleware import BLTCompressionMiddleware
            return BLTCompressionMiddleware
        elif middleware_name == 'blt_middleware_loader':
            from middleware.blt_middleware_loader import BLTMiddlewareLoader
            return BLTMiddlewareLoader
        else:
            logger.warning(f"Unknown middleware: {middleware_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import middleware class {middleware_name}: {e}")
        return None


async def register_middleware():
    """Register middleware components."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    middleware_components = [
        ('hybrid_middleware', 'Hybrid RAG+LLM middleware for multi-modal processing'),
        ('voxsigil_middleware', 'Core VoxSigil middleware for voice processing'),
        ('voxsigil_blt_middleware', 'BLT-specific middleware for language optimization'),
        ('blt_compression_middleware', 'Compression middleware for BLT data optimization'),
        ('blt_middleware_loader', 'Dynamic middleware loading and management system'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🔄 Starting registration of {len(middleware_components)} middleware components...")
    
    for middleware_name, description in middleware_components:
        try:
            # Import the middleware class
            middleware_class = import_middleware_class(middleware_name)
            if middleware_class is None:
                logger.warning(f"Skipping middleware {middleware_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = MiddlewareModuleAdapter(
                module_id=f'middleware_{middleware_name}',
                middleware_class=middleware_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'middleware_{middleware_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered middleware: {middleware_name}")
            
        except Exception as e:
            logger.error(f"Failed to register middleware {middleware_name}: {str(e)}")
            failed_count += 1
    
    logger.info(f"🎉 Middleware registration complete: {registered_count}/{len(middleware_components)} successful")
    
    return {
        'total_middleware': len(middleware_components),
        'registered': registered_count,
        'failed': failed_count,
        'success_rate': f"{(registered_count/len(middleware_components))*100:.1f}%"
    }


async def register_single_middleware(middleware_name: str, description: str = None):
    """Register a single middleware component."""
    try:
        from Vanta import get_vanta_core_instance
        
        vanta = get_vanta_core_instance()
        
        # Import the middleware class
        middleware_class = import_middleware_class(middleware_name)
        if middleware_class is None:
            raise ValueError(f"Failed to import middleware class: {middleware_name}")
        
        # Create adapter
        adapter = MiddlewareModuleAdapter(
            module_id=f'middleware_{middleware_name}',
            middleware_class=middleware_class,
            description=description or f"Middleware: {middleware_name}"
        )
        
        # Register with Vanta
        await vanta.register_module(f'middleware_{middleware_name}', adapter)
        
        logger.info(f"✅ Successfully registered middleware: {middleware_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register middleware {middleware_name}: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("Starting middleware components registration...")
        results = await register_middleware()
        
        print("\n" + "="*50)
        print("🔄 MIDDLEWARE COMPONENTS REGISTRATION RESULTS")
        print("="*50)
        print(f"✅ Success Rate: {results['success_rate']}")
        print(f"📊 Middleware Registered: {results['registered']}/{results['total_middleware']}")
        if results['failed'] > 0:
            print(f"⚠️ Failed Middleware: {results['failed']}")
        print("="*50)
        
    asyncio.run(main())
