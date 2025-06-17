# middleware/vanta_registration.py
"""
Middleware Module Vanta Registration - HOLO-1.5 Enhanced

This module provides registration capabilities for the middleware module
with the Vanta orchestrator system using HOLO-1.5 cognitive mesh integration.

Components registered:
- HybridMiddleware: Advanced hybrid BLT-enhanced middleware for processing
- BLTMiddleware: Byte Latent Transformer middleware components
- VoxSigilMiddleware: Core VoxSigil middleware processing
- MessageEnhancement: Context enhancement and processing middleware

HOLO-1.5 Integration: Full cognitive mesh integration with middleware processing capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Type

# HOLO-1.5 imports
import sys
import os
# Add the parent directory to the path to ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module

# Configure logging
logger = logging.getLogger(__name__)


@vanta_core_module(
    name="middleware_processor",
    subsystem="communication",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="HOLO-1.5 enhanced middleware processing and communication layer with hybrid BLT capabilities",
    capabilities=[
        "hybrid_middleware_processing",
        "blt_enhanced_communication", 
        "context_enhancement",
        "message_routing",
        "adaptive_compression",
        "entropy_based_routing",
        "patch_processing",
        "sigil_enrichment"
    ],
    dependencies=["core", "BLT", "VoxSigilRag"],
    cognitive_load=2.8,
    symbolic_depth=3,
    collaboration_patterns=[
        "message_enhancement",
        "hybrid_processing",
        "adaptive_routing"
    ]
)
class MiddlewareModule(BaseCore):
    """
    Middleware Module - Communication and processing middleware with cognitive mesh integration
    
    HOLO-1.5 Enhanced Features:
    - Hybrid BLT-enhanced middleware processing with cognitive mesh coordination
    - Advanced message routing and context enhancement
    - Entropy-based adaptive processing with symbolic reasoning
    - Cognitive mesh-aware communication layer optimization
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.cognitive_mesh_role = CognitiveMeshRole.PROCESSOR
        self.module_name = "middleware"
        
    async def initialize_subsystem(self):
        """Initialize middleware subsystem with HOLO-1.5 cognitive mesh"""
        try:
            # Initialize middleware capabilities
            self.logger.info("🔄 Initializing Middleware Module with HOLO-1.5 cognitive mesh")
            
            # Set up cognitive mesh middleware capabilities
            await self._setup_hybrid_middleware()
            await self._initialize_blt_components()
            await self._setup_message_enhancement()
            await self._initialize_communication_layer()
            
            self.logger.info("✅ Middleware Module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Middleware Module: {e}")
            return False
    
    async def _setup_hybrid_middleware(self):
        """Set up hybrid middleware with BLT enhancement"""
        try:
            # Initialize hybrid middleware configuration
            self.hybrid_config = {
                "entropy_threshold": 0.25,
                "blt_hybrid_weight": 0.7,
                "cache_ttl_seconds": 300,
                "log_level": "INFO"
            }
            
            # Register hybrid processing capabilities
            await self._register_processing_capabilities()
            
            self.logger.info("🧠 Hybrid middleware initialized with cognitive mesh")
            
        except Exception as e:
            self.logger.error(f"Failed to setup hybrid middleware: {e}")
    
    async def _initialize_blt_components(self):
        """Initialize BLT middleware components"""
        try:
            # Set up BLT processing capabilities
            self.blt_capabilities = {
                "byte_latent_processing": True,
                "patch_aware_validation": True,
                "entropy_based_compression": True,
                "sigil_patch_encoding": True
            }
            
            # Register BLT enhancement capabilities
            await self._register_blt_capabilities()
            
            self.logger.info("🔧 BLT middleware components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BLT components: {e}")
    
    async def _setup_message_enhancement(self):
        """Set up message context enhancement"""
        try:
            # Initialize context enhancement capabilities
            self.enhancement_config = {
                "rag_enabled": True,
                "max_rag_results": 5,
                "context_optimization": True,
                "recency_boost": True
            }
            
            # Register enhancement capabilities
            await self._register_enhancement_capabilities()
            
            self.logger.info("💬 Message enhancement capabilities initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup message enhancement: {e}")
    
    async def _initialize_communication_layer(self):
        """Initialize communication layer"""
        try:
            # Set up communication protocols
            self.communication_protocols = {
                "message_routing": True,
                "protocol_adaptation": True,
                "cross_domain_translation": True,
                "error_recovery": True
            }
            
            # Register communication capabilities
            await self._register_communication_capabilities()
            
            self.logger.info("📡 Communication layer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication layer: {e}")
    
    async def _register_processing_capabilities(self):
        """Register processing capabilities with cognitive mesh"""
        # Register hybrid processing capabilities
        pass
    
    async def _register_blt_capabilities(self):
        """Register BLT capabilities with cognitive mesh"""
        # Register BLT enhancement capabilities
        pass
    
    async def _register_enhancement_capabilities(self):
        """Register enhancement capabilities with cognitive mesh"""
        # Register message enhancement capabilities
        pass
    
    async def _register_communication_capabilities(self):
        """Register communication capabilities with cognitive mesh"""
        # Register communication layer capabilities
        pass
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message through middleware with cognitive enhancement"""
        try:
            # Apply cognitive mesh processing
            enhanced_message = await self._apply_cognitive_enhancement(message)
            
            # Route through appropriate middleware
            processed_message = await self._route_message(enhanced_message)
            
            return processed_message
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return message
    
    async def _apply_cognitive_enhancement(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cognitive mesh enhancement to message"""
        # Implement cognitive enhancement logic
        return message
    
    async def _route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message through appropriate middleware"""
        # Implement message routing logic
        return message
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get middleware module status"""
        return {
            "module_name": self.module_name,
            "cognitive_mesh_role": self.cognitive_mesh_role.value,
            "hybrid_middleware": getattr(self, 'hybrid_config', None) is not None,
            "blt_components": getattr(self, 'blt_capabilities', None) is not None,
            "message_enhancement": getattr(self, 'enhancement_config', None) is not None,
            "communication_layer": getattr(self, 'communication_protocols', None) is not None,
            "holo15_enabled": True
        }

# Legacy adapter for backwards compatibility
class MiddlewareModuleAdapter:
    """Legacy adapter for registering middleware components as Vanta modules."""
    
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


# Registration function called by master orchestrator
async def register(vanta_core):
    """
    Register Middleware Module with UnifiedVantaCore master orchestrator
    
    This function is called automatically during system initialization
    to integrate the middleware module with the cognitive mesh.
    """
    middleware_module = MiddlewareModule()
    await vanta_core.register_module("middleware", middleware_module)
    return middleware_module
