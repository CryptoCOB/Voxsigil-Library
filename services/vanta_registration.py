# services/vanta_registration.py
"""
Service Connectors Registration with Vanta
==========================================

Registers service connector components that bridge external systems
and services with the Vanta orchestrator.

Components:
- MemoryServiceConnector: UnifiedMemoryInterface service bridge
"""

import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

logger = logging.getLogger("Vanta.ServicesRegistration")


class ServiceModuleAdapter:
    """Adapter for registering service connector components as Vanta modules."""
    
    def __init__(self, module_id: str, service_class: Type, description: str):
        self.module_id = module_id
        self.service_class = service_class
        self.description = description
        self.service_instance = None
        self.capabilities = []
        
    async def initialize(self, vanta_core):
        """Initialize the service instance with vanta core."""
        try:
            # Try to initialize with appropriate parameters
            if hasattr(self.service_class, '__init__'):
                import inspect
                sig = inspect.signature(self.service_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params:
                    self.service_instance = self.service_class(vanta_core=vanta_core)
                elif 'core' in params:
                    self.service_instance = self.service_class(core=vanta_core)
                elif 'config' in params:
                    # For services that take config
                    config = self._get_service_config()
                    self.service_instance = self.service_class(config=config)
                else:
                    self.service_instance = self.service_class()
            else:
                self.service_instance = self.service_class()
                
            # If the service has an initialize method, call it
            if hasattr(self.service_instance, 'initialize'):
                await self.service_instance.initialize()
            elif hasattr(self.service_instance, 'setup'):
                self.service_instance.setup()
            elif hasattr(self.service_instance, 'connect'):
                await self.service_instance.connect()
                
            logger.info(f"Service {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize service {self.module_id}: {e}")
            return False
            
    def _get_service_config(self) -> Dict[str, Any]:
        """Get service-specific configuration."""
        module_name = self.module_id.lower()
        config = {
            "service_type": module_name,
            "enable_logging": True,
            "enable_monitoring": True,
            "timeout": 30
        }
        
        if 'memory' in module_name:
            config.update({
                "memory_pool_size": 1000,
                "enable_persistence": True,
                "cache_strategy": "lru",
                "max_memory_mb": 512
            })
        elif 'connector' in module_name:
            config.update({
                "connection_pool_size": 10,
                "retry_attempts": 3,
                "health_check_interval": 60
            })
            
        return config
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the service."""
        if not self.service_instance:
            return {"error": f"Service {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate service method
            if hasattr(self.service_instance, 'process'):
                result = await self.service_instance.process(request)
            elif hasattr(self.service_instance, 'handle'):
                result = await self.service_instance.handle(request)
            elif hasattr(self.service_instance, 'execute'):
                result = await self.service_instance.execute(request)
            elif hasattr(self.service_instance, 'service_request'):
                result = await self.service_instance.service_request(request)
            elif hasattr(self.service_instance, 'connect_service'):
                result = await self.service_instance.connect_service(request)
            else:
                # For services without specific processing methods
                result = {"message": f"Service {self.module_id} processed request"}
                
            return {"service": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in service {self.module_id}: {e}")
            return {"error": str(e)}
            
    async def check_health(self) -> Dict[str, Any]:
        """Check service health status."""
        try:
            status = {
                "service_id": self.module_id,
                "status": "healthy" if self.service_instance else "not_initialized",
                "capabilities": self._extract_capabilities(),
                "initialized": self.service_instance is not None
            }
            
            # Check if service has health check method
            if self.service_instance:
                if hasattr(self.service_instance, 'check_health'):
                    health_info = await self.service_instance.check_health()
                    status.update({"health_details": health_info})
                elif hasattr(self.service_instance, 'is_connected'):
                    connection_status = self.service_instance.is_connected()
                    status.update({"connected": connection_status})
                elif hasattr(self.service_instance, 'get_status'):
                    service_status = self.service_instance.get_status()
                    status.update({"service_status": service_status})
                
            return status
        except Exception as e:
            logger.error(f"Health check failed for service {self.module_id}: {e}")
            return {"service_id": self.module_id, "status": "error", "error": str(e)}
            
    async def disconnect(self):
        """Disconnect and cleanup service resources."""
        try:
            if self.service_instance:
                if hasattr(self.service_instance, 'disconnect'):
                    await self.service_instance.disconnect()
                elif hasattr(self.service_instance, 'close'):
                    await self.service_instance.close()
                elif hasattr(self.service_instance, 'cleanup'):
                    self.service_instance.cleanup()
                    
            logger.info(f"Service {self.module_id} disconnected successfully")
        except Exception as e:
            logger.error(f"Error disconnecting service {self.module_id}: {e}")
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get service metadata for Vanta registration."""
        metadata = {
            "type": "service",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "service_class": self.service_class.__name__,
            "connection_type": self._get_connection_type()
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on service type."""
        module_name = self.module_id.lower()
        capabilities = ["service", "connector"]
        
        if 'memory' in module_name:
            capabilities.extend(['memory_management', 'data_persistence', 'caching', 'storage'])
        elif 'connector' in module_name:
            capabilities.extend(['external_connection', 'bridge_service', 'integration'])
        elif 'api' in module_name:
            capabilities.extend(['api_gateway', 'request_routing', 'protocol_translation'])
        elif 'database' in module_name:
            capabilities.extend(['database_connection', 'query_execution', 'transaction_management'])
            
        # Add common service capabilities
        capabilities.extend(['connection_management', 'status_monitoring', 'resource_pooling'])
        
        return capabilities
        
    def _get_connection_type(self) -> str:
        """Get the primary connection type for this service."""
        module_name = self.module_id.lower()
        
        if 'memory' in module_name:
            return "memory_service"
        elif 'database' in module_name:
            return "database_connection"
        elif 'api' in module_name:
            return "api_service"
        elif 'connector' in module_name:
            return "bridge_connector"
        else:
            return "generic_service"


def import_service_class(service_name: str):
    """Dynamically import a service class."""
    try:
        if service_name == 'memory_service_connector':
            from services.memory_service_connector import MemoryServiceConnector
            return MemoryServiceConnector
        else:
            logger.warning(f"Unknown service: {service_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import service class {service_name}: {e}")
        return None


async def register_services():
    """Register service connector components."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    service_components = [
        ('memory_service_connector', 'Memory service connector for UnifiedMemoryInterface integration'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🔌 Starting registration of {len(service_components)} service components...")
    
    for service_name, description in service_components:
        try:
            # Import the service class
            service_class = import_service_class(service_name)
            if service_class is None:
                logger.warning(f"Skipping service {service_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = ServiceModuleAdapter(
                module_id=f'service_{service_name}',
                service_class=service_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'service_{service_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered service: {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {str(e)}")
            failed_count += 1
    
    logger.info(f"🎉 Service registration complete: {registered_count}/{len(service_components)} successful")
    
    return {
        'total_services': len(service_components),
        'registered': registered_count,
        'failed': failed_count,
        'success_rate': f"{(registered_count/len(service_components))*100:.1f}%" if len(service_components) > 0 else "N/A"
    }


async def register_single_service(service_name: str, description: str = None):
    """Register a single service component."""
    try:
        from Vanta import get_vanta_core_instance
        
        vanta = get_vanta_core_instance()
        
        # Import the service class
        service_class = import_service_class(service_name)
        if service_class is None:
            raise ValueError(f"Failed to import service class: {service_name}")
        
        # Create adapter
        adapter = ServiceModuleAdapter(
            module_id=f'service_{service_name}',
            service_class=service_class,
            description=description or f"Service: {service_name}"
        )
        
        # Register with Vanta
        await vanta.register_module(f'service_{service_name}', adapter)
        
        logger.info(f"✅ Successfully registered service: {service_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register service {service_name}: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio
    
    async def main():
        logger.info("Starting service components registration...")
        results = await register_services()
        
        print("\n" + "="*50)
        print("🔌 SERVICE COMPONENTS REGISTRATION RESULTS")
        print("="*50)
        print(f"✅ Success Rate: {results['success_rate']}")
        print(f"📊 Services Registered: {results['registered']}/{results['total_services']}")
        if results['failed'] > 0:
            print(f"⚠️ Failed Services: {results['failed']}")
        print("="*50)
        
    asyncio.run(main())
