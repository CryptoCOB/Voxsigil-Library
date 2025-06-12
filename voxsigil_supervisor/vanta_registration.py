# voxsigil_supervisor/vanta_registration.py
"""
VoxSigil Supervisor System Registration with Vanta
==================================================

Registers VoxSigil supervisor components and engines with the Vanta orchestrator
for centralized supervision and coordination.
"""

import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

logger = logging.getLogger("Vanta.SupervisorRegistration")


class SupervisorModuleAdapter:
    """Adapter for registering supervisor components as Vanta modules."""
    
    def __init__(self, module_id: str, component_class: Type, description: str):
        self.module_id = module_id
        self.component_class = component_class
        self.description = description
        self.component_instance = None
        self.capabilities = ['supervision', 'coordination', 'monitoring']
        
    async def initialize(self, vanta_core):
        """Initialize the supervisor component with vanta core."""
        try:
            # Try to initialize with vanta_core parameter first
            if hasattr(self.component_class, '__init__'):
                import inspect
                sig = inspect.signature(self.component_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params or 'core' in params:
                    self.component_instance = self.component_class(vanta_core=vanta_core)
                elif 'supervisor_config' in params or 'config' in params:
                    # For supervisor components that take config
                    config = {
                        'max_concurrent_tasks': 10,
                        'supervision_interval': 1.0,
                        'health_check_interval': 5.0,
                        'logging_level': 'INFO'
                    }
                    self.component_instance = self.component_class(config=config)
                else:
                    self.component_instance = self.component_class()
            else:
                self.component_instance = self.component_class()
                
            # If the supervisor component has an initialize method, call it
            if hasattr(self.component_instance, 'initialize'):
                await self.component_instance.initialize()
            elif hasattr(self.component_instance, 'setup'):
                self.component_instance.setup()
            elif hasattr(self.component_instance, 'start'):
                self.component_instance.start()
                
            logger.info(f"Supervisor component {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize supervisor component {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a supervision request through the component."""
        if not self.component_instance:
            return {"error": f"Supervisor component {self.module_id} not initialized"}
            
        try:
            # Route request to component's supervise method if available
            if hasattr(self.component_instance, 'supervise'):
                result = await self.component_instance.supervise(request)
                return {"supervisor": self.module_id, "result": result}
            elif hasattr(self.component_instance, 'process'):
                result = await self.component_instance.process(request)
                return {"supervisor": self.module_id, "result": result}
            elif hasattr(self.component_instance, 'handle_request'):
                result = self.component_instance.handle_request(request)
                return {"supervisor": self.module_id, "result": result}
            else:
                return {"error": f"Supervisor component {self.module_id} has no processing method"}
        except Exception as e:
            logger.error(f"Error processing request in supervisor component {self.module_id}: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the supervisor component."""
        status = {
            "module_id": self.module_id,
            "initialized": self.component_instance is not None,
            "capabilities": self.capabilities,
            "description": self.description
        }
        
        # Add component-specific status if available
        if self.component_instance and hasattr(self.component_instance, 'get_status'):
            try:
                status["component_status"] = self.component_instance.get_status()
            except Exception as e:
                status["component_status"] = f"Error getting status: {e}"
                
        return status


def import_supervisor_class(component_name: str):
    """Import a supervisor component class by name."""
    try:
        if component_name == 'supervisor_engine':
            from .supervisor_engine import SupervisorEngine
            return SupervisorEngine
        elif component_name == 'supervisor_engine_compat':
            from .supervisor_engine_compat import SupervisorEngineCompat
            return SupervisorEngineCompat
        elif component_name == 'supervisor_wrapper':
            from .supervisor_wrapper import SupervisorWrapper
            return SupervisorWrapper
        elif component_name == 'blt_supervisor_integration':
            from .blt_supervisor_integration import BLTSupervisorIntegration
            return BLTSupervisorIntegration
        else:
            # Try dynamic import
            module_name = f"voxsigil_supervisor.{component_name}"
            module = __import__(module_name, fromlist=[component_name])
            
            # Try common class naming patterns
            class_names = [
                # Exact name
                component_name,
                # CamelCase variations
                ''.join(word.capitalize() for word in component_name.split('_')),
                # With common suffixes
                f"{component_name.title().replace('_', '')}",
                f"{component_name.title().replace('_', '')}Engine",
                f"{component_name.title().replace('_', '')}Supervisor",
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
        logger.warning(f"Could not import supervisor component {component_name}: {e}")
        return None


async def register_supervisor():
    """Register all VoxSigil supervisor components with Vanta."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    supervisor_components = [
        ('supervisor_engine', 'Main supervisor engine'),
        ('supervisor_engine_compat', 'Supervisor engine compatibility layer'),
        ('supervisor_wrapper', 'Supervisor wrapper for legacy components'),
        ('blt_supervisor_integration', 'BLT supervisor integration'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"👁️ Starting registration of {len(supervisor_components)} supervisor components...")
    
    for component_name, description in supervisor_components:
        try:
            # Import the supervisor component class
            component_class = import_supervisor_class(component_name)
            if component_class is None:
                logger.warning(f"Skipping supervisor component {component_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = SupervisorModuleAdapter(
                module_id=f'supervisor_{component_name}',
                component_class=component_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'supervisor_{component_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered supervisor component: {component_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register supervisor component {component_name}: {e}")
            failed_count += 1
    
    logger.info(f"🎉 Supervisor registration complete: {registered_count} successful, {failed_count} failed")
    return registered_count, failed_count


if __name__ == "__main__":
    import asyncio
    asyncio.run(register_supervisor())
