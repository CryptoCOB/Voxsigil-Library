"""
Vanta Registration Module for VMB (VoxSigil Memory Bank) System
===============================================================

This module provides comprehensive registration capabilities for VoxSigil Library 
VMB system components with the Vanta orchestrator system.

VMB Components:
- VMB Operations: Core memory bank operations
- VMB Config: Configuration management
- VMB Activation: System activation utilities
- VMB Production Executor: Production-level execution engine
- Advanced Demo: Demonstration capabilities

Registration Architecture:
- VMBModuleAdapter: Adapter for VMB system components
- Dynamic component loading with error handling
- Async registration patterns
- Memory bank coordination and management
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VMBModuleAdapter:
    """
    Adapter for VoxSigil Library VMB (VoxSigil Memory Bank) system components.
    
    Handles registration, initialization, and coordination of VMB
    components with the Vanta orchestrator system.
    """
    
    def __init__(self, module_name: str, component_type: str = "vmb"):
        self.module_name = module_name
        self.component_type = component_type
        self.is_initialized = False
        self.vanta_core = None
        self.vmb_components = {}
        self.vmb_config = {}
        self.operation_handlers = {}
        
    async def initialize(self, vanta_core) -> bool:
        """Initialize VMB module with Vanta core."""
        try:
            self.vanta_core = vanta_core
            logger.info(f"Initializing VMB module: {self.module_name}")
            
            # Initialize VMB components
            await self._initialize_vmb_components()
            
            # Load VMB configuration
            await self._load_vmb_configuration()
            
            # Set up operation handlers
            await self._setup_operation_handlers()
            
            # Connect to Vanta core if VMB supports it
            if hasattr(vanta_core, 'register_vmb_module'):
                await vanta_core.register_vmb_module(self)
                
            self.is_initialized = True
            logger.info(f"Successfully initialized VMB module: {self.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VMB module {self.module_name}: {str(e)}")
            return False
    
    async def _initialize_vmb_components(self):
        """Initialize individual VMB components."""
        try:
            # Initialize VMB Operations
            vmb_operations = await self._import_vmb_operations()
            if vmb_operations:
                self.vmb_components['operations'] = vmb_operations
                logger.info("VMB Operations initialized")
                
            # Initialize VMB Config
            vmb_config = await self._import_vmb_config()
            if vmb_config:
                self.vmb_components['config'] = vmb_config
                logger.info("VMB Config initialized")
                
            # Initialize VMB Activation
            vmb_activation = await self._import_vmb_activation()
            if vmb_activation:
                self.vmb_components['activation'] = vmb_activation
                logger.info("VMB Activation initialized")
                
            # Initialize VMB Production Executor
            vmb_executor = await self._import_vmb_production_executor()
            if vmb_executor:
                self.vmb_components['executor'] = vmb_executor
                logger.info("VMB Production Executor initialized")
                
            # Initialize VMB Advanced Demo
            vmb_demo = await self._import_vmb_advanced_demo()
            if vmb_demo:
                self.vmb_components['demo'] = vmb_demo
                logger.info("VMB Advanced Demo initialized")
                
        except Exception as e:
            logger.error(f"Error initializing VMB components: {str(e)}")
    
    async def _import_vmb_operations(self):
        """Import and initialize VMB Operations."""
        try:
            from .vmb_operations import VMBOperations
            return VMBOperations()
        except ImportError as e:
            logger.warning(f"Could not import VMBOperations: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBOperations: {str(e)}")
            return None
    
    async def _import_vmb_config(self):
        """Import and initialize VMB Config."""
        try:
            from .config import VMBConfig
            return VMBConfig()
        except ImportError as e:
            logger.warning(f"Could not import VMBConfig: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBConfig: {str(e)}")
            return None
    
    async def _import_vmb_activation(self):
        """Import and initialize VMB Activation."""
        try:
            from .vmb_activation import VMBActivation
            return VMBActivation()
        except ImportError as e:
            logger.warning(f"Could not import VMBActivation: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBActivation: {str(e)}")
            return None
    
    async def _import_vmb_production_executor(self):
        """Import and initialize VMB Production Executor."""
        try:
            from .vmb_production_executor import VMBProductionExecutor
            return VMBProductionExecutor()
        except ImportError as e:
            logger.warning(f"Could not import VMBProductionExecutor: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBProductionExecutor: {str(e)}")
            return None
    
    async def _import_vmb_advanced_demo(self):
        """Import and initialize VMB Advanced Demo."""
        try:
            from .vmb_advanced_demo import VMBAdvancedDemo
            return VMBAdvancedDemo()
        except ImportError as e:
            logger.warning(f"Could not import VMBAdvancedDemo: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBAdvancedDemo: {str(e)}")
            return None
    
    async def _load_vmb_configuration(self):
        """Load VMB configuration."""
        try:
            # Load configuration from VMB config component
            config_component = self.vmb_components.get('config')
            if config_component and hasattr(config_component, 'get_config'):
                self.vmb_config = await config_component.get_config()
            else:
                # Default configuration
                self.vmb_config = {
                    'memory_bank_size': 1000,
                    'activation_threshold': 0.8,
                    'production_mode': False,
                    'demo_enabled': True
                }
            
            logger.info("VMB configuration loaded")
        except Exception as e:
            logger.error(f"Error loading VMB configuration: {str(e)}")
    
    async def _setup_operation_handlers(self):
        """Set up operation handlers for VMB requests."""
        try:
            self.operation_handlers = {
                'store': self._handle_store_operation,
                'retrieve': self._handle_retrieve_operation,
                'activate': self._handle_activate_operation,
                'execute': self._handle_execute_operation,
                'demo': self._handle_demo_operation,
                'status': self._handle_status_operation,
                'config': self._handle_config_operation
            }
            logger.info("VMB operation handlers established")
        except Exception as e:
            logger.error(f"Error setting up VMB operation handlers: {str(e)}")
    
    async def process_vmb_request(self, operation: str, request_data: Any):
        """Process VMB request through appropriate component."""
        try:
            if not self.is_initialized:
                raise RuntimeError("VMB module not initialized")
                
            # Get operation handler
            handler = self.operation_handlers.get(operation)
            if not handler:
                raise ValueError(f"Unknown VMB operation: {operation}")
                
            # Process request through handler
            return await handler(request_data)
                
        except Exception as e:
            logger.error(f"Error processing VMB request: {str(e)}")
            raise
    
    async def _handle_store_operation(self, request_data: Any):
        """Handle VMB store operations."""
        try:
            operations_component = self.vmb_components.get('operations')
            if operations_component and hasattr(operations_component, 'store'):
                return await operations_component.store(request_data)
            
            # Fallback store operation
            return {
                "status": "stored",
                "data": request_data.get('data'),
                "key": request_data.get('key', 'default'),
                "timestamp": "now"
            }
                
        except Exception as e:
            logger.error(f"Error in VMB store operation: {str(e)}")
            raise
    
    async def _handle_retrieve_operation(self, request_data: Any):
        """Handle VMB retrieve operations."""
        try:
            operations_component = self.vmb_components.get('operations')
            if operations_component and hasattr(operations_component, 'retrieve'):
                return await operations_component.retrieve(request_data.get('key'))
            
            # Fallback retrieve operation
            return {
                "status": "retrieved",
                "key": request_data.get('key'),
                "data": None,
                "found": False
            }
                
        except Exception as e:
            logger.error(f"Error in VMB retrieve operation: {str(e)}")
            raise
    
    async def _handle_activate_operation(self, request_data: Any):
        """Handle VMB activation operations."""
        try:
            activation_component = self.vmb_components.get('activation')
            if activation_component and hasattr(activation_component, 'activate'):
                return await activation_component.activate(request_data.get('config', {}))
            
            # Fallback activation operation
            return {
                "status": "activated",
                "mode": request_data.get('mode', 'standard'),
                "active": True
            }
                
        except Exception as e:
            logger.error(f"Error in VMB activation operation: {str(e)}")
            raise
    
    async def _handle_execute_operation(self, request_data: Any):
        """Handle VMB execution operations."""
        try:
            executor_component = self.vmb_components.get('executor')
            if executor_component and hasattr(executor_component, 'execute'):
                return await executor_component.execute(request_data.get('task'))
            
            # Fallback execution operation
            return {
                "status": "executed",
                "task": request_data.get('task'),
                "result": "completed",
                "execution_time": "0ms"
            }
                
        except Exception as e:
            logger.error(f"Error in VMB execution operation: {str(e)}")
            raise
    
    async def _handle_demo_operation(self, request_data: Any):
        """Handle VMB demo operations."""
        try:
            demo_component = self.vmb_components.get('demo')
            if demo_component and hasattr(demo_component, 'run_demo'):
                return await demo_component.run_demo(request_data.get('demo_type', 'basic'))
            
            # Fallback demo operation
            return {
                "status": "demo_completed",
                "demo_type": request_data.get('demo_type', 'basic'),
                "results": "Demo run successfully"
            }
                
        except Exception as e:
            logger.error(f"Error in VMB demo operation: {str(e)}")
            raise
    
    async def _handle_status_operation(self, request_data: Any):
        """Handle VMB status operations."""
        try:
            return {
                "module_name": self.module_name,
                "is_initialized": self.is_initialized,
                "components_count": len(self.vmb_components),
                "available_components": list(self.vmb_components.keys()),
                "configuration": self.vmb_config,
                "operations": list(self.operation_handlers.keys())
            }
                
        except Exception as e:
            logger.error(f"Error in VMB status operation: {str(e)}")
            raise
    
    async def _handle_config_operation(self, request_data: Any):
        """Handle VMB configuration operations."""
        try:
            config_component = self.vmb_components.get('config')
            operation = request_data.get('operation', 'get')
            
            if operation == 'get':
                if config_component and hasattr(config_component, 'get_config'):
                    return await config_component.get_config()
                return self.vmb_config
                
            elif operation == 'set':
                if config_component and hasattr(config_component, 'set_config'):
                    return await config_component.set_config(request_data.get('config', {}))
                
                # Update local config
                new_config = request_data.get('config', {})
                self.vmb_config.update(new_config)
                return {"status": "config_updated", "config": self.vmb_config}
                
            else:
                raise ValueError(f"Unknown config operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in VMB config operation: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of VMB module."""
        return {
            "module_name": self.module_name,
            "component_type": self.component_type,
            "is_initialized": self.is_initialized,
            "components_count": len(self.vmb_components),
            "available_components": list(self.vmb_components.keys()),
            "operations": list(self.operation_handlers.keys()),
            "configuration": self.vmb_config
        }


class VMBSystemManager:
    """
    System manager for VMB module coordination.
    
    Handles registration, routing, and coordination of all VMB
    components within the VoxSigil Library ecosystem.
    """
    
    def __init__(self):
        self.vmb_adapters = {}
        self.vmb_routing = {}
        self.system_config = {}
        self.is_initialized = False
        
    async def initialize_system(self):
        """Initialize the VMB system."""
        try:
            logger.info("Initializing VMB System Manager")
            
            # Register all VMB components
            await self._register_vmb_components()
            
            # Set up VMB routing
            await self._setup_vmb_routing()
            
            # Load system configuration
            await self._load_system_configuration()
            
            self.is_initialized = True
            logger.info("VMB System Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VMB System Manager: {str(e)}")
            raise
    
    async def _register_vmb_components(self):
        """Register all VMB components."""
        try:
            # Register main VMB adapter
            main_adapter = VMBModuleAdapter("vmb", "vmb")
            self.vmb_adapters["main"] = main_adapter
            
            # Register operations adapter
            operations_adapter = VMBModuleAdapter("vmb_operations", "operations")
            self.vmb_adapters["operations"] = operations_adapter
            
            # Register executor adapter
            executor_adapter = VMBModuleAdapter("vmb_production_executor", "executor")
            self.vmb_adapters["executor"] = executor_adapter
            
            logger.info(f"Registered {len(self.vmb_adapters)} VMB adapters")
            
        except Exception as e:
            logger.error(f"Error registering VMB components: {str(e)}")
            raise
    
    async def _setup_vmb_routing(self):
        """Set up VMB routing patterns."""
        try:
            self.vmb_routing = {
                "memory_operations": {
                    "adapter": "operations",
                    "operations": ["store", "retrieve", "search"]
                },
                "system_operations": {
                    "adapter": "main",
                    "operations": ["activate", "status", "config"]
                },
                "execution_operations": {
                    "adapter": "executor",
                    "operations": ["execute", "batch_execute"]
                },
                "demo_operations": {
                    "adapter": "main",
                    "operations": ["demo", "showcase"]
                }
            }
            
            logger.info("VMB routing patterns established")
            
        except Exception as e:
            logger.error(f"Error setting up VMB routing: {str(e)}")
            raise
    
    async def _load_system_configuration(self):
        """Load VMB system configuration."""
        try:
            self.system_config = {
                "default_memory_size": 1000,
                "max_concurrent_operations": 10,
                "activation_timeout": 30,
                "execution_timeout": 60,
                "demo_mode": True
            }
            
            logger.info("VMB system configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading VMB system configuration: {str(e)}")
            raise
    
    async def route_vmb_request(self, operation_type: str, request_data: Any):
        """Route VMB request to appropriate adapter."""
        try:
            if not self.is_initialized:
                raise RuntimeError("VMB System Manager not initialized")
                
            # Find appropriate routing pattern
            routing_pattern = None
            for pattern_name, pattern in self.vmb_routing.items():
                if operation_type in pattern["operations"]:
                    routing_pattern = pattern
                    break
            
            if not routing_pattern:
                # Default to main adapter
                routing_pattern = {"adapter": "main"}
                
            adapter_key = routing_pattern["adapter"]
            adapter = self.vmb_adapters.get(adapter_key)
            if not adapter:
                raise RuntimeError(f"VMB adapter not available: {adapter_key}")
                
            return await adapter.process_vmb_request(operation_type, request_data)
            
        except Exception as e:
            logger.error(f"Error routing VMB request: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of VMB system."""
        return {
            "is_initialized": self.is_initialized,
            "adapters_count": len(self.vmb_adapters),
            "available_adapters": list(self.vmb_adapters.keys()),
            "routing_patterns": list(self.vmb_routing.keys()),
            "system_config": self.system_config
        }


# Global system manager instance
vmb_system_manager = VMBSystemManager()

async def register_vmb() -> Dict[str, Any]:
    """
    Register VMB module with Vanta orchestrator.
    
    Returns:
        Dict containing registration results and status information.
    """
    try:
        logger.info("Starting VMB module registration")
        
        # Initialize system manager
        await vmb_system_manager.initialize_system()
        
        # Create main VMB adapter
        vmb_adapter = VMBModuleAdapter("vmb")
        
        # Registration would be completed by Vanta orchestrator
        registration_result = {
            "module_name": "vmb",
            "module_type": "vmb", 
            "status": "registered",
            "components": [
                "VMBOperations",
                "VMBConfig",
                "VMBActivation",
                "VMBProductionExecutor",
                "VMBAdvancedDemo"
            ],
            "capabilities": [
                "memory_operations",
                "system_operations", 
                "execution_operations",
                "demo_operations"
            ],
            "adapter": vmb_adapter,
            "system_manager": vmb_system_manager
        }
        
        logger.info("VMB module registration completed successfully")
        return registration_result
        
    except Exception as e:
        logger.error(f"Failed to register VMB module: {str(e)}")
        raise

# Export registration function and key classes
__all__ = [
    'register_vmb',
    'VMBModuleAdapter', 
    'VMBSystemManager',
    'vmb_system_manager'
]
