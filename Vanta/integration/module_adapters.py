"""
Vanta Integration Layer
======================

Provides standardized adapters and integration utilities for connecting
existing modules to the Vanta orchestrator system.

Key Components:
- Base module adapters for common patterns
- Legacy system integration helpers
- Configuration management
- Health monitoring utilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from abc import abstractmethod
from datetime import datetime

from ..interfaces import ModuleAdapterProtocol, VantaProtocol
from ..core.orchestrator import vanta_orchestrator


class BaseModuleAdapter(ModuleAdapterProtocol):
    """
    Base implementation of ModuleAdapterProtocol providing common functionality
    for integrating existing modules with Vanta.
    """
    
    def __init__(
        self,
        module_id: str,
        module_info: Dict[str, Any],
        capabilities: List[str],
        target_module: Any = None
    ):
        self._module_id = module_id
        self._module_info = module_info
        self._capabilities = capabilities
        self._target_module = target_module
        self._vanta_client: Optional[VantaProtocol] = None
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{module_id}")
        
        # Health metrics
        self._health_metrics = {
            'status': 'initializing',
            'last_health_check': None,
            'error_count': 0,
            'request_count': 0,
            'success_rate': 1.0
        }
    
    @property
    def module_id(self) -> str:
        return self._module_id
    
    @property
    def module_info(self) -> Dict[str, Any]:
        return {
            **self._module_info,
            'initialized': self._initialized,
            'health_status': self._health_metrics['status']
        }
    
    @property
    def capabilities(self) -> List[str]:
        return self._capabilities.copy()
    
    async def initialize(
        self,
        vanta_client: VantaProtocol,
        config: Dict[str, Any]
    ) -> bool:
        """Initialize module adapter with Vanta client."""
        try:
            self._logger.info(f"Initializing module adapter: {self._module_id}")
            
            self._vanta_client = vanta_client
            self._config = config
            
            # Initialize target module if provided
            if self._target_module and hasattr(self._target_module, 'initialize'):
                if asyncio.iscoroutinefunction(self._target_module.initialize):
                    await self._target_module.initialize(config)
                else:
                    self._target_module.initialize(config)
            
            # Perform adapter-specific initialization
            await self._adapter_initialize()
            
            self._initialized = True
            self._health_metrics['status'] = 'healthy'
            self._logger.info(f"Successfully initialized module: {self._module_id}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize module {self._module_id}: {str(e)}")
            self._health_metrics['status'] = 'error'
            self._health_metrics['error_count'] += 1
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown module adapter."""
        try:
            self._logger.info(f"Shutting down module adapter: {self._module_id}")
            
            # Perform adapter-specific cleanup
            await self._adapter_shutdown()
            
            # Shutdown target module if provided
            if self._target_module and hasattr(self._target_module, 'shutdown'):
                if asyncio.iscoroutinefunction(self._target_module.shutdown):
                    await self._target_module.shutdown()
                else:
                    self._target_module.shutdown()
            
            self._initialized = False
            self._health_metrics['status'] = 'shutdown'
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to shutdown module {self._module_id}: {str(e)}")
            return False
    
    async def handle_request(
        self,
        request_type: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming request from Vanta."""
        self._health_metrics['request_count'] += 1
        
        try:
            # Validate request
            if request_type not in self._capabilities:
                raise ValueError(f"Unsupported request type: {request_type}")
            
            # Route to appropriate handler
            result = await self._handle_request_internal(request_type, payload, context)
            
            # Update success rate
            self._update_success_rate(True)
            
            return {
                'success': True,
                'result': result,
                'module_id': self._module_id,
                'request_type': request_type
            }
            
        except Exception as e:
            self._logger.error(f"Request handling error in {self._module_id}: {str(e)}")
            self._health_metrics['error_count'] += 1
            self._update_success_rate(False)
            
            return {
                'success': False,
                'error': str(e),
                'module_id': self._module_id,
                'request_type': request_type
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            # Perform adapter-specific health check
            health_data = await self._adapter_health_check()
            
            # Check target module health if available
            if self._target_module and hasattr(self._target_module, 'health_check'):
                if asyncio.iscoroutinefunction(self._target_module.health_check):
                    target_health = await self._target_module.health_check()
                else:
                    target_health = self._target_module.health_check()
                
                health_data['target_module_health'] = target_health
            
            # Update health status
            self._health_metrics['last_health_check'] = datetime.utcnow().isoformat()
            
            if health_data.get('status') == 'healthy':
                self._health_metrics['status'] = 'healthy'
            elif health_data.get('status') == 'degraded':
                self._health_metrics['status'] = 'degraded'
            else:
                self._health_metrics['status'] = 'unhealthy'
            
            return {
                **self._health_metrics,
                **health_data
            }
            
        except Exception as e:
            self._logger.error(f"Health check failed for {self._module_id}: {str(e)}")
            self._health_metrics['status'] = 'error'
            self._health_metrics['error_count'] += 1
            
            return {
                **self._health_metrics,
                'error': str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get module performance metrics."""
        try:
            # Get adapter-specific metrics
            metrics = await self._adapter_get_metrics()
            
            # Get target module metrics if available
            if self._target_module and hasattr(self._target_module, 'get_metrics'):
                if asyncio.iscoroutinefunction(self._target_module.get_metrics):
                    target_metrics = await self._target_module.get_metrics()
                else:
                    target_metrics = self._target_module.get_metrics()
                
                metrics['target_module_metrics'] = target_metrics
            
            return {
                'module_id': self._module_id,
                'health_metrics': self._health_metrics,
                'adapter_metrics': metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get metrics for {self._module_id}: {str(e)}")
            return {
                'module_id': self._module_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _update_success_rate(self, success: bool) -> None:
        """Update success rate using exponential moving average."""
        if success:
            self._health_metrics['success_rate'] = (
                0.9 * self._health_metrics['success_rate'] + 0.1 * 1.0
            )
        else:
            self._health_metrics['success_rate'] = (
                0.9 * self._health_metrics['success_rate'] + 0.1 * 0.0
            )
    
    # Abstract methods for subclasses to implement
    
    async def _adapter_initialize(self) -> None:
        """Adapter-specific initialization logic."""
        pass
    
    async def _adapter_shutdown(self) -> None:
        """Adapter-specific shutdown logic."""
        pass
    
    @abstractmethod
    async def _handle_request_internal(
        self,
        request_type: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Handle the actual request logic."""
        pass
    
    async def _adapter_health_check(self) -> Dict[str, Any]:
        """Adapter-specific health check logic."""
        return {'status': 'healthy'}
    
    async def _adapter_get_metrics(self) -> Dict[str, Any]:
        """Adapter-specific metrics collection."""
        return {}


class LegacyModuleAdapter(BaseModuleAdapter):
    """
    Adapter for integrating legacy modules that don't follow
    the new interface patterns.
    """
    
    def __init__(
        self,
        module_id: str,
        legacy_module: Any,
        method_mapping: Dict[str, str],
        module_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize legacy module adapter.
        
        Args:
            module_id: Unique identifier for the module
            legacy_module: The legacy module instance
            method_mapping: Maps request types to legacy module methods
            module_info: Optional module metadata
        """
        capabilities = list(method_mapping.keys())
        info = module_info or {
            'name': module_id,
            'type': 'legacy',
            'description': f'Legacy module adapter for {module_id}'
        }
        
        super().__init__(module_id, info, capabilities, legacy_module)
        self._method_mapping = method_mapping
    
    async def _handle_request_internal(
        self,
        request_type: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Handle request by calling mapped legacy method."""
        if request_type not in self._method_mapping:
            raise ValueError(f"No method mapping for request type: {request_type}")
        
        method_name = self._method_mapping[request_type]
        
        if not hasattr(self._target_module, method_name):
            raise AttributeError(f"Legacy module has no method: {method_name}")
        
        method = getattr(self._target_module, method_name)
        
        # Call method with appropriate arguments
        if asyncio.iscoroutinefunction(method):
            return await method(**payload)
        else:
            return method(**payload)
    
    async def _adapter_health_check(self) -> Dict[str, Any]:
        """Check if legacy module methods are accessible."""
        try:
            # Verify all mapped methods exist
            for request_type, method_name in self._method_mapping.items():
                if not hasattr(self._target_module, method_name):
                    return {
                        'status': 'unhealthy',
                        'error': f'Missing method: {method_name}'
                    }
            
            return {'status': 'healthy', 'legacy_methods_verified': True}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class ClassBasedAdapter(BaseModuleAdapter):
    """
    Adapter for integrating class-based modules that implement
    specific interface patterns.
    """
    
    def __init__(
        self,
        module_id: str,
        module_class: type,
        interface_mapping: Dict[str, str],
        init_args: Optional[Dict[str, Any]] = None,
        module_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize class-based adapter.
        
        Args:
            module_id: Unique identifier for the module
            module_class: The class to instantiate
            interface_mapping: Maps request types to class methods
            init_args: Arguments for class initialization
            module_info: Optional module metadata
        """
        capabilities = list(interface_mapping.keys())
        info = module_info or {
            'name': module_id,
            'type': 'class_based',
            'class_name': module_class.__name__,
            'description': f'Class-based adapter for {module_class.__name__}'
        }
        
        self._module_class = module_class
        self._interface_mapping = interface_mapping
        self._init_args = init_args or {}
        
        super().__init__(module_id, info, capabilities)
    
    async def _adapter_initialize(self) -> None:
        """Initialize the target class instance."""
        try:
            self._target_module = self._module_class(**self._init_args)
            
            # Call initialize method if it exists
            if hasattr(self._target_module, 'initialize'):
                init_method = getattr(self._target_module, 'initialize')
                if asyncio.iscoroutinefunction(init_method):
                    await init_method()
                else:
                    init_method()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize class {self._module_class.__name__}: {str(e)}")
    
    async def _handle_request_internal(
        self,
        request_type: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Handle request by calling mapped class method."""
        if request_type not in self._interface_mapping:
            raise ValueError(f"No interface mapping for request type: {request_type}")
        
        method_name = self._interface_mapping[request_type]
        
        if not hasattr(self._target_module, method_name):
            raise AttributeError(f"Module class has no method: {method_name}")
        
        method = getattr(self._target_module, method_name)
        
        # Call method with appropriate arguments
        if asyncio.iscoroutinefunction(method):
            return await method(**payload)
        else:
            return method(**payload)


class ModuleRegistry:
    """
    Registry for managing module adapters and their integration
    with the Vanta orchestrator.
    """
    
    def __init__(self):
        self._registered_adapters: Dict[str, BaseModuleAdapter] = {}
        self._logger = logging.getLogger(__name__)
    
    async def register_legacy_module(
        self,
        module_id: str,
        legacy_module: Any,
        method_mapping: Dict[str, str],
        module_info: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a legacy module with Vanta."""
        try:
            adapter = LegacyModuleAdapter(
                module_id, legacy_module, method_mapping, module_info
            )
            
            success = await vanta_orchestrator.register_module(
                module_id, adapter, config
            )
            
            if success:
                self._registered_adapters[module_id] = adapter
                self._logger.info(f"Registered legacy module: {module_id}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to register legacy module {module_id}: {str(e)}")
            return False
    
    async def register_class_based_module(
        self,
        module_id: str,
        module_class: type,
        interface_mapping: Dict[str, str],
        init_args: Optional[Dict[str, Any]] = None,
        module_info: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a class-based module with Vanta."""
        try:
            adapter = ClassBasedAdapter(
                module_id, module_class, interface_mapping, init_args, module_info
            )
            
            success = await vanta_orchestrator.register_module(
                module_id, adapter, config
            )
            
            if success:
                self._registered_adapters[module_id] = adapter
                self._logger.info(f"Registered class-based module: {module_id}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to register class-based module {module_id}: {str(e)}")
            return False
    
    async def register_custom_adapter(
        self,
        module_id: str,
        adapter: BaseModuleAdapter,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a custom adapter with Vanta."""
        try:
            success = await vanta_orchestrator.register_module(
                module_id, adapter, config
            )
            
            if success:
                self._registered_adapters[module_id] = adapter
                self._logger.info(f"Registered custom adapter: {module_id}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to register custom adapter {module_id}: {str(e)}")
            return False
    
    async def unregister_module(self, module_id: str) -> bool:
        """Unregister a module from Vanta."""
        try:
            success = await vanta_orchestrator.unregister_module(module_id)
            
            if success and module_id in self._registered_adapters:
                del self._registered_adapters[module_id]
                self._logger.info(f"Unregistered module: {module_id}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to unregister module {module_id}: {str(e)}")
            return False
    
    def get_registered_modules(self) -> List[str]:
        """Get list of registered module IDs."""
        return list(self._registered_adapters.keys())
    
    def get_adapter(self, module_id: str) -> Optional[BaseModuleAdapter]:
        """Get adapter instance for a module."""
        return self._registered_adapters.get(module_id)


# Global module registry instance
module_registry = ModuleRegistry()
