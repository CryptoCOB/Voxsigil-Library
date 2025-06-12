"""
Config Module Vanta Registration

This module provides registration capabilities for the Configuration module
with the Vanta orchestrator system.

Components registered:
- ProductionConfig: Production environment configuration
- ConfigManager: Configuration management and validation
- EnvironmentConfig: Environment-specific configurations
- DefaultConfig: Default configuration templates

HOLO-1.5 Integration: Configuration mesh integration with dynamic config loading.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# Configure logging
logger = logging.getLogger(__name__)

class ConfigModuleAdapter:
    """Adapter for integrating Config module with Vanta orchestrator."""
    
    def __init__(self):
        self.module_id = "config"
        self.display_name = "Configuration Management Module"
        self.version = "1.0.0"
        self.description = "Configuration management and environment settings for VoxSigil Library"
        
        # Config component instances
        self.production_config = None
        self.config_manager = None
        self.environment_config = None
        self.initialized = False
        
        # Available config components
        self.available_components = {
            'production_config': 'production_config.ProductionConfig',
            'config_manager': 'config_manager.ConfigManager',
            'environment_config': 'environment_config.EnvironmentConfig'
        }
        
    async def initialize(self, vanta_core):
        """Initialize the Config module with vanta core."""
        try:
            logger.info(f"Initializing Config module with Vanta core...")
            
            # Import and initialize Production Config
            try:
                config_module = importlib.import_module('config.production_config')
                ProductionConfigClass = getattr(config_module, 'ProductionConfig', None)
                
                if ProductionConfigClass:
                    try:
                        self.production_config = ProductionConfigClass(vanta_core=vanta_core)
                    except TypeError:
                        self.production_config = ProductionConfigClass()
                        if hasattr(self.production_config, 'set_vanta_core'):
                            self.production_config.set_vanta_core(vanta_core)
                        elif hasattr(self.production_config, 'vanta_core'):
                            self.production_config.vanta_core = vanta_core
                else:
                    # Fallback: create a basic config manager
                    self.production_config = self._create_basic_config_manager(vanta_core)
                    
            except ImportError:
                logger.warning("Production config not found, creating basic config manager")
                self.production_config = self._create_basic_config_manager(vanta_core)
            
            # Try to import other config components
            try:
                config_manager_module = importlib.import_module('config.config_manager')
                ConfigManagerClass = getattr(config_manager_module, 'ConfigManager')
                
                try:
                    self.config_manager = ConfigManagerClass(vanta_core=vanta_core)
                except TypeError:
                    self.config_manager = ConfigManagerClass()
                    if hasattr(self.config_manager, 'set_vanta_core'):
                        self.config_manager.set_vanta_core(vanta_core)
                        
            except (ImportError, AttributeError):
                logger.info("Config manager not found, using production config")
                self.config_manager = self.production_config
            
            # Initialize async components if they have async_init methods
            if hasattr(self.production_config, 'async_init'):
                await self.production_config.async_init()
                
            if self.config_manager and hasattr(self.config_manager, 'async_init'):
                await self.config_manager.async_init()
            
            self.initialized = True
            logger.info(f"Config module initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Config module: {e}")
            return False
    
    def _create_basic_config_manager(self, vanta_core):
        """Create a basic config manager for fallback."""
        class BasicConfigManager:
            def __init__(self, vanta_core=None):
                self.vanta_core = vanta_core
                self.config_data = {}
                
            def get_config(self, key: str, default: Any = None) -> Any:
                return self.config_data.get(key, default)
                
            def set_config(self, key: str, value: Any):
                self.config_data[key] = value
                
            def load_config(self, config_path: str = None):
                # Basic config loading logic
                import yaml
                import os
                
                if not config_path:
                    config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
                
                try:
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            self.config_data = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load config from {config_path}: {e}")
                    self.config_data = {}
                    
        return BasicConfigManager(vanta_core)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for Config module operations."""
        if not self.initialized:
            return {"error": "Config module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'get_config':
                return await self._handle_get_config(request)
            elif request_type == 'set_config':
                return await self._handle_set_config(request)
            elif request_type == 'load_config':
                return await self._handle_load_config(request)
            elif request_type == 'validate_config':
                return await self._handle_validate_config(request)
            elif request_type == 'get_environment':
                return await self._handle_get_environment(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing Config request: {e}")
            return {"error": str(e)}
    
    async def _handle_get_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration retrieval requests."""
        config_key = request.get('key')
        default_value = request.get('default')
        
        try:
            if hasattr(self.production_config, 'get_config'):
                value = self.production_config.get_config(config_key, default_value)
            else:
                value = getattr(self.production_config, config_key, default_value)
            
            return {"result": value, "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to get config: {e}"}
    
    async def _handle_set_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration setting requests."""
        config_key = request.get('key')
        config_value = request.get('value')
        
        if config_key is None:
            return {"error": "Missing config key"}
        
        try:
            if hasattr(self.production_config, 'set_config'):
                self.production_config.set_config(config_key, config_value)
            else:
                setattr(self.production_config, config_key, config_value)
            
            return {"result": "Config updated", "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to set config: {e}"}
    
    async def _handle_load_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration loading requests."""
        config_path = request.get('config_path')
        
        try:
            if hasattr(self.production_config, 'load_config'):
                result = self.production_config.load_config(config_path)
            else:
                result = "Config loading not supported"
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to load config: {e}"}
    
    async def _handle_validate_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration validation requests."""
        config_data = request.get('config_data', {})
        
        try:
            if hasattr(self.production_config, 'validate_config'):
                is_valid = self.production_config.validate_config(config_data)
            else:
                # Basic validation
                is_valid = isinstance(config_data, dict)
            
            return {"result": {"valid": is_valid}, "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to validate config: {e}"}
    
    async def _handle_get_environment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environment information requests."""
        try:
            import os
            
            environment_info = {
                "environment": os.getenv('ENVIRONMENT', 'development'),
                "debug": os.getenv('DEBUG', 'false').lower() == 'true',
                "log_level": os.getenv('LOG_LEVEL', 'INFO'),
                "vanta_core_enabled": hasattr(self, 'vanta_core') and self.vanta_core is not None
            }
            
            return {"result": environment_info, "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to get environment info: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Config module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "configuration_management",
                "environment_detection",
                "config_validation",
                "dynamic_configuration",
                "production_settings"
            ],
            "supported_operations": [
                "get_config",
                "set_config",
                "load_config",
                "validate_config",
                "get_environment"
            ],
            "components": list(self.available_components.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "PROCESSOR",
            "symbolic_depth": 2
        }
    
    async def shutdown(self):
        """Shutdown the Config module gracefully."""
        try:
            logger.info("Shutting down Config module...")
            
            # Shutdown components that support it
            if self.production_config and hasattr(self.production_config, 'shutdown'):
                if asyncio.iscoroutinefunction(self.production_config.shutdown):
                    await self.production_config.shutdown()
                else:
                    self.production_config.shutdown()
            
            if self.config_manager and hasattr(self.config_manager, 'shutdown'):
                if asyncio.iscoroutinefunction(self.config_manager.shutdown):
                    await self.config_manager.shutdown()
                else:
                    self.config_manager.shutdown()
            
            self.initialized = False
            logger.info("Config module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Config module shutdown: {e}")


# Registration function for the master orchestrator
async def register_config_module(vanta_core) -> ConfigModuleAdapter:
    """Register the Config module with Vanta orchestrator."""
    logger.info("Registering Config module with Vanta orchestrator...")
    
    adapter = ConfigModuleAdapter()
    success = await adapter.initialize(vanta_core)
    
    if success:
        logger.info("Config module registration successful")
    else:
        logger.error("Config module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['ConfigModuleAdapter', 'register_config_module']
