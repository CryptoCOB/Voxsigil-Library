#!/usr/bin/env python
"""
Utils Module Vanta Registration - HOLO-1.5 Enhanced

This module provides registration capabilities for the utilities module
with the Vanta orchestrator system using HOLO-1.5 cognitive mesh integration.

Components registered:
- DataUtils: Data processing and transformation utilities
- FileUtils: File system operations and management
- LoggingUtils: Enhanced logging and monitoring utilities
- NetworkUtils: Network communication and connectivity utilities
- ConfigUtils: Configuration management and validation
- CryptoUtils: Cryptographic operations and security utilities

HOLO-1.5 Integration: Full cognitive mesh integration with utility processing capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# HOLO-1.5 imports
import sys
import os
# Add the parent directory to the path to ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module

# Configure logging
logger = logging.getLogger(__name__)


@vanta_core_module(
    name="utils_module_processor",
    subsystem="utilities",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="Utility module processor for data processing, file operations, and system utilities",
    capabilities=[
        "data_processing",
        "file_management",
        "logging_enhancement",
        "network_operations",
        "configuration_management",
        "cryptographic_operations",
        "system_utilities",
        "validation_services"
    ],
    cognitive_load=2.0,
    symbolic_depth=1,
    collaboration_patterns=[
        "utility_processing",
        "resource_management",
        "system_integration",
        "service_orchestration"
    ],
)
class UtilsModuleAdapter(BaseCore):
    """
    Utils Module Adapter with HOLO-1.5 integration.
    Provides comprehensive utility processing, file management,
    and system utility coordination for the VoxSigil ecosystem.
    """
    
    def __init__(self, vanta_core: Any, config: Dict[str, Any]):
        super().__init__(vanta_core, config)
        
        self.module_id = "utils"
        self.display_name = "Utilities Module"
        self.version = "1.0.0"
        self.description = "Comprehensive utility module for data processing, file operations, and system utilities"
        
        # Utility component instances
        self.data_utils = None
        self.file_utils = None
        self.logging_utils = None
        self.network_utils = None
        self.config_utils = None
        self.crypto_utils = None
        self.initialized = False
        
        # Available utility components
        self.available_components = {
            'data_utils': 'data_utils.DataUtils',
            'file_utils': 'file_utils.FileUtils',
            'logging_utils': 'logging_utils.LoggingUtils',
            'network_utils': 'network_utils.NetworkUtils',
            'config_utils': 'config_utils.ConfigUtils',
            'crypto_utils': 'crypto_utils.CryptoUtils',
            'system_utils': 'system_utils.SystemUtils',
            'validation_utils': 'validation_utils.ValidationUtils'
        }
        
        self.logger = config.get("logger", logging.getLogger(__name__))
        self.logger.setLevel(config.get("log_level", logging.INFO))
    
    async def initialize(self) -> bool:
        """Initialize the Utils module with vanta core."""
        try:
            self.logger.info(f"Initializing Utils module with Vanta core...")
            
            # Initialize core utilities first
            await self._initialize_data_utils()
            await self._initialize_file_utils()
            await self._initialize_logging_utils()
            await self._initialize_network_utils()
            await self._initialize_config_utils()
            await self._initialize_crypto_utils()
            
            # Register with HOLO-1.5 cognitive mesh
            if hasattr(self.vanta_core, "register_component"):
                self.vanta_core.register_component(
                    "utils_module_processor",
                    self,
                    {"type": "utility_service", "cognitive_role": "PROCESSOR"}
                )
            
            self.initialized = True
            self.logger.info(f"Utils module initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Utils module: {e}")
            return False
    
    async def _initialize_data_utils(self):
        """Initialize data processing utilities."""
        try:
            data_utils_module = importlib.import_module('utils.data_utils')
            DataUtilsClass = getattr(data_utils_module, 'DataUtils', None)
            
            if DataUtilsClass:
                try:
                    self.data_utils = DataUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.data_utils = DataUtilsClass()
                    if hasattr(self.data_utils, 'set_vanta_core'):
                        self.data_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.data_utils, 'async_init'):
                    await self.data_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Data utils not available: {e}")
            self.data_utils = None
    
    async def _initialize_file_utils(self):
        """Initialize file management utilities."""
        try:
            file_utils_module = importlib.import_module('utils.file_utils')
            FileUtilsClass = getattr(file_utils_module, 'FileUtils', None)
            
            if FileUtilsClass:
                try:
                    self.file_utils = FileUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.file_utils = FileUtilsClass()
                    if hasattr(self.file_utils, 'set_vanta_core'):
                        self.file_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.file_utils, 'async_init'):
                    await self.file_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"File utils not available: {e}")
            self.file_utils = None
    
    async def _initialize_logging_utils(self):
        """Initialize logging enhancement utilities."""
        try:
            logging_utils_module = importlib.import_module('utils.logging_utils')
            LoggingUtilsClass = getattr(logging_utils_module, 'LoggingUtils', None)
            
            if LoggingUtilsClass:
                try:
                    self.logging_utils = LoggingUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.logging_utils = LoggingUtilsClass()
                    if hasattr(self.logging_utils, 'set_vanta_core'):
                        self.logging_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.logging_utils, 'async_init'):
                    await self.logging_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Logging utils not available: {e}")
            self.logging_utils = None
    
    async def _initialize_network_utils(self):
        """Initialize network utilities."""
        try:
            network_utils_module = importlib.import_module('utils.network_utils')
            NetworkUtilsClass = getattr(network_utils_module, 'NetworkUtils', None)
            
            if NetworkUtilsClass:
                try:
                    self.network_utils = NetworkUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.network_utils = NetworkUtilsClass()
                    if hasattr(self.network_utils, 'set_vanta_core'):
                        self.network_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.network_utils, 'async_init'):
                    await self.network_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Network utils not available: {e}")
            self.network_utils = None
    
    async def _initialize_config_utils(self):
        """Initialize configuration utilities."""
        try:
            config_utils_module = importlib.import_module('utils.config_utils')
            ConfigUtilsClass = getattr(config_utils_module, 'ConfigUtils', None)
            
            if ConfigUtilsClass:
                try:
                    self.config_utils = ConfigUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.config_utils = ConfigUtilsClass()
                    if hasattr(self.config_utils, 'set_vanta_core'):
                        self.config_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.config_utils, 'async_init'):
                    await self.config_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Config utils not available: {e}")
            self.config_utils = None
    
    async def _initialize_crypto_utils(self):
        """Initialize cryptographic utilities."""
        try:
            crypto_utils_module = importlib.import_module('utils.crypto_utils')
            CryptoUtilsClass = getattr(crypto_utils_module, 'CryptoUtils', None)
            
            if CryptoUtilsClass:
                try:
                    self.crypto_utils = CryptoUtilsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.crypto_utils = CryptoUtilsClass()
                    if hasattr(self.crypto_utils, 'set_vanta_core'):
                        self.crypto_utils.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.crypto_utils, 'async_init'):
                    await self.crypto_utils.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Crypto utils not available: {e}")
            self.crypto_utils = None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for utility operations."""
        if not self.initialized:
            return {"error": "Utils module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'data_processing':
                return await self._handle_data_processing(request)
            elif request_type == 'file_operations':
                return await self._handle_file_operations(request)
            elif request_type == 'logging_enhancement':
                return await self._handle_logging_enhancement(request)
            elif request_type == 'network_operations':
                return await self._handle_network_operations(request)
            elif request_type == 'config_management':
                return await self._handle_config_management(request)
            elif request_type == 'crypto_operations':
                return await self._handle_crypto_operations(request)
            elif request_type == 'validation':
                return await self._handle_validation(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            self.logger.error(f"Error processing utils request: {e}")
            return {"error": str(e)}
    
    async def _handle_data_processing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing requests."""
        if not self.data_utils:
            return {"error": "Data utils not available"}
        
        operation = request.get('operation')
        data = request.get('data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.data_utils, operation):
                handler = getattr(self.data_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data, **params)
                else:
                    result = handler(data, **params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Data processing failed: {e}"}
    
    async def _handle_file_operations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operation requests."""
        if not self.file_utils:
            return {"error": "File utils not available"}
        
        operation = request.get('operation')
        file_path = request.get('file_path')
        params = request.get('params', {})
        
        try:
            if hasattr(self.file_utils, operation):
                handler = getattr(self.file_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(file_path, **params)
                else:
                    result = handler(file_path, **params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"File operation failed: {e}"}
    
    async def _handle_logging_enhancement(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logging enhancement requests."""
        if not self.logging_utils:
            return {"error": "Logging utils not available"}
        
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.logging_utils, operation):
                handler = getattr(self.logging_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**params)
                else:
                    result = handler(**params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Logging operation failed: {e}"}
    
    async def _handle_network_operations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network operation requests."""
        if not self.network_utils:
            return {"error": "Network utils not available"}
        
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.network_utils, operation):
                handler = getattr(self.network_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**params)
                else:
                    result = handler(**params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Network operation failed: {e}"}
    
    async def _handle_config_management(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration management requests."""
        if not self.config_utils:
            return {"error": "Config utils not available"}
        
        operation = request.get('operation')
        config_data = request.get('config_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.config_utils, operation):
                handler = getattr(self.config_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(config_data, **params)
                else:
                    result = handler(config_data, **params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Config operation failed: {e}"}
    
    async def _handle_crypto_operations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cryptographic operation requests."""
        if not self.crypto_utils:
            return {"error": "Crypto utils not available"}
        
        operation = request.get('operation')
        data = request.get('data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.crypto_utils, operation):
                handler = getattr(self.crypto_utils, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data, **params)
                else:
                    result = handler(data, **params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Crypto operation failed: {e}"}
    
    async def _handle_validation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation requests."""
        validation_type = request.get('validation_type')
        data = request.get('data')
        rules = request.get('rules', {})
        
        try:
            # Import validation utilities
            validation_module = importlib.import_module('utils.validation_utils')
            ValidationUtilsClass = getattr(validation_module, 'ValidationUtils')
            
            validator = ValidationUtilsClass()
            if hasattr(validator, validation_type):
                handler = getattr(validator, validation_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data, **rules)
                else:
                    result = handler(data, **rules)
            else:
                result = {"warning": f"Validation type {validation_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Validation failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Utils module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "data_processing",
                "file_management",
                "logging_enhancement",
                "network_operations",
                "configuration_management",
                "cryptographic_operations",
                "system_utilities",
                "validation_services"
            ],
            "supported_operations": [
                "data_processing",
                "file_operations",
                "logging_enhancement",
                "network_operations",
                "config_management",
                "crypto_operations",
                "validation"
            ],
            "components": list(self.available_components.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "PROCESSOR",
            "symbolic_depth": 1
        }
    
    async def shutdown(self):
        """Shutdown the Utils module gracefully."""
        try:
            self.logger.info("Shutting down Utils module...")
            
            # Shutdown all utility components that support it
            for util_name in ['data_utils', 'file_utils', 'logging_utils', 
                             'network_utils', 'config_utils', 'crypto_utils']:
                util_instance = getattr(self, util_name, None)
                if util_instance and hasattr(util_instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(util_instance.shutdown):
                        await util_instance.shutdown()
                    else:
                        util_instance.shutdown()
            
            self.initialized = False
            self.logger.info("Utils module shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Utils module shutdown: {e}")


# Registration function for the master orchestrator
async def register_utils_module(vanta_core) -> UtilsModuleAdapter:
    """Register the Utils module with Vanta orchestrator."""
    logger.info("Registering Utils module with Vanta orchestrator...")
    
    adapter = UtilsModuleAdapter(vanta_core, {})
    success = await adapter.initialize()
    
    if success:
        logger.info("Utils module registration successful")
    else:
        logger.error("Utils module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['UtilsModuleAdapter', 'register_utils_module']
