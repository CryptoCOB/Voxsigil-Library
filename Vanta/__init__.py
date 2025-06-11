"""
Vanta Main Initialization
========================

Main entry point for the Vanta orchestrator system. Handles system startup,
module discovery, and coordination of the entire VoxSigil Library ecosystem.

Usage:
    from Vanta import VantaSystem
    
    # Initialize and start Vanta
    vanta = VantaSystem()
    await vanta.initialize()
    await vanta.start_system()
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import importlib.util
import json
from datetime import datetime

from .core.orchestrator import vanta_orchestrator
from .core.fallback_implementations import initialize_fallbacks
from .integration.module_adapters import module_registry
from .interfaces import (
    BaseRagInterface,
    BaseLlmInterface,
    BaseMemoryInterface,
    BaseAgentInterface,
    VantaProtocol,
    ModuleAdapterProtocol
)


class VantaSystem:
    """
    Main Vanta system coordinator handling initialization,
    module discovery, and system lifecycle management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._system_status = 'uninitialized'
        self._logger = logging.getLogger(__name__)
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for Vanta system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('vanta_system.log')
            ]
        )
        
        self._logger.info("Vanta System logging initialized")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the Vanta system with configuration."""
        try:
            self._logger.info("Initializing Vanta System...")
            self._system_status = 'initializing'
            
            # Load configuration
            await self._load_configuration(config)
            
            # Initialize fallback implementations
            await initialize_fallbacks()
            self._logger.info("Fallback implementations initialized")
            
            # Discover and register modules
            await self._discover_modules()
            
            # Run startup hooks
            await self._run_startup_hooks()
            
            self._system_status = 'initialized'
            self._logger.info("Vanta System initialization complete")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Vanta System: {str(e)}")
            self._system_status = 'error'
            return False
    
    async def _load_configuration(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load system configuration from file or provided dict."""
        if config:
            self._config = config
        elif self._config_path and os.path.exists(self._config_path):
            with open(self._config_path, 'r') as f:
                self._config = json.load(f)
        else:
            # Default configuration
            self._config = {
                'system': {
                    'name': 'VoxSigil Library',
                    'version': '1.0.0',
                    'environment': 'development'
                },
                'modules': {
                    'auto_discovery': True,
                    'discovery_paths': [
                        'agents/',
                        'ARC/',
                        'ART/', 
                        'BLT/',
                        'core/',
                        'engines/',
                        'llm/',
                        'memory/',
                        'middleware/',
                        'training/'
                    ]
                },
                'orchestrator': {
                    'request_timeout': 30.0,
                    'health_check_interval': 60,
                    'max_retries': 3
                },
                'fallback': {
                    'enabled': True,
                    'reliability_threshold': 0.5
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'vanta_system.log'
                }
            }
        
        self._logger.info(f"Configuration loaded: {self._config.get('system', {}).get('name', 'VoxSigil Library')}")
    
    async def _discover_modules(self) -> None:
        """Automatically discover and register modules in the workspace."""
        if not self._config.get('modules', {}).get('auto_discovery', True):
            self._logger.info("Module auto-discovery disabled")
            return
        
        discovery_paths = self._config.get('modules', {}).get('discovery_paths', [])
        workspace_root = Path('.')
        
        self._logger.info(f"Discovering modules in paths: {discovery_paths}")
        
        for path in discovery_paths:
            module_path = workspace_root / path
            if module_path.exists() and module_path.is_dir():
                await self._discover_modules_in_path(module_path)
    
    async def _discover_modules_in_path(self, module_path: Path) -> None:
        """Discover and register modules in a specific path."""
        try:
            # Look for module configuration files
            config_files = [
                module_path / 'vanta_module.json',
                module_path / 'module_config.json',
                module_path / '__vanta__.json'
            ]
            
            module_config = None
            for config_file in config_files:
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        module_config = json.load(f)
                    break
            
            if module_config:
                await self._register_configured_module(module_path, module_config)
            else:
                # Attempt automatic module detection
                await self._auto_detect_module(module_path)
                
        except Exception as e:
            self._logger.error(f"Error discovering modules in {module_path}: {str(e)}")
    
    async def _register_configured_module(
        self, 
        module_path: Path, 
        config: Dict[str, Any]
    ) -> None:
        """Register a module with explicit configuration."""
        try:
            module_id = config.get('module_id', module_path.name)
            module_type = config.get('type', 'auto')
            
            self._logger.info(f"Registering configured module: {module_id}")
            
            if module_type == 'legacy':
                await self._register_legacy_module(module_path, config)
            elif module_type == 'class_based':
                await self._register_class_based_module(module_path, config)
            else:
                await self._auto_detect_module(module_path)
            
        except Exception as e:
            self._logger.error(f"Failed to register configured module in {module_path}: {str(e)}")
    
    async def _register_legacy_module(
        self, 
        module_path: Path, 
        config: Dict[str, Any]
    ) -> None:
        """Register a legacy module using configuration."""
        module_id = config.get('module_id', module_path.name)
        module_file = config.get('module_file', '__init__.py')
        method_mapping = config.get('method_mapping', {})
        
        # Import the legacy module
        spec = importlib.util.spec_from_file_location(
            module_id, 
            module_path / module_file
        )
        if spec and spec.loader:
            legacy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy_module)
            
            # Get the main class or use the module itself
            main_class = config.get('main_class')
            if main_class:
                target = getattr(legacy_module, main_class)()
            else:
                target = legacy_module
            
            await module_registry.register_legacy_module(
                module_id,
                target,
                method_mapping,
                config.get('module_info', {}),
                config.get('init_config', {})
            )
    
    async def _register_class_based_module(
        self, 
        module_path: Path, 
        config: Dict[str, Any]
    ) -> None:
        """Register a class-based module using configuration."""
        module_id = config.get('module_id', module_path.name)
        module_file = config.get('module_file', '__init__.py')
        class_name = config.get('class_name')
        interface_mapping = config.get('interface_mapping', {})
        init_args = config.get('init_args', {})
        
        # Import the module class
        spec = importlib.util.spec_from_file_location(
            module_id, 
            module_path / module_file
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            module_class = getattr(module, class_name)
            
            await module_registry.register_class_based_module(
                module_id,
                module_class,
                interface_mapping,
                init_args,
                config.get('module_info', {}),
                config.get('init_config', {})
            )
    
    async def _auto_detect_module(self, module_path: Path) -> None:
        """Attempt to automatically detect and register a module."""
        try:
            module_id = module_path.name
            self._logger.info(f"Auto-detecting module: {module_id}")
            
            # Look for common patterns
            init_file = module_path / '__init__.py'
            main_file = module_path / f"{module_id}.py"
            
            if init_file.exists():
                await self._try_import_module(module_id, init_file)
            elif main_file.exists():
                await self._try_import_module(module_id, main_file)
            else:
                # Look for Python files in the directory
                python_files = list(module_path.glob('*.py'))
                if python_files:
                    await self._try_import_module(module_id, python_files[0])
                
        except Exception as e:
            self._logger.warning(f"Could not auto-detect module in {module_path}: {str(e)}")
    
    async def _try_import_module(self, module_id: str, module_file: Path) -> None:
        """Try to import and analyze a module for registration."""
        try:
            spec = importlib.util.spec_from_file_location(module_id, module_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Analyze module for interfaces and classes
                await self._analyze_and_register_module(module_id, module)
                
        except Exception as e:
            self._logger.debug(f"Failed to import {module_file}: {str(e)}")
    
    async def _analyze_and_register_module(self, module_id: str, module: Any) -> None:
        """Analyze imported module and attempt registration."""
        try:
            # Look for classes that implement known interfaces
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if isinstance(attr, type) and hasattr(attr, '__bases__'):
                    # Check if class implements any of our base interfaces
                    base_names = [base.__name__ for base in attr.__bases__]
                    
                    if any(name in ['BaseRagInterface', 'BaseLlmInterface', 
                                  'BaseMemoryInterface', 'BaseAgentInterface'] 
                           for name in base_names):
                        
                        # Generate interface mapping based on class methods
                        interface_mapping = self._generate_interface_mapping(attr)
                        
                        if interface_mapping:
                            await module_registry.register_class_based_module(
                                f"{module_id}_{attr_name.lower()}",
                                attr,
                                interface_mapping,
                                {},
                                {
                                    'name': f"{module_id} {attr_name}",
                                    'type': 'auto_detected',
                                    'source_file': str(module.__file__)
                                }
                            )
                            
                            self._logger.info(f"Auto-registered class-based module: {module_id}_{attr_name.lower()}")
                            
        except Exception as e:
            self._logger.debug(f"Failed to analyze module {module_id}: {str(e)}")
    
    def _generate_interface_mapping(self, cls: type) -> Dict[str, str]:
        """Generate interface mapping for a class based on its methods."""
        mapping = {}
        
        # Common method patterns
        method_patterns = {
            'retrieve_documents': 'retrieve_documents',
            'index_document': 'index_document',
            'generate_text': 'generate_text',
            'get_embeddings': 'get_embeddings',
            'store_memory': 'store_memory',
            'retrieve_memory': 'retrieve_memory',
            'execute_task': 'execute_task',
            'load_model': 'load_model'
        }
        
        for method_name in dir(cls):
            if not method_name.startswith('_') and callable(getattr(cls, method_name)):
                if method_name in method_patterns:
                    mapping[method_name] = method_name
        
        return mapping
    
    async def _run_startup_hooks(self) -> None:
        """Run registered startup hooks."""
        for hook in self._startup_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                self._logger.error(f"Startup hook failed: {str(e)}")
    
    async def start_system(self) -> bool:
        """Start the Vanta system and all registered modules."""
        try:
            if self._system_status != 'initialized':
                raise RuntimeError("System must be initialized before starting")
            
            self._logger.info("Starting Vanta System...")
            self._system_status = 'starting'
            
            # Start periodic health checks
            asyncio.create_task(self._health_check_loop())
            
            self._system_status = 'running'
            self._logger.info("Vanta System started successfully")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start Vanta System: {str(e)}")
            self._system_status = 'error'
            return False
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for all modules."""
        interval = self._config.get('orchestrator', {}).get('health_check_interval', 60)
        
        while self._system_status == 'running':
            try:
                system_status = await vanta_orchestrator.get_system_status()
                self._logger.debug(f"System health check: {system_status['overall_health']}")
                
                # Log any unhealthy modules
                for module_id, health in system_status.get('module_health', {}).items():
                    if health.get('status') != 'healthy':
                        self._logger.warning(f"Unhealthy module detected: {module_id} - {health}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self._logger.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(interval)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the Vanta system."""
        try:
            self._logger.info("Shutting down Vanta System...")
            self._system_status = 'shutting_down'
            
            # Run shutdown hooks
            for hook in self._shutdown_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    self._logger.error(f"Shutdown hook failed: {str(e)}")
            
            # Shutdown orchestrator (which will shutdown all modules)
            await vanta_orchestrator.shutdown()
            
            self._system_status = 'shutdown'
            self._logger.info("Vanta System shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during shutdown: {str(e)}")
            self._system_status = 'error'
    
    def add_startup_hook(self, hook: Callable) -> None:
        """Add a function to be called during system startup."""
        self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: Callable) -> None:
        """Add a function to be called during system shutdown."""
        self._shutdown_hooks.append(hook)
    
    def get_system_status(self) -> str:
        """Get current system status."""
        return self._system_status
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status including all modules."""
        if self._system_status == 'running':
            orchestrator_status = await vanta_orchestrator.get_system_status()
        else:
            orchestrator_status = {}
        
        return {
            'system_status': self._system_status,
            'config': self._config.get('system', {}),
            'registered_modules': module_registry.get_registered_modules(),
            'orchestrator_status': orchestrator_status,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_vanta_client(self, module_id: str):
        """Get Vanta client for external module registration."""
        from .core.orchestrator import VantaClient
        return VantaClient(vanta_orchestrator, module_id)


# Convenience functions for easy system management

async def create_vanta_system(config_path: Optional[str] = None) -> VantaSystem:
    """Create and initialize a new Vanta system."""
    vanta = VantaSystem(config_path)
    await vanta.initialize()
    return vanta

async def start_vanta_system(config_path: Optional[str] = None) -> VantaSystem:
    """Create, initialize, and start a new Vanta system."""
    vanta = await create_vanta_system(config_path)
    await vanta.start_system()
    return vanta


# Global system instance for simple usage
_global_vanta_system: Optional[VantaSystem] = None

async def get_global_vanta_system() -> VantaSystem:
    """Get or create the global Vanta system instance."""
    global _global_vanta_system
    
    if _global_vanta_system is None:
        _global_vanta_system = await start_vanta_system()
    
    return _global_vanta_system
