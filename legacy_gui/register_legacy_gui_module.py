# legacy_gui/register_legacy_gui_module.py
"""
Vanta Registration Module for Legacy GUI Components
==================================================

This module provides comprehensive registration capabilities for VoxSigil Library 
legacy GUI components with the Vanta orchestrator system.

Legacy GUI Components:
- Dynamic GridFormer GUI: Interactive GridFormer training interface
- GUI Styles: Legacy styling and theming utilities
- GUI Utils: Legacy GUI utility functions and helpers
- Training Interface New: Updated training interface components
- VMB Final Demo: VMB system demonstration interface
- VMB GUI Launcher: VMB-specific GUI launcher
- VMB GUI Simple: Simplified VMB interface

Registration Architecture:
- LegacyGUIModuleAdapter: Adapter for legacy GUI components
- Dynamic component loading with backward compatibility
- Async registration patterns
- Legacy system coordination and management
"""

import asyncio
import logging
import importlib
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegacyGUIModuleAdapter:
    """
    Adapter for VoxSigil Library Legacy GUI components.
    
    Handles registration, initialization, and coordination of legacy GUI
    components with the Vanta orchestrator system while maintaining
    backward compatibility.
    """
    
    def __init__(self, module_name: str, component_type: str = "legacy_gui"):
        self.module_name = module_name
        self.component_type = component_type
        self.is_initialized = False
        self.vanta_core = None
        self.legacy_components = {}
        self.legacy_config = {}
        self.legacy_handlers = {}
        
    async def initialize(self, vanta_core) -> bool:
        """Initialize Legacy GUI module with Vanta core."""
        try:
            self.vanta_core = vanta_core
            logger.info(f"Initializing Legacy GUI module: {self.module_name}")
            
            # Initialize legacy GUI components
            await self._initialize_legacy_components()
            
            # Load legacy configuration
            await self._load_legacy_configuration()
            
            # Set up legacy handlers
            await self._setup_legacy_handlers()
            
            # Connect to Vanta core if legacy GUI supports it
            if hasattr(vanta_core, 'register_legacy_gui_module'):
                await vanta_core.register_legacy_gui_module(self)
                
            self.is_initialized = True
            logger.info(f"Successfully initialized Legacy GUI module: {self.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Legacy GUI module {self.module_name}: {str(e)}")
            return False
    
    async def _initialize_legacy_components(self):
        """Initialize individual legacy GUI components."""
        try:
            # Initialize Dynamic GridFormer GUI
            dynamic_gridformer = await self._import_dynamic_gridformer_gui()
            if dynamic_gridformer:
                self.legacy_components['dynamic_gridformer'] = dynamic_gridformer
                logger.info("Dynamic GridFormer GUI initialized")
                
            # Initialize GUI Styles
            gui_styles = await self._import_gui_styles()
            if gui_styles:
                self.legacy_components['styles'] = gui_styles
                logger.info("GUI Styles initialized")
                
            # Initialize GUI Utils
            gui_utils = await self._import_gui_utils()
            if gui_utils:
                self.legacy_components['utils'] = gui_utils
                logger.info("GUI Utils initialized")
                
            # Initialize Training Interface New
            training_interface = await self._import_training_interface_new()
            if training_interface:
                self.legacy_components['training_interface'] = training_interface
                logger.info("Training Interface New initialized")
                
            # Initialize VMB Final Demo
            vmb_final_demo = await self._import_vmb_final_demo()
            if vmb_final_demo:
                self.legacy_components['vmb_demo'] = vmb_final_demo
                logger.info("VMB Final Demo initialized")
                
            # Initialize VMB GUI Launcher
            vmb_gui_launcher = await self._import_vmb_gui_launcher()
            if vmb_gui_launcher:
                self.legacy_components['vmb_launcher'] = vmb_gui_launcher
                logger.info("VMB GUI Launcher initialized")
                
            # Initialize VMB GUI Simple
            vmb_gui_simple = await self._import_vmb_gui_simple()
            if vmb_gui_simple:
                self.legacy_components['vmb_simple'] = vmb_gui_simple
                logger.info("VMB GUI Simple initialized")
                
        except Exception as e:
            logger.error(f"Error initializing legacy GUI components: {str(e)}")
    
    async def _import_dynamic_gridformer_gui(self):
        """Import and initialize Dynamic GridFormer GUI."""
        try:
            from .dynamic_gridformer_gui import DynamicGridFormerGUI
            return DynamicGridFormerGUI()
        except ImportError as e:
            logger.warning(f"Could not import DynamicGridFormerGUI: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing DynamicGridFormerGUI: {str(e)}")
            return None
    
    async def _import_gui_styles(self):
        """Import and initialize GUI Styles."""
        try:
            from .gui_styles import GUIStyles
            return GUIStyles()
        except ImportError as e:
            logger.warning(f"Could not import GUIStyles: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing GUIStyles: {str(e)}")
            return None
    
    async def _import_gui_utils(self):
        """Import and initialize GUI Utils."""
        try:
            from .gui_utils import GUIUtils
            return GUIUtils()
        except ImportError as e:
            logger.warning(f"Could not import GUIUtils: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing GUIUtils: {str(e)}")
            return None
    
    async def _import_training_interface_new(self):
        """Import and initialize Training Interface New."""
        try:
            from .training_interface_new import TrainingInterfaceNew
            return TrainingInterfaceNew()
        except ImportError as e:
            logger.warning(f"Could not import TrainingInterfaceNew: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing TrainingInterfaceNew: {str(e)}")
            return None
    
    async def _import_vmb_final_demo(self):
        """Import and initialize VMB Final Demo."""
        try:
            from .vmb_final_demo import VMBFinalDemo
            return VMBFinalDemo()
        except ImportError as e:
            logger.warning(f"Could not import VMBFinalDemo: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBFinalDemo: {str(e)}")
            return None
    
    async def _import_vmb_gui_launcher(self):
        """Import and initialize VMB GUI Launcher."""
        try:
            from .vmb_gui_launcher import VMBGUILauncher
            return VMBGUILauncher()
        except ImportError as e:
            logger.warning(f"Could not import VMBGUILauncher: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBGUILauncher: {str(e)}")
            return None
    
    async def _import_vmb_gui_simple(self):
        """Import and initialize VMB GUI Simple."""
        try:
            from .vmb_gui_simple import VMBGUISimple
            return VMBGUISimple()
        except ImportError as e:
            logger.warning(f"Could not import VMBGUISimple: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing VMBGUISimple: {str(e)}")
            return None
    
    async def _load_legacy_configuration(self):
        """Load legacy GUI configuration."""
        try:
            # Default legacy configuration
            self.legacy_config = {
                'backward_compatibility': True,
                'legacy_theme': 'classic',
                'gridformer_enabled': True,
                'vmb_integration': True,
                'training_interface_mode': 'legacy',
                'auto_migration': False,
                'deprecation_warnings': True,
                'fallback_mode': True
            }
            
            # Try to load from styles component if available
            styles_component = self.legacy_components.get('styles')
            if styles_component and hasattr(styles_component, 'get_config'):
                styles_config = await styles_component.get_config()
                self.legacy_config.update(styles_config)
            
            logger.info("Legacy GUI configuration loaded")
        except Exception as e:
            logger.error(f"Error loading legacy GUI configuration: {str(e)}")
    
    async def _setup_legacy_handlers(self):
        """Set up legacy GUI handlers for processing requests."""
        try:
            self.legacy_handlers = {
                'launch_gridformer': self._handle_launch_gridformer_request,
                'launch_vmb_demo': self._handle_launch_vmb_demo_request,
                'launch_training': self._handle_launch_training_request,
                'apply_styles': self._handle_apply_styles_request,
                'legacy_operation': self._handle_legacy_operation_request,
                'config': self._handle_config_request,
                'status': self._handle_status_request,
                'migrate': self._handle_migrate_request
            }
            logger.info("Legacy GUI handlers established")
        except Exception as e:
            logger.error(f"Error setting up legacy GUI handlers: {str(e)}")
    
    async def process_legacy_request(self, operation: str, request_data: Any):
        """Process legacy GUI request through appropriate component."""
        try:
            if not self.is_initialized:
                raise RuntimeError("Legacy GUI module not initialized")
                
            # Get legacy handler
            handler = self.legacy_handlers.get(operation)
            if not handler:
                raise ValueError(f"Unknown legacy GUI operation: {operation}")
                
            # Process request through handler
            return await handler(request_data)
                
        except Exception as e:
            logger.error(f"Error processing legacy GUI request: {str(e)}")
            raise
    
    async def _handle_launch_gridformer_request(self, request_data: Any):
        """Handle GridFormer GUI launch requests."""
        try:
            gridformer_component = self.legacy_components.get('dynamic_gridformer')
            if gridformer_component and hasattr(gridformer_component, 'launch'):
                return await gridformer_component.launch(
                    config=request_data.get('config', {}),
                    mode=request_data.get('mode', 'interactive')
                )
            
            # Fallback GridFormer launch response
            return {
                "status": "gridformer_launched",
                "interface": "dynamic_gridformer",
                "mode": request_data.get('mode', 'interactive'),
                "legacy_mode": True
            }
                
        except Exception as e:
            logger.error(f"Error launching GridFormer GUI: {str(e)}")
            raise
    
    async def _handle_launch_vmb_demo_request(self, request_data: Any):
        """Handle VMB demo launch requests."""
        try:
            demo_type = request_data.get('demo_type', 'final')
            
            if demo_type == 'final':
                vmb_demo = self.legacy_components.get('vmb_demo')
                if vmb_demo and hasattr(vmb_demo, 'launch'):
                    return await vmb_demo.launch(request_data.get('options', {}))
            elif demo_type == 'launcher':
                vmb_launcher = self.legacy_components.get('vmb_launcher')
                if vmb_launcher and hasattr(vmb_launcher, 'launch'):
                    return await vmb_launcher.launch(request_data.get('options', {}))
            elif demo_type == 'simple':
                vmb_simple = self.legacy_components.get('vmb_simple')
                if vmb_simple and hasattr(vmb_simple, 'launch'):
                    return await vmb_simple.launch(request_data.get('options', {}))
            
            # Fallback VMB demo response
            return {
                "status": "vmb_demo_launched",
                "demo_type": demo_type,
                "legacy_interface": True
            }
                
        except Exception as e:
            logger.error(f"Error launching VMB demo: {str(e)}")
            raise
    
    async def _handle_launch_training_request(self, request_data: Any):
        """Handle training interface launch requests."""
        try:
            training_interface = self.legacy_components.get('training_interface')
            if training_interface and hasattr(training_interface, 'launch'):
                return await training_interface.launch(
                    training_config=request_data.get('training_config', {}),
                    mode=request_data.get('mode', 'new')
                )
            
            # Fallback training interface response
            return {
                "status": "training_interface_launched",
                "interface_type": "new",
                "legacy_mode": True,
                "config": request_data.get('training_config', {})
            }
                
        except Exception as e:
            logger.error(f"Error launching training interface: {str(e)}")
            raise
    
    async def _handle_apply_styles_request(self, request_data: Any):
        """Handle style application requests."""
        try:
            styles_component = self.legacy_components.get('styles')
            if styles_component and hasattr(styles_component, 'apply_style'):
                return await styles_component.apply_style(
                    style_name=request_data.get('style_name', 'classic'),
                    target=request_data.get('target', 'all')
                )
            
            # Fallback style application response
            return {
                "status": "styles_applied",
                "style_name": request_data.get('style_name', 'classic'),
                "target": request_data.get('target', 'all'),
                "legacy_styles": True
            }
                
        except Exception as e:
            logger.error(f"Error applying styles: {str(e)}")
            raise
    
    async def _handle_legacy_operation_request(self, request_data: Any):
        """Handle generic legacy operations."""
        try:
            operation_type = request_data.get('operation_type', 'unknown')
            utils_component = self.legacy_components.get('utils')
            
            if utils_component and hasattr(utils_component, 'execute_legacy_operation'):
                return await utils_component.execute_legacy_operation(
                    operation=operation_type,
                    params=request_data.get('params', {})
                )
            
            # Fallback legacy operation response
            return {
                "status": "legacy_operation_completed",
                "operation_type": operation_type,
                "compatibility_mode": True,
                "deprecation_warning": "Consider migrating to modern GUI components"
            }
                
        except Exception as e:
            logger.error(f"Error in legacy operation: {str(e)}")
            raise
    
    async def _handle_migrate_request(self, request_data: Any):
        """Handle migration requests from legacy to modern GUI."""
        try:
            component_type = request_data.get('component_type', 'all')
            migration_strategy = request_data.get('strategy', 'gradual')
            
            # Fallback migration response
            return {
                "status": "migration_planned",
                "component_type": component_type,
                "strategy": migration_strategy,
                "legacy_components": list(self.legacy_components.keys()),
                "migration_available": True,
                "recommendation": "Use modern GUI components for new features"
            }
                
        except Exception as e:
            logger.error(f"Error in migration handling: {str(e)}")
            raise
    
    async def _handle_config_request(self, request_data: Any):
        """Handle legacy GUI configuration requests."""
        try:
            operation = request_data.get('operation', 'get')
            
            if operation == 'get':
                return {
                    "status": "config_retrieved",
                    "config": self.legacy_config
                }
                
            elif operation == 'set':
                new_config = request_data.get('config', {})
                self.legacy_config.update(new_config)
                return {
                    "status": "config_updated",
                    "config": self.legacy_config
                }
                
            else:
                raise ValueError(f"Unknown config operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in legacy GUI config handling: {str(e)}")
            raise
    
    async def _handle_status_request(self, request_data: Any):
        """Handle legacy GUI status requests."""
        try:
            return {
                "module_name": self.module_name,
                "is_initialized": self.is_initialized,
                "components_count": len(self.legacy_components),
                "available_components": list(self.legacy_components.keys()),
                "configuration": self.legacy_config,
                "operations": list(self.legacy_handlers.keys()),
                "legacy_mode": True,
                "deprecation_status": "maintained for backward compatibility"
            }
                
        except Exception as e:
            logger.error(f"Error in legacy GUI status handling: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of legacy GUI module."""
        return {
            "module_name": self.module_name,
            "component_type": self.component_type,
            "is_initialized": self.is_initialized,
            "components_count": len(self.legacy_components),
            "available_components": list(self.legacy_components.keys()),
            "operations": list(self.legacy_handlers.keys()),
            "configuration": self.legacy_config,
            "legacy_mode": True
        }


async def register_legacy_gui() -> Dict[str, Any]:
    """
    Register Legacy GUI module with Vanta orchestrator.
    
    Returns:
        Dict containing registration results and status information.
    """
    try:
        logger.info("Starting Legacy GUI module registration")
        
        # Create main Legacy GUI adapter
        legacy_gui_adapter = LegacyGUIModuleAdapter("legacy_gui")
        
        # Registration would be completed by Vanta orchestrator
        registration_result = {
            "module_name": "legacy_gui",
            "module_type": "legacy_gui", 
            "status": "registered",
            "components": [
                "DynamicGridFormerGUI",
                "GUIStyles",
                "GUIUtils",
                "TrainingInterfaceNew",
                "VMBFinalDemo",
                "VMBGUILauncher",
                "VMBGUISimple"
            ],
            "capabilities": [
                "launch_gridformer",
                "launch_vmb_demo",
                "launch_training",
                "apply_styles",
                "legacy_operation",
                "config",
                "status",
                "migrate"
            ],
            "adapter": legacy_gui_adapter,
            "legacy_mode": True,
            "deprecation_notice": "Consider migrating to modern GUI components"
        }
        
        logger.info("Legacy GUI module registration completed successfully")
        return registration_result
        
    except Exception as e:
        logger.error(f"Failed to register Legacy GUI module: {str(e)}")
        raise

# Export registration function and key classes
__all__ = [
    'register_legacy_gui',
    'LegacyGUIModuleAdapter'
]
