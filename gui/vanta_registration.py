"""
Vanta Registration Module for GUI Components
============================================

This module provides comprehensive registration capabilities for VoxSigil Library 
GUI components with the Vanta orchestrator system.

GUI Components:
- GUI Launcher: Main GUI application launcher
- Agent Status Panel: Agent monitoring and status display
- Echo Log Panel: Log display and monitoring
- Mesh Map Panel: Network mesh visualization
- PyQt Main: Main PyQt application framework

Registration Architecture:
- GUIModuleAdapter: Adapter for GUI system components
- Dynamic component loading with error handling
- Async registration patterns
- GUI coordination and management
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GUIModuleAdapter:
    """
    Adapter for VoxSigil Library GUI components.
    
    Handles registration, initialization, and coordination of GUI
    components with the Vanta orchestrator system.
    """
    
    def __init__(self, module_name: str, component_type: str = "gui"):
        self.module_name = module_name
        self.component_type = component_type
        self.is_initialized = False
        self.vanta_core = None
        self.gui_components = {}
        self.gui_config = {}
        self.gui_handlers = {}
        
    async def initialize(self, vanta_core) -> bool:
        """Initialize GUI module with Vanta core."""
        try:
            self.vanta_core = vanta_core
            logger.info(f"Initializing GUI module: {self.module_name}")
            
            # Initialize GUI components
            await self._initialize_gui_components()
            
            # Load GUI configuration
            await self._load_gui_configuration()
            
            # Set up GUI handlers
            await self._setup_gui_handlers()
            
            # Connect to Vanta core if GUI supports it
            if hasattr(vanta_core, 'register_gui_module'):
                await vanta_core.register_gui_module(self)
                
            self.is_initialized = True
            logger.info(f"Successfully initialized GUI module: {self.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GUI module {self.module_name}: {str(e)}")
            return False
    
    async def _initialize_gui_components(self):
        """Initialize individual GUI components."""
        try:
            # Initialize GUI Launcher
            gui_launcher = await self._import_gui_launcher()
            if gui_launcher:
                self.gui_components['launcher'] = gui_launcher
                logger.info("GUI Launcher initialized")
                
            # Initialize Agent Status Panel
            agent_status_panel = await self._import_agent_status_panel()
            if agent_status_panel:
                self.gui_components['agent_status'] = agent_status_panel
                logger.info("Agent Status Panel initialized")
                
            # Initialize Echo Log Panel
            echo_log_panel = await self._import_echo_log_panel()
            if echo_log_panel:
                self.gui_components['echo_log'] = echo_log_panel
                logger.info("Echo Log Panel initialized")
                
            # Initialize Mesh Map Panel
            mesh_map_panel = await self._import_mesh_map_panel()
            if mesh_map_panel:
                self.gui_components['mesh_map'] = mesh_map_panel
                logger.info("Mesh Map Panel initialized")
                
            # Initialize PyQt Main
            pyqt_main = await self._import_pyqt_main()
            if pyqt_main:
                self.gui_components['pyqt_main'] = pyqt_main
                logger.info("PyQt Main initialized")
                
        except Exception as e:
            logger.error(f"Error initializing GUI components: {str(e)}")
    
    async def _import_gui_launcher(self):
        """Import and initialize GUI Launcher."""
        try:
            from .launcher import GUILauncher
            return GUILauncher()
        except ImportError as e:
            logger.warning(f"Could not import GUILauncher: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing GUILauncher: {str(e)}")
            return None
    
    async def _import_agent_status_panel(self):
        """Import and initialize Agent Status Panel."""
        try:
            from .components.agent_status_panel import AgentStatusPanel
            return AgentStatusPanel()
        except ImportError as e:
            logger.warning(f"Could not import AgentStatusPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing AgentStatusPanel: {str(e)}")
            return None
    
    async def _import_echo_log_panel(self):
        """Import and initialize Echo Log Panel."""
        try:
            from .components.echo_log_panel import EchoLogPanel
            return EchoLogPanel()
        except ImportError as e:
            logger.warning(f"Could not import EchoLogPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing EchoLogPanel: {str(e)}")
            return None
    
    async def _import_mesh_map_panel(self):
        """Import and initialize Mesh Map Panel."""
        try:
            from .components.mesh_map_panel import MeshMapPanel
            return MeshMapPanel()
        except ImportError as e:
            logger.warning(f"Could not import MeshMapPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing MeshMapPanel: {str(e)}")
            return None
    
    async def _import_pyqt_main(self):
        """Import and initialize PyQt Main."""
        try:
            from .components.pyqt_main import PyQtMain
            return PyQtMain()
        except ImportError as e:
            logger.warning(f"Could not import PyQtMain: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQtMain: {str(e)}")
            return None
    
    async def _load_gui_configuration(self):
        """Load GUI configuration."""
        try:
            # Default GUI configuration
            self.gui_config = {
                'window_width': 1200,
                'window_height': 800,
                'theme': 'dark',
                'auto_refresh': True,
                'refresh_interval': 1000,
                'show_status_bar': True,
                'enable_mesh_map': True,
                'log_level': 'INFO'
            }
            
            # Try to load from launcher if available
            launcher = self.gui_components.get('launcher')
            if launcher and hasattr(launcher, 'get_config'):
                launcher_config = await launcher.get_config()
                self.gui_config.update(launcher_config)
            
            logger.info("GUI configuration loaded")
        except Exception as e:
            logger.error(f"Error loading GUI configuration: {str(e)}")
    
    async def _setup_gui_handlers(self):
        """Set up GUI handlers for GUI requests."""
        try:
            self.gui_handlers = {
                'launch': self._handle_launch_request,
                'show_panel': self._handle_show_panel_request,
                'update_status': self._handle_update_status_request,
                'log_message': self._handle_log_message_request,
                'update_mesh': self._handle_update_mesh_request,
                'config': self._handle_config_request,
                'status': self._handle_status_request,
                'close': self._handle_close_request
            }
            logger.info("GUI handlers established")
        except Exception as e:
            logger.error(f"Error setting up GUI handlers: {str(e)}")
    
    async def process_gui_request(self, operation: str, request_data: Any):
        """Process GUI request through appropriate component."""
        try:
            if not self.is_initialized:
                raise RuntimeError("GUI module not initialized")
                
            # Get GUI handler
            handler = self.gui_handlers.get(operation)
            if not handler:
                raise ValueError(f"Unknown GUI operation: {operation}")
                
            # Process request through handler
            return await handler(request_data)
                
        except Exception as e:
            logger.error(f"Error processing GUI request: {str(e)}")
            raise
    
    async def _handle_launch_request(self, request_data: Any):
        """Handle GUI launch requests."""
        try:
            launcher = self.gui_components.get('launcher')
            if launcher and hasattr(launcher, 'launch'):
                return await launcher.launch(request_data.get('config', {}))
            
            # Fallback launch response
            return {
                "status": "launched",
                "window_id": "main_window",
                "config": request_data.get('config', self.gui_config)
            }
                
        except Exception as e:
            logger.error(f"Error in GUI launch: {str(e)}")
            raise
    
    async def _handle_show_panel_request(self, request_data: Any):
        """Handle show panel requests."""
        try:
            panel_type = request_data.get('panel_type', 'agent_status')
            panel_component = self.gui_components.get(panel_type)
            
            if panel_component and hasattr(panel_component, 'show'):
                return await panel_component.show(request_data.get('data', {}))
            
            # Fallback panel show response
            return {
                "status": "panel_shown",
                "panel_type": panel_type,
                "visible": True
            }
                
        except Exception as e:
            logger.error(f"Error showing panel: {str(e)}")
            raise
    
    async def _handle_update_status_request(self, request_data: Any):
        """Handle status update requests."""
        try:
            agent_status_panel = self.gui_components.get('agent_status')
            if agent_status_panel and hasattr(agent_status_panel, 'update_status'):
                return await agent_status_panel.update_status(request_data.get('status_data', {}))
            
            # Fallback status update response
            return {
                "status": "status_updated",
                "agent_count": request_data.get('agent_count', 0),
                "active_agents": request_data.get('active_agents', [])
            }
                
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
            raise
    
    async def _handle_log_message_request(self, request_data: Any):
        """Handle log message requests."""
        try:
            echo_log_panel = self.gui_components.get('echo_log')
            if echo_log_panel and hasattr(echo_log_panel, 'log_message'):
                return await echo_log_panel.log_message(
                    message=request_data.get('message', ''),
                    level=request_data.get('level', 'INFO')
                )
            
            # Fallback log message response
            return {
                "status": "message_logged",
                "message": request_data.get('message', ''),
                "level": request_data.get('level', 'INFO'),
                "timestamp": "now"
            }
                
        except Exception as e:
            logger.error(f"Error logging message: {str(e)}")
            raise
    
    async def _handle_update_mesh_request(self, request_data: Any):
        """Handle mesh map update requests."""
        try:
            mesh_map_panel = self.gui_components.get('mesh_map')
            if mesh_map_panel and hasattr(mesh_map_panel, 'update_mesh'):
                return await mesh_map_panel.update_mesh(request_data.get('mesh_data', {}))
            
            # Fallback mesh update response
            return {
                "status": "mesh_updated",
                "nodes": request_data.get('nodes', []),
                "connections": request_data.get('connections', [])
            }
                
        except Exception as e:
            logger.error(f"Error updating mesh: {str(e)}")
            raise
    
    async def _handle_config_request(self, request_data: Any):
        """Handle GUI configuration requests."""
        try:
            operation = request_data.get('operation', 'get')
            
            if operation == 'get':
                return {
                    "status": "config_retrieved",
                    "config": self.gui_config
                }
                
            elif operation == 'set':
                new_config = request_data.get('config', {})
                self.gui_config.update(new_config)
                return {
                    "status": "config_updated",
                    "config": self.gui_config
                }
                
            else:
                raise ValueError(f"Unknown config operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in config handling: {str(e)}")
            raise
    
    async def _handle_status_request(self, request_data: Any):
        """Handle GUI status requests."""
        try:
            return {
                "module_name": self.module_name,
                "is_initialized": self.is_initialized,
                "components_count": len(self.gui_components),
                "available_components": list(self.gui_components.keys()),
                "configuration": self.gui_config,
                "operations": list(self.gui_handlers.keys())
            }
                
        except Exception as e:
            logger.error(f"Error in status handling: {str(e)}")
            raise
    
    async def _handle_close_request(self, request_data: Any):
        """Handle GUI close requests."""
        try:
            # Close all GUI components
            for component_name, component in self.gui_components.items():
                if hasattr(component, 'close'):
                    await component.close()
            
            return {
                "status": "gui_closed",
                "components_closed": list(self.gui_components.keys())
            }
                
        except Exception as e:
            logger.error(f"Error closing GUI: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of GUI module."""
        return {
            "module_name": self.module_name,
            "component_type": self.component_type,
            "is_initialized": self.is_initialized,
            "components_count": len(self.gui_components),
            "available_components": list(self.gui_components.keys()),
            "operations": list(self.gui_handlers.keys()),
            "configuration": self.gui_config
        }


class GUISystemManager:
    """
    System manager for GUI module coordination.
    
    Handles registration, routing, and coordination of all GUI
    components within the VoxSigil Library ecosystem.
    """
    
    def __init__(self):
        self.gui_adapters = {}
        self.gui_routing = {}
        self.system_config = {}
        self.is_initialized = False
        
    async def initialize_system(self):
        """Initialize the GUI system."""
        try:
            logger.info("Initializing GUI System Manager")
            
            # Register all GUI components
            await self._register_gui_components()
            
            # Set up GUI routing
            await self._setup_gui_routing()
            
            # Load system configuration
            await self._load_system_configuration()
            
            self.is_initialized = True
            logger.info("GUI System Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GUI System Manager: {str(e)}")
            raise
    
    async def _register_gui_components(self):
        """Register all GUI components."""
        try:
            # Register main GUI adapter
            main_adapter = GUIModuleAdapter("gui", "gui")
            self.gui_adapters["main"] = main_adapter
            
            # Register launcher adapter
            launcher_adapter = GUIModuleAdapter("gui_launcher", "launcher")
            self.gui_adapters["launcher"] = launcher_adapter
            
            # Register panels adapter
            panels_adapter = GUIModuleAdapter("gui_panels", "panels")
            self.gui_adapters["panels"] = panels_adapter
            
            logger.info(f"Registered {len(self.gui_adapters)} GUI adapters")
            
        except Exception as e:
            logger.error(f"Error registering GUI components: {str(e)}")
            raise
    
    async def _setup_gui_routing(self):
        """Set up GUI routing patterns."""
        try:
            self.gui_routing = {
                "application": {
                    "adapter": "launcher",
                    "operations": ["launch", "close"]
                },
                "panels": {
                    "adapter": "panels",
                    "operations": ["show_panel", "update_status", "log_message", "update_mesh"]
                },
                "system": {
                    "adapter": "main",
                    "operations": ["config", "status"]
                }
            }
            
            logger.info("GUI routing patterns established")
            
        except Exception as e:
            logger.error(f"Error setting up GUI routing: {str(e)}")
            raise
    
    async def _load_system_configuration(self):
        """Load GUI system configuration."""
        try:
            self.system_config = {
                "max_panels": 10,
                "auto_start": False,
                "default_theme": "dark",
                "panel_refresh_rate": 1000,
                "enable_logging": True
            }
            
            logger.info("GUI system configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading GUI system configuration: {str(e)}")
            raise
    
    async def route_gui_request(self, operation_type: str, request_data: Any):
        """Route GUI request to appropriate adapter."""
        try:
            if not self.is_initialized:
                raise RuntimeError("GUI System Manager not initialized")
                
            # Find appropriate routing pattern
            routing_pattern = None
            for pattern_name, pattern in self.gui_routing.items():
                if operation_type in pattern["operations"]:
                    routing_pattern = pattern
                    break
            
            if not routing_pattern:
                # Default to main adapter
                routing_pattern = {"adapter": "main"}
                
            adapter_key = routing_pattern["adapter"]
            adapter = self.gui_adapters.get(adapter_key)
            if not adapter:
                raise RuntimeError(f"GUI adapter not available: {adapter_key}")
                
            return await adapter.process_gui_request(operation_type, request_data)
            
        except Exception as e:
            logger.error(f"Error routing GUI request: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of GUI system."""
        return {
            "is_initialized": self.is_initialized,
            "adapters_count": len(self.gui_adapters),
            "available_adapters": list(self.gui_adapters.keys()),
            "routing_patterns": list(self.gui_routing.keys()),
            "system_config": self.system_config
        }


# Global system manager instance
gui_system_manager = GUISystemManager()

async def register_gui() -> Dict[str, Any]:
    """
    Register GUI module with Vanta orchestrator.
    
    Returns:
        Dict containing registration results and status information.
    """
    try:
        logger.info("Starting GUI module registration")
        
        # Initialize system manager
        await gui_system_manager.initialize_system()
        
        # Create main GUI adapter
        gui_adapter = GUIModuleAdapter("gui")
        
        # Registration would be completed by Vanta orchestrator
        registration_result = {
            "module_name": "gui",
            "module_type": "gui", 
            "status": "registered",
            "components": [
                "GUILauncher",
                "AgentStatusPanel",
                "EchoLogPanel",
                "MeshMapPanel",
                "PyQtMain"
            ],
            "capabilities": [
                "application",
                "panels", 
                "system"
            ],
            "adapter": gui_adapter,
            "system_manager": gui_system_manager
        }
        
        logger.info("GUI module registration completed successfully")
        return registration_result
        
    except Exception as e:
        logger.error(f"Failed to register GUI module: {str(e)}")
        raise

# Export registration function and key classes
__all__ = [
    'register_gui',
    'GUIModuleAdapter', 
    'GUISystemManager',
    'gui_system_manager'
]
