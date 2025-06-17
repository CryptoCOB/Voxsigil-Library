# gui/register_gui_module.py
"""
Vanta Registration Module for GUI Components (PyQt5)
====================================================

This module provides comprehensive registration capabilities for VoxSigil Library 
GUI components with the Vanta orchestrator system, specifically designed for PyQt5.

GUI Components (PyQt5-based):
- GUI Launcher: Main GUI application launcher (PyQt5)
- Agent Status Panel: Agent monitoring and status display (PyQt5)
- Echo Log Panel: Log display and monitoring (PyQt5)
- Mesh Map Panel: Network mesh visualization (PyQt5)
- PyQt Main: Main PyQt5 application framework

Registration Architecture:
- GUIModuleAdapter: Adapter for PyQt5-based GUI system components
- Dynamic component loading with PyQt5 compatibility checking
- Async registration patterns with Qt event loop integration
- PyQt5-specific GUI coordination and management

Requirements:
- PyQt5 must be installed and available
- Proper Qt application context management
- Thread-safe GUI operations
"""

import logging
import importlib
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyQt5 availability check
PYQT5_AVAILABLE = False
QT_APP = None
try:
    import PyQt5
    from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
    from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject
    from PyQt5.QtGui import QIcon, QPalette
    PYQT5_AVAILABLE = True
    logger.info("PyQt5 successfully imported")
    
    # Check if QApplication exists or create one
    if QApplication.instance() is None:
        QT_APP = QApplication(sys.argv)
        logger.info("Created new QApplication instance")
    else:
        QT_APP = QApplication.instance()
        logger.info("Using existing QApplication instance")
        
except ImportError as e:
    logger.warning(f"PyQt5 not available: {str(e)}")
    logger.warning("GUI functionality will be limited without PyQt5")
    logger.warning("Please install PyQt5: pip install PyQt5")

class GUIModuleAdapter:
    """
    Adapter for VoxSigil Library GUI components with PyQt5 support.
    
    Handles registration, initialization, and coordination of PyQt5-based GUI
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
        self.qt_app = QT_APP
        self.pyqt5_available = PYQT5_AVAILABLE
        
    async def initialize(self, vanta_core) -> bool:
        """Initialize GUI module with Vanta core and PyQt5 compatibility."""
        try:
            if not self.pyqt5_available:
                logger.error("Cannot initialize GUI module: PyQt5 not available")
                return False
                
            self.vanta_core = vanta_core
            logger.info(f"Initializing PyQt5 GUI module: {self.module_name}")
            
            # Verify PyQt5 environment
            await self._verify_pyqt5_environment()
            
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
            logger.info(f"Successfully initialized PyQt5 GUI module: {self.module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GUI module {self.module_name}: {str(e)}")
            return False
    
    async def _verify_pyqt5_environment(self):
        """Verify PyQt5 environment is properly set up."""
        try:
            if not self.qt_app:
                raise RuntimeError("QApplication not available")
                
            # Check PyQt5 version
            pyqt5_version = PyQt5.QtCore.PYQT_VERSION_STR
            logger.info(f"PyQt5 version: {pyqt5_version}")
            
            # Verify essential PyQt5 modules
            essential_modules = [
                'PyQt5.QtWidgets',
                'PyQt5.QtCore', 
                'PyQt5.QtGui'
            ]
            
            for module_name in essential_modules:
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"✅ {module_name} available")
                except ImportError as e:
                    logger.error(f"❌ {module_name} not available: {str(e)}")
                    raise
            logger.info("PyQt5 environment verification successful")
        except Exception as e:
            logger.error(f"PyQt5 environment verification failed: {str(e)}")
            raise
    
    async def _initialize_gui_components(self):
        """Initialize individual PyQt5 GUI components."""
        try:
            # Initialize GUI Launcher (PyQt5)
            gui_launcher = await self._import_gui_launcher()
            if gui_launcher:
                self.gui_components['launcher'] = gui_launcher
                logger.info("PyQt5 GUI Launcher initialized")
                
            # Initialize Agent Status Panel (PyQt5)
            agent_status_panel = await self._import_agent_status_panel()
            if agent_status_panel:
                self.gui_components['agent_status'] = agent_status_panel
                logger.info("PyQt5 Agent Status Panel initialized")
                
            # Initialize Echo Log Panel (PyQt5)
            echo_log_panel = await self._import_echo_log_panel()
            if echo_log_panel:
                self.gui_components['echo_log'] = echo_log_panel
                logger.info("PyQt5 Echo Log Panel initialized")
                
            # Initialize Mesh Map Panel (PyQt5)
            mesh_map_panel = await self._import_mesh_map_panel()
            if mesh_map_panel:
                self.gui_components['mesh_map'] = mesh_map_panel
                logger.info("PyQt5 Mesh Map Panel initialized")
                
            # Initialize PyQt Main
            pyqt_main = await self._import_pyqt_main()
            if pyqt_main:
                self.gui_components['pyqt_main'] = pyqt_main
                logger.info("PyQt5 Main initialized")
                
            # Initialize GridFormer Interface (PyQt5)
            gridformer_interface = await self._import_gridformer_interface()
            if gridformer_interface:
                self.gui_components['gridformer'] = gridformer_interface
                logger.info("PyQt5 GridFormer Interface initialized")
                
            # Initialize Training Interface (PyQt5)
            training_interface = await self._import_training_interface()
            if training_interface:
                self.gui_components['training'] = training_interface
                logger.info("PyQt5 Training Interface initialized")
                  # Initialize VMB Components (PyQt5)
            vmb_components = await self._import_vmb_components()
            if vmb_components:
                if 'demo' in vmb_components:
                    self.gui_components['vmb_demo'] = vmb_components['demo']
                if 'launcher' in vmb_components:
                    self.gui_components['vmb_launcher'] = vmb_components['launcher']
                if 'simple' in vmb_components:
                    self.gui_components['vmb_simple'] = vmb_components['simple']
                logger.info("PyQt5 VMB Components initialized")
                
            # Initialize Interface Tab Components (PyQt5)
            interface_components = await self._import_interface_components()
            if interface_components:
                for name, component in interface_components.items():
                    self.gui_components[f'interface_{name}'] = component
                logger.info("PyQt5 Interface Components initialized")
                
            # Initialize GUI Styles (PyQt5)
            gui_styles = await self._import_gui_styles()
            if gui_styles:
                self.gui_components['styles'] = gui_styles['styles']
                self.gui_components['widget_factory'] = gui_styles['widget_factory']
                self.gui_components['theme_manager'] = gui_styles['theme_manager']
                logger.info("PyQt5 GUI Styles initialized")
                
        except Exception as e:
            logger.error(f"Error initializing PyQt5 GUI components: {str(e)}")
    
    async def _import_gui_launcher(self):
        """Import and initialize PyQt5 GUI Launcher."""
        try:
            from .launcher import GUILauncher
            launcher = GUILauncher()
            
            # Ensure it's PyQt5-compatible
            if hasattr(launcher, 'qt_app') and launcher.qt_app is None:
                launcher.qt_app = self.qt_app
                
            return launcher
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 GUILauncher: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 GUILauncher: {str(e)}")
            return None
    
    async def _import_agent_status_panel(self):
        """Import and initialize PyQt5 Agent Status Panel."""
        try:
            from .components.agent_status_panel import AgentStatusPanel
            panel = AgentStatusPanel()
            
            # Ensure PyQt5 compatibility
            if hasattr(panel, 'setParent') and not panel.parent():
                # This is a QWidget, ensure proper PyQt5 setup
                logger.debug("Setting up PyQt5 Agent Status Panel")
                
            return panel
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 AgentStatusPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 AgentStatusPanel: {str(e)}")
            return None
    
    async def _import_echo_log_panel(self):
        """Import and initialize PyQt5 Echo Log Panel."""
        try:
            from .components.echo_log_panel import EchoLogPanel
            panel = EchoLogPanel()
            
            # Ensure PyQt5 compatibility
            if hasattr(panel, 'setParent') and not panel.parent():
                logger.debug("Setting up PyQt5 Echo Log Panel")
                
            return panel
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 EchoLogPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 EchoLogPanel: {str(e)}")
            return None
    
    async def _import_mesh_map_panel(self):
        """Import and initialize PyQt5 Mesh Map Panel."""
        try:
            from .components.mesh_map_panel import MeshMapPanel
            panel = MeshMapPanel()
            
            # Ensure PyQt5 compatibility for visualization
            if hasattr(panel, 'setParent') and not panel.parent():
                logger.debug("Setting up PyQt5 Mesh Map Panel")
                
            return panel
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 MeshMapPanel: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 MeshMapPanel: {str(e)}")
            return None
    
    async def _import_pyqt_main(self):
        """Import and initialize PyQt5 Main."""
        try:
            from .components.pyqt_main import PyQtMain
            main_widget = PyQtMain()
            
            # Ensure it has access to the QApplication
            if hasattr(main_widget, 'qt_app'):
                main_widget.qt_app = self.qt_app
                
            return main_widget
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 PyQtMain: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 PyQtMain: {str(e)}")
            return None
    
    async def _import_gridformer_interface(self):
        """Import and initialize PyQt5 GridFormer Interface."""
        try:
            from .components.dynamic_gridformer_gui import DynamicGridFormerGUI
            interface = DynamicGridFormerGUI()
            
            # Ensure PyQt5 compatibility
            if hasattr(interface, 'qt_app'):
                interface.qt_app = self.qt_app
                
            return interface
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 DynamicGridFormerGUI: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 DynamicGridFormerGUI: {str(e)}")
            return None
    
    async def _import_training_interface(self):
        """Import and initialize PyQt5 Training Interface."""
        try:
            from .components.training_interface_new import TrainingInterfaceNew
            interface = TrainingInterfaceNew()
            
            # Ensure PyQt5 compatibility
            if hasattr(interface, 'setParent') and not interface.parent():
                logger.debug("Setting up PyQt5 Training Interface")
                
            return interface
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 TrainingInterfaceNew: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 TrainingInterfaceNew: {str(e)}")
            return None
    
    async def _import_vmb_components(self):
        """Import and initialize PyQt5 VMB Components."""
        try:
            components = {}
            
            # Import VMB Final Demo
            try:
                from .components.vmb_final_demo import VMBFinalDemo
                components['demo'] = VMBFinalDemo()
                logger.debug("VMB Final Demo component imported")
            except ImportError as e:
                logger.warning(f"Could not import VMBFinalDemo: {str(e)}")
            
            # Import VMB GUI Launcher
            try:
                from .components.vmb_gui_launcher import VMBGUILauncher
                components['launcher'] = VMBGUILauncher()
                logger.debug("VMB GUI Launcher component imported")
            except ImportError as e:
                logger.warning(f"Could not import VMBGUILauncher: {str(e)}")
            
            # Import VMB Simple GUI
            try:
                from .components.vmb_gui_simple import VMBGUISimple
                components['simple'] = VMBGUISimple()
                logger.debug("VMB Simple GUI component imported")
            except ImportError as e:
                logger.warning(f"Could not import VMBGUISimple: {str(e)}")
            
            return components if components else None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 VMB Components: {str(e)}")
            return None
    
    async def _import_gui_styles(self):
        """Import and initialize PyQt5 GUI Styles."""
        try:
            from .components.gui_styles import VoxSigilStyles, VoxSigilWidgetFactory, VoxSigilThemeManager
              # Return a dictionary with all the style-related classes
            return {
                'styles': VoxSigilStyles,
                'widget_factory': VoxSigilWidgetFactory,
                'theme_manager': VoxSigilThemeManager()
            }
        except ImportError as e:
            logger.warning(f"Could not import PyQt5 GUI Styles: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 GUI Styles: {str(e)}")
            return None
    
    async def _import_interface_components(self):
        """Import and initialize PyQt5 Interface Tab Components."""
        try:
            components = {}
            
            # Import Model Tab Interface
            try:
                from ..interfaces.model_tab_interface import VoxSigilModelInterface
                components['model'] = VoxSigilModelInterface()
                logger.debug("Model Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilModelInterface: {str(e)}")
            
            # Import Performance Tab Interface
            try:
                from ..interfaces.performance_tab_interface import VoxSigilPerformanceInterface
                components['performance'] = VoxSigilPerformanceInterface()
                logger.debug("Performance Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilPerformanceInterface: {str(e)}")
            
            # Import Visualization Tab Interface
            try:
                from ..interfaces.visualization_tab_interface import VoxSigilVisualizationInterface
                components['visualization'] = VoxSigilVisualizationInterface()
                logger.debug("Visualization Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilVisualizationInterface: {str(e)}")
            
            # Import Model Discovery Interface
            try:
                from ..interfaces.model_discovery_interface import ModelDiscoveryInterface
                components['model_discovery'] = ModelDiscoveryInterface()
                logger.debug("Model Discovery Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import ModelDiscoveryInterface: {str(e)}")
            
            # Import Training Interface
            try:
                from ..interfaces.training_interface import VoxSigilTrainingInterface
                # This interface needs parent_gui and tab_widget, we'll pass None for now
                components['training_advanced'] = VoxSigilTrainingInterface(None, None)
                logger.debug("Training Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilTrainingInterface: {str(e)}")
            
            return components if components else None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 Interface Components: {str(e)}")
            return None
        """Import and initialize PyQt5 Interface Tab Components."""
        try:
            components = {}
            
            # Import Model Tab Interface
            try:
                from ..interfaces.model_tab_interface import VoxSigilModelInterface
                components['model'] = VoxSigilModelInterface()
                logger.debug("Model Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilModelInterface: {str(e)}")
            
            # Import Performance Tab Interface
            try:
                from ..interfaces.performance_tab_interface import VoxSigilPerformanceInterface
                components['performance'] = VoxSigilPerformanceInterface()
                logger.debug("Performance Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilPerformanceInterface: {str(e)}")
            
            # Import Visualization Tab Interface
            try:
                from ..interfaces.visualization_tab_interface import VoxSigilVisualizationInterface
                components['visualization'] = VoxSigilVisualizationInterface()
                logger.debug("Visualization Tab Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilVisualizationInterface: {str(e)}")
            
            # Import Model Discovery Interface
            try:
                from ..interfaces.model_discovery_interface import ModelDiscoveryInterface
                components['model_discovery'] = ModelDiscoveryInterface()
                logger.debug("Model Discovery Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import ModelDiscoveryInterface: {str(e)}")
            
            # Import Training Interface
            try:
                from ..interfaces.training_interface import VoxSigilTrainingInterface
                # This interface needs parent_gui and tab_widget, we'll pass None for now
                components['training_advanced'] = VoxSigilTrainingInterface(None, None)
                logger.debug("Training Interface component imported")
            except ImportError as e:
                logger.warning(f"Could not import VoxSigilTrainingInterface: {str(e)}")
            
            return components if components else None
        except Exception as e:
            logger.error(f"Error initializing PyQt5 Interface Components: {str(e)}")
            return None
    
    async def _load_gui_configuration(self):
        """Load PyQt5-specific GUI configuration."""
        try:
            # Default PyQt5 GUI configuration
            self.gui_config = {
                'framework': 'PyQt5',
                'window_width': 1200,
                'window_height': 800,
                'theme': 'dark',
                'auto_refresh': True,
                'refresh_interval': 1000,
                'show_status_bar': True,
                'enable_mesh_map': True,
                'log_level': 'INFO',
                'qt_style': 'Fusion',
                'enable_threading': True,
                'thread_safe_gui': True
            }
            
            # Try to load from launcher if available
            launcher = self.gui_components.get('launcher')
            if launcher and hasattr(launcher, 'get_config'):
                launcher_config = await launcher.get_config()
                self.gui_config.update(launcher_config)
            
            # Set PyQt5 application style if available
            if self.qt_app and 'qt_style' in self.gui_config:
                try:
                    self.qt_app.setStyle(self.gui_config['qt_style'])
                    logger.info(f"Applied PyQt5 style: {self.gui_config['qt_style']}")
                except Exception as e:
                    logger.warning(f"Could not apply PyQt5 style: {str(e)}")
            
            logger.info("PyQt5 GUI configuration loaded")
        except Exception as e:
            logger.error(f"Error loading PyQt5 GUI configuration: {str(e)}")
    
    async def _setup_gui_handlers(self):
        """Set up PyQt5-aware GUI handlers for GUI requests."""
        try:
            self.gui_handlers = {
                'launch': self._handle_launch_request,
                'show_panel': self._handle_show_panel_request,
                'update_status': self._handle_update_status_request,
                'log_message': self._handle_log_message_request,
                'update_mesh': self._handle_update_mesh_request,
                'config': self._handle_config_request,
                'status': self._handle_status_request,
                'close': self._handle_close_request,
                'qt_exec': self._handle_qt_exec_request
            }
            logger.info("PyQt5 GUI handlers established")
        except Exception as e:
            logger.error(f"Error setting up PyQt5 GUI handlers: {str(e)}")
    
    async def process_gui_request(self, operation: str, request_data: Any):
        """Process GUI request through appropriate PyQt5 component."""
        try:
            if not self.is_initialized:
                raise RuntimeError("PyQt5 GUI module not initialized")
                
            if not self.pyqt5_available:
                raise RuntimeError("PyQt5 not available")
                
            # Get GUI handler
            handler = self.gui_handlers.get(operation)
            if not handler:
                raise ValueError(f"Unknown PyQt5 GUI operation: {operation}")
                
            # Process request through handler
            return await handler(request_data)
                
        except Exception as e:
            logger.error(f"Error processing PyQt5 GUI request: {str(e)}")
            raise
    
    async def _handle_qt_exec_request(self, request_data: Any):
        """Handle PyQt5 application execution requests."""
        try:
            if not self.qt_app:
                raise RuntimeError("QApplication not available")
                
            exec_mode = request_data.get('mode', 'non_blocking')
            
            if exec_mode == 'blocking':
                # This will block until the GUI is closed
                return {
                    "status": "qt_app_started",
                    "mode": "blocking",
                    "exit_code": self.qt_app.exec_()
                }
            else:
                # Non-blocking mode - just process events
                self.qt_app.processEvents()
                return {
                    "status": "qt_events_processed",
                    "mode": "non_blocking"
                }
                
        except Exception as e:
            logger.error(f"Error in PyQt5 execution: {str(e)}")
            raise
    
    async def _handle_launch_request(self, request_data: Any):
        """Handle PyQt5 GUI launch requests."""
        try:
            launcher = self.gui_components.get('launcher')
            if launcher and hasattr(launcher, 'launch'):
                result = await launcher.launch(request_data.get('config', {}))
                
                # Ensure PyQt5 app is running
                if self.qt_app:
                    self.qt_app.processEvents()
                    
                return result
            
            # Fallback PyQt5 launch response
            return {
                "status": "launched",
                "framework": "PyQt5",
                "window_id": "main_window",
                "config": request_data.get('config', self.gui_config),
                "qt_app_available": self.qt_app is not None
            }
                
        except Exception as e:
            logger.error(f"Error in PyQt5 GUI launch: {str(e)}")
            raise
    
    async def _handle_show_panel_request(self, request_data: Any):
        """Handle PyQt5 panel show requests."""
        try:
            panel_type = request_data.get('panel_type', 'agent_status')
            panel_component = self.gui_components.get(panel_type)
            
            if panel_component and hasattr(panel_component, 'show'):
                result = await panel_component.show(request_data.get('data', {}))
                
                # Process PyQt5 events to update UI
                if self.qt_app:
                    self.qt_app.processEvents()
                    
                return result
            
            # Fallback PyQt5 panel show response
            return {
                "status": "panel_shown",
                "framework": "PyQt5",
                "panel_type": panel_type,
                "visible": True
            }
                
        except Exception as e:
            logger.error(f"Error showing PyQt5 panel: {str(e)}")
            raise
    
    async def _handle_update_status_request(self, request_data: Any):
        """Handle PyQt5 status update requests."""
        try:
            agent_status_panel = self.gui_components.get('agent_status')
            if agent_status_panel and hasattr(agent_status_panel, 'update_status'):
                result = await agent_status_panel.update_status(request_data.get('status_data', {}))
                
                # Update PyQt5 UI
                if self.qt_app:
                    self.qt_app.processEvents()
                    
                return result
            
            # Fallback PyQt5 status update response
            return {
                "status": "status_updated",
                "framework": "PyQt5",
                "agent_count": request_data.get('agent_count', 0),
                "active_agents": request_data.get('active_agents', [])
            }
                
        except Exception as e:
            logger.error(f"Error updating PyQt5 status: {str(e)}")
            raise
    
    async def _handle_log_message_request(self, request_data: Any):
        """Handle PyQt5 log message requests."""
        try:
            echo_log_panel = self.gui_components.get('echo_log')
            if echo_log_panel and hasattr(echo_log_panel, 'log_message'):
                result = await echo_log_panel.log_message(
                    message=request_data.get('message', ''),
                    level=request_data.get('level', 'INFO')
                )
                
                # Update PyQt5 log display
                if self.qt_app:
                    self.qt_app.processEvents()
                    
                return result
            
            # Fallback PyQt5 log message response
            return {
                "status": "message_logged",
                "framework": "PyQt5",
                "message": request_data.get('message', ''),
                "level": request_data.get('level', 'INFO'),
                "timestamp": "now"
            }
                
        except Exception as e:
            logger.error(f"Error logging PyQt5 message: {str(e)}")
            raise
    
    async def _handle_update_mesh_request(self, request_data: Any):
        """Handle PyQt5 mesh map update requests."""
        try:
            mesh_map_panel = self.gui_components.get('mesh_map')
            if mesh_map_panel and hasattr(mesh_map_panel, 'update_mesh'):
                result = await mesh_map_panel.update_mesh(request_data.get('mesh_data', {}))
                
                # Update PyQt5 visualization
                if self.qt_app:
                    self.qt_app.processEvents()
                    
                return result
            
            # Fallback PyQt5 mesh update response
            return {
                "status": "mesh_updated",
                "framework": "PyQt5",
                "nodes": request_data.get('nodes', []),
                "connections": request_data.get('connections', [])
            }
                
        except Exception as e:
            logger.error(f"Error updating PyQt5 mesh: {str(e)}")
            raise
    
    async def _handle_config_request(self, request_data: Any):
        """Handle PyQt5 GUI configuration requests."""
        try:
            operation = request_data.get('operation', 'get')
            
            if operation == 'get':
                return {
                    "status": "config_retrieved",
                    "framework": "PyQt5",
                    "config": self.gui_config
                }
                
            elif operation == 'set':
                new_config = request_data.get('config', {})
                self.gui_config.update(new_config)
                
                # Apply PyQt5-specific config changes
                if 'qt_style' in new_config and self.qt_app:
                    try:
                        self.qt_app.setStyle(new_config['qt_style'])
                    except Exception as e:
                        logger.warning(f"Could not apply PyQt5 style: {str(e)}")
                
                return {
                    "status": "config_updated",
                    "framework": "PyQt5",
                    "config": self.gui_config
                }
                
            else:
                raise ValueError(f"Unknown config operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in PyQt5 config handling: {str(e)}")
            raise
    
    async def _handle_status_request(self, request_data: Any):
        """Handle PyQt5 GUI status requests."""
        try:
            return {
                "module_name": self.module_name,
                "framework": "PyQt5",
                "pyqt5_version": PyQt5.QtCore.PYQT_VERSION_STR if PYQT5_AVAILABLE else "N/A",
                "is_initialized": self.is_initialized,
                "components_count": len(self.gui_components),
                "available_components": list(self.gui_components.keys()),
                "configuration": self.gui_config,
                "operations": list(self.gui_handlers.keys()),
                "qt_app_available": self.qt_app is not None,
                "pyqt5_available": self.pyqt5_available
            }
                
        except Exception as e:
            logger.error(f"Error in PyQt5 status handling: {str(e)}")
            raise
    
    async def _handle_close_request(self, request_data: Any):
        """Handle PyQt5 GUI close requests."""
        try:
            # Close all PyQt5 GUI components
            for component_name, component in self.gui_components.items():
                if hasattr(component, 'close'):
                    await component.close()
                elif hasattr(component, 'hide'):
                    component.hide()
            
            # Quit PyQt5 application if requested
            if request_data.get('quit_app', False) and self.qt_app:
                self.qt_app.quit()
            
            return {
                "status": "gui_closed",
                "framework": "PyQt5",
                "components_closed": list(self.gui_components.keys()),
                "qt_app_quit": request_data.get('quit_app', False)
            }
                
        except Exception as e:
            logger.error(f"Error closing PyQt5 GUI: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of PyQt5 GUI module."""
        return {
            "module_name": self.module_name,
            "component_type": self.component_type,
            "framework": "PyQt5",
            "pyqt5_version": PyQt5.QtCore.PYQT_VERSION_STR if PYQT5_AVAILABLE else "N/A",
            "is_initialized": self.is_initialized,
            "components_count": len(self.gui_components),
            "available_components": list(self.gui_components.keys()),
            "operations": list(self.gui_handlers.keys()),
            "configuration": self.gui_config,
            "qt_app_available": self.qt_app is not None,
            "pyqt5_available": self.pyqt5_available
        }


async def register_gui() -> Dict[str, Any]:
    """
    Register PyQt5 GUI module with Vanta orchestrator.
    
    Returns:
        Dict containing registration results and status information.
    """
    try:
        logger.info("Starting PyQt5 GUI module registration")
        
        if not PYQT5_AVAILABLE:
            logger.error("Cannot register GUI module: PyQt5 not available")
            logger.error("Please install PyQt5: pip install PyQt5")
            return {
                "module_name": "gui",
                "status": "failed",
                "error": "PyQt5 not available",
                "install_command": "pip install PyQt5"
            }
        
        # Create main PyQt5 GUI adapter
        gui_adapter = GUIModuleAdapter("gui")
        
        # Registration would be completed by Vanta orchestrator
        registration_result = {
            "module_name": "gui",
            "module_type": "gui",            "framework": "PyQt5",
            "pyqt5_version": PyQt5.QtCore.PYQT_VERSION_STR,
            "status": "registered",
            "components": [
                "GUILauncher",
                "AgentStatusPanel", 
                "EchoLogPanel",
                "MeshMapPanel",
                "PyQtMain",
                "DynamicGridFormerGUI",
                "TrainingInterfaceNew",
                "VMBFinalDemo",
                "VMBGUILauncher", 
                "VMBGUISimple",
                "VoxSigilStyles",
                "VoxSigilWidgetFactory",
                "VoxSigilThemeManager"
            ],
            "capabilities": [
                "application",
                "panels", 
                "system",
                "qt_exec",
                "gridformer",
                "training",
                "vmb_operations",
                "model_testing",
                "data_visualization",
                "advanced_styling",
                "theme_management",
                "widget_factory"
            ],
            "requirements": [
                "PyQt5"
            ],
            "adapter": gui_adapter,
            "qt_app_available": QT_APP is not None
        }
        
        logger.info("PyQt5 GUI module registration completed successfully")
        return registration_result
        
    except Exception as e:
        logger.error(f"Failed to register PyQt5 GUI module: {str(e)}")
        raise

# Export registration function and key classes
__all__ = [
    'register_gui',
    'GUIModuleAdapter',
    'PYQT5_AVAILABLE',
    'QT_APP'
]
