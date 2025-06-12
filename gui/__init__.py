"""PyQt5 based GUI package with comprehensive VoxSigil components."""

# Core PyQt5 components
from .components.pyqt_main import VoxSigilMainWindow, launch
from .components.agent_status_panel import AgentStatusPanel
from .components.echo_log_panel import EchoLogPanel
from .components.mesh_map_panel import MeshMapPanel

# Advanced GUI components  
from .components.gui_styles import (
    VoxSigilStyles, VoxSigilWidgetFactory, VoxSigilThemeManager,
    AnimatedToolTip, VoxSigilGUIUtils
)

# Application interfaces
try:
    from .components.dynamic_gridformer_gui import DynamicGridFormerGUI
except ImportError:
    DynamicGridFormerGUI = None

try:
    from .components.training_interface_new import TrainingInterfaceNew
except ImportError:
    TrainingInterfaceNew = None

# Interface tab components
try:
    from ..interfaces.model_tab_interface import VoxSigilModelInterface
    from ..interfaces.performance_tab_interface import VoxSigilPerformanceInterface
    from ..interfaces.visualization_tab_interface import VoxSigilVisualizationInterface
    from ..interfaces.model_discovery_interface import ModelDiscoveryInterface
    from ..interfaces.training_interface import VoxSigilTrainingInterface
except ImportError:
    VoxSigilModelInterface = None
    VoxSigilPerformanceInterface = None
    VoxSigilVisualizationInterface = None
    ModelDiscoveryInterface = None
    VoxSigilTrainingInterface = None

# VMB Components
try:
    from .components.vmb_final_demo import VMBFinalDemo
    from .components.vmb_gui_launcher import VMBGUILauncher
    from .components.vmb_gui_simple import VMBGUISimple
except ImportError:
    VMBFinalDemo = None
    VMBGUILauncher = None
    VMBGUISimple = None

# Main launcher
from .launcher import launch_gui_with_fallback

# Registration module
from .register_gui_module import register_gui

__all__ = [
    # Core components
    "VoxSigilMainWindow", 
    "launch",
    "AgentStatusPanel",
    "EchoLogPanel", 
    "MeshMapPanel",
    
    # Styling and utilities
    "VoxSigilStyles",
    "VoxSigilWidgetFactory", 
    "VoxSigilThemeManager",
    "AnimatedToolTip",
    "VoxSigilGUIUtils",
    
    # Application interfaces
    "DynamicGridFormerGUI",
    "TrainingInterfaceNew",
    
    # Interface tab components
    "VoxSigilModelInterface",
    "VoxSigilPerformanceInterface", 
    "VoxSigilVisualizationInterface",
    "ModelDiscoveryInterface",
    "VoxSigilTrainingInterface",
    
    # VMB components
    "VMBFinalDemo",
    "VMBGUILauncher",
    "VMBGUISimple",
    
    # Launchers
    "launch_gui_with_fallback",
    
    # Registration
    "register_gui"
]
