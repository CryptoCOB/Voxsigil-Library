"""
VoxSigil GUI Components - UI components for the GridFormer and Vanta systems
"""

# Import all interface components for easy access
# Note: Real DynamicGridFormerGUI is now fixed and can be imported
from .gui.components.pyqt_main import VoxSigilMainWindow, launch

# from .model_discovery_interface import ModelDiscoveryInterface
# from .gui.model_tab_interface import VoxSigilModelInterface
# from .neural_interface import NeuralInterface
# from .gui.performance_tab_interface import VoxSigilPerformanceInterface
# from .gui.testing_tab_interface import VoxSigilTestingInterface
# from .training_interface import VoxSigilTrainingInterface
# from .gui.visualization_tab_interface import VoxSigilVisualizationInterface
# from .visualization_utils import GridVisualizer, PerformanceVisualizer
# Import new enhanced components that are working
from .enhanced_testing_interface import EnhancedVoxSigilTestingInterface
from .voxsigil_integration import (
    get_voxsigil_integration,
    initialize_voxsigil_integration,
)

# Export all components - Real DynamicGridFormerGUI is now working!
__all__ = [
    "DynamicGridFormerGUI",
    "EnhancedVoxSigilTestingInterface",
    "initialize_voxsigil_integration",
    "get_voxsigil_integration",
]
