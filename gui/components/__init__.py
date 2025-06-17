"""
VoxSigil GUI Components Package
Contains all GUI components for the VoxSigil interface.
"""

# Import enhanced components directly (skip main unified for now due to syntax issues)
try:
    from .enhanced_model_tab import EnhancedModelTab
except ImportError:
    EnhancedModelTab = None

try:
    from .enhanced_model_discovery_tab import EnhancedModelDiscoveryTab
except ImportError:
    EnhancedModelDiscoveryTab = None

try:
    from .enhanced_visualization_tab import EnhancedVisualizationTab
except ImportError:
    EnhancedVisualizationTab = None

# Skip the main unified for now due to syntax errors
VoxSigilMainWindow = None

__all__ = [
    "VoxSigilMainWindow",
    "EnhancedModelTab",
    "EnhancedModelDiscoveryTab",
    "EnhancedVisualizationTab",
]
