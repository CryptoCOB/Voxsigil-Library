# VMB (Voxsigil Memory Braid) Module
# This module contains all VMB-related functionality including:
# - Memory activation and operations
# - Configuration and status management
# - Production execution and testing
# - Advanced demonstrations and reporting

__version__ = "1.0.0"
__author__ = "VoxSigil Library"

# Import main VMB components
from .vmb_operations import *
from .vmb_activation import *
from .vmb_config_status import *

__all__ = [
    "vmb_operations",
    "vmb_activation", 
    "vmb_config_status",
    "vmb_status",
    "vmb_production_executor",
    "vmb_advanced_demo",
    "vmb_completion_report",
    "vmb_final_status",
    "vmb_import_test"
]
