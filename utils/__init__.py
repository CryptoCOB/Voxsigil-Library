# Utilities Module
# Contains utility functions and helpers for the Voxsigil Library

__version__ = "1.0.0"

from .numpy_resolver import *
from .path_helper import *
from .visualization_utils import *

__all__ = [
    "safe_import_numpy",
    "safe_array", 
    "safe_zeros",
    "safe_concatenate",
    "add_project_root_to_path",
    "get_project_root"
]
