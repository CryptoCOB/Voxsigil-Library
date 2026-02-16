"""
Utility functions and helpers for the MetaConsciousness framework.

This package provides various utility functions used throughout the framework,
including path setup, logging, tracing, configuration, file I/O, formatting,
mathematical utilities, and more.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List, Union

# Initialize logger
logger = logging.getLogger(__name__)

# Import key utilities for easy access
try:
    from .path_setup import setup_project_path, find_project_root
    from .log_event import log_event, log_structured_event
    from .trace import add_trace_event, get_trace_history, clear_trace_history
    from .math_utils import safe_division, calculate_ema, cosine_similarity, normalize_vector
    from .time_utils import format_timestamp, calculate_elapsed_time, get_current_timestamp
    from .file_io import safe_read_file, safe_write_file, ensure_directory
    from .registry_helper import safely_register_component, get_component
    from .config import load_config, save_config, get_config_value
    
    # Functional Feature 1: Add version information
    __version__ = '0.2.1'
    
    # Functional Feature 2: Track loaded utilities
    _loaded_utilities = {
        'path_setup': True,
        'log_event': True,
        'trace': True,
        'math_utils': True,
        'time_utils': True,
        'file_io': True,
        'registry_helper': True,
        'config': True
    }
    
    # Functional Feature 3: List all re-exported functions
    __all__ = [
        # Path utilities
        'setup_project_path',
        'find_project_root',
        
        # Logging and tracing
        'log_event',
        'log_structured_event',
        'add_trace_event',
        'get_trace_history',
        'clear_trace_history',
        
        # Component interaction
        'safely_register_component',
        'get_component',
        
        # File and directory operations
        'ensure_directory',
        'safe_read_file',
        'safe_write_file',
        
        # Configuration
        'load_config',
        'save_config',
        'get_config_value',
        
        # Time utilities
        'format_timestamp',
        'calculate_elapsed_time',
        'get_current_timestamp',
        
        # Math utilities
        'safe_division',
        'calculate_ema',
        'cosine_similarity',
        'normalize_vector',
        'calculate_moving_average'
    ]
    
except ImportError as e:
    # Encapsulated Feature 1: Graceful import error handling
    logger.warning(f"Could not import one or more utilities: {e}")
    _loaded_utilities = {}
    __all__ = []

# Encapsulated Feature 2: Utility availability check
def is_utility_available(utility_name: str) -> bool:
    """
    Check if a specific utility module is available.
    
    Args:
        utility_name: Name of the utility module to check
        
    Returns:
        True if the utility is available, False otherwise
    """
    return _loaded_utilities.get(utility_name, False)

# Encapsulated Feature 3: Get available utilities
def get_available_utilities() -> List[str]:
    """
    Get a list of available utility modules.
    
    Returns:
        List of available utility module names
    """
    return list(_loaded_utilities.keys())

# Encapsulated Feature 4: Initialize logger for other modules
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the MetaConsciousness namespace.
    
    Args:
        name: Logger name suffix (will be appended to 'metaconsciousness.utils')
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"metaconsciousness.utils.{name}")

# Encapsulated Feature 5: Project path verification
def verify_project_path() -> bool:
    """
    Verify that the MetaConsciousness project path is properly set up.
    
    Returns:
        True if the path is correctly set up, False otherwise
    """
    if 'setup_project_path' not in globals():
        logger.warning("Path setup utility not available")
        return False
    
    try:
        return setup_project_path()
    except Exception as e:
        logger.error(f"Failed to verify project path: {e}")
        return False

def calculate_moving_average(data, window_size=5) -> None:
    """
    Calculate the moving average of a data series.
    
    Args:
        data: List or array of numeric values
        window_size: Size of the moving window
        
    Returns:
        List of moving averages with the same length as input
    """
    import numpy as np
    
    if not data:
        return []
    
    # Convert to numpy array for easier calculations
    data_array = np.array(data)
    
    # Calculate moving average
    result = np.zeros_like(data_array, dtype=float)
    
    for i in range(len(data_array)):
        # Calculate window boundaries
        start = max(0, i - window_size + 1)
        # Extract window and calculate average
        window = data_array[start:i+1]
        result[i] = np.mean(window)
    
    # Convert back to list for broader compatibility
    return result.tolist()

# If imported directly, try to set up the project path
if __name__ != "__main__" and is_utility_available('path_setup'):
    verify_project_path()
