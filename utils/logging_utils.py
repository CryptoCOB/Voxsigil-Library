# voxsigil_supervisor/utils/logging_utils.py
"""
Logging utilities for the VoxSigil Supervisor.
"""
import logging
import sys
from typing import Optional # Added Optional typing

SUPERVISOR_LOGGER_NAME = "VoxSigilSupervisor" # Corrected to VoxSigil

def setup_supervisor_logging(level: int = logging.INFO, log_file_path: Optional[str] = None): # Added type hint for level
    """
    Configures a standardized logger for the supervisor package.
    Should ideally be called once by the application using the supervisor.
    """
    logger = logging.getLogger(SUPERVISOR_LOGGER_NAME)
    if logger.hasHandlers(): 
        # Optional: clear existing handlers if re-configuring, or just return if already configured
        # logger.handlers.clear() 
        return logger # Already configured

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s') # Changed module to filename

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file_path:
        try:
            fh = logging.FileHandler(log_file_path, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            # Use a basic print here as logger might not be fully set up for errors during setup
            print(f"Error setting up file handler for VoxSigil logging: {e}")
    
    return logger

def get_supervisor_logger(module_name_str: str) -> logging.Logger: # Renamed for clarity
    """
    Gets a logger instance namespaced under the main supervisor logger.
    e.g., VoxSigilSupervisor.interfaces.rag
    """
    # Extract the actual module name part from a potentially longer string (like __name__)
    clean_module_name = module_name_str.split('.')[-1]
    return logging.getLogger(f"{SUPERVISOR_LOGGER_NAME}.{clean_module_name}")