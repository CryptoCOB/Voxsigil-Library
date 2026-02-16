"""
Helper functions for logging, I/O, and formatting.
"""
import time
import datetime
import json
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MetaConsciousness")

def format_timestamp(timestamp=None) -> None:
    """
    Format a timestamp for logging and tracing.

    Args:
        timestamp: Unix timestamp (or current time if None)

    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()

    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def log_event(msg, level="INFO") -> None:
    """
    Log an event with the specified level.

    Args:
        msg: Message to log
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    log_level = level_map.get(level.upper(), logging.INFO)
    logger.log(log_level, msg)

    # Print to console for interactive use
    if log_level >= logging.INFO:
        print(f"[{format_timestamp()}] {msg}")

def ensure_directory(directory_path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log_event(f"Error creating directory {directory_path}: {e}", "ERROR")
        return False

def save_json(data, filepath) -> None:
    """
    Save data as JSON to a file.

    Args:
        data: Data to save
        filepath: Path to save the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        log_event(f"Error saving JSON: {e}", "ERROR")
        return False

def load_json(filepath) -> None:
    """
    Load JSON data from a file.

    Args:
        filepath: Path to the file

    Returns:
        dict: Loaded data, or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(filepath):
            log_event(f"File not found: {filepath}", "WARNING")
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data
    except Exception as e:
        log_event(f"Error loading JSON from {filepath}: {e}", "ERROR")
        return None

def calculate_moving_average(data, window_size=5) -> None:
    """
    Calculate moving average of a data series.

    Args:
        data: List of numeric values
        window_size: Size of the moving window

    Returns:
        list: Moving averages
    """
    if not data:
        return []

    if len(data) < window_size:
        # If we have less data than window size, use all available data
        window_size = len(data)

    result = []
    for i in range(len(data)):
        if i < window_size - 1:
            # For the first few elements, use all available data
            result.append(sum(data[:i+1]) / (i+1))
        else:
            # For later elements, use the sliding window
            result.append(sum(data[i-window_size+1:i+1]) / window_size)

    return result

"""
Main utilities module (for backward compatibility).
"""
from MetaConsciousness.utils.log_event import log_event, get_recent_logs, clear_logs

# Re-export all functions
__all__ = ['log_event', 'get_recent_logs', 'clear_logs']

# Add any utility functions that might be used directly from this file below
