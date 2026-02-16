"""
General Utilities

This module provides miscellaneous utility functions for the MetaConsciousness framework.
"""

import datetime
import json
import os
from pathlib import Path
import sys
import hashlib
import random
import string
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

# User Input Utilities

def confirm_action(prompt: str) -> bool:
    """
    Asks the user for confirmation (y/n).
    
    Args:
        prompt: Question to ask the user
        
    Returns:
        True if user confirms (y), False if user declines (n or empty)
    """
    while True:
        try:
            response = input(f"{prompt} [y/N]: ").lower().strip()
            if response == 'y': 
                return True
            elif response == 'n' or response == '': 
                return False  # Default No
            else: 
                print("Invalid input. Please enter 'y' or 'n'.")
        except EOFError:
            logger.warning("EOFError reading confirmation, defaulting to 'No'.")
            return False

def get_relative_path(filepath: Path, root_dir: Path) -> str:
    """
    Gets the path relative to the project root.
    
    Args:
        filepath: The file path to convert
        root_dir: The root directory to make the path relative to
        
    Returns:
        The relative path as a string
    """
    try:
        return str(filepath.relative_to(root_dir))
    except ValueError:
        return str(filepath)

def retry(func: Callable[..., T], 
         max_attempts: int = 3, 
         delay: float = 1.0,
         backoff_factor: float = 2.0,
         exceptions: tuple = (Exception,)) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Factor to increase delay by after each attempt
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Result of the function
        
    Raises:
        Last exception encountered if all attempts fail
    """
    attempt = 0
    current_delay = delay
    
    while attempt < max_attempts:
        try:
            return func()
        except exceptions as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
                
            logger.warning(f"Attempt {attempt} failed with error: {e}. Retrying in {current_delay}s...")
            time.sleep(current_delay)
            current_delay *= backoff_factor

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a random ID.
    
    Args:
        prefix: Optional prefix
        length: Length of random part
        
    Returns:
        Generated ID
    """
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}{random_part}"

def calculate_hash(data: Union[str, bytes, Dict[str, Any]]) -> str:
    """
    Calculate SHA-256 hash of data.
    
    Args:
        data: Data to hash (string, bytes, or dict)
        
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.sha256()
    
    if isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    elif isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, dict):
        # Sort keys for consistent hashing
        hasher.update(str(sorted(data.items())).encode('utf-8'))
    else:
        raise TypeError(f"Unsupported data type for hashing: {type(data)}")
        
    return hasher.hexdigest()

def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if a string is a valid UUID.
    
    Args:
        uuid_string: UUID string to check
        
    Returns:
        True if valid UUID, False otherwise
    """
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(uuid_pattern.match(uuid_string))

def flatten_dict(nested_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        nested_dict: Nested dictionary
        separator: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    result = {}
    
    def _flatten(d, prefix=''):
        for key, value in d.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                result[new_key] = value
    
    _flatten(nested_dict)
    return result

def deep_get(dictionary: Dict[str, Any], 
            keys: Union[str, List[str]], 
            default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using a dot-separated path.
    
    Args:
        dictionary: Dictionary to get value from
        keys: Dot-separated path or list of keys
        default: Default value if path not found
        
    Returns:
        Value or default
    """
    if isinstance(keys, str):
        keys = keys.split('.')
        
    current = dictionary
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
        
    return current

def deep_set(dictionary: Dict[str, Any], 
            keys: Union[str, List[str]], 
            value: Any) -> Dict[str, Any]:
    """
    Set a value in a nested dictionary using a dot-separated path.
    
    Args:
        dictionary: Dictionary to set value in
        keys: Dot-separated path or list of keys
        value: Value to set
        
    Returns:
        Modified dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
        
    current = dictionary
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
        
    current[keys[-1]] = value
    return dictionary

def chunks(lst: List[T], n: int) -> List[List[T]]:
    """
    Split a list into chunks of size n.
    
    Args:
        lst: List to split
        n: Chunk size
        
    Returns:
        List of chunks
    """
    if n <= 0:
        raise ValueError("Chunk size must be positive")
        
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def is_running_in_notebook() -> bool:
    """
    Check if code is running in a Jupyter notebook.
    
    Returns:
        True if running in a notebook, False otherwise
    """
    try:
        # Check if 'get_ipython' exists and its class name contains 'ZMQ'
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False

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

# JSON State Utility Functions
def _load_json_state(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads state dictionary from a JSON file with error handling."""
    logger = logging.getLogger(__name__)
    try:
        if not os.path.exists(filepath):
            logger.warning(f"State file not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Successfully loaded state from {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load state from {filepath}: {e}")
        return None

def _save_json_state(state: Dict[str, Any], filepath: str) -> bool:
    """Saves state dictionary to a JSON file with error handling."""
    logger = logging.getLogger(__name__)
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Serialize to JSON with indentation for readability
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)  # Use str as fallback for non-serializable objects

        logger.info(f"Successfully saved state to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save state to {filepath}: {e}")
        return False

def _safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Performs division safely, returning default if denominator is zero or near-zero."""
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator
