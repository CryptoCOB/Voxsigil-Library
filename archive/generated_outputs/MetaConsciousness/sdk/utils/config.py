"""
Configuration Utilities

This module provides utilities for loading and managing configuration files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)

def load_config(file_path: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration file
        default_config: Default configuration to use if file not found or error occurs
        
    Returns:
        Configuration dictionary
    """
    default_config = default_config or {}
    
    if not os.path.exists(file_path):
        logger.warning(f"Configuration file not found: {file_path}. Using default configuration.")
        return default_config.copy()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {file_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration file {file_path}: {e}")
        return default_config.copy()
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {e}")
        return default_config.copy()

def save_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the configuration
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write to temp file first for atomicity
        temp_path = f"{file_path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Rename temp file to target (atomic on most OSes)
        os.replace(temp_path, file_path)
        
        logger.info(f"Saved configuration to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {e}")
        return False

def get_config_value(config: Dict[str, Any], key_path: Union[str, List[str]], 
                    default: Any = None) -> Any:
    """
    Get a value from a nested configuration dictionary using a dot-separated path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "server.port") or list of keys
        default: Default value to return if key not found
        
    Returns:
        Configuration value or default
    """
    if isinstance(key_path, str):
        key_path = key_path.split('.')
    
    current = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current

def set_config_value(config: Dict[str, Any], key_path: Union[str, List[str]], 
                    value: Any) -> bool:
    """
    Set a value in a nested configuration dictionary using a dot-separated path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "server.port") or list of keys
        value: Value to set
        
    Returns:
        True if value was set, False otherwise
    """
    if isinstance(key_path, str):
        key_path = key_path.split('.')
    
    current = config
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            logger.error(f"Cannot set {'.'.join(key_path)} because {key} is not a dictionary")
            return False
        current = current[key]
    
    current[key_path[-1]] = value
    return True

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any],
                 deep_merge: bool = True) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
        deep_merge: Whether to do a deep merge of nested dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, override_value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(override_value, dict) and deep_merge:
            # Recursively merge dictionaries
            result[key] = merge_configs(result[key], override_value, deep_merge)
        else:
            # Override or add value
            result[key] = override_value
    
    return result

def validate_config(config: Dict[str, Any], 
                   schema: Dict[str, Tuple[type, Any]]) -> List[str]:
    """
    Validate a configuration against a schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary mapping keys to (type, default) tuples
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    for key, (expected_type, default) in schema.items():
        if key not in config:
            if default is not None:
                # Add default value
                config[key] = default
            else:
                errors.append(f"Missing required key: {key}")
        elif not isinstance(config[key], expected_type):
            errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(config[key]).__name__}")
    
    return errors
