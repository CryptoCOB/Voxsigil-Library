"""
Formatting Utilities

This module provides utilities for formatting text and data.
"""

import json
import logging
import textwrap
import datetime
from typing import Dict, Any, List, Optional, Union
import os

# Configure logger
logger = logging.getLogger(__name__)

def format_timestamp(timestamp: Optional[float] = None, 
                    fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a Unix timestamp into a human-readable string.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch). If None, uses current time.
        fmt: String format for the timestamp
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().timestamp()
    
    try:
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        return dt_object.strftime(fmt)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid timestamp value for formatting: {e}")
        return "Invalid Timestamp"

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds into a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "Invalid Duration"
    
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} h"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def format_json(data: Any, indent: int = 2, sort_keys: bool = False) -> str:
    """
    Format an object as a JSON string.
    
    Args:
        data: JSON-serializable data
        indent: JSON indentation level
        sort_keys: Whether to sort dictionary keys
        
    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent, sort_keys=sort_keys)
    except Exception as e:
        logger.warning(f"Error formatting data as JSON: {e}")
        return str(data)

def format_list(items: List[Any], separator: str = ", ", 
               prefix: str = "", suffix: str = "") -> str:
    """
    Format a list of items as a string.
    
    Args:
        items: List of items
        separator: Separator between items
        prefix: String to prepend
        suffix: String to append
        
    Returns:
        Formatted string
    """
    if not items:
        return f"{prefix}{suffix}"
    
    return f"{prefix}{separator.join(str(item) for item in items)}{suffix}"

def format_dict(data: Dict[str, Any], key_value_sep: str = ": ", 
               item_sep: str = ", ", compact: bool = False) -> str:
    """
    Format a dictionary as a string.
    
    Args:
        data: Dictionary to format
        key_value_sep: Separator between keys and values
        item_sep: Separator between items
        compact: Whether to use compact format
        
    Returns:
        Formatted string
    """
    if not data:
        return "{}"
    
    if compact:
        items = [f"{k}{key_value_sep}{str(v)}" for k, v in data.items()]
        return "{" + item_sep.join(items) + "}"
    else:
        return format_json(data)

def wrap_text(text: str, width: int = 80, indent: str = "") -> str:
    """
    Wrap text to a specified width.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        indent: String to use for indentation
        
    Returns:
        Wrapped text
    """
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)

def format_error(error: Exception, include_traceback: bool = False) -> str:
    """
    Format an exception as a string.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include traceback
        
    Returns:
        Formatted error string
    """
    if include_traceback:
        import traceback
        return f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"
    else:
        return f"{type(error).__name__}: {str(error)}"

def truncate_string(s: str, max_length: int = 100, ellipsis: str = "...") -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        ellipsis: String to append when truncated
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(ellipsis)] + ellipsis

def ensure_directory(directory_path: str) -> bool:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        True if directory exists or was created, False on failure
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception:
        return False
