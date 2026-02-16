"""
Time Utilities

This module provides time-related utility functions for the MetaConsciousness framework.
"""

import time
import datetime
import logging
from typing import Optional, Union, Tuple

# Configure logger
logger = logging.getLogger(__name__)

def get_current_timestamp() -> float:
    """
    Get the current Unix timestamp.
    
    Returns:
        Current Unix timestamp (seconds since epoch)
    """
    return time.time()

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
        timestamp = time.time()
    
    try:
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        return dt_object.strftime(fmt)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid timestamp value for formatting: {e}")
        return "Invalid Timestamp"

def parse_timestamp(timestamp_str: str, 
                  fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[float]:
    """
    Parse a timestamp string into a Unix timestamp.
    
    Args:
        timestamp_str: Timestamp string
        fmt: String format of the timestamp
        
    Returns:
        Unix timestamp or None if parsing fails
    """
    try:
        dt_object = datetime.datetime.strptime(timestamp_str, fmt)
        return dt_object.timestamp()
    except ValueError as e:
        logger.warning(f"Error parsing timestamp string '{timestamp_str}': {e}")
        return None

def calculate_elapsed_time(start_time: float) -> float:
    """
    Calculate elapsed time since a starting timestamp.
    
    Args:
        start_time: Starting Unix timestamp
        
    Returns:
        Elapsed time in seconds
    """
    return time.time() - start_time

def format_elapsed_time(elapsed_seconds: float) -> str:
    """
    Format elapsed time in seconds into a human-readable string.
    
    Args:
        elapsed_seconds: Elapsed time in seconds
        
    Returns:
        Formatted elapsed time string
    """
    if elapsed_seconds < 0:
        return "Invalid Duration"
    
    if elapsed_seconds < 0.001:
        return f"{elapsed_seconds * 1000000:.2f} μs"
    elif elapsed_seconds < 1:
        return f"{elapsed_seconds * 1000:.2f} ms"
    elif elapsed_seconds < 60:
        return f"{elapsed_seconds:.2f} s"
    elif elapsed_seconds < 3600:
        minutes = int(elapsed_seconds / 60)
        seconds = elapsed_seconds % 60
        return f"{minutes} min {seconds:.2f} s"
    elif elapsed_seconds < 86400:
        hours = int(elapsed_seconds / 3600)
        minutes = int((elapsed_seconds % 3600) / 60)
        seconds = elapsed_seconds % 60
        return f"{hours} h {minutes} min {seconds:.2f} s"
    else:
        days = int(elapsed_seconds / 86400)
        hours = int((elapsed_seconds % 86400) / 3600)
        minutes = int((elapsed_seconds % 3600) / 60)
        return f"{days} days {hours} h {minutes} min"

def get_timestamp_components(timestamp: Optional[float] = None) -> Tuple[int, int, int, int, int, int]:
    """
    Get individual components of a timestamp.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch). If None, uses current time.
        
    Returns:
        Tuple of (year, month, day, hour, minute, second)
    """
    if timestamp is None:
        timestamp = time.time()
    
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    return (dt_object.year, dt_object.month, dt_object.day, 
            dt_object.hour, dt_object.minute, dt_object.second)

def create_timestamp(year: int, month: int, day: int, 
                    hour: int = 0, minute: int = 0, second: int = 0) -> float:
    """
    Create a Unix timestamp from individual components.
    
    Args:
        year: Year
        month: Month (1-12)
        day: Day of month
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)
        
    Returns:
        Unix timestamp
    """
    try:
        dt_object = datetime.datetime(year, month, day, hour, minute, second)
        return dt_object.timestamp()
    except ValueError as e:
        logger.warning(f"Error creating timestamp: {e}")
        return 0.0

def is_same_day(timestamp1: float, timestamp2: float) -> bool:
    """
    Check if two timestamps are on the same day.
    
    Args:
        timestamp1: First timestamp
        timestamp2: Second timestamp
        
    Returns:
        True if both timestamps are on the same day (in local time)
    """
    dt1 = datetime.datetime.fromtimestamp(timestamp1)
    dt2 = datetime.datetime.fromtimestamp(timestamp2)
    
    return (dt1.year == dt2.year and 
            dt1.month == dt2.month and 
            dt1.day == dt2.day)
