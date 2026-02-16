"""
Event Logging Utilities

This module provides centralized event logging utilities for the MetaConsciousness framework.
"""

import logging
import json
import time
import threading
import os
import traceback
from typing import Dict, Any, Optional, Union, List, Callable

# Configure logger
logger = logging.getLogger(__name__)

# Thread-local storage for event context
_event_context = threading.local()
_event_context.context = {}

# Global lock for thread safety
_log_lock = threading.Lock()

# Functional Feature 1: Log levels mapping
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def set_event_context(key: str, value: Any) -> None:
    """
    Set a value in the event context that will be included in all subsequent event logs.
    
    Args:
        key: Context key
        value: Context value
    """
    if not hasattr(_event_context, 'context'):
        _event_context.context = {}
    _event_context.context[key] = value

def clear_event_context() -> None:
    """Clear the current event context."""
    _event_context.context = {}

def get_event_context() -> Dict[str, Any]:
    """
    Get the current event context.
    
    Returns:
        Dictionary with the current event context
    """
    return getattr(_event_context, 'context', {}).copy()

# Functional Feature 2: Add context manager for temporary context
class EventContext:
    """Context manager for temporarily setting event context values."""
    
    def __init__(self, **context_values) -> None:
        """
        Initialize with context values to set temporarily.
        
        Args:
            **context_values: Key-value pairs to add to the event context
        """
        self.context_values = context_values
        self.previous_values = {}
        
    def __enter__(self):
        """Set temporary context values, saving previous values."""
        if not hasattr(_event_context, 'context'):
            _event_context.context = {}
            
        # Save previous values and set new ones
        for key, value in self.context_values.items():
            self.previous_values[key] = _event_context.context.get(key)
            _event_context.context[key] = value
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore previous context values."""
        for key, value in self.previous_values.items():
            if value is None:
                # Remove key if it wasn't there before
                _event_context.context.pop(key, None)
            else:
                # Restore previous value
                _event_context.context[key] = value

def log_event(event_name: str, metadata: Optional[Dict[str, Any]] = None, 
              level: str = "INFO", include_context: bool = True) -> Dict[str, Any]:
    """
    Log an event with structured metadata.
    
    Args:
        event_name: Name of the event
        metadata: Additional metadata for the event
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        include_context: Whether to include the event context
        
    Returns:
        Event data dictionary
    """
    metadata = metadata or {}
    log_level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    
    event_data = {
        "event": event_name,
        "timestamp": time.time(),
        "metadata": metadata
    }
    
    # Add event context if available and requested
    if include_context:
        context = get_event_context()
        if context:
            event_data["context"] = context
    
    # Log with appropriate level
    with _log_lock:
        logger.log(log_level, f"Event: {event_name} | {json.dumps(metadata)}")
    
    # Try to notify the event system via SDKContext
    try:
        from MetaConsciousness.core.context import SDKContext
        event_system = SDKContext.get("event_system")
        if event_system and hasattr(event_system, 'notify'):
            event_system.notify(event_name, event_data)
    except ImportError:
        pass  # SDKContext not available
    except AttributeError:
        pass  # Event system not available or doesn't have notify method
    except Exception as e:
        logger.debug(f"Error notifying event system: {e}")
    
    return event_data

def log_structured_event(event_data: Dict[str, Any], level: str = "INFO") -> Dict[str, Any]:
    """
    Log a pre-structured event.
    
    Args:
        event_data: Pre-structured event data
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Updated event data dictionary
    """
    if "timestamp" not in event_data:
        event_data["timestamp"] = time.time()
    
    log_level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    
    # Log with appropriate level
    with _log_lock:
        logger.log(log_level, f"Event: {event_data.get('event', 'UNNAMED')} | {json.dumps(event_data)}")
    
    return event_data

# Functional Feature 3: Add event filters
def create_event_filter(filter_func: Callable[[Dict[str, Any]], bool]) -> Callable:
    """
    Create a filter function for events.
    
    Args:
        filter_func: Function that takes an event data dictionary and returns True if the event should be processed
        
    Returns:
        A decorator that can be applied to an event handler function
    """
    def decorator(handler_func):
        def wrapper(event_data):
            if filter_func(event_data):
                return handler_func(event_data)
            return None
        return wrapper
    return decorator

def configure_event_logging(log_file: Optional[str] = None, 
                           log_level: int = logging.INFO,
                           format_str: Optional[str] = None) -> None:
    """
    Configure event logging.
    
    Args:
        log_file: Path to log file (None for console only)
        log_level: Logging level
        format_str: Log format string
    """
    root_logger = logging.getLogger("metaconsciousness")
    root_logger.setLevel(log_level)
    
    # Default format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            logger.info(f"Event logging configured with file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up log file: {e}")

# Encapsulated Feature 1: Thread ID tracking
def _get_current_thread_info() -> Dict[str, Any]:
    """
    Get information about the current thread.
    
    Returns:
        Dictionary with thread information
    """
    thread = threading.current_thread()
    return {
        "thread_id": thread.ident,
        "thread_name": thread.name,
        "is_main_thread": thread is threading.main_thread()
    }

# Encapsulated Feature 2: Exception formatter
def _format_exception(exc: Exception) -> Dict[str, Any]:
    """
    Format an exception for inclusion in event data.
    
    Args:
        exc: Exception to format
        
    Returns:
        Dictionary with formatted exception information
    """
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc()
    }

# Encapsulated Feature 3: Message formatter
def _format_log_message(event_name: str, metadata: Dict[str, Any]) -> str:
    """
    Format a log message consistently.
    
    Args:
        event_name: Name of the event
        metadata: Event metadata
        
    Returns:
        Formatted message string
    """
    try:
        return f"Event: {event_name} | {json.dumps(metadata, default=str)}"
    except Exception:
        return f"Event: {event_name} | {metadata}"

# Encapsulated Feature 4: Rate limiting
_last_event_times = {}
_rate_limit_lock = threading.Lock()

def _check_rate_limit(event_name: str, 
                     min_interval: float = 1.0) -> bool:
    """
    Check if an event should be rate-limited.
    
    Args:
        event_name: Name of the event
        min_interval: Minimum interval between events in seconds
        
    Returns:
        True if event should be processed, False if it should be rate-limited
    """
    current_time = time.time()
    
    with _rate_limit_lock:
        last_time = _last_event_times.get(event_name, 0)
        if current_time - last_time < min_interval:
            return False
            
        _last_event_times[event_name] = current_time
        return True

# Encapsulated Feature 5: Event broadcasting
_event_subscribers = {}
_subscribers_lock = threading.Lock()

def subscribe_to_event(event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
    """
    Subscribe to an event.
    
    Args:
        event_name: Name of the event to subscribe to (or '*' for all events)
        handler: Function to call when the event occurs
    """
    with _subscribers_lock:
        if event_name not in _event_subscribers:
            _event_subscribers[event_name] = []
        _event_subscribers[event_name].append(handler)

def unsubscribe_from_event(event_name: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
    """
    Unsubscribe from an event.
    
    Args:
        event_name: Name of the event
        handler: Handler function to remove
        
    Returns:
        True if handler was removed, False if not found
    """
    with _subscribers_lock:
        if event_name in _event_subscribers:
            try:
                _event_subscribers[event_name].remove(handler)
                return True
            except ValueError:
                return False
    return False

def broadcast_event(event_data: Dict[str, Any]) -> None:
    """
    Broadcast an event to all subscribers.
    
    Args:
        event_data: Event data to broadcast
    """
    event_name = event_data.get("event", "")
    
    with _subscribers_lock:
        # Call specific event handlers
        handlers = _event_subscribers.get(event_name, [])
        # Also call wildcard handlers
        handlers.extend(_event_subscribers.get("*", []))
    
    # Call handlers outside the lock to prevent deadlocks
    for handler in handlers:
        try:
            handler(event_data)
        except Exception as e:
            logger.error(f"Error in event handler for {event_name}: {e}")
