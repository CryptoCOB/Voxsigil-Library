"""
Trace Utilities

This module provides utilities for managing trace history in the MetaConsciousness framework.
"""

import time
import threading
import logging
import json
from collections import deque
from typing import Dict, Any, List, Optional, Deque, Union

# Configure logger
logger = logging.getLogger(__name__)

# Thread-safe trace history
_trace_history_lock = threading.Lock()
_trace_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

# Functional Feature 1: Add category support for grouping traces
_category_histories: Dict[str, Deque[Dict[str, Any]]] = {}
_category_locks: Dict[str, threading.Lock] = {}

def add_trace_event(event_name: str, metadata: Optional[Dict[str, Any]] = None, 
                    category: Optional[str] = None) -> Dict[str, Any]:
    """
    Add an event to the trace history.
    
    Args:
        event_name: Name of the event
        metadata: Additional metadata for the event
        category: Optional category to group related traces
        
    Returns:
        Trace event data
    """
    metadata = metadata or {}
    
    # Create trace event
    trace_event = {
        "event": event_name,
        "timestamp": time.time(),
        "metadata": metadata,
        "thread_id": threading.get_ident()
    }
    
    # Try to get component info from SDKContext
    try:
        from MetaConsciousness.core.context import SDKContext
        current_context = SDKContext.get("current_context")
        if current_context:
            trace_event["context"] = current_context
    except (ImportError, AttributeError):
        pass  # SDKContext not available or doesn't have current_context
    
    # Add to main history with thread safety
    with _trace_history_lock:
        _trace_history.append(trace_event)
    
    # Add to category history if specified
    if category:
        # Create category history if not exists
        if category not in _category_histories:
            with _trace_history_lock:  # Ensure thread-safe initialization
                if category not in _category_histories:
                    _category_histories[category] = deque(maxlen=1000)
                    _category_locks[category] = threading.Lock()
        
        # Add to category history
        with _category_locks[category]:
            _category_histories[category].append(trace_event)
    
    return trace_event

# Functional Feature 2: Add named trace sessions
_trace_sessions: Dict[str, List[Dict[str, Any]]] = {}
_session_locks: Dict[str, threading.Lock] = {}

def start_trace_session(session_name: str) -> str:
    """
    Start a new trace session for grouping related trace events.
    
    Args:
        session_name: Name for the trace session
        
    Returns:
        Session name (same as input)
    """
    with _trace_history_lock:
        if session_name not in _trace_sessions:
            _trace_sessions[session_name] = []
            _session_locks[session_name] = threading.Lock()
    
    logger.debug(f"Started trace session: {session_name}")
    return session_name

def add_to_trace_session(session_name: str, event_name: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Add an event to a specific trace session.
    
    Args:
        session_name: Name of the trace session
        event_name: Name of the event
        metadata: Additional metadata for the event
        
    Returns:
        Trace event data or None if session doesn't exist
    """
    if session_name not in _trace_sessions:
        logger.warning(f"Trace session not found: {session_name}")
        return None
    
    # Create trace event
    trace_event = {
        "event": event_name,
        "timestamp": time.time(),
        "metadata": metadata or {},
        "thread_id": threading.get_ident(),
        "session": session_name
    }
    
    # Add to main history
    with _trace_history_lock:
        _trace_history.append(trace_event)
    
    # Add to session
    with _session_locks[session_name]:
        _trace_sessions[session_name].append(trace_event)
    
    return trace_event

def end_trace_session(session_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    End a trace session and return its events.
    
    Args:
        session_name: Name of the trace session
        
    Returns:
        List of trace events in the session or None if session doesn't exist
    """
    if session_name not in _trace_sessions:
        logger.warning(f"Trace session not found: {session_name}")
        return None
    
    with _session_locks[session_name]:
        session_events = _trace_sessions[session_name]
        del _trace_sessions[session_name]
    
    del _session_locks[session_name]
    
    logger.debug(f"Ended trace session {session_name} with {len(session_events)} events")
    return session_events

# Functional Feature 3: Export traces to JSON
def export_traces_to_json(filepath: str, 
                         filter_event: Optional[str] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> bool:
    """
    Export trace history to a JSON file.
    
    Args:
        filepath: Path to save the JSON file
        filter_event: Filter events by name
        start_time: Filter events after this timestamp
        end_time: Filter events before this timestamp
        
    Returns:
        True if export was successful, False otherwise
    """
    try:
        traces = get_trace_history(filter_event=filter_event, 
                                   start_time=start_time, 
                                   end_time=end_time)
        
        # Create export data structure
        export_data = {
            "export_timestamp": time.time(),
            "trace_count": len(traces),
            "traces": traces
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(traces)} trace events to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting traces to {filepath}: {e}")
        return False

def get_trace_history(limit: Optional[int] = None, 
                     filter_event: Optional[str] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Get events from the trace history with optional filtering.
    
    Args:
        limit: Maximum number of events to return
        filter_event: Filter events by name
        start_time: Filter events after this timestamp
        end_time: Filter events before this timestamp
        
    Returns:
        List of trace events
    """
    with _trace_history_lock:
        # Create a copy to avoid thread safety issues
        events = list(_trace_history)
    
    # Apply filters
    if filter_event:
        events = [e for e in events if e.get("event") == filter_event]
    
    if start_time:
        events = [e for e in events if e.get("timestamp", 0) >= start_time]
    
    if end_time:
        events = [e for e in events if e.get("timestamp", float("inf")) <= end_time]
    
    # Apply limit after filtering
    if limit and limit > 0:
        events = events[-limit:]
    
    return events

def get_category_trace_history(category: str,
                              limit: Optional[int] = None,
                              filter_event: Optional[str] = None,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Get events from a specific category trace history with optional filtering.
    
    Args:
        category: Category name
        limit: Maximum number of events to return
        filter_event: Filter events by name
        start_time: Filter events after this timestamp
        end_time: Filter events before this timestamp
        
    Returns:
        List of trace events or empty list if category doesn't exist
    """
    if category not in _category_histories:
        return []
    
    with _category_locks[category]:
        # Create a copy to avoid thread safety issues
        events = list(_category_histories[category])
    
    # Apply filters
    if filter_event:
        events = [e for e in events if e.get("event") == filter_event]
    
    if start_time:
        events = [e for e in events if e.get("timestamp", 0) >= start_time]
    
    if end_time:
        events = [e for e in events if e.get("timestamp", float("inf")) <= end_time]
    
    # Apply limit after filtering
    if limit and limit > 0:
        events = events[-limit:]
    
    return events

def clear_trace_history() -> None:
    """Clear the trace history."""
    with _trace_history_lock:
        _trace_history.clear()
    logger.debug("Trace history cleared")

def clear_category_trace_history(category: str) -> bool:
    """
    Clear a specific category trace history.
    
    Args:
        category: Category name
        
    Returns:
        True if category existed and was cleared, False otherwise
    """
    if category not in _category_histories:
        return False
    
    with _category_locks[category]:
        _category_histories[category].clear()
    
    logger.debug(f"Trace history for category '{category}' cleared")
    return True

def set_trace_history_limit(max_size: int) -> None:
    """
    Set the maximum size of the trace history.
    
    Args:
        max_size: Maximum number of events to keep
    """
    if max_size <= 0:
        logger.warning(f"Invalid trace history size: {max_size}. Must be positive.")
        return
    
    global _trace_history
    
    with _trace_history_lock:
        # Create a new deque with the new max size and copy existing items
        new_history = deque(list(_trace_history), maxlen=max_size)
        _trace_history = new_history
    
    logger.debug(f"Trace history limit set to {max_size}")

def set_category_trace_history_limit(category: str, max_size: int) -> bool:
    """
    Set the maximum size of a category trace history.
    
    Args:
        category: Category name
        max_size: Maximum number of events to keep
        
    Returns:
        True if category existed and was updated, False otherwise
    """
    if max_size <= 0:
        logger.warning(f"Invalid trace history size: {max_size}. Must be positive.")
        return False
    
    if category not in _category_histories:
        return False
    
    with _category_locks[category]:
        # Create a new deque with the new max size and copy existing items
        new_history = deque(list(_category_histories[category]), maxlen=max_size)
        _category_histories[category] = new_history
    
    logger.debug(f"Trace history limit for category '{category}' set to {max_size}")
    return True

def get_trace_summary() -> Dict[str, Any]:
    """
    Get a summary of the trace history.
    
    Returns:
        Dictionary with trace summary
    """
    with _trace_history_lock:
        if not _trace_history:
            return {"count": 0, "oldest": None, "newest": None, "event_types": {}}
        
        # Count event types
        event_counts = {}
        for event in _trace_history:
            event_type = event.get("event", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "count": len(_trace_history),
            "oldest": _trace_history[0]["timestamp"] if _trace_history else None,
            "newest": _trace_history[-1]["timestamp"] if _trace_history else None,
            "event_types": event_counts
        }

# Encapsulated Feature 1: Thread identification helper
def _get_thread_identifier() -> Dict[str, Any]:
    """
    Get identifier information for the current thread.
    
    Returns:
        Dictionary with thread information
    """
    thread = threading.current_thread()
    return {
        "id": thread.ident,
        "name": thread.name,
        "is_main": thread is threading.main_thread()
    }

# Encapsulated Feature 2: Trace data sanitization
def _sanitize_trace_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize trace data to ensure it can be serialized to JSON.
    
    Args:
        data: Trace data to sanitize
        
    Returns:
        Sanitized copy of the data
    """
    if not isinstance(data, dict):
        return {"value": str(data)}
    
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _sanitize_trace_data(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [_sanitize_trace_data(item) if isinstance(item, dict) else str(item) for item in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        else:
            result[key] = str(value)
    
    return result

# Encapsulated Feature 3: Trace ID generator
_next_trace_id = 0
_trace_id_lock = threading.Lock()

def _generate_trace_id() -> int:
    """
    Generate a unique trace ID.
    
    Returns:
        Unique trace ID
    """
    global _next_trace_id
    with _trace_id_lock:
        trace_id = _next_trace_id
        _next_trace_id += 1
    return trace_id

# Encapsulated Feature 4: Event type stats
def _get_event_type_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for each event type.
    
    Returns:
        Dictionary mapping event types to their statistics
    """
    event_stats = {}
    
    with _trace_history_lock:
        if not _trace_history:
            return {}
        
        for event in _trace_history:
            event_type = event.get("event", "unknown")
            
            if event_type not in event_stats:
                event_stats[event_type] = {
                    "count": 0,
                    "first_timestamp": float("inf"),
                    "last_timestamp": 0
                }
            
            stats = event_stats[event_type]
            stats["count"] += 1
            stats["first_timestamp"] = min(stats["first_timestamp"], event.get("timestamp", 0))
            stats["last_timestamp"] = max(stats["last_timestamp"], event.get("timestamp", 0))
    
    # Calculate frequencies
    total_events = sum(stats["count"] for stats in event_stats.values())
    if total_events > 0:
        for stats in event_stats.values():
            stats["frequency"] = stats["count"] / total_events
    
    return event_stats

# Encapsulated Feature 5: Performance monitoring
_operation_times: Dict[str, List[float]] = {}
_operation_times_lock = threading.Lock()

def _record_operation_time(operation: str, duration: float) -> None:
    """
    Record the duration of an operation for performance monitoring.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
    """
    with _operation_times_lock:
        if operation not in _operation_times:
            _operation_times[operation] = []
        _operation_times[operation].append(duration)
        
        # Keep only the last 100 times
        if len(_operation_times[operation]) > 100:
            _operation_times[operation] = _operation_times[operation][-100:]

def _get_operation_stats() -> Dict[str, Dict[str, float]]:
    """
    Get statistics for operation times.
    
    Returns:
        Dictionary mapping operations to their statistics
    """
    stats = {}
    
    with _operation_times_lock:
        for operation, times in _operation_times.items():
            if not times:
                continue
                
            stats[operation] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "count": len(times)
            }
    
    return stats
