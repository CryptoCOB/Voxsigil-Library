"""
Registry Helper Utilities

This module provides utilities for safely interacting with the SDK's central registry
through the core.context SDKContext.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Union

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

def update_utility_registry() -> bool:
    """
    Ensure that all utility functions are properly registered with SDKContext.
    
    This function is used to update the registry after fixes to prevent
    component loading failures.
    
    Returns:
        Success status
    """
    try:
        # Import SDKContext
        from MetaConsciousness.core.context import SDKContext
        from MetaConsciousness.utils import format_timestamp, load_json, save_json
        
        # Register utility functions directly
        SDKContext.register("utils.format_timestamp", format_timestamp)
        SDKContext.register("utils.load_json", load_json)
        SDKContext.register("utils.save_json", save_json)
        
        # Also create a utils bundle for easier access
        utils_bundle = {
            "format_timestamp": format_timestamp,
            "load_json": load_json,
            "save_json": save_json
        }
        SDKContext.register("utils", utils_bundle)
        
        logger.info("Updated utility functions registry")
        return True
    except ImportError as e:
        logger.warning(f"Could not import required modules for utility registry update: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating utility registry: {e}")
        return False

def safely_register_component(name: str, component: Any, overwrite: bool = False) -> bool:
    """
    Safely register a component in the SDKContext registry.
    
    Args:
        name: Name of the component in the registry
        component: Component instance to register
        overwrite: Whether to overwrite if component already exists
        
    Returns:
        True if registration was successful, False otherwise
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Check if component already exists
        existing = SDKContext.get(name)
        if existing is not None:
            if not overwrite:
                logger.warning(f"Component '{name}' already registered. Use overwrite=True to replace it.")
                return False
            logger.info(f"Overwriting existing component '{name}'")
        
        # Register the component
        SDKContext.register(name, component)
        logger.info(f"Successfully registered component '{name}'")
        return True
    except ImportError:
        logger.error("Failed to import SDKContext. Is the core module available?")
        return False
    except Exception as e:
        logger.error(f"Error registering component '{name}': {e}")
        return False

def get_component(name: str, default: Any = None) -> Any:
    """
    Safely get a component from the SDKContext registry.
    
    Args:
        name: Name of the component in the registry
        default: Default value to return if component not found
        
    Returns:
        Component instance or default value if not found
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        return SDKContext.get(name, default)
    except ImportError:
        logger.error("Failed to import SDKContext. Is the core module available?")
        return default
    except Exception as e:
        logger.error(f"Error getting component '{name}': {e}")
        return default

def list_components() -> List[str]:
    """
    Get a list of all registered component names.
    
    Returns:
        List of component names or empty list if error
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Try different methods based on SDKContext implementation
        if hasattr(SDKContext, 'get_registry_keys'):
            return SDKContext.get_registry_keys()
        elif hasattr(SDKContext, 'get_registry'):
            registry = SDKContext.get_registry()
            return list(registry.keys())
        elif hasattr(SDKContext, '_registry'):
            # Direct access to private attribute (less ideal)
            registry = SDKContext._registry
            return list(registry.keys())
        else:
            logger.warning("Could not determine how to list registry components.")
            return []
    except ImportError:
        logger.error("Failed to import SDKContext. Is the core module available?")
        return []
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        return []

def register_multiple(components: Dict[str, Any], overwrite: bool = False) -> Dict[str, bool]:
    """
    Register multiple components at once.
    
    Args:
        components: Dictionary mapping names to component instances
        overwrite: Whether to overwrite existing components
        
    Returns:
        Dictionary mapping names to registration success status
    """
    results = {}
    for name, component in components.items():
        results[name] = safely_register_component(name, component, overwrite)
    return results

def unregister_component(name: str) -> bool:
    """
    Unregister a component from the SDKContext registry.
    
    Args:
        name: Name of the component to unregister
        
    Returns:
        True if unregistration was successful, False otherwise
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Check if component exists
        if SDKContext.get(name) is None:
            logger.warning(f"Component '{name}' not found in registry.")
            return False
        
        # Unregister by setting to None
        SDKContext.register(name, None)
        logger.info(f"Successfully unregistered component '{name}'")
        return True
    except ImportError:
        logger.error("Failed to import SDKContext. Is the core module available?")
        return False
    except Exception as e:
        logger.error(f"Error unregistering component '{name}': {e}")
        return False

def get_typed_component(name: str, expected_type: type, default: T = None) -> Union[Any, T]:
    """
    Get a component and verify its type.
    
    Args:
        name: Name of the component in the registry
        expected_type: Expected type of the component
        default: Default value to return if component not found or wrong type
        
    Returns:
        Component instance or default value if not found or wrong type
    """
    component = get_component(name, None)
    
    if component is None:
        logger.warning(f"Component '{name}' not found in registry.")
        return default
    
    if not isinstance(component, expected_type):
        logger.warning(f"Component '{name}' has unexpected type: {type(component).__name__}, expected: {expected_type.__name__}")
        return default
    
    return component

def get_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Get a snapshot of the current registry state with component metadata.
    
    Returns:
        Dictionary with registry state information
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        components = list_components()
        snapshot = {
            "timestamp": time.time(),
            "component_count": len(components),
            "components": {}
        }
        
        for name in components:
            component = SDKContext.get(name)
            if component is not None:
                snapshot["components"][name] = {
                    "type": type(component).__name__,
                    "has_initialize": hasattr(component, "initialize") and callable(getattr(component, "initialize")),
                    "has_shutdown": hasattr(component, "shutdown") and callable(getattr(component, "shutdown")),
                    "has_status": hasattr(component, "get_status") and callable(getattr(component, "get_status"))
                }
            else:
                snapshot["components"][name] = {
                    "type": "None",
                    "is_registered": True
                }
        
        return snapshot
    except ImportError:
        logger.error("Failed to import SDKContext. Is the core module available?")
        return {"error": "SDKContext not available", "timestamp": time.time(), "component_count": 0, "components": {}}
    except Exception as e:
        logger.error(f"Error creating registry snapshot: {e}")
        return {"error": str(e), "timestamp": time.time(), "component_count": 0, "components": {}}

def check_component_health(name: str) -> Dict[str, Any]:
    """
    Check the health of a specific component.
    
    Args:
        name: Name of the component to check
        
    Returns:
        Dictionary with health check results
    """
    component = get_component(name)
    
    if component is None:
        return {
            "name": name,
            "status": "not_found",
            "healthy": False
        }
    
    result = {
        "name": name,
        "type": type(component).__name__,
        "healthy": True
    }
    
    # Check for get_health method
    if hasattr(component, "get_health") and callable(getattr(component, "get_health")):
        try:
            health = component.get_health()
            result["health_info"] = health
            if isinstance(health, dict) and "healthy" in health:
                result["healthy"] = health["healthy"]
        except Exception as e:
            result["health_error"] = str(e)
            result["healthy"] = False
    
    # Check for get_status method
    if hasattr(component, "get_status") and callable(getattr(component, "get_status")):
        try:
            status = component.get_status()
            result["status_info"] = status
        except Exception as e:
            result["status_error"] = str(e)
            result["healthy"] = False
    
    return result

def safely_call_component_method(name: str, method_name: str, *args, **kwargs) -> Tuple[bool, Any]:
    """
    Safely call a method on a registered component.
    
    Args:
        name: Name of the component in the registry
        method_name: Name of the method to call
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
        
    Returns:
        Tuple of (success, result or error message)
    """
    component = get_component(name)
    
    if component is None:
        return False, f"Component '{name}' not found in registry."
    
    if not hasattr(component, method_name):
        return False, f"Component '{name}' has no method '{method_name}'."
    
    method = getattr(component, method_name)
    if not callable(method):
        return False, f"'{method_name}' is not a callable method on component '{name}'."
    
    try:
        result = method(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Error calling method '{method_name}' on component '{name}': {e}")
        return False, str(e)
