"""
Component Linker Module

This module provides utilities for connecting and managing dependencies between
components in the MetaConsciousness framework.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class ComponentLinker:
    """
    Manages connections and dependencies between components.
    
    This class helps keep track of component relationships, supports dependency
    injection, and provides utilities for dependency resolution.
    """
    
    def __init__(self) -> None:
        """Initialize the component linker."""
        self.components = {}  # name -> component
        self.dependencies = {}  # name -> [dependencies]
        self.dependents = {}  # name -> [dependents]
        self.connections = {}  # (source, target) -> connection_info
        self.registration_time = {}  # name -> registration_time
    
    def register_component(self, name: str, component: Any) -> bool:
        """
        Register a component with the linker.
        
        Args:
            name: Component name
            component: Component instance
            
        Returns:
            True if registration was successful, False otherwise
        """
        if name in self.components:
            logger.warning(f"Component '{name}' already registered, overwriting")
        
        self.components[name] = component
        self.registration_time[name] = time.time()
        
        if name not in self.dependencies:
            self.dependencies[name] = []
        if name not in self.dependents:
            self.dependents[name] = []
        
        logger.info(f"Registered component: {name}")
        return True
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the linker.
        
        Args:
            name: Component name
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if name not in self.components:
            logger.warning(f"Component '{name}' not registered")
            return False
        
        # Remove connections involving this component
        connections_to_remove = []
        for (source, target) in self.connections:
            if source == name or target == name:
                connections_to_remove.append((source, target))
        
        for connection in connections_to_remove:
            del self.connections[connection]
        
        # Remove component
        del self.components[name]
        del self.registration_time[name]
        
        # Clean up dependencies
        for dependency in self.dependencies.get(name, []):
            if name in self.dependents.get(dependency, []):
                self.dependents[dependency].remove(name)
        
        for dependent in self.dependents.get(name, []):
            if name in self.dependencies.get(dependent, []):
                self.dependencies[dependent].remove(name)
        
        if name in self.dependencies:
            del self.dependencies[name]
        if name in self.dependents:
            del self.dependents[name]
        
        logger.info(f"Unregistered component: {name}")
        return True
    
    def connect_components(self, source: str, target: str, 
                          connection_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish a connection between two components.
        
        Args:
            source: Source component name
            target: Target component name
            connection_info: Optional metadata about the connection
            
        Returns:
            True if connection was successful, False otherwise
        """
        if source not in self.components:
            logger.warning(f"Source component '{source}' not registered")
            return False
        
        if target not in self.components:
            logger.warning(f"Target component '{target}' not registered")
            return False
        
        # Update dependencies
        if target not in self.dependencies[source]:
            self.dependencies[source].append(target)
        
        if source not in self.dependents[target]:
            self.dependents[target].append(source)
        
        # Store connection
        self.connections[(source, target)] = connection_info or {}
        
        logger.info(f"Connected components: {source} -> {target}")
        return True
    
    def disconnect_components(self, source: str, target: str) -> bool:
        """
        Remove a connection between two components.
        
        Args:
            source: Source component name
            target: Target component name
            
        Returns:
            True if disconnection was successful, False otherwise
        """
        if (source, target) not in self.connections:
            logger.warning(f"No connection from '{source}' to '{target}'")
            return False
        
        # Remove connection
        del self.connections[(source, target)]
        
        # Update dependencies
        if target in self.dependencies.get(source, []):
            self.dependencies[source].remove(target)
        
        if source in self.dependents.get(target, []):
            self.dependents[target].remove(source)
        
        logger.info(f"Disconnected components: {source} -> {target}")
        return True
    
    def get_component(self, name: str) -> Any:
        """
        Get a registered component.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def get_dependencies(self, name: str) -> List[str]:
        """
        Get the dependencies of a component.
        
        Args:
            name: Component name
            
        Returns:
            List of dependency names
        """
        return self.dependencies.get(name, [])
    
    def get_dependents(self, name: str) -> List[str]:
        """
        Get the dependents of a component.
        
        Args:
            name: Component name
            
        Returns:
            List of dependent names
        """
        return self.dependents.get(name, [])
    
    def get_connection_info(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a connection.
        
        Args:
            source: Source component name
            target: Target component name
            
        Returns:
            Connection information or None if no connection exists
        """
        return self.connections.get((source, target))
    
    def resolve_dependencies(self, name: str) -> List[str]:
        """
        Resolve the dependency order for a component.
        
        Args:
            name: Component name
            
        Returns:
            List of component names in dependency order
        """
        if name not in self.components:
            logger.warning(f"Component '{name}' not registered")
            return []
        
        # Use depth-first search to resolve dependencies
        visited = set()
        order = []
        
        def dfs(component):
            if component in visited:
                return
            
            visited.add(component)
            
            for dependency in self.dependencies.get(component, []):
                dfs(dependency)
            
            order.append(component)
        
        dfs(name)
        return order
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect dependency cycles in the component graph.
        
        Returns:
            List of cycles (each cycle is a list of component names)
        """
        # Use a modified DFS to detect cycles
        all_cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    all_cycles.append(path[cycle_start:] + [neighbor])
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in self.components:
            if node not in visited:
                dfs(node, [])
        
        return all_cycles
    
    def get_component_graph(self) -> Dict[str, Any]:
        """
        Get a representation of the component dependency graph.
        
        Returns:
            Dictionary with graph information
        """
        return {
            "nodes": list(self.components.keys()),
            "edges": [(source, target) for source, target in self.connections],
            "components": {name: {
                "dependencies": self.dependencies.get(name, []),
                "dependents": self.dependents.get(name, []),
                "registration_time": self.registration_time.get(name)
            } for name in self.components}
        }
    
    def initialize_component(self, name: str, init_args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a component with its dependencies.
        
        Args:
            name: Component name
            init_args: Optional initialization arguments
            
        Returns:
            True if initialization was successful, False otherwise
        """
        component = self.get_component(name)
        if not component:
            logger.warning(f"Component '{name}' not registered")
            return False
        
        # Resolve dependencies
        dependency_order = self.resolve_dependencies(name)
        
        # Remove the component itself from the dependency order
        if name in dependency_order:
            dependency_order.remove(name)
        
        # Initialize dependencies first
        for dependency in dependency_order:
            if hasattr(self.components[dependency], "initialize"):
                try:
                    self.components[dependency].initialize()
                except Exception as e:
                    logger.error(f"Error initializing dependency '{dependency}': {e}")
                    return False
        
        # Initialize the component
        init_args = init_args or {}
        try:
            if hasattr(component, "initialize"):
                component.initialize(**init_args)
            logger.info(f"Initialized component: {name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing component '{name}': {e}")
            return False

    def shutdown_component(self, name: str, shutdown_dependents: bool = True) -> bool:
        """
        Shutdown a component and optionally its dependents.
        
        Args:
            name: Component name
            shutdown_dependents: Whether to also shutdown components that depend on this one
            
        Returns:
            True if shutdown was successful, False otherwise
        """
        component = self.get_component(name)
        if not component:
            logger.warning(f"Component '{name}' not registered")
            return False
        
        # Shutdown dependents first if requested
        if shutdown_dependents:
            for dependent in self.get_dependents(name):
                self.shutdown_component(dependent, True)
        
        # Shutdown the component
        try:
            if hasattr(component, "shutdown"):
                component.shutdown()
            logger.info(f"Shutdown component: {name}")
            return True
        except Exception as e:
            logger.error(f"Error shutting down component '{name}': {e}")
            return False
    
    def validate_component_interface(self, name: str, required_methods: List[str]) -> bool:
        """
        Validate that a component implements required methods.
        
        Args:
            name: Component name
            required_methods: List of method names that should be implemented
            
        Returns:
            True if component implements all required methods, False otherwise
        """
        component = self.get_component(name)
        if not component:
            logger.warning(f"Component '{name}' not registered")
            return False
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(component, method_name) or not callable(getattr(component, method_name)):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.warning(f"Component '{name}' is missing required methods: {missing_methods}")
            return False
        
        return True
    
    def broadcast_event(self, event_name: str, event_data: Any = None) -> Dict[str, Any]:
        """
        Broadcast an event to all components that have a handle_event method.
        
        Args:
            event_name: Name of the event
            event_data: Optional event data
            
        Returns:
            Dictionary mapping component names to their response
        """
        results = {}
        for name, component in self.components.items():
            if hasattr(component, "handle_event") and callable(getattr(component, "handle_event")):
                try:
                    results[name] = component.handle_event(event_name, event_data)
                except Exception as e:
                    logger.error(f"Error broadcasting event '{event_name}' to component '{name}': {e}")
                    results[name] = {"error": str(e)}
        
        return results

# Create a global instance for convenience
linker = ComponentLinker()
