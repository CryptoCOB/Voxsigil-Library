"""
SDK Diagnostics Module

This module provides utilities for diagnosing and troubleshooting the
MetaConsciousness SDK and its components.
"""

import logging
import time
import os
import sys
import json
import importlib
import platform
from typing import Dict, Any, List, Optional, Union, Set, Tuple

# Configure logger
logger = logging.getLogger(__name__)

def get_sdk_info() -> Dict[str, Any]:
    """
    Get basic information about the SDK environment.
    
    Returns:
        Dictionary with SDK information
    """
    info = {
        "timestamp": time.time(),
        "python_version": sys.version,
        "platform": sys.platform,
        "path": sys.path,
        "executable": sys.executable,
        "modules_loaded": list(sorted(sys.modules.keys()))
    }
    
    # Try to get additional MetaConsciousness-specific info
    try:
        from MetaConsciousness import __version__
        info["version"] = __version__
    except (ImportError, AttributeError):
        info["version"] = "unknown"
    
    # Try to get package location
    try:
        import MetaConsciousness
        info["package_location"] = os.path.dirname(os.path.abspath(MetaConsciousness.__file__))
    except (ImportError, AttributeError):
        info["package_location"] = "unknown"
    
    # Add more detailed platform info
    info["platform_details"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    
    # Add Python package information
    info["python_packages"] = {
        "numpy": _get_package_version("numpy"),
        "requests": _get_package_version("requests"),
        "logging": _get_package_version("logging")
    }
    
    return info

def _get_package_version(package_name: str) -> str:
    """
    Helper function to get package version.
    
    Args:
        package_name: Name of the package
        
    Returns:
        Version string or "not installed"
    """
    try:
        return importlib.import_module(package_name).__version__
    except (ImportError, AttributeError):
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except Exception:
            return "not installed"

def check_component_availability() -> Dict[str, bool]:
    """
    Check which components are available in the SDK.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    components = {
        "core": False,
        "memory": False,
        "art": False,
        "agent": False,
        "frameworks": False,
        "interface": False
    }
    
    # Check core components
    try:
        import MetaConsciousness.core
        components["core"] = True
    except ImportError:
        pass
    
    # Check memory components
    try:
        import MetaConsciousness.memory
        components["memory"] = True
    except ImportError:
        pass
    
    # Check art components
    try:
        import MetaConsciousness.art
        components["art"] = True
    except ImportError:
        pass
    
    # Check agent components
    try:
        import MetaConsciousness.agent
        components["agent"] = True
    except ImportError:
        pass
    
    # Check frameworks components
    try:
        import MetaConsciousness.frameworks
        components["frameworks"] = True
    except ImportError:
        pass
    
    # Check interface components
    try:
        import MetaConsciousness.interface
        components["interface"] = True
    except ImportError:
        pass
    
    # Add checks for additional components
    try:
        import MetaConsciousness.omega3
        components["omega3"] = True
    except ImportError:
        components["omega3"] = False
        
    try:
        import MetaConsciousness.rag
        components["rag"] = True
    except ImportError:
        components["rag"] = False
        
    try:
        import MetaConsciousness.narrative
        components["narrative"] = True
    except ImportError:
        components["narrative"] = False
    
    return components

def get_context_registry() -> Dict[str, Any]:
    """
    Get the current state of the SDKContext registry.
    
    Returns:
        Dictionary with registry information or error message
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Attempt to get registry (private attribute)
        # This is implementation-dependent and might change
        registry = {}
        
        if hasattr(SDKContext, 'get_registry'):
            # Preferred way if available
            registry = SDKContext.get_registry()
        elif hasattr(SDKContext, '_registry'):
            # Direct access (be cautious)
            registry = SDKContext._registry.copy()
        elif hasattr(SDKContext, 'get_all'):
            # Alternative method if available
            registry = SDKContext.get_all()
        else:
            # Fallback: enumerate all known keys and try to get them
            # This is not efficient but might work as last resort
            known_keys = [
                "core", "memory_cluster", "art_cluster", "agent_cluster",
                "compression_cluster", "narrative_cluster", "metacognitive",
                "path_homotopy_functor", "sheaf_functor", "utils"
            ]
            registry = {k: SDKContext.get(k) for k in known_keys if SDKContext.get(k) is not None}
        
        # Return registry with types and summaries
        return {
            "registry_size": len(registry),
            "keys": list(registry.keys()),
            "types": {k: type(v).__name__ for k, v in registry.items()},
            "registry_summary": {
                k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                for k, v in registry.items()
            }
        }
    except ImportError:
        return {"error": "SDKContext not available"}
    except Exception as e:
        return {"error": f"Error accessing registry: {e}"}

def diagnose_imports() -> Dict[str, Any]:
    """
    Diagnose import-related issues with the SDK.
    
    Returns:
        Dictionary with import diagnostics
    """
    results = {
        "timestamp": time.time(),
        "import_checks": [],
        "environment": {
            "sys_path": sys.path,
            "python_version": sys.version
        }
    }
    
    # Key modules to check
    modules_to_check = [
        "MetaConsciousness",
        "MetaConsciousness.core",
        "MetaConsciousness.core.context",
        "MetaConsciousness.memory",
        "MetaConsciousness.utils",
        "MetaConsciousness.art",
        "MetaConsciousness.frameworks",
        "MetaConsciousness.agent"
    ]
    
    # Add additional modules to check
    additional_modules = [
        "MetaConsciousness.omega3",
        "MetaConsciousness.omega3.agent",
        "MetaConsciousness.rag",
        "MetaConsciousness.narrative",
        "MetaConsciousness.memory.persistence"
    ]
    
    modules_to_check.extend(additional_modules)
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            results["import_checks"].append({
                "module": module_name,
                "status": "success"
            })
        except ImportError as e:
            results["import_checks"].append({
                "module": module_name,
                "status": "error",
                "message": str(e)
            })
    
    # Check for circular imports
    circular_imports = []
    for module_name in sys.modules:
        if module_name.startswith("MetaConsciousness"):
            module = sys.modules[module_name]
            if hasattr(module, "__name__") and hasattr(module, "__file__"):
                # Try to detect circular import symptoms
                if getattr(module, "__file__", None) is None:
                    circular_imports.append(module_name)
    
    results["potential_circular_imports"] = circular_imports
    
    return results

def check_component_health(component_name: str) -> Dict[str, Any]:
    """
    Check the health of a specific component.
    
    Args:
        component_name: Name of the component to check
        
    Returns:
        Dictionary with component health information
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Get the component
        component = SDKContext.get(component_name)
        
        if component is None:
            return {
                "name": component_name,
                "status": "not_found",
                "message": f"Component '{component_name}' not found in SDKContext"
            }
        
        # Check for health method
        if hasattr(component, "get_health") and callable(component.get_health):
            health = component.get_health()
            return {
                "name": component_name,
                "status": "healthy" if health.get("healthy", False) else "unhealthy",
                "health": health
            }
        
        # Check for status method
        if hasattr(component, "get_status") and callable(component.get_status):
            status = component.get_status()
            return {
                "name": component_name,
                "status": "healthy",  # Assume healthy if status is available
                "component_status": status
            }
        
        # Basic check for common attributes
        attributes = {}
        for attr in ["initialized", "status", "state", "config"]:
            if hasattr(component, attr):
                value = getattr(component, attr)
                attributes[attr] = value if not isinstance(value, dict) else "dict"
        
        return {
            "name": component_name,
            "status": "unknown",
            "type": type(component).__name__,
            "attributes": attributes
        }
    except ImportError:
        return {
            "name": component_name,
            "status": "error",
            "message": "SDKContext not available"
        }
    except Exception as e:
        return {
            "name": component_name,
            "status": "error",
            "message": f"Error checking component: {e}"
        }

def check_component_dependencies(component_name: str) -> Dict[str, Any]:
    """
    Check the dependencies of a specific component.
    
    Args:
        component_name: Name of the component to check
        
    Returns:
        Dictionary with component dependency information
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Get the component
        component = SDKContext.get(component_name)
        
        if component is None:
            return {
                "name": component_name,
                "status": "not_found",
                "message": f"Component '{component_name}' not found in SDKContext"
            }
        
        # Check for get_dependencies method
        dependencies = []
        if hasattr(component, "get_dependencies") and callable(component.get_dependencies):
            try:
                dependencies = component.get_dependencies()
            except Exception as e:
                return {
                    "name": component_name,
                    "status": "error",
                    "message": f"Error getting dependencies: {e}",
                    "dependencies": []
                }
        
        # Infer dependencies from attributes
        inferred_dependencies = []
        dependency_attrs = ["controller", "agent", "memory", "art", "router", "engine"]
        for attr in dependency_attrs:
            if hasattr(component, attr):
                dep = getattr(component, attr)
                if dep is not None:
                    inferred_dependencies.append({
                        "name": attr,
                        "type": type(dep).__name__
                    })
        
        return {
            "name": component_name,
            "status": "success",
            "explicit_dependencies": dependencies,
            "inferred_dependencies": inferred_dependencies
        }
    except ImportError:
        return {
            "name": component_name,
            "status": "error",
            "message": "SDKContext not available"
        }
    except Exception as e:
        return {
            "name": component_name,
            "status": "error",
            "message": f"Error checking dependencies: {e}"
        }

def check_all_components_health() -> Dict[str, Dict[str, Any]]:
    """
    Check the health of all registered components.
    
    Returns:
        Dictionary mapping component names to health information
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Get all registered components
        registry_info = get_context_registry()
        component_names = registry_info.get("keys", [])
        
        # Check health of each component
        health_info = {}
        for name in component_names:
            health_info[name] = check_component_health(name)
        
        return health_info
    except ImportError:
        return {"error": "SDKContext not available"}
    except Exception as e:
        return {"error": f"Error checking component health: {e}"}

def diagnose_component_connections() -> Dict[str, Any]:
    """
    Diagnose connections between components.
    
    Returns:
        Dictionary with connection diagnostics
    """
    try:
        from MetaConsciousness.core.context import SDKContext
        
        # Get component registry
        registry_info = get_context_registry()
        component_names = registry_info.get("keys", [])
        
        connections = {}
        
        # Standard expected connections
        expected_connections = {
            "meta_reflex": ["art_controller", "omega3_agent"],
            "memory_cluster": ["episodic_memory", "semantic_memory"],
            "art_cluster": ["art_controller", "art_trainer"],
            "agent_cluster": ["agent", "monitor"]
        }
        
        # Check expected connections
        for source, targets in expected_connections.items():
            source_component = SDKContext.get(source)
            if source_component is None:
                connections[source] = {
                    "status": "not_found",
                    "message": f"Source component '{source}' not found"
                }
                continue
            
            source_connections = {}
            for target in targets:
                target_component = SDKContext.get(target)
                if target_component is None:
                    source_connections[target] = {
                        "status": "target_not_found",
                        "message": f"Target component '{target}' not found"
                    }
                    continue
                
                # Look for attribute references
                connection_found = False
                for attr_name in dir(source_component):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr_value = getattr(source_component, attr_name)
                        if attr_value is target_component:
                            source_connections[target] = {
                                "status": "connected",
                                "attribute": attr_name
                            }
                            connection_found = True
                            break
                    except Exception:
                        pass
                
                if not connection_found:
                    source_connections[target] = {
                        "status": "not_connected",
                        "message": f"No connection found from '{source}' to '{target}'"
                    }
            
            connections[source] = source_connections
        
        return {
            "status": "success",
            "connections": connections
        }
    except ImportError:
        return {
            "status": "error",
            "message": "SDKContext not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error diagnosing connections: {e}"
        }

def visualize_component_graph(output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a visualization of the component dependency graph.
    
    Args:
        output_file: Optional file to save the visualization (if graphviz is available)
        
    Returns:
        Dictionary with graph information
    """
    try:
        # Get component connections
        connections = diagnose_component_connections()
        if connections.get("status") != "success":
            return connections
        
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Create nodes and edges
        for source, targets in connections.get("connections", {}).items():
            if source not in graph_data["nodes"]:
                graph_data["nodes"].append(source)
            
            for target, info in targets.items():
                if target not in graph_data["nodes"]:
                    graph_data["nodes"].append(target)
                
                if info.get("status") == "connected":
                    graph_data["edges"].append({
                        "source": source,
                        "target": target,
                        "label": info.get("attribute", "")
                    })
        
        # Try to generate visualization if output_file is specified
        if output_file:
            try:
                import graphviz
                dot = graphviz.Digraph("Component Graph")
                
                for node in graph_data["nodes"]:
                    dot.node(node)
                
                for edge in graph_data["edges"]:
                    dot.edge(edge["source"], edge["target"], label=edge["label"])
                
                dot.render(output_file, format="png", cleanup=True)
                graph_data["visualization"] = f"{output_file}.png"
            except ImportError:
                graph_data["visualization_error"] = "graphviz not available"
            except Exception as e:
                graph_data["visualization_error"] = f"Error generating visualization: {e}"
        
        return {
            "status": "success",
            "graph": graph_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating component graph: {e}"
        }

def create_diagnostic_report(output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive diagnostic report.
    
    Args:
        output_file: Optional file to save the report (JSON)
        
    Returns:
        Dictionary with full diagnostic information
    """
    report = {
        "timestamp": time.time(),
        "sdk_info": get_sdk_info(),
        "components_available": check_component_availability(),
        "import_diagnostics": diagnose_imports(),
        "context_registry": get_context_registry(),
        "component_connections": diagnose_component_connections()
    }
    
    # Check health of key components
    component_health = {}
    for component in ["core", "memory_cluster", "art_cluster", "agent_cluster", "omega3_agent"]:
        component_health[component] = check_component_health(component)
    
    report["component_health"] = component_health
    
    # Check dependencies
    component_dependencies = {}
    for component in ["meta_reflex", "art_controller", "memory_cluster", "omega3_agent"]:
        component_dependencies[component] = check_component_dependencies(component)
    
    report["component_dependencies"] = component_dependencies
    
    # Add system information
    report["system_info"] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }
    
    # Save report if requested
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Diagnostic report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving diagnostic report to {output_file}: {e}")
    
    return report

def run_quick_diagnostic() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run a quick diagnostic test on the SDK.
    
    Returns:
        Tuple of (success, message, details)
    """
    try:
        sdk_available = False
        core_available = False
        context_available = False
        registry_available = False
        
        # Check if SDK is available
        try:
            import MetaConsciousness
            sdk_available = True
        except ImportError:
            return False, "MetaConsciousness SDK not found in Python path", {"sdk_available": False}
        
        # Check if core is available
        try:
            import MetaConsciousness.core
            core_available = True
        except ImportError:
            return False, "MetaConsciousness core module not available", {
                "sdk_available": sdk_available,
                "core_available": False
            }
        
        # Check if context is available
        try:
            from MetaConsciousness.core.context import SDKContext
            context_available = True
        except ImportError:
            return False, "SDKContext not available", {
                "sdk_available": sdk_available,
                "core_available": core_available,
                "context_available": False
            }
        
        # Check if registry is accessible
        try:
            registry_info = get_context_registry()
            if "error" not in registry_info:
                registry_available = True
            else:
                return False, f"SDKContext registry not accessible: {registry_info['error']}", {
                    "sdk_available": sdk_available,
                    "core_available": core_available,
                    "context_available": context_available,
                    "registry_available": False
                }
        except Exception as e:
            return False, f"Error accessing SDKContext registry: {e}", {
                "sdk_available": sdk_available,
                "core_available": core_available,
                "context_available": context_available,
                "registry_available": False
            }
        
        # Check component availability
        components_available = check_component_availability()
        missing_components = [name for name, available in components_available.items() if not available]
        
        if missing_components:
            return True, f"MetaConsciousness SDK available but missing components: {', '.join(missing_components)}", {
                "sdk_available": sdk_available,
                "core_available": core_available,
                "context_available": context_available,
                "registry_available": registry_available,
                "components_available": components_available,
                "missing_components": missing_components
            }
        
        return True, "MetaConsciousness SDK available and all core components found", {
            "sdk_available": sdk_available,
            "core_available": core_available,
            "context_available": context_available,
            "registry_available": registry_available,
            "components_available": components_available
        }
    except Exception as e:
        return False, f"Error during quick diagnostic: {e}", {"error": str(e)}

if __name__ == "__main__":
    # When run directly, perform a full diagnostic and print a summary
    success, message, details = run_quick_diagnostic()
    print(f"Quick diagnostic: {'SUCCESS' if success else 'FAILURE'}")
    print(f"Message: {message}")
    
    if success:
        print("\nDetailed component availability:")
        for component, available in details.get("components_available", {}).items():
            print(f"  {component}: {'✓' if available else '✗'}")
        
        # Create and save a full report
        report_file = os.path.join(os.path.dirname(__file__), "sdk_diagnostic_report.json")
        report = create_diagnostic_report(report_file)
        print(f"\nFull diagnostic report saved to: {report_file}")
    else:
        print("\nDiagnosis failed. Please check your installation.")
        for key, value in details.items():
            print(f"  {key}: {value}")
