"""
Cluster Scanner Module

This module provides utilities for scanning component clusters in the MetaConsciousness framework
and identifying orphaned modules or components.
"""

import os
import sys
import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
import time
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable

# Configure logger
logger = logging.getLogger(__name__)

def get_project_root() -> Optional[Path]:
    """
    Find the project root directory.
    
    Returns:
        Path to the project root, or None if not found
    """
    try:
        from MetaConsciousness.utils.path_setup import find_project_root
        return find_project_root()
    except ImportError:
        # Fallback implementation if path_setup is not available
        current_path = Path(__file__).resolve().parent
        for parent in [current_path] + list(current_path.parents):
            if (parent / "MetaConsciousness").is_dir():
                return parent
        return None

def get_metaconsciousness_modules() -> Dict[str, Dict[str, Any]]:
    """
    Get all modules under the MetaConsciousness package.
    
    Returns:
        Dictionary mapping module names to module information
    """
    result = {}
    
    try:
        import MetaConsciousness
    except ImportError:
        logger.error("MetaConsciousness package not found")
        return result
    
    metaconsciousness_path = Path(MetaConsciousness.__file__).parent
    
    for module_info in pkgutil.iter_modules([str(metaconsciousness_path)]):
        module_name = module_info.name
        if not module_name.startswith("_"):  # Skip private modules
            try:
                module = importlib.import_module(f"MetaConsciousness.{module_name}")
                result[module_name] = {
                    "is_package": module_info.ispkg,
                    "path": str(metaconsciousness_path / module_name),
                    "module": module,
                    "submodules": []
                }
                
                # Scan submodules if it's a package
                if module_info.ispkg:
                    subpackage_path = metaconsciousness_path / module_name
                    for submodule_info in pkgutil.iter_modules([str(subpackage_path)]):
                        submodule_name = submodule_info.name
                        if not submodule_name.startswith("_"):  # Skip private modules
                            try:
                                submodule = importlib.import_module(f"MetaConsciousness.{module_name}.{submodule_name}")
                                result[module_name]["submodules"].append({
                                    "name": submodule_name,
                                    "is_package": submodule_info.ispkg,
                                    "path": str(subpackage_path / submodule_name),
                                    "module": submodule
                                })
                            except ImportError as e:
                                logger.warning(f"Could not import submodule {module_name}.{submodule_name}: {e}")
                
            except ImportError as e:
                logger.warning(f"Could not import module {module_name}: {e}")
    
    return result

def find_cluster_components() -> Dict[str, Dict[str, Any]]:
    """
    Find all cluster components in the framework.
    
    Returns:
        Dictionary mapping cluster names to component information
    """
    clusters = {}
    
    try:
        # Try to get cluster components from SDKContext
        from MetaConsciousness.core.context import SDKContext
        
        # Common cluster component names
        cluster_names = [
            "core", "memory_cluster", "art_cluster", "agent_cluster",
            "omega3", "compression_cluster", "narrative_cluster", "metacognitive"
        ]
        
        for name in cluster_names:
            component = SDKContext.get(name)
            if component is not None:
                clusters[name] = {
                    "component": component,
                    "type": type(component).__name__,
                    "location": inspect.getmodule(component).__file__ if inspect.getmodule(component) else "unknown",
                    "registered": True
                }
    except ImportError:
        logger.warning("Could not import SDKContext to check for registered clusters")
    
    # Also scan for cluster classes that might not be registered
    try:
        modules = get_metaconsciousness_modules()
        
        for module_name, module_info in modules.items():
            if module_name.endswith("_cluster") or module_name in ["core", "memory", "art", "agent"]:
                module = module_info["module"]
                
                # Find cluster classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and obj.__module__.startswith("MetaConsciousness") and 
                        ("Cluster" in name or "Controller" in name or "Manager" in name)):
                        
                        # Skip if already registered
                        registered = any(cluster["component"].__class__ == obj for cluster in clusters.values() 
                                        if "component" in cluster)
                        
                        if not registered:
                            clusters[f"{module_name}.{name}"] = {
                                "class": obj,
                                "type": name,
                                "location": inspect.getmodule(obj).__file__ if inspect.getmodule(obj) else "unknown",
                                "registered": False
                            }
    except Exception as e:
        logger.error(f"Error scanning for cluster classes: {e}")
    
    return clusters

def detect_orphaned_modules() -> List[Dict[str, Any]]:
    """
    Detect modules that are not imported or used by any other module.
    
    Returns:
        List of orphaned module information
    """
    orphaned_modules = []
    
    try:
        import MetaConsciousness
    except ImportError:
        logger.error("MetaConsciousness package not found")
        return orphaned_modules
    
    metaconsciousness_path = Path(MetaConsciousness.__file__).parent
    
    # Get all Python files in the project
    python_files = []
    for root, _, files in os.walk(metaconsciousness_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                python_files.append(os.path.join(root, file))
    
    # Import patterns to look for in each file
    used_modules = set()
    
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Look for import statements
                import_lines = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("import ") or line.startswith("from "):
                        import_lines.append(line)
                
                # Extract module names
                for line in import_lines:
                    if line.startswith("import "):
                        modules = line[7:].split(",")
                        for module in modules:
                            module = module.strip().split(" as ")[0]
                            if module.startswith("MetaConsciousness."):
                                used_modules.add(module)
                    elif line.startswith("from "):
                        parts = line.split(" import ")
                        if len(parts) == 2 and parts[0].startswith("from "):
                            module = parts[0][5:].strip()
                            if module.startswith("MetaConsciousness."):
                                used_modules.add(module)
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
    
    # Get all actual modules
    all_modules = set()
    for module_name, module_info in get_metaconsciousness_modules().items():
        all_modules.add(f"MetaConsciousness.{module_name}")
        for submodule in module_info.get("submodules", []):
            all_modules.add(f"MetaConsciousness.{module_name}.{submodule['name']}")
    
    # Find orphaned modules
    for module in all_modules:
        if module not in used_modules and not module.endswith("__init__"):
            try:
                mod = importlib.import_module(module)
                orphaned_modules.append({
                    "name": module,
                    "location": mod.__file__ if hasattr(mod, "__file__") else "unknown",
                    "reason": "Not imported by any other module"
                })
            except ImportError as e:
                orphaned_modules.append({
                    "name": module,
                    "location": "unknown",
                    "reason": f"Not importable: {e}"
                })
    
    return orphaned_modules

def analyze_module_dependencies() -> Dict[str, Any]:
    """
    Analyze the dependency graph of modules.
    
    Returns:
        Dictionary with dependency analysis
    """
    result = {
        "modules": {},
        "dependencies": {},
        "import_cycles": []
    }
    
    try:
        import MetaConsciousness
    except ImportError:
        logger.error("MetaConsciousness package not found")
        return result
    
    metaconsciousness_path = Path(MetaConsciousness.__file__).parent
    
    # Get all Python files in the project
    python_files = []
    for root, _, files in os.walk(metaconsciousness_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                rel_path = os.path.relpath(os.path.join(root, file), metaconsciousness_path.parent)
                module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                python_files.append((os.path.join(root, file), module_path))
    
    # Process each file
    for file_path, module_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Track dependencies for this module
                dependencies = []
                
                # Look for import statements
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("import "):
                        modules = line[7:].split(",")
                        for module in modules:
                            module = module.strip().split(" as ")[0]
                            if module.startswith("MetaConsciousness."):
                                dependencies.append(module)
                    elif line.startswith("from "):
                        parts = line.split(" import ")
                        if len(parts) == 2 and parts[0].startswith("from "):
                            module = parts[0][5:].strip()
                            if module.startswith("MetaConsciousness."):
                                dependencies.append(module)
                
                # Store data
                result["modules"][module_path] = {
                    "file": file_path,
                    "imports_count": len(dependencies)
                }
                
                result["dependencies"][module_path] = dependencies
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
    
    # Find import cycles
    try:
        cycles = []
        visited = {}  # 0: not visited, 1: visiting, 2: visited
        rec_stack = {}
        
        def find_cycles(node, path=[]):
            nonlocal cycles, visited, rec_stack
            
            if node not in visited:
                visited[node] = 0
                rec_stack[node] = False
            
            visited[node] = 1  # Mark as visiting
            rec_stack[node] = True
            
            for neighbor in result["dependencies"].get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = 0
                
                if visited[neighbor] == 0:
                    new_path = path + [node]
                    if find_cycles(neighbor, new_path):
                        return True
                elif visited[neighbor] == 1 and rec_stack.get(neighbor, False):
                    # Found a cycle
                    cycle_path = path + [node, neighbor]
                    cycles.append(cycle_path)
                    return True
            
            visited[node] = 2  # Mark as visited
            rec_stack[node] = False
            return False
        
        # Check each node
        for node in result["modules"]:
            if node not in visited:
                find_cycles(node)
        
        result["import_cycles"] = cycles
    except Exception as e:
        logger.error(f"Error finding import cycles: {e}")
    
    return result

def scan_clusters(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Scan the MetaConsciousness framework for clusters and modules.
    
    Args:
        config: Optional configuration dictionary with scan parameters
        
    Returns:
        Dictionary with scan results
    """
    config = config or {}
    ignore_modules = set(config.get("ignore_modules", []))
    only_scan_modules = set(config.get("only_scan_modules", []))
    include_orphaned = config.get("include_orphaned", True)
    include_dependencies = config.get("include_dependencies", True)
    
    result = {
        "timestamp": time.time(),
        "clusters": {},
        "modules": {},
        "orphaned_modules": [],
        "dependencies": {}
    }
    
    # Find clusters
    clusters = find_cluster_components()
    result["clusters"] = {name: {k: v for k, v in info.items() if k != "component" and k != "class"} 
                         for name, info in clusters.items()}
    
    # Get modules
    modules = get_metaconsciousness_modules()
    for module_name, module_info in modules.items():
        if (not only_scan_modules or module_name in only_scan_modules) and module_name not in ignore_modules:
            result["modules"][module_name] = {
                "is_package": module_info["is_package"],
                "path": module_info["path"],
                "submodules_count": len(module_info.get("submodules", [])),
                "submodules": [sm["name"] for sm in module_info.get("submodules", [])]
            }
    
    # Get orphaned modules if requested
    if include_orphaned:
        result["orphaned_modules"] = detect_orphaned_modules()
    
    # Get dependencies if requested
    if include_dependencies:
        dependency_analysis = analyze_module_dependencies()
        result["dependencies"] = dependency_analysis["dependencies"]
        result["import_cycles"] = dependency_analysis["import_cycles"]
    
    return result

def create_cluster_report(output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive report about cluster components.
    
    Args:
        output_file: Optional file to save the report (JSON)
        
    Returns:
        Dictionary with cluster information
    """
    import json
    
    report = scan_clusters()
    
    if output_file:
        try:
            import os
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Cluster report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving cluster report to {output_file}: {e}")
    
    return report

def generate_cluster_diagram(output_file: Optional[str] = None) -> Optional[str]:
    """
    Generate a diagram showing cluster relationships.
    
    Args:
        output_file: Optional file to save the diagram (if graphviz is available)
        
    Returns:
        Path to the saved diagram or None if graphviz not available
    """
    try:
        import graphviz
    except ImportError:
        logger.warning("graphviz not available, cannot generate diagram")
        return None
    
    try:
        # Get clusters and dependencies
        clusters = find_cluster_components()
        dependencies = analyze_module_dependencies()
        
        # Create graph
        dot = graphviz.Digraph("MetaConsciousness Clusters")
        
        # Add cluster nodes
        for name, info in clusters.items():
            node_attrs = {
                "shape": "box",
                "style": "filled",
                "fillcolor": "lightblue" if info.get("registered", False) else "lightgray"
            }
            dot.node(name, name, **node_attrs)
        
        # Add dependencies
        for source, targets in dependencies["dependencies"].items():
            for target in targets:
                # Simplify module names for clarity
                source_simple = source.replace("MetaConsciousness.", "").split(".")[0]
                target_simple = target.replace("MetaConsciousness.", "").split(".")[0]
                
                # Only show edges between clusters
                if (source_simple in clusters or source_simple.endswith("_cluster") and
                    target_simple in clusters or target_simple.endswith("_cluster")):
                    dot.edge(source_simple, target_simple)
        
        # Render the diagram
        if output_file:
            output_path = dot.render(output_file, format="png", cleanup=True)
            logger.info(f"Cluster diagram saved to {output_path}")
            return output_path
        else:
            return dot.pipe().decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating cluster diagram: {e}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # When run directly, generate a report and diagram
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Scanning MetaConsciousness clusters...")
    report = create_cluster_report(os.path.join(output_dir, "cluster_report.json"))
    print(f"Found {len(report['clusters'])} clusters and {len(report['modules'])} modules")
    
    if report.get("orphaned_modules"):
        print(f"Found {len(report['orphaned_modules'])} potentially orphaned modules")
    
    if report.get("import_cycles"):
        print(f"Warning: Found {len(report['import_cycles'])} import cycles")
    
    print("Generating cluster diagram...")
    diagram_path = generate_cluster_diagram(os.path.join(output_dir, "cluster_diagram"))
    if diagram_path:
        print(f"Diagram saved to: {diagram_path}")
