"""
Path Helper Module for VoxSigil Project

This module helps manage import paths for the VoxSigil project components.
It ensures that components in different directories can properly import
each other without circular dependencies or path issues.
"""

import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger("path_helper")


def add_project_root_to_path():
    """
    Add the project root directory to the Python path.
    This allows imports from any submodule to access any other submodule
    by using the full package path (e.g., 'from ART.art_controller import ARTController').
    """
    # Get the absolute path to the Sigil project root directory
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Add to sys.path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        return True
    return False


def setup_voxsigil_imports():
    """
    Setup all necessary import paths for VoxSigil components.
    Call this at the top of modules that need to import across the project.
    """
    added = add_project_root_to_path()
    return added


def verify_module_paths(module_names):
    """
    Verify that the specified modules can be imported.

    Args:
        module_names (list): List of module names to verify

    Returns:
        dict: Dictionary with module names as keys and boolean availability status as values
    """
    results = {}
    for module_name in module_names:
        try:
            __import__(module_name)
            results[module_name] = True
        except ImportError:
            results[module_name] = False

    return results


def setup_voxsigil_component_paths():
    """
    Ensures that all VoxSigil component paths are properly set up.
    This is particularly important for integration between components like
    VantaCore and VantaSigilSupervisor.

    Returns:
        dict: Dictionary with component paths and their availability status
    """
    # Add project root to path first
    add_project_root_to_path()

    # Get the project root directory
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Define important component paths
    component_paths = {
        "Vanta": project_root / "Vanta",
        "Vanta.core": project_root / "Vanta" / "core",
        "Vanta.integration": project_root / "Vanta" / "integration",
        "Vanta.interfaces": project_root / "Vanta" / "interfaces",
        "ART": project_root / "ART",
        "BLT": project_root / "BLT",
        "Voxsigil-Library": project_root / "Voxsigil-Library",
        "tools": project_root / "tools",
        "utils": project_root / "utils",
    }

    # Check if paths exist
    component_status = {}
    for name, path in component_paths.items():
        component_status[name] = path.exists()

    return component_status


def verify_sigil_mode_dependencies():
    """
    Verify that all dependencies for sigil_mode are available.

    Returns:
        dict: Status of sigil mode dependencies
    """
    dependencies = {}

    # Check if VantaSigilSupervisor can be imported
    try:
        from Vanta.integration.vanta_supervisor import VantaSigilSupervisor

        dependencies["VantaSigilSupervisor"] = True
    except ImportError as e:
        dependencies["VantaSigilSupervisor"] = False
        dependencies["VantaSigilSupervisor_error"] = str(e)

    # Check if RealSupervisorConnector can be imported
    try:
        from Vanta.interfaces.real_supervisor_connector import RealSupervisorConnector

        dependencies["RealSupervisorConnector"] = True
    except ImportError as e:
        dependencies["RealSupervisorConnector"] = False
        dependencies["RealSupervisorConnector_error"] = str(e)

    # Check optional dependencies
    optional_deps = [
        "ART.art_manager",
        "Vanta.core.sleep_time_compute",
        "BLT.blt_encoder",
        "BLT.hybrid_middleware",
    ]

    for dep in optional_deps:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError as e:
            dependencies[dep] = False
            dependencies[f"{dep}_error"] = str(e)

    return dependencies


def create_sigil_supervisor_instance(
    rag_interface=None, llm_interface=None, memory_interface=None
):
    """
    Create a VantaSigilSupervisor instance with fallback interfaces if needed.

    Args:
        rag_interface: RAG interface instance (optional)
        llm_interface: LLM interface instance (optional)
        memory_interface: Memory interface instance (optional)

    Returns:
        VantaSigilSupervisor instance or None if dependencies unavailable
    """
    dependencies = verify_sigil_mode_dependencies()

    if not dependencies.get("VantaSigilSupervisor", False):
        return None

    try:
        from Vanta.integration.vanta_supervisor import VantaSigilSupervisor

        # Create fallback interfaces if not provided
        if rag_interface is None:
            # Create a simple fallback RAG interface
            class FallbackRagInterface:
                def retrieve_context(self, query, context=None):
                    return f"No RAG available for query: {query}"

            rag_interface = FallbackRagInterface()

        if llm_interface is None:
            # Create a simple fallback LLM interface
            class FallbackLlmInterface:
                def generate_response(
                    self, query, context=None, system_prompt=None, **kwargs
                ):
                    return f"No LLM available for query: {query}", {}, {}

            llm_interface = FallbackLlmInterface()

        # Create supervisor instance
        supervisor = VantaSigilSupervisor(
            rag_interface=rag_interface,
            llm_interface=llm_interface,
            memory_interface=memory_interface,
            enable_adaptive=True,
            enable_echo_harmonization=True,
        )

        return supervisor

    except Exception as e:
        logger.error(f"Failed to create VantaSigilSupervisor: {e}")
        return None


# If this module is run directly, it will print the path information
if __name__ == "__main__":
    added = setup_voxsigil_imports()
    print(f"Project root added to sys.path: {added}")
    print(f"Current sys.path: {sys.path}")

    # Test sigil mode dependencies
    print("\nSigil mode dependencies:")
    deps = verify_sigil_mode_dependencies()
    for dep, status in deps.items():
        print(f"  {dep}: {status}")

    # Test component paths
    print("\nComponent paths:")
    paths = setup_voxsigil_component_paths()
    for path, exists in paths.items():
        print(f"  {path}: {'✓' if exists else '✗'}")
