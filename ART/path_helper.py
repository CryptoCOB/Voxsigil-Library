"""
Path Helper Module for ART Project Components

This module helps manage import paths for the ART project components.
It ensures that components in different directories can properly import
each other without circular dependencies or path issues.
"""

import os
import sys
from pathlib import Path


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


def setup_art_imports():
    """
    Setup all necessary import paths for ART components.
    Call this at the top of modules that need to import across the project.
    """
    added = add_project_root_to_path()
    return added


# If this module is run directly, it will print the path information
if __name__ == "__main__":
    added = setup_art_imports()
    print(f"Project root added to sys.path: {added}")
    print(f"Current sys.path: {sys.path}")
