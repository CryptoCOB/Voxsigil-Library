"""
Path setup utility

Provides functions to ensure Python path includes project root.
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

def add_project_root_to_path() -> str:
    """
    Add the project root directory to the Python path.
    
    Returns:
        Path to the project root directory that was added
    """
    # Start at the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up the directory tree until we find the project root
    # (which contains the MetaConsciousness package)
    project_root = current_dir
    while True:
        # Check if we've reached the project root
        if os.path.isdir(os.path.join(project_root, "MetaConsciousness")):
            break
            
        # Go up one directory
        parent = os.path.dirname(project_root)
        if parent == project_root:  # We've reached the filesystem root
            project_root = os.path.dirname(current_dir)  # Default to parent of current dir
            break
        
        project_root = parent
    
    # Add to sys.path if not already present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to sys.path: {project_root}")
    
    return project_root

# For backward compatibility
setup_project_path = add_project_root_to_path
