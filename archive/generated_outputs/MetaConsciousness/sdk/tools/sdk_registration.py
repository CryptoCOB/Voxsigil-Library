"""
SDK Registration Fixer

This script detects and fixes missing or incorrect SDKContext registrations.
"""

import os
import sys
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sdk_registration")

def find_project_root() -> None:
    """Find the project root directory (containing MetaConsciousness)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start from the script directory and go up until we find project root
    current_dir = script_dir
    while True:
        if os.path.isdir(os.path.join(current_dir, "MetaConsciousness")):
            return current_dir
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached filesystem root
            logger.error("Could not find project root directory")
            return None
        
        current_dir = parent_dir

def scan_sdk_registrations(root_dir) -> None:
    """
    Scan for SDK registrations and potential issues.
    
    Args:
        root_dir: Project root directory
        
    Returns:
        Dictionary with scan results
    """
    if not root_dir:
        return {"error": "Project root not found"}
    
    metaconsciousness_dir = os.path.join(root_dir, "MetaConsciousness")
    if not os.path.isdir(metaconsciousness_dir):
        return {"error": f"MetaConsciousness directory not found at {metaconsciousness_dir}"}
    
    results = {
        "registrations": [],
        "potential_missing_registrations": [],
        "duplicate_registrations": [],
        "files_scanned": 0
    }
    
    # Patterns to detect
    registration_pattern = re.compile(r'SDKContext\.register\(\s*[\'"]([^\'"]+)[\'"]\s*,\s*(\w+)\s*\)', re.MULTILINE)
    class_pattern = re.compile(r'class\s+(\w+)', re.MULTILINE)
    
    # Keep track of registrations
    registrations = {}
    
    # Walk through the project
    for dirpath, dirnames, filenames in os.walk(metaconsciousness_dir):
        # Process Python files
        for filename in [f for f in filenames if f.endswith('.py')]:
            results["files_scanned"] += 1
            filepath = os.path.join(dirpath, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find class definitions
                classes = class_pattern.findall(content)
                
                # Find registrations
                file_registrations = registration_pattern.findall(content)
                if file_registrations:
                    for reg_name, reg_var in file_registrations:
                        reg_info = {
                            "name": reg_name,
                            "variable": reg_var,
                            "file": filepath
                        }
                        
                        results["registrations"].append(reg_info)
                        
                        # Check for duplicates
                        if reg_name in registrations:
                            if registrations[reg_name]["variable"] != reg_var or registrations[reg_name]["file"] != filepath:
                                results["duplicate_registrations"].append({
                                    "name": reg_name,
                                    "registrations": [
                                        registrations[reg_name],
                                        reg_info
                                    ]
                                })
                        else:
                            registrations[reg_name] = reg_info
                
                # Look for potential missing registrations
                if "SDKContext" in content and classes:
                    has_registrations = "SDKContext.register" in content
                    # If the file contains SDKContext import and class definitions
                    # but no registrations, it might be missing registrations
                    if not has_registrations:
                        results["potential_missing_registrations"].append({
                            "file": filepath,
                            "classes": classes
                        })
            
            except Exception as e:
                logger.warning(f"Error processing file {filepath}: {e}")
    
    return results

def main() -> None:
    """Main entry point."""
    logger.info("Starting SDK registration scan...")
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        logger.error("Could not find project root, aborting.")
        return 1
    
    # Scan project
    scan_results = scan_sdk_registrations(project_root)
    
    # Log scan results
    logger.info(f"Scanned {scan_results.get('files_scanned', 0)} files")
    logger.info(f"Found {len(scan_results.get('registrations', []))} SDK registrations")
    logger.info(f"Found {len(scan_results.get('duplicate_registrations', []))} duplicate registrations")
    logger.info(f"Found {len(scan_results.get('potential_missing_registrations', []))} potential missing registrations")
    
    # Report duplicate registrations
    if scan_results.get('duplicate_registrations'):
        logger.info("\nDuplicate registrations:")
        for dup in scan_results['duplicate_registrations']:
            logger.info(f"  - '{dup['name']}' registered in multiple places:")
            for reg in dup['registrations']:
                logger.info(f"      * {reg['file']} as {reg['variable']}")
    
    # Report potential missing registrations
    if scan_results.get('potential_missing_registrations'):
        logger.info("\nPotential missing registrations:")
        for missing in scan_results['potential_missing_registrations']:
            logger.info(f"  - {missing['file']} contains classes {', '.join(missing['classes'])} but no SDKContext registrations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
