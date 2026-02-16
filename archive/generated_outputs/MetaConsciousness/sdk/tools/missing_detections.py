"""
Missing Detections Fixer

This script detects and fixes missing imports, modules, and classes in the codebase.
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
logger = logging.getLogger("missing_detections")

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

def scan_project_files(root_dir) -> None:
    """
    Scan project files for potential issues.
    
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
        "missing_init_files": [],
        "missing_imports": [],
        "circular_imports": [],
        "files_scanned": 0
    }
    
    # Pattern to detect common import issues
    import_pattern = re.compile(r'^\s*(?:from|import)\s+([.\w]+)', re.MULTILINE)
    circular_pattern = re.compile(r'^\s*(?:from|import)\s+([.\w]+).*# Circular', re.MULTILINE)
    
    # Walk through the project
    for dirpath, dirnames, filenames in os.walk(metaconsciousness_dir):
        # Check for missing __init__.py
        if dirpath != metaconsciousness_dir and "__init__.py" not in filenames:
            results["missing_init_files"].append(dirpath)
        
        # Process Python files
        for filename in [f for f in filenames if f.endswith('.py')]:
            results["files_scanned"] += 1
            filepath = os.path.join(dirpath, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports
                imports = import_pattern.findall(content)
                
                # Find circular imports
                circular_imports = circular_pattern.findall(content)
                if circular_imports:
                    results["circular_imports"].append({
                        "file": filepath,
                        "imports": circular_imports
                    })
                
                # Check for potentially missing imports
                for imp in imports:
                    if imp.startswith('MetaConsciousness.'):
                        module_path = imp.replace('MetaConsciousness.', '')
                        module_path = module_path.split('.')[0]  # Get first component
                        
                        # Check if the imported module exists
                        module_dir = os.path.join(metaconsciousness_dir, module_path)
                        if not os.path.exists(module_dir) and not os.path.exists(f"{module_dir}.py"):
                            results["missing_imports"].append({
                                "file": filepath,
                                "import": imp,
                                "module": module_path
                            })
            
            except Exception as e:
                logger.warning(f"Error processing file {filepath}: {e}")
    
    return results

def fix_missing_init_files(missing_dirs) -> None:
    """
    Create missing __init__.py files.
    
    Args:
        missing_dirs: List of directories missing __init__.py
        
    Returns:
        Count of fixed directories
    """
    fixed_count = 0
    
    for directory in missing_dirs:
        init_path = os.path.join(directory, "__init__.py")
        
        try:
            # Create empty __init__.py
            with open(init_path, 'w') as f:
                f.write('"""Auto-generated __init__.py file."""\n')
            
            logger.info(f"Created __init__.py in {directory}")
            fixed_count += 1
            
        except Exception as e:
            logger.error(f"Error creating __init__.py in {directory}: {e}")
    
    return fixed_count

def main() -> int:
    """Main entry point."""
    logger.info("Starting missing detections scan...")
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        logger.error("Could not find project root, aborting.")
        return 1
    
    # Scan project
    scan_results = scan_project_files(project_root)
    
    # Log scan results
    logger.info(f"Scanned {scan_results.get('files_scanned', 0)} files")
    logger.info(f"Found {len(scan_results.get('missing_init_files', []))} directories missing __init__.py")
    logger.info(f"Found {len(scan_results.get('missing_imports', []))} potentially missing imports")
    logger.info(f"Found {len(scan_results.get('circular_imports', []))} circular imports")
    
    # Fix missing __init__.py files
    if scan_results.get('missing_init_files'):
        fixed_count = fix_missing_init_files(scan_results['missing_init_files'])
        logger.info(f"Fixed {fixed_count} directories with missing __init__.py files")
    
    # Report missing imports (no automatic fix)
    if scan_results.get('missing_imports'):
        logger.info("\nPotentially missing imports:")
        for imp in scan_results['missing_imports']:
            logger.info(f"  - {imp['import']} in {imp['file']}")
    
    # Report circular imports (no automatic fix)
    if scan_results.get('circular_imports'):
        logger.info("\nCircular imports:")
        for circ in scan_results['circular_imports']:
            logger.info(f"  - {circ['file']}: {', '.join(circ['imports'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
