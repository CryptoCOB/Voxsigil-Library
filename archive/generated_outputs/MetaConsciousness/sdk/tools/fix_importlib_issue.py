"""
Fix Importlib Issue

This script addresses issues with importlib.util by ensuring the correct importlib
version is used or providing backward-compatible alternatives.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_importlib")

def check_importlib_compatibility() -> None:
    """Check if importlib.util is available in the current Python environment."""
    try:
        import importlib.util
        logger.info(f"importlib.util is available (Python {sys.version})")
        return True
    except (ImportError, AttributeError):
        logger.warning(f"importlib.util is NOT available (Python {sys.version})")
        return False

def fix_script_imports(script_path) -> None:
    """
    Fix imports in a script to use backward-compatible importlib methods.
    
    Args:
        script_path: Path to the script to fix
        
    Returns:
        True if fixed, False otherwise
    """
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if the script uses importlib.util
        if "importlib.util" not in content:
            logger.info(f"No importlib.util references found in {script_path}")
            return False
        
        # Make a backup
        backup_path = f"{script_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup: {backup_path}")
        
        # Replace importlib.util.spec_from_file_location with alternative
        new_content = content.replace(
            "import importlib\n",
            "import importlib\nimport imp  # For backward compatibility\n"
        )
        
        new_content = new_content.replace(
            "spec = importlib.util.spec_from_file_location",
            "# Backward compatible module loading\ntry:\n    import importlib.util\n    spec = importlib.util.spec_from_file_location"
        )
        
        new_content = new_content.replace(
            "module = importlib.util.module_from_spec(spec)",
            "    module = importlib.util.module_from_spec(spec)"
        )
        
        new_content = new_content.replace(
            "spec.loader.exec_module(module)",
            "    spec.loader.exec_module(module)\nexcept (ImportError, AttributeError):\n    # Fallback for older Python versions\n    module = imp.load_source(script_name, script_file_path)"
        )
        
        # Write the fixed content
        with open(script_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Fixed importlib usage in {script_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing {script_path}: {e}")
        return False

def main() -> int:
    """Main entry point.
    
    Returns:
        0 on success, 1 on errors
    """
    logger.info("Starting importlib compatibility check...")
    
    # Check importlib compatibility
    has_importlib_util = check_importlib_compatibility()
    
    if has_importlib_util:
        logger.info("This Python environment has importlib.util - no fixes needed.")
        return 0
    
    # Fix the main fixer script
    master_fixer_path = os.path.join(os.path.dirname(__file__), "fix_all_issues.py")
    if os.path.exists(master_fixer_path):
        success = fix_script_imports(master_fixer_path)
        if success:
            logger.info("Successfully fixed master fixer script")
        else:
            logger.warning("Could not fix master fixer script")
    else:
        logger.error(f"Master fixer script not found at {master_fixer_path}")
    
    # Look for other scripts that might use importlib.util
    tools_dir = os.path.dirname(__file__)
    logger.info(f"Scanning scripts in {tools_dir}...")
    
    fixed_count = 0
    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename != "fix_importlib_issue.py":
            script_path = os.path.join(tools_dir, filename)
            if fix_script_imports(script_path):
                fixed_count += 1
    
    logger.info(f"Fixed importlib usage in {fixed_count} scripts")
    return 0

if __name__ == "__main__":
    sys.exit(main())
