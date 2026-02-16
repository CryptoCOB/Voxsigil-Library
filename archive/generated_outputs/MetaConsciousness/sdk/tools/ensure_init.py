#!/usr/bin/env python
"""
Ensure __init__.py Files

This script scans Python directories in the MetaConsciousness codebase
and creates missing __init__.py files to ensure proper package structure.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import centralized utility functions
try:
    from MetaConsciousness.utils.utils import confirm_action, get_relative_path
    centralized_utils_available = True
except ImportError:
    centralized_utils_available = False
    logger = logging.getLogger("ensure_init_fixer")
    logger.warning("Could not import centralized utility functions, using local implementations.")

# --- Logger Setup ---
def _setup_logger(level=logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configures the logging for the script."""
    logger = logging.getLogger("ensure_init_fixer")
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Logging detailed output to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up log file at {log_file}: {e}")

    return logger

logger = _setup_logger()

# --- Utility Functions ---
def _find_project_root(start_path: Path, marker: str = "MetaConsciousness") -> Path:
    """Finds the project root directory containing the marker directory."""
    current = start_path.resolve()
    while True:
        if (current / marker).is_dir():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    logger.warning(f"Could not find project root containing '{marker}'. Using start path parent.")
    return start_path.parent.parent  # Go up two levels from tools directory

def _write_init_file(filepath: Path, content: str = "") -> bool:
    """Writes content to an __init__.py file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        return False

def _confirm_action(prompt: str) -> bool:
    """Asks the user for confirmation (y/n) using centralized utility if available."""
    if centralized_utils_available:
        return confirm_action(prompt)
    else:
        # Fallback to local implementation if centralized version not available
        while True:
            try:
                response = input(f"{prompt} [y/N]: ").lower().strip()
                if response == 'y': return True
                elif response == 'n' or response == '': return False
                else: print("Invalid input. Please enter 'y' or 'n'.")
            except EOFError:
                logger.warning("EOFError reading confirmation, defaulting to 'No'.")
                return False

def _get_relative_path(filepath: Path, root_dir: Path) -> str:
    """Gets the path relative to the project root using centralized utility if available."""
    if centralized_utils_available:
        return get_relative_path(filepath, root_dir)
    else:
        # Fallback to local implementation if centralized version not available
        try:
            return str(filepath.relative_to(root_dir))
        except ValueError:
            return str(filepath)

# --- Ensure __init__.py Functions ---
def ensure_init_files(metaconsciousness_dir: Path, dry_run: bool, interactive: bool) -> List[str]:
    """Creates missing __init__.py files in Python directories."""
    created_files = []
    checked_dirs = set()

    # Find all Python files first to identify directories that need __init__.py
    python_files = list(metaconsciousness_dir.rglob("*.py"))

    # Process each directory containing Python files
    for py_file in python_files:
        parent_dir = py_file.parent

        # Traverse up from each Python file to ensure all parent directories have __init__.py
        current_dir = parent_dir
        while current_dir.is_relative_to(metaconsciousness_dir):
            if current_dir in checked_dirs:
                break

            checked_dirs.add(current_dir)
            init_file = current_dir / "__init__.py"

            if not init_file.exists():
                logger.info(f"Missing __init__.py in: {_get_relative_path(current_dir, metaconsciousness_dir.parent)}")

                if dry_run:
                    logger.info(f"[Dry Run] Would create: {init_file.name}")
                    created_files.append(str(init_file))
                    current_dir = current_dir.parent
                    continue

                if interactive and not _confirm_action(f"Create {init_file.name} in {current_dir.name}?"):
                    logger.info(f"Skipping creation in {current_dir.name} due to user decision")
                    current_dir = current_dir.parent
                    continue

                # Create the file with a standard header comment
                content = f"""# Auto-generated by ensure_init.py
\"\"\"
{current_dir.name} package.
\"\"\"
"""
                if _write_init_file(init_file, content):
                    logger.info(f"✓ Created: {init_file}")
                    created_files.append(str(init_file))
                else:
                    logger.error(f"Failed to create {init_file}")

            # Move up to parent directory
            current_dir = current_dir.parent

    return created_files

# --- Main Function ---
def main() -> int:
    """Main function to parse arguments and run the fixer."""
    parser = argparse.ArgumentParser(description="Ensure __init__.py files exist in Python packages.")
    parser.add_argument("--root", default=None, help="Project root directory.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show changes without creating files.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Confirm each file creation.")
    parser.add_argument("--log-file", default="ensure_init.log", help="Log file path.")
    parser.add_argument("--verbose", "-v", action="store_const", dest="log_level", const=logging.DEBUG, default=logging.INFO, help="Enable verbose output.")

    args = parser.parse_args()

    # Re-initialize logger with args
    global logger
    logger = _setup_logger(level=args.log_level, log_file=args.log_file)

    root_dir = Path(args.root).resolve() if args.root else _find_project_root(Path(__file__).parent.parent)
    metaconsciousness_dir = root_dir / "MetaConsciousness"

    if not metaconsciousness_dir.is_dir():
        logger.critical(f"❌ Error: MetaConsciousness directory not found at {metaconsciousness_dir}")
        return 1

    logger.info(f"Starting __init__.py file scan for {metaconsciousness_dir}")

    # Run ensure init files function
    created_files = ensure_init_files(
        metaconsciousness_dir,
        args.dry_run,
        args.interactive
    )

    # Summary
    logger.info(f"Total __init__.py files created: {len(created_files)}")

    return 0 if created_files else 1  # Return 0 if files created, 1 if no files created

if __name__ == "__main__":
    sys.exit(main())
