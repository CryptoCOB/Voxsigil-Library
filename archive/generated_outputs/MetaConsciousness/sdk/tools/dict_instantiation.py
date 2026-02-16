#!/usr/bin/env python
"""
Dictionary Instantiation Fixer

This script scans Python files in the MetaConsciousness codebase
and fixes improper dictionary/list/tuple instantiations by replacing
type constructor calls with literal syntax where appropriate.
"""
import os
import sys
import re
import logging
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Logger Setup ---
def _setup_logger(level=logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configures the logging for the script."""
    logger = logging.getLogger("dict_instantiation_fixer")
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

def _read_file_content(filepath: Path) -> Optional[str]:
    """Reads file content with UTF-8 encoding and error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

def _write_file_content(filepath: Path, content: str) -> bool:
    """Writes content to a file with UTF-8 encoding and error handling."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        return False

def _create_backup(filepath: Path) -> bool:
    """Creates a backup copy of the file with a .bak extension."""
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    try:
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error creating backup for {filepath}: {e}")
        return False

def _confirm_action(prompt: str) -> bool:
    """Asks the user for confirmation (y/n)."""
    while True:
        try:
            response = input(f"{prompt} [y/N]: ").lower().strip()
            if response == 'y': return True
            elif response == 'n' or response == '': return False
            else: print("Invalid input. Please enter 'y' or 'n'.")
        except EOFError:
            logger.warning("EOFError reading confirmation, defaulting to 'No'.")
            return False

# --- Fix Dictionary Instantiation Functions ---
def fix_dict_instantiations(metaconsciousness_dir: Path, dry_run: bool, interactive: bool, backup: bool) -> List[str]:
    """Fixes improper dictionary/list/tuple instantiations."""
    fixed_files = []

    # Define patterns to search for and their replacements
    patterns = [
        (r'Dict\(\)', '{}'),
        (r'List\(\)', '[]'),
        (r'Tuple\(\)', '()'),
        (r'Set\(\)', 'set()'),
        # More complex pattern for Dict with arguments
        (r'Dict\(([^)]*)\)', r'{\1}'),
        # More complex pattern for List with arguments
        (r'List\(([^)]*)\)', r'[\1]'),
        # More complex pattern for Tuple with arguments
        (r'Tuple\(([^)]*)\)', r'(\1)')
    ]

    # Scan all Python files
    for filepath in metaconsciousness_dir.rglob("*.py"):
        if filepath.name.endswith("_test.py") or "__pycache__" in str(filepath):
            continue

        content = _read_file_content(filepath)
        if content is None:
            continue

        original_content = content
        modified = False
        fix_count = 0

        # Apply all patterns
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                # For complex patterns, simple replacement might not be sufficient
                # For demonstration, we'll use a simple approach
                content = re.sub(pattern, replacement, content)
                if content != original_content:
                    modified = True
                    fix_count += 1

        if modified:
            logger.info(f"Found {fix_count} dict/list/tuple instantiation issue(s) in {filepath.name}")

            if dry_run:
                logger.info(f"[Dry Run] Would fix {filepath.name}")
                fixed_files.append(str(filepath))
                continue

            if interactive and not _confirm_action(f"Fix {fix_count} issue(s) in {filepath.name}?"):
                logger.info(f"Skipping {filepath.name} due to user decision")
                continue

            if backup:
                _create_backup(filepath)

            if _write_file_content(filepath, content):
                logger.info(f"✓ Fixed {filepath.name}")
                fixed_files.append(str(filepath))
            else:
                logger.error(f"Failed to write changes to {filepath.name}")

    return fixed_files

# --- Main Function ---
def main() -> int:
    """Main function to parse arguments and run the fixer."""
    parser = argparse.ArgumentParser(description="Fix dictionary/list/tuple instantiation issues.")
    parser.add_argument("--root", default=None, help="Project root directory.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show changes without modifying files.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Confirm each modification.")
    parser.add_argument("--no-backup", action="store_false", dest="backup", default=True, help="Disable creation of .bak files.")
    parser.add_argument("--log-file", default="dict_instantiation.log", help="Log file path.")
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

    logger.info(f"Starting dictionary instantiation scan for {metaconsciousness_dir}")

    # Run fix functions
    fixed_files = fix_dict_instantiations(
        metaconsciousness_dir,
        args.dry_run,
        args.interactive,
        args.backup
    )

    # Summary
    logger.info(f"Total files fixed: {len(fixed_files)}")

    return 0 if fixed_files else 1  # Return 0 if files fixed, 1 if no files fixed

if __name__ == "__main__":
    sys.exit(main())
