#!/usr/bin/env python
"""
Fix Remaining Issues

This script handles miscellaneous cleanup tasks that are not covered by other fixers
such as file encoding, whitespace normalization, and other code quality issues.
"""
import os
import sys
import re
import logging
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Set, Tuple

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Logger Setup ---
def _setup_logger(level=logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configures the logging for the script."""
    logger = logging.getLogger("remaining_issues_fixer")
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
    return start_path.parent

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

# --- Fix Functions ---
def fix_whitespace_issues(metaconsciousness_dir: Path, dry_run: bool, interactive: bool) -> List[str]:
    """
    Fix common whitespace issues like trailing whitespace and inconsistent line endings.
    """
    fixed_files = []

    # Scan Python files
    for filepath in metaconsciousness_dir.rglob("*.py"):
        if filepath.name.endswith("_test.py") or "__pycache__" in str(filepath):
            continue

        content = _read_file_content(filepath)
        if content is None:
            continue

        original_content = content
        modified = False

        # Fix trailing whitespace
        modified_content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

        # Normalize line endings to Unix style (\n)
        if '\r\n' in modified_content:
            modified_content = modified_content.replace('\r\n', '\n')
            modified = True

        # Ensure file ends with a single newline
        if not modified_content.endswith('\n'):
            modified_content += '\n'
            modified = True
        elif modified_content.endswith('\n\n'):
            modified_content = modified_content.rstrip('\n') + '\n'
            modified = True

        if modified_content != original_content:
            logger.info(f"Found whitespace issues in {filepath.name}")

            if dry_run:
                logger.info(f"[Dry Run] Would fix whitespace in {filepath.name}")
                fixed_files.append(str(filepath))
                continue

            if interactive and not _confirm_action(f"Fix whitespace issues in {filepath.name}?"):
                logger.info(f"Skipping {filepath.name} due to user decision")
                continue

            if _write_file_content(filepath, modified_content):
                logger.info(f"✓ Fixed whitespace in {filepath.name}")
                fixed_files.append(str(filepath))
            else:
                logger.error(f"Failed to write changes to {filepath.name}")

    return fixed_files

def fix_file_encoding_issues(metaconsciousness_dir: Path, dry_run: bool, interactive: bool) -> List[str]:
    """
    Ensures files are properly UTF-8 encoded.
    """
    # Implementation would be similar to fix_whitespace_issues but checking for encoding issues
    # For now, return empty list as this is more complex to implement safely
    logger.info("File encoding check not implemented yet.")
    return []

# --- Main Function ---
def main() -> int:
    """Main function to parse arguments and run the fixer."""
    parser = argparse.ArgumentParser(description="Fix remaining miscellaneous issues.")
    parser.add_argument("--root", default=None, help="Project root directory.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show changes without modifying files.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Confirm each modification.")
    parser.add_argument("--log-file", default="remaining_issues.log", help="Log file path.")
    parser.add_argument("--verbose", "-v", action="store_const", dest="log_level", const=logging.DEBUG, default=logging.INFO, help="Enable verbose output.")

    args = parser.parse_args()

    # Re-initialize logger with args
    global logger
    logger = _setup_logger(level=args.log_level, log_file=args.log_file)

    # Correct the project root path resolution
    root_dir = Path(args.root).resolve() if args.root else _find_project_root(Path(__file__).parent.parent)
    metaconsciousness_dir = root_dir / "MetaConsciousness"

    if not metaconsciousness_dir.is_dir():
        logger.critical(f"❌ Error: MetaConsciousness directory not found at {metaconsciousness_dir}")
        return 1

    logger.info(f"Starting fix of remaining issues for {metaconsciousness_dir}")

    # Run fix functions
    all_fixed_files = []

    # Fix whitespace issues
    fixed_files = fix_whitespace_issues(
        metaconsciousness_dir,
        args.dry_run,
        args.interactive
    )
    all_fixed_files.extend(fixed_files)

    # Fix file encoding issues
    fixed_files = fix_file_encoding_issues(
        metaconsciousness_dir,
        args.dry_run,
        args.interactive
    )
    all_fixed_files.extend(fixed_files)

    # Summary
    logger.info(f"Total files fixed: {len(all_fixed_files)}")

    return 0 if all_fixed_files else 1  # Return 0 if files fixed, 1 if no files fixed

if __name__ == "__main__":
    sys.exit(main())
