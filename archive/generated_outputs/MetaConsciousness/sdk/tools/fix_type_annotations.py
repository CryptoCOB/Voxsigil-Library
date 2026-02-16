#!/usr/bin/env python
"""
Fix Type Annotation Issues

This script fixes type annotation issues in the codebase, particularly:
- Dict instantiation instead of dict literals
- Correct imports from type_definitions instead of types
- Proper type annotations in function signatures
"""
import os
import sys
import glob
import re
from typing import List, Dict, Any, Set, Optional

def main() -> None:
    """Run type annotation fixer."""
    print("🔍 Scanning for type annotation issues...")

    # Get project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    metaconsciousness_dir = os.path.join(root_dir, "MetaConsciousness")

    # Ensure we're using the right directories
    if not os.path.exists(metaconsciousness_dir):
        print(f"❌ Error: MetaConsciousness directory not found at {metaconsciousness_dir}")
        return 1

    # Track all fixed files
    fixed_files = []

    # Fix Dict instantiation issues ({} -> {})
    dict_fixes = fix_dict_instantiation(metaconsciousness_dir)
    fixed_files.extend(dict_fixes)

    # Fix type import references (types -> type_definitions)
    type_imports = fix_type_imports(metaconsciousness_dir)
    fixed_files.extend(type_imports)

    # Add missing type annotations
    annotation_fixes = add_missing_annotations(metaconsciousness_dir)
    fixed_files.extend(annotation_fixes)

    # Print summary
    unique_fixed = set(fixed_files)
    print(f"\n✅ Fixed {len(unique_fixed)} files with type annotation issues:")
    for file in sorted(unique_fixed):
        rel_path = os.path.relpath(file, root_dir)
        print(f"  - {rel_path}")

    return 0

def fix_dict_instantiation(root_dir: str) -> List[str]:
    """
    Fix Dict instantiation issues ({} -> {}).

    Args:
        root_dir: Root directory to scan

    Returns:
        List of fixed file paths
    """
    fixed_files = []

    # Find all Python files
    for filepath in glob.glob(f"{root_dir}/**/*.py", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for direct instantiation of type annotations
        modified_content = content

        # Fix {}, [], etc.
        modified_content = re.sub(
            r'Dict\(\s*\)',  # {}
            r'{}',           # {}
            modified_content
        )

        modified_content = re.sub(
            r'List\(\s*\)',  # []
            r'[]',           # []
            modified_content
        )

        modified_content = re.sub(
            r'Tuple\(\s*\)',  # ()
            r'()',            # ()
            modified_content
        )

        modified_content = re.sub(
            r'Set\(\s*\)',    # set()
            r'set()',         # set()
            modified_content
        )

        # Fix annotations used as variables
        modified_content = re.sub(
            r'(\w+)\s*=\s*Dict',
            r'\1 = dict',
            modified_content
        )

        modified_content = re.sub(
            r'(\w+)\s*=\s*List',
            r'\1 = list',
            modified_content
        )

        # Fix instantiations with arguments
        modified_content = re.sub(
            r'Dict\(([^)]+)\)',
            r'{\1}',
            modified_content
        )

        # Write changes back if modified
        if modified_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            fixed_files.append(filepath)
            print(f"✓ Fixed type instantiation in {os.path.basename(filepath)}")

    return fixed_files

def fix_type_imports(root_dir: str) -> List[str]:
    """
    Fix imports from types to type_definitions.

    Args:
        root_dir: Root directory to scan

    Returns:
        List of fixed file paths
    """
    fixed_files = []

    # Find all Python files
    for filepath in glob.glob(f"{root_dir}/**/*.py", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # Fix imports from types to type_definitions
        if re.search(r'from [.\w]*core\.types import', content):
            content = re.sub(
                r'from ([.\w]*)core\.types import',
                r'from \1core.type_definitions import',
                content
            )
            modified = True

        # Also fix direct imports
        if re.search(r'import [.\w]*core\.types', content):
            content = re.sub(
                r'import ([.\w]*)core\.types',
                r'import \1core.type_definitions',
                content
            )
            modified = True

        # Fix references to types
        if re.search(r'[.\w]*core\.types\.', content):
            content = re.sub(
                r'([.\w]*)core\.types\.',
                r'\1core.type_definitions.',
                content
            )
            modified = True

        # Write changes back if modified
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            fixed_files.append(filepath)
            print(f"✓ Fixed type imports in {os.path.basename(filepath)}")

    return fixed_files

def add_missing_annotations(root_dir: str) -> List[str]:
    """
    Add missing type annotations.

    Args:
        root_dir: Root directory to scan

    Returns:
        List of fixed file paths
    """
    fixed_files = []

    # Find all Python files
    for filepath in glob.glob(f"{root_dir}/**/*.py", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # Look for functions with no return type
        # This is a simple heuristic - only adds -> None for functions with docstrings
        for match in re.finditer(r'def (\w+)\(([^)]*)\):\s*\n\s+"""', content):
            func_name = match.group(1)
            params = match.group(2)

            if '->' not in params:
                # Add return type if missing
                replacement = f'def {func_name}({params}) -> None:\n    """'
                content = content.replace(match.group(0), replacement)
                modified = True

        # Write changes back if modified
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            fixed_files.append(filepath)
            print(f"✓ Added missing return type annotations in {os.path.basename(filepath)}")

    return fixed_files

if __name__ == "__main__":
    sys.exit(main())
