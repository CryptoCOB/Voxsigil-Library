#!/usr/bin/env python
"""
System Hook Audit Tool

This script scans the MetaConsciousness codebase to ensure that all modules
are correctly hooked into the MetaCore, Reflex Layer, or Agent context.
"""
import os
import sys
import ast
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from MetaConsciousness.utils.path_setup import get_project_root

def scan_for_hook_issues(project_root: Path, ignore_patterns: List[str] = None) -> Dict[str, Any]:
    """
    Scan Python files to identify modules that may not be properly connected.

    Args:
        project_root: Root directory to scan
        ignore_patterns: List of patterns to ignore

    Returns:
        Dictionary with scan results
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            "test_",
            "_test.py",
            "setup.py",
            "path_setup.py",
            "system_hook_audit.py"
        ]

    hook_issues = []
    total_files = 0

    # Recursively scan Python files
    for dirpath, _, files in os.walk(project_root):
        dirpath_str = str(dirpath)

        # Skip ignored directories
        if any(ignore in dirpath_str for ignore in ignore_patterns):
            continue

        for file in files:
            if not file.endswith('.py'):
                continue

            # Skip ignored files
            if any(ignore in file for ignore in ignore_patterns):
                continue

            total_files += 1
            filepath = os.path.join(dirpath, file)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=filepath)

                    # Look for imports from relevant modules
                    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
                    hooks_found = any(
                        imp.module and (
                            'meta_core' in imp.module or
                            'cluster' in imp.module or
                            'reflex' in imp.module or
                            'context' in imp.module
                        ) for imp in imports
                    )

                    # Also check for direct imports
                    direct_imports = [node for node in tree.body if isinstance(node, ast.Import)]
                    for imp in direct_imports:
                        for name in imp.names:
                            if any(hook in name.name for hook in ['MetaCore', 'Cluster', 'Reflex', 'Context']):
                                hooks_found = True

                    if not hooks_found:
                        hook_issues.append(filepath)
            except Exception as e:
                print(f"Error analyzing {filepath}: {e}")

    return {
        "total_files": total_files,
        "hook_issues": hook_issues
    }

def check_registration(hook_issues: List[str]) -> Dict[str, Any]:
    """
    Check if modules with potential hook issues are registered in the context.

    Args:
        hook_issues: List of files with potential hook issues

    Returns:
        Dictionary with registration check results
    """
    try:
        # Import here to avoid circular imports
        from MetaConsciousness.context import SDKContext

        registered_items = SDKContext.list_frameworks() + SDKContext.list_functors()
        registered_keys = []

        for item in registered_items:
            if item.startswith("functor_") or item.startswith("framework_"):
                key = item.split("_", 1)[1]
                registered_keys.append(key)

        # Check if any of the potentially unhooked modules are registered
        missing_hooks = []
        for path in hook_issues:
            filename = os.path.basename(path)
            module_name = os.path.splitext(filename)[0]

            # Check if module name or similar is in registered keys
            is_registered = any(key in module_name or module_name in key for key in registered_keys)

            if not is_registered:
                missing_hooks.append(path)

        return {
            "registered_keys": registered_keys,
            "missing_hooks": missing_hooks
        }
    except ImportError as e:
        print(f"Error importing SDKContext: {e}")
        return {
            "error": str(e),
            "registered_keys": [],
            "missing_hooks": hook_issues  # Default to all issues if we can't check registration
        }

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Audit system hooks in the MetaConsciousness codebase")
    parser.add_argument("--project-root", default=None, help="Project root directory")
    parser.add_argument("--report-file", default="hook_audit_report.json", help="Output report file")
    parser.add_argument("--ignore", nargs="+", default=None, help="Additional patterns to ignore")

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = get_project_root() / "MetaConsciousness"

    print(f"Scanning project at: {project_root}")

    # Build ignore patterns
    ignore_patterns = [
        "__pycache__",
        "test_",
        "_test.py",
        "setup.py",
        "path_setup.py",
        "system_hook_audit.py"
    ]

    if args.ignore:
        ignore_patterns.extend(args.ignore)

    # Run the scan
    scan_results = scan_for_hook_issues(project_root, ignore_patterns)

    print(f"Scanned {scan_results['total_files']} Python files")
    print(f"Found {len(scan_results['hook_issues'])} files with potential hook issues")

    # Check registration
    reg_results = check_registration(scan_results['hook_issues'])

    # Prepare final report
    report = {
        "total_files_scanned": scan_results['total_files'],
        "potential_hook_issues": len(scan_results['hook_issues']),
        "unhooked_modules": len(reg_results['missing_hooks']),
        "registered_keys": reg_results['registered_keys'],
        "hook_issue_files": [os.path.relpath(p, project_root.parent) for p in scan_results['hook_issues']],
        "missing_hook_files": [os.path.relpath(p, project_root.parent) for p in reg_results['missing_hooks']]
    }

    # Print summary
    print("\n=== SYSTEM HOOK AUDIT SUMMARY ===")
    print(f"Total files scanned: {report['total_files_scanned']}")
    print(f"Files lacking direct hooks: {report['potential_hook_issues']}")
    print(f"Unhooked modules not registered: {report['unhooked_modules']}")

    if report['missing_hook_files']:
        print("\n❌ Unhooked or orphaned modules found:")
        for issue in report['missing_hook_files']:
            print(f" - {issue}")
    else:
        print("\n✅ All modules correctly registered with MetaCore, Reflex, or Clusters.")

    # Save report
    if args.report_file:
        with open(args.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report_file}")

    # Return appropriate exit code
    return 1 if report['missing_hook_files'] else 0

if __name__ == "__main__":
    sys.exit(main())
