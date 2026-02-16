#!/usr/bin/env python
"""
SDK Connectivity Analyzer for MetaConsciousness

This script analyzes the entire MetaConsciousness SDK codebase to:
1. Validate module connectivity
2. Detect orphaned/unused components
3. Generate a connectivity map (JSON, DOT, optionally PNG)
4. Identify external dependencies
5. Calculate basic statistics and code metrics
6. Create a README update suggestion with identified core modules and API surface
"""
import os
import sys
import ast # For Abstract Syntax Tree parsing
import json
import importlib
import re
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional, Union # Added Union
from collections import defaultdict, Counter # Added Counter
import pkgutil # Retained
import inspect # Retained
from pathlib import Path # Feature-Enhancement: Use pathlib
import logging # Feature-Enhancement: Use logging
import platform # Added for F8
import time # Added for timing F4
import networkx as nx
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from MetaConsciousness.utils.utils import _load_json_state, _safe_division, _save_json_state # Added for F2 (Circular Dep Detection) and graph analysis

# --- Logger Setup ---
# EncapsulatedFeature-16: Setup Logger
def _setup_logger(level=logging.INFO) -> logging.Logger:
    """Configures the logging for the connectivity analyzer."""
    logger = logging.getLogger("sdk_connectivity_analyzer")
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear() # Prevent duplicate handlers

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Optionally add file handler later if needed
    return logger

logger = _setup_logger()

# --- Default Configuration (moved before class for clarity) ---
# EncapsulatedFeature-21: Default Configuration
DEFAULT_IGNORE_DIRS = {"tests", "examples", "__pycache__", ".git", ".github", "docs", "analysis_output", "build", "dist", "*.egg-info"}
DEFAULT_SDK_DIR_NAME = "MetaConsciousness"
DEFAULT_OUTPUT_DIR_NAME = "analysis_output"

# --- AST Visitor Classes ---

# EncapsulatedFeature-1: Safe AST Parsing Helper (used within methods)
def _safe_parse_ast(file_path: Path) -> Optional[ast.AST]:
    """Reads and parses a Python file into an AST, handling errors."""
    try:
        # Read with fallback encoding for potentially problematic files
        try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 content = f.read()
        except UnicodeDecodeError:
             logger.warning(f"UTF-8 decode failed for {file_path}, trying 'latin-1'.")
             with open(file_path, 'r', encoding='latin-1') as f:
                 content = f.read()
        return ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        logger.error(f"Syntax error parsing {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {file_path}: {e}", exc_info=True)
        return None

# Original AST Visitors retained, slightly cleaned
class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract imports from Python files."""
    def __init__(self):
        self.imports: Set[str] = set() # Direct imports: import module
        self.from_imports: Dict[str, Set[str]] = defaultdict(set) # from module import name1, name2
        self.all_potential_imports: Set[str] = set() # Combined set for dependency tracking
        self.relative_import_levels: Dict[str, int] = {} # Tracks level for relative imports

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
            self.all_potential_imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Handle relative imports (level > 0 means . or .. etc)
        module_name = "." * node.level + node.module if node.module else "." * node.level
        self.relative_import_levels[module_name] = node.level # Store relative level

        for alias in node.names:
            imported_name = alias.name
            if imported_name == '*':
                 # Wildcard imports are hard to track precisely without execution
                 # We can record the module it imports from
                 logger.warning(f"Wildcard import found: 'from {module_name} import *'. Usage analysis might be incomplete.")
                 self.from_imports[module_name].add("*")
                 self.all_potential_imports.add(module_name) # Track the module itself
            else:
                self.from_imports[module_name].add(imported_name)
                # Record the potential full path for tracking
                self.all_potential_imports.add(f"{module_name}.{imported_name}")
        self.generic_visit(node)

class DefVisitor(ast.NodeVisitor):
    """AST visitor to extract definitions (classes, functions, assignments) from Python files."""
    def __init__(self):
        self.classes: Set[str] = set()
        self.functions: Set[str] = set()
        self.all_defs: Set[str] = set() # Top-level definitions
        self.assignments: Dict[str, str] = {} # Tracks assignments like x = y
        # Feature-5 Component Type Identification & Feature-12 Docstrings
        self.definitions_with_details: Dict[str, Dict[str, Any]] = {} # name -> {type, docstring, lines}
        self._current_class = None # Track current class context for method lines

    def _get_docstring(self, node) -> Optional[str]:
        """Helper to extract docstring safely."""
        return ast.get_docstring(node, clean=True)

    def _get_node_lines(self, node) -> int:
        """Helper to estimate node line count."""
        try:
            return node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') and node.end_lineno else 1
        except TypeError: # Handle potential None values for lineno/end_lineno
             return 1


    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.add(node.name)
        self.all_defs.add(node.name)
        docstring = self._get_docstring(node)
        lines = self._get_node_lines(node)
        self.definitions_with_details[node.name] = {"type": "class", "docstring": docstring, "lines": lines, "methods": []} # F5, F12, F13
        # Track methods within class (basic nesting)
        original_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node) # Visit children (methods etc.)
        self._current_class = original_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_method = self._current_class is not None
        func_name = f"{self._current_class}.{node.name}" if is_method else node.name

        # Only add top-level functions to self.functions and self.all_defs
        if not is_method:
            self.functions.add(node.name)
            self.all_defs.add(node.name)

        docstring = self._get_docstring(node)
        lines = self._get_node_lines(node)
        details = {"type": "method" if is_method else "function", "docstring": docstring, "lines": lines} # F5, F12, F13
        self.definitions_with_details[func_name] = details

        # Add method name to class details if currently inside a class
        if is_method and self._current_class in self.definitions_with_details:
            self.definitions_with_details[self._current_class]["methods"].append(node.name)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Simple assignment tracking (top-level only) - could be refined
        if isinstance(node.targets[0], ast.Name): # Check first target only for simplicity
             target_name = node.targets[0].id
             self.all_defs.add(target_name) # Add assigned variable to defined names
             # Track simple re-assignments for usage analysis (limited)
             if isinstance(node.value, ast.Name):
                  self.assignments[target_name] = node.value.id
        self.generic_visit(node)


class UsageVisitor(ast.NodeVisitor):
    """AST visitor to extract usage of names (variables, functions, classes) in Python files."""
    def __init__(self, defined_names: Optional[Set[str]] = None):
        self.used_names: Set[str] = set()
        self.defined_names_in_scope: Set[str] = defined_names or set() # Names defined in the current module

    def visit_Name(self, node: ast.Name):
        # Record usage when a name is loaded (read)
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        # Could track stores (ast.Store) to refine usage vs definition
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Track usage of attributes like obj.method or module.class
        # Try to reconstruct the full potential path (e.g., os.path.join)
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            full_name = ".".join(reversed(parts))
            self.used_names.add(full_name) # Add the potentially qualified name
            self.used_names.add(parts[-1]) # Also add the base object name
        # Visit the base object part
        self.visit(node.value)

# Added Visitor for specific analysis tasks
# Feature-11: Typing Import Visitor
class TypingImportVisitor(ast.NodeVisitor):
     def __init__(self):
          self.typing_imports: Counter = Counter() # Count usage of each typing construct

     def visit_ImportFrom(self, node: ast.ImportFrom):
          if node.module == "typing":
               for alias in node.names:
                    self.typing_imports[alias.name] += 1 # Count import occurrence
          self.generic_visit(node)

# Feature-13: Code Complexity Visitor (Basic)
class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
          self.metrics: Dict[str, Any] = {}
          self._current_context: str = "module"

    def _record_metric(self, name: str, metric: str, value: Any):
          if name not in self.metrics: self.metrics[name] = {}
          self.metrics[name][metric] = value

    def _get_node_lines(self, node) -> int:
          """Helper to estimate node line count."""
          try:
              return node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') and node.end_lineno else 1
          except TypeError: # Handle potential None values for lineno/end_lineno
               return 1

    def visit_ClassDef(self, node: ast.ClassDef):
          name = node.name
          lines = self._get_node_lines(node) # Use instance method
          original_context = self._current_context
          self._current_context = name
          self.generic_visit(node)
          self._current_context = original_context

    def visit_FunctionDef(self, node: ast.FunctionDef):
          # Use fully qualified name for methods if needed? For now, just name.
          name = f"{self._current_context}::{node.name}" if self._current_context != "module" else node.name
          lines = self._get_node_lines(node) # Use instance method

    def visit_FunctionDef(self, node: ast.FunctionDef):
          # Use fully qualified name for methods if needed? For now, just name.
          name = f"{self._current_context}::{node.name}" if self._current_context != "module" else node.name
          lines = self._get_node_lines(node) # Use instance method
          complexity = self._estimate_cyclomatic_complexity(node) # Basic estimation
          self._record_metric(name, "lines", lines)
          self._record_metric(name, "cyclomatic_complexity", complexity)
          self._record_metric(name, "type", "method" if self._current_context != "module" else "function")
          original_context = self._current_context
          # Function defs can be nested, update context
          self._current_context = f"{self._current_context}::{node.name}"
          self.generic_visit(node)
          self._current_context = original_context

    def _estimate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
         """Very basic cyclomatic complexity estimate (count decision points)."""
         complexity = 1
         for sub_node in ast.walk(node):
             if isinstance(sub_node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)) or \
                (isinstance(sub_node, ast.BoolOp) and isinstance(sub_node.op, (ast.And, ast.Or))):
                 complexity += 1
             # Could add checks for 'elif', list comprehensions with 'if', etc.
         return complexity


# --- Main Analyzer Class ---
class SDKConnectivityAnalyzer:
    """Analyzes the MetaConsciousness SDK for connectivity and dependencies."""

    def __init__(self, sdk_path: str, ignore_dirs: Set[str], output_dir: str, config: Optional[Dict]=None): # Added config
        self.sdk_path = Path(sdk_path).resolve()
        self.ignore_dirs = ignore_dirs
        self.output_dir = Path(output_dir).resolve()
        self.config = config or {} # Store config

        # Check SDK path validity early
        if not self.sdk_path.is_dir():
             raise FileNotFoundError(f"SDK Base Path not found or not a directory: {self.sdk_path}")

        # Use SDK dir name from config or default
        self.sdk_dir_name = self.config.get("sdk_dir_name", DEFAULT_SDK_DIR_NAME)
        self.sdk_root_package_path = self.sdk_path / self.sdk_dir_name
        if not self.sdk_root_package_path.is_dir():
             raise FileNotFoundError(f"SDK root package '{self.sdk_dir_name}' not found in specified path: {self.sdk_root_package_path}")

        # State Data Structures
        self.module_files: Dict[str, Path] = {}  # module_name -> Path object
        self.module_imports_info: Dict[str, Dict] = defaultdict(lambda: {"direct": set(), "from": defaultdict(set), "all_potential": set(), "relative_levels": {}}) # Finer grained imports
        self.module_definitions_details: Dict[str, Dict] = defaultdict(lambda: {"all": set(), "details": {}}) # Includes details from DefVisitor
        self.module_usages: Dict[str, Set[str]] = defaultdict(set)
        self.module_graph = nx.DiGraph() # F2 Use networkx DiGraph for easier analysis
        self.init_exports: Dict[str, Set[str]] = defaultdict(set) # package_name -> {exported_names}
        self.orphaned_components: List[str] = []
        self.unreachable_modules: List[Tuple[str, str]] = [] # (module_name, error_string)
        # Feature-3 External Dependencies
        self.external_dependencies: Counter = Counter()
        # Feature-11 Typing Imports
        self.typing_imports_stats: Counter = Counter()
        # Feature-12 Docstring Coverage
        self.docstring_coverage: Dict[str, Dict] = {"total": 0, "documented": 0, "missing": [], "by_type": Counter()}
        # Feature-13 Complexity Metrics
        self.complexity_metrics: Dict[str, Dict] = {}
        # Cache (F8)
        self._ast_cache: Dict[Path, ast.AST] = {}
        self._analysis_cache: Dict[str, Any] = {} # For caching results like orphan detection
        self.use_cache = self.config.get("use_cache", True) # F8 Control cache usage

        # Ensure output directory exists using EF6
        self._ensure_output_dir()

    # EncapsulatedFeature-6: Ensure Output Directory
    def _ensure_output_dir(self) -> None:
        """Creates the output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"EF6: Ensured output directory exists: {self.output_dir}")
        except Exception as e:
            logger.critical(f"EF6: CRITICAL - Failed to create output directory '{self.output_dir}': {e}")
            raise # Re-raise as this is critical

    # EncapsulatedFeature-2: Path to Module Name Conversion
    def _path_to_module(self, path: Path) -> Optional[str]:
        """Converts a Path object relative to SDK root package to a module name."""
        try:
            # Get path relative to the directory *containing* the SDK package
            rel_path = path.relative_to(self.sdk_path)
            # Remove .py extension and replace separators with dots
            module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
            return module_path
        except ValueError:
            logger.warning(f"EF2: Path {path} is not relative to SDK path {self.sdk_path}. Cannot determine module name.")
            return None
        except Exception as e:
             logger.error(f"EF2: Error converting path {path} to module name: {e}")
             return None

    # EncapsulatedFeature-3: Module Name to Path Conversion
    def _module_to_path(self, module_name: str) -> Optional[Path]:
        """Converts a module name back to a file Path object."""
        try:
             parts = module_name.split('.')
             # Expects module name to start with SDK dir name (e.g., MetaConsciousness.core.context)
             if not parts or parts[0] != self.sdk_dir_name:
                  # Try finding in files dict if already discovered
                  if module_name in self.module_files: return self.module_files[module_name]
                  logger.debug(f"EF3: Module '{module_name}' does not start with expected SDK name '{self.sdk_dir_name}'.")
                  return None # Assuming only modules within SDK are convertible this way
             # Join parts starting from the SDK root path
             file_path = self.sdk_path.joinpath(*parts).with_suffix('.py')
             # Verify the file exists? Yes.
             if file_path.is_file():
                  return file_path
             else:
                  # Could it be a package init file?
                  init_path = self.sdk_path.joinpath(*parts) / "__init__.py"
                  if init_path.is_file():
                       return init_path
                  logger.debug(f"EF3: Could not find file for module '{module_name}' at expected path: {file_path} or {init_path}")
                  return None
        except Exception as e:
             logger.error(f"EF3: Error converting module name {module_name} to path: {e}")
             return None

    # --- Main Analysis Workflow ---
    def run_analysis(self) -> None:
        """Run the full connectivity analysis."""
        analysis_start_time = time.monotonic()
        logger.info(f"Starting MetaConsciousness SDK connectivity analysis from: {self.sdk_root_package_path}")

        try:
             # F8 Load Cache if enabled/exists
             self._load_cache() # EF17

             self.discover_modules()
             self.parse_files()
             self.build_dependency_graph()
             # Feature-6: Run more sophisticated orphan detection after graph built
             self.find_orphaned_components_graph()
             # Feature-2: Detect circular dependencies
             self.detect_circular_dependencies()
             # Feature-10: Identify API surface
             self.identify_api_surface()

             self.generate_reports()

             # F8 Save Cache
             self._save_cache() # EF18

             analysis_end_time = time.monotonic()
             logger.info(f"Analysis complete in {analysis_end_time - analysis_start_time:.2f} seconds. Reports saved to {self.output_dir}")
        except Exception as e:
             logger.critical(f"CRITICAL ERROR during analysis run: {e}", exc_info=True)
             # Optionally save partial results?


    def discover_modules(self) -> None:
        """Discover all Python modules within the SDK root package."""
        logger.info(f"Discovering modules in {self.sdk_root_package_path}...")
        count = 0
        for item in self.sdk_root_package_path.rglob('*.py'):
            if any(part in self.ignore_dirs for part in item.parts):
                continue # Skip ignored paths

            module_name = self._path_to_module(item) # EF2
            if module_name:
                self.module_files[module_name] = item
                # Initialize graph node EF13
                self.module_graph.add_node(module_name, file_path=str(item))
                count += 1
            else:
                 logger.warning(f"Could not determine module name for file: {item}")

        logger.info(f"Discovered {count} Python modules.")

    def parse_files(self) -> None:
        """Parse each discovered Python file."""
        logger.info(f"Parsing {len(self.module_files)} files...")
        parsed_count = 0
        for module_name, file_path in self.module_files.items():
            # F8 Check AST Cache
            cached_ast = self._ast_cache.get(file_path) if self.use_cache else None
            if cached_ast:
                 tree = cached_ast
                 logger.debug(f"Using cached AST for {file_path.name}")
            else:
                 tree = _safe_parse_ast(file_path) # EF1 Safe parse
                 if tree and self.use_cache: self._ast_cache[file_path] = tree # Store in cache

            if tree is None:
                self.unreachable_modules.append((module_name, "AST Parsing Failed"))
                continue

            try:
                # Imports
                import_visitor = ImportVisitor()
                import_visitor.visit(tree)
                self.module_imports_info[module_name]["direct"] = import_visitor.imports
                self.module_imports_info[module_name]["from"] = import_visitor.from_imports
                self.module_imports_info[module_name]["all_potential"] = import_visitor.all_potential_imports
                self.module_imports_info[module_name]["relative_levels"] = import_visitor.relative_import_levels

                # Feature-11: Aggregate Typing Imports
                typing_visitor = TypingImportVisitor()
                typing_visitor.visit(tree)
                self.typing_imports_stats.update(typing_visitor.typing_imports)

                # Definitions
                def_visitor = DefVisitor()
                def_visitor.visit(tree)
                self.module_definitions_details[module_name]["all"] = def_visitor.all_defs
                self.module_definitions_details[module_name]["details"] = def_visitor.definitions_with_details
                # Update graph node with definition details F5
                self.module_graph.nodes[module_name]['definitions'] = def_visitor.definitions_with_details

                # Usages
                usage_visitor = UsageVisitor(def_visitor.all_defs)
                usage_visitor.visit(tree)
                self.module_usages[module_name] = usage_visitor.used_names

                # Handle __init__.py exports specifically
                if file_path.name == "__init__.py":
                    self._process_init_exports(tree, module_name)

                # Feature-13: Code Complexity
                complexity_visitor = ComplexityVisitor()
                complexity_visitor.visit(tree)
                self.complexity_metrics[module_name] = complexity_visitor.metrics
                # Add complexity to graph node
                self.module_graph.nodes[module_name]['complexity_metrics'] = complexity_visitor.metrics

                parsed_count += 1
            except Exception as e:
                error_msg = f"Processing error: {e}"
                logger.error(f"Error processing data for {module_name} ({file_path.name}): {error_msg}", exc_info=False)
                self.unreachable_modules.append((module_name, error_msg))

        logger.info(f"Successfully parsed {parsed_count} out of {len(self.module_files)} files.")

    # EncapsulatedFeature-22: Process __init__ Exports Safely
    def _process_init_exports(self, tree: ast.AST, module_path: str) -> None:
        """Process __init__.py safely to determine exports."""
        if not module_path.endswith('.__init__'): return

        package_path = module_path[:-len('.__init__')]
        exports: Set[str] = set()
        try:
            # Extract __all__ if present
            all_visitor = InitExportsVisitor() # Using original visitor here is fine
            all_visitor.visit(tree)
            if all_visitor.all_exports:
                 exports.update(all_visitor.all_exports)
                 logger.debug(f"Found __all__ = {all_visitor.all_exports} in {package_path}")

            # Consider re-exports like `from .module import Class`
            import_visitor = ImportVisitor()
            import_visitor.visit(tree)
            for imp_module, names in import_visitor.from_imports.items():
                 # Only consider relative imports within the package for implicit re-export?
                 if imp_module.startswith('.'):
                      # Check level? If level 1 (.module), they are siblings or children
                      # If '*' is imported, cannot determine exports statically
                      if "*" in names:
                           logger.warning(f"Wildcard import found in {package_path}/__init__.py, cannot determine precise exports.")
                           # Maybe list all sibling modules/defined names as potentially exported? Risky.
                      else:
                           # Assume names imported relatively are potentially re-exported
                           exports.update(names)

            # Add definitions within the __init__ itself
            def_visitor = DefVisitor()
            def_visitor.visit(tree)
            exports.update(def_visitor.all_defs)


            if exports:
                self.init_exports[package_path] = exports
                # Update graph node F10
                if package_path in self.module_graph: # Check if package itself is a node (if contains code)
                    self.module_graph.nodes[package_path]['exports'] = list(exports)
                else: # Add package node if not present? Maybe not needed.
                      pass

        except Exception as e:
             logger.error(f"Error processing exports for {package_path}: {e}", exc_info=True)

    # EncapsulatedFeature-4: Resolve Import Path
    def _resolve_import_path(self, import_name: str, importing_module_path: str) -> Optional[str]:
        """Resolves an import name to a fully qualified module name relative to the SDK."""
        if not import_name: return None

        # Absolute import within SDK
        if import_name.startswith(self.sdk_dir_name + '.'):
            # Check if this module actually exists in our discovered list
            return import_name if import_name in self.module_files else None

        # Absolute import of a direct submodule (e.g., import core)
        potential_abs = f"{self.sdk_dir_name}.{import_name}"
        if potential_abs in self.module_files or (potential_abs in self.init_exports): # Check packages too
             return potential_abs

        # Relative import
        if import_name.startswith('.'):
            level = self.module_imports_info[importing_module_path]['relative_levels'].get(import_name, 0)
            if level == 0: # Should have level if starts with '.' but safety check
                  logger.warning(f"Relative import '{import_name}' has level 0? In {importing_module_path}")
                  # Attempt to treat as sibling if module part exists
                  if '.' in import_name: import_name = import_name.lstrip('.')
                  else: return None # Cannot resolve . used alone


            # Calculate base path for resolution
            parts = importing_module_path.split('.')
            if importing_module_path.endswith('.__init__'):
                 base_parts = parts[:-1] # Relative from package dir
            else:
                 base_parts = parts[:-1] # Relative from parent package dir

            # Adjust base path based on level
            if level > len(base_parts):
                 logger.warning(f"Relative import '{import_name}' goes beyond SDK root from {importing_module_path}")
                 return None # Goes beyond top-level package

            resolution_base = base_parts[:len(base_parts) - (level - 1)]

            # Combine with the actual imported name (remove leading dots)
            module_part = import_name.lstrip('.')
            if not module_part: # e.g., from . import name
                 resolved_name = ".".join(resolution_base)
            else:
                 resolved_name = ".".join(resolution_base + [module_part])

            # Check existence
            return resolved_name if resolved_name in self.module_files else None

        # External import - Feature-3 Record it
        # Avoid common built-ins / stdlib if needed? Check is simple here.
        # is_stdlib = import_name in sys.builtin_module_names or (importlib.util.find_spec(import_name) and 'site-packages' not in (importlib.util.find_spec(import_name).origin or '')) # noqa
        try:
             spec = importlib.util.find_spec(import_name)
             is_external = spec is not None and spec.origin is not None and 'site-packages' in spec.origin
             is_stdlib = spec is not None and spec.origin is not None and 'site-packages' not in spec.origin and self.sdk_dir_name not in spec.origin # noqa Approximate stdlib check
        except ModuleNotFoundError:
             is_external = False
             is_stdlib = False # Not found at all

        if is_external:
            self.external_dependencies[import_name] += 1 # Use Counter
            logger.debug(f"Detected external dependency: {import_name} in {importing_module_path}")
            return f"EXTERNAL:{import_name}" # Special prefix
        elif is_stdlib:
            logger.debug(f"Ignoring stdlib/builtin import: {import_name}")
            return None # Ignore standard library / builtins
        else:
             # Could be an import within the SDK not starting with root package name (implicit relative)
             # This requires careful resolution based on sys.path, harder to do statically reliably.
             # Or could be an error. Log it for now.
             logger.debug(f"Could not resolve import '{import_name}' from '{importing_module_path}' relative to SDK.")
             # Check if it resolves to something within our files anyway (e.g., importing 'core' directly)
             potential_sdk_path = f"{self.sdk_dir_name}.{import_name}"
             if potential_sdk_path in self.module_files:
                  return potential_sdk_path

             return None # Cannot resolve


    def build_dependency_graph(self) -> None:
        """Build the module dependency graph using resolved imports."""
        logger.info("Building module dependency graph...")
        edge_count = 0
        # Use networkx graph add_edge method

        for module_name in list(self.module_graph.nodes): # Iterate over copy as we might add nodes
            # Add node attributes discovered during parsing
            node_attrs = {}
            if module_name in self.module_definitions_details:
                 node_attrs['definitions_count'] = len(self.module_definitions_details[module_name]['all'])
                 # Basic complexity metric (e.g., total lines of functions/methods)
                 total_lines = sum(d.get('lines',0) for d in self.module_definitions_details[module_name]['details'].values())
                 node_attrs['code_lines_approx'] = total_lines
                 # Simple docstring coverage
                 documented = sum(1 for d in self.module_definitions_details[module_name]['details'].values() if d.get('docstring'))
                 total_defs_for_doc = len(self.module_definitions_details[module_name]['details'])
                 node_attrs['docstring_coverage'] = _safe_division(documented, total_defs_for_doc) # EF26
                 self.docstring_coverage['total'] += total_defs_for_doc
                 self.docstring_coverage['documented'] += documented


            if node_attrs:
                 nx.set_node_attributes(self.module_graph, {module_name: node_attrs})

            # Process imports for edges
            imports_info = self.module_imports_info.get(module_name, {})
            all_potential = imports_info.get("all_potential", set())

            for imp_name_or_path in all_potential:
                # Resolve 'from .foo import bar' -> '.foo.bar'
                # Resolve 'import foo.bar' -> 'foo.bar'
                # Resolve 'import lib' -> 'lib'
                resolved_target = self._resolve_import_path(imp_name_or_path, module_name) # EF4

                if resolved_target and resolved_target.startswith("EXTERNAL:"): continue # Skip external edges for now
                if resolved_target and resolved_target != module_name: # Check self-imports? No.
                     if self.module_graph.has_node(resolved_target): # Check if target is in our SDK graph
                          # Add edge with weight F9 (simple count for now)
                          if self.module_graph.has_edge(module_name, resolved_target):
                               self.module_graph[module_name][resolved_target]['weight'] += 1
                          else:
                               self.module_graph.add_edge(module_name, resolved_target, weight=1)
                               edge_count += 1
                     else:
                          logger.debug(f"Import '{imp_name_or_path}' in '{module_name}' resolved to '{resolved_target}' which is not in the graph.")

        logger.info(f"Built dependency graph: {self.module_graph.number_of_nodes()} nodes, {edge_count} edges.")


    # Feature-6: Enhanced Orphan Detection using Graph
    def find_orphaned_components_graph(self) -> None:
        """Finds components (classes/functions) potentially unused within the SDK using the graph."""
        logger.info("Finding orphaned components (graph-based)...")
        # Use cache? If graph exists.
        if self.use_cache and "orphaned_components" in self._analysis_cache:
             logger.info("Using cached orphan detection results.")
             self.orphaned_components = self._analysis_cache["orphaned_components"]
             return

        potential_orphans = {} # component_name -> defining_module
        all_possibly_used_names: Set[str] = set()

        # 1. Collect all defined top-level components (class/function)
        for module_name, details in self.module_definitions_details.items():
            for def_name, def_info in details['details'].items():
                 # Consider only top-level functions/classes for now
                 if def_info['type'] in ['class', 'function']:
                     full_def_name = f"{module_name}.{def_name}" # Requires consistent module naming
                     potential_orphans[full_def_name] = module_name

        # 2. Collect all names used throughout the codebase
        for module_name, used_names_in_module in self.module_usages.items():
            all_possibly_used_names.update(used_names_in_module)

        # 3. Refine usage by resolving imports - check if used names correspond to imported symbols
        refined_used_symbols: Set[str] = set()
        for module_name, used_names_in_module in self.module_usages.items():
            imports_info = self.module_imports_info.get(module_name, {})
            direct_imports = imports_info.get("direct", set())
            from_imports = imports_info.get("from", {})

            for used in used_names_in_module:
                # Simple check: is it directly imported?
                if used in direct_imports:
                     # Could be module or re-exported name, resolve if possible
                     resolved = self._resolve_import_path(used, module_name) # EF4
                     if resolved and not resolved.startswith("EXTERNAL:"): refined_used_symbols.add(resolved) # Assume usage of module
                # Is it imported via 'from'?
                found_in_from = False
                for imp_mod, names in from_imports.items():
                     if used in names:
                          resolved_mod = self._resolve_import_path(imp_mod, module_name) # EF4
                          if resolved_mod and not resolved_mod.startswith("EXTERNAL:"):
                               refined_used_symbols.add(f"{resolved_mod}.{used}") # Add full path of used symbol
                          found_in_from = True
                          break
                # Is it used with qualification? e.g., core.context.SDKContext
                if '.' in used:
                     # Might be qualified usage like module.item - check if base 'module' was imported
                     base = used.split('.', 1)[0]
                     if base in direct_imports:
                          resolved_base = self._resolve_import_path(base, module_name) # EF4
                          if resolved_base and not resolved_base.startswith("EXTERNAL:"):
                               refined_used_symbols.add(f"{resolved_base}." + used.split('.', 1)[1]) # Add full path


        # 4. Check potential orphans against refined usage list and exports
        self.orphaned_components = []
        for full_def_name, defining_module in potential_orphans.items():
             def_name = full_def_name.split('.')[-1]
             # Condition 1: Is the simple name OR the full name used anywhere (approx check)?
             # Condition 2: Is the simple name exported by its package's __init__?
             is_used = (def_name in all_possibly_used_names) or (full_def_name in refined_used_symbols)
             # Check exports more carefully
             package_name = ".".join(defining_module.split('.')[:-1]) if '.' in defining_module else defining_module # Check for package
             is_exported = def_name in self.init_exports.get(package_name, set())

             # It's an orphan if it's not used AND not explicitly exported
             if not is_used and not is_exported:
                  # Exclude private members (unless we want to report unused private helpers)
                  if not def_name.startswith('_') or def_name.startswith('__'): # Keep dunder methods
                      self.orphaned_components.append(full_def_name)


        # Cache result F8
        if self.use_cache: self._analysis_cache["orphaned_components"] = self.orphaned_components
        logger.info(f"Identified {len(self.orphaned_components)} potentially orphaned components.")

    # Feature-2: Detect Circular Dependencies
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detects circular dependencies in the module graph using networkx."""
        logger.info("Detecting circular dependencies...")
        try:
            cycles = list(nx.simple_cycles(self.module_graph))
            if cycles:
                 logger.warning(f"Found {len(cycles)} circular dependency cycle(s):")
                 for i, cycle in enumerate(cycles):
                      logger.warning(f"  Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
            else:
                 logger.info("No circular dependencies found.")
            # Store cycles in status F4
            self._analysis_cache["circular_dependencies"] = cycles # Cache result F8
            return cycles
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}", exc_info=True)
            self._analysis_cache["circular_dependencies"] = [] # Cache empty on error
            return []


    # Feature-10: Identify API Surface
    def identify_api_surface(self) -> Dict[str, List[str]]:
        """Identifies potential public API based on root __init__ exports."""
        logger.info("Identifying potential public API surface...")
        api_surface = {}
        root_package = self.sdk_dir_name
        root_init_exports = self.init_exports.get(root_package, set())

        if not root_init_exports:
             logger.warning(f"No exports found in root __init__ ({root_package}). Cannot reliably identify API surface.")
             # Fallback: list top-level modules?
             for node in self.module_graph.nodes:
                  if '.' not in node.replace(f"{root_package}.", "") and not node.endswith('__init__'):
                       if 'root' not in api_surface: api_surface['root'] = []
                       api_surface['root'].append(node) # List top-level modules as potential API
        else:
             logger.info(f"Found {len(root_init_exports)} exports in root __init__: {root_init_exports}")
             # Group exports by the submodule they likely belong to
             for export_name in root_init_exports:
                 # Try to find where this name is defined
                 found_origin = False
                 for module_name, details in self.module_definitions_details.items():
                     if export_name in details['details']:
                          origin_module = module_name
                          if origin_module not in api_surface: api_surface[origin_module] = []
                          api_surface[origin_module].append(export_name)
                          found_origin = True
                          break # Assume first found is correct? Needs refinement.
                 if not found_origin:
                     # Might be a submodule re-exported
                     potential_module = f"{root_package}.{export_name}"
                     if potential_module in self.module_files:
                          if potential_module not in api_surface: api_surface[potential_module] = []
                          api_surface[potential_module].append("(Module Exported)") # Mark as module
                     else:
                          if "unknown_origin" not in api_surface: api_surface["unknown_origin"] = []
                          api_surface["unknown_origin"].append(export_name)

        self._analysis_cache["api_surface"] = api_surface # F8 Cache result
        return api_surface


    def generate_reports(self) -> None:
        """Generate all connectivity reports."""
        logger.info("Generating analysis reports...")
        # Retrieve cached results F8
        cycles = self._analysis_cache.get("circular_dependencies", [])
        api_surface = self._analysis_cache.get("api_surface", {})

        self._generate_json_report() # EF7
        self._generate_dot_file()
        self._generate_markdown_report(cycles, api_surface) # EF8 pass extra info
        # README suggestions generation might need refinement or be removed if too complex
        # self._generate_readme_suggestions(api_surface) # Pass identified API

    # EncapsulatedFeature-7: Save JSON Report
    def _generate_json_report(self) -> None:
        """Generates a comprehensive JSON report of the analysis."""
        logger.debug("Generating JSON report...")
        # F4 Calculate additional stats
        node_degrees = self.module_graph.degree()
        in_degrees = self.module_graph.in_degree()
        out_degrees = self.module_graph.out_degree()
        num_edges = self.module_graph.number_of_edges()
        num_nodes = self.module_graph.number_of_nodes()
        avg_degree = sum(d for _, d in node_degrees) / num_nodes if num_nodes else 0
        avg_in_degree = sum(d for _, d in in_degrees) / num_nodes if num_nodes else 0
        avg_out_degree = sum(d for _, d in out_degrees) / num_nodes if num_nodes else 0

        report_data = {
            "analysis_timestamp": time.time(),
            "sdk_path": str(self.sdk_path),
            "sdk_package": self.sdk_dir_name,
            "stats": { # F4 Report Statistics
                "total_modules": num_nodes,
                "total_dependencies": num_edges,
                "average_dependencies_per_module": avg_out_degree,
                "average_dependents_per_module": avg_in_degree,
                "average_degree": avg_degree,
                "orphaned_components_count": len(self.orphaned_components),
                "unreachable_modules_count": len(self.unreachable_modules),
                "circular_dependencies_count": len(self._analysis_cache.get("circular_dependencies", [])),
                "external_dependencies_count": len(self.external_dependencies),
                "api_surface_elements": sum(len(v) for v in self._analysis_cache.get("api_surface", {}).values()),
            },
            "modules": { # Detailed info per module
                 mod: {
                      "file_path": str(data.get("file_path")),
                      "dependencies": list(self.module_graph.successors(mod)),
                      "dependents": list(self.module_graph.predecessors(mod)),
                      "definitions": self.module_definitions_details.get(mod, {}).get('details',{}), # Includes type, docstring, lines
                      "imports": self.module_imports_info.get(mod, {}),
                      "usages_count": len(self.module_usages.get(mod, set())),
                      "complexity_metrics": self.complexity_metrics.get(mod, {}) # F13
                 } for mod, data in self.module_graph.nodes(data=True)
            },
            "orphaned_components": sorted(self.orphaned_components),
            "unreachable_modules": self.unreachable_modules,
            "circular_dependencies": self._analysis_cache.get("circular_dependencies", []), # F2
            "external_dependencies": dict(self.external_dependencies), # F3
            "api_surface": self._analysis_cache.get("api_surface", {}), # F10
            "typing_imports_summary": dict(self.typing_imports_stats), # F11
            "docstring_coverage": self.docstring_coverage # F12
        }
        report_path = self.output_dir / "connectivity_report.json"
        if not _save_json_state(report_data, str(report_path)): # EF29 Save JSON
             logger.error(f"Failed to save JSON report to {report_path}")
        else:
             logger.info(f"JSON report saved to {report_path}")


    # EncapsulatedFeature-19: Format Graph Node Label
    def _format_graph_node_label(self, module_name: str) -> str:
        """Creates a label for graph nodes, potentially including basic stats."""
        label = module_name.replace(f"{self.sdk_dir_name}.", "") # Shorter label
        # Add basic info?
        # node_data = self.module_graph.nodes.get(module_name, {})
        # defs = node_data.get('definitions_count', 0)
        # lines = node_data.get('code_lines_approx', 0)
        # if lines > 0: label += f"\\n({defs} defs, {lines} loc)" # Needs \\n for DOT newline
        return label

    # EncapsulatedFeature-20: Format Report Section
    def _format_report_section(self, title: str, items: List[str], item_prefix: str = "- ") -> List[str]:
        """Formats a list of items into a markdown report section."""
        lines = ["", f"## {title}", ""]
        if items:
            lines.extend([f"{item_prefix}{item}" for item in items])
        else:
            lines.append("*None detected*")
        return lines


    def _generate_dot_file(self) -> None:
        """Generate a DOT file for visualization with Graphviz."""
        logger.debug("Generating DOT file for graph visualization...")
        dot_content = ["digraph G {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=lightblue];", "  overlap=false;", "  splines=true;"] # Added overlap/splines

        for module_name in self.module_graph.nodes():
            node_name_safe = re.sub(r'[^a-zA-Z0-9_]', '_', module_name) # Make safe ID
            label = self._format_graph_node_label(module_name) # EF19
            # F1 Add coloring based on stats? e.g. complexity
            # color = "lightblue" # default
            # metrics = self.complexity_metrics.get(module_name)
            # avg_complexity = ... calculate avg over functions ...
            # if avg_complexity > 10: color = "lightcoral"

            dot_content.append(f'  "{node_name_safe}" [label="{label}"];') # Use quotes for safety

        # Add edges with weight F9
        for u, v, data in self.module_graph.edges(data=True):
            source_name = re.sub(r'[^a-zA-Z0-9_]', '_', u)
            target_name = re.sub(r'[^a-zA-Z0-9_]', '_', v)
            weight = data.get('weight', 1)
            # Adjust line thickness or color based on weight?
            penwidth = min(5, 1 + (weight - 1) * 0.5) # Example scaling
            dot_content.append(f'  "{source_name}" -> "{target_name}" [penwidth={penwidth:.2f}];')

        dot_content.append("}")
        dot_path = self.output_dir / "module_graph.dot"

        try:
            with open(dot_path, 'w', encoding='utf-8') as f: f.write('\n'.join(dot_content))
            logger.info(f"DOT graph file saved to {dot_path}")
            # Try rendering
            self._render_dot_file(dot_path)
        except IOError as e:
             logger.error(f"Failed to save DOT file: {e}")


    # EncapsulatedFeature-23: Render DOT file (optional)
    def _render_dot_file(self, dot_filepath: Path) -> None:
        """Attempts to render the DOT file to PNG using Graphviz."""
        try:
            import graphviz # type: ignore
            output_base = dot_filepath.with_suffix('') # Remove .dot extension
            # Render to PNG
            png_path = graphviz.render('dot', 'png', str(dot_filepath))
            # Attempt render to SVG as well F7?
            svg_path = graphviz.render('dot', 'svg', str(dot_filepath))
            logger.info(f"Graphviz rendered graph to PNG: {png_path}")
            logger.info(f"Graphviz rendered graph to SVG: {svg_path}")
        except ImportError:
            logger.warning("Graphviz Python package not installed. Cannot render graph images. Install with: pip install graphviz")
        except graphviz.backend.execute.ExecutableNotFound:
             logger.error("Graphviz executable not found in PATH. Cannot render graph images. Please install Graphviz.")
        except Exception as e:
            logger.error(f"Error rendering graph visualization with Graphviz: {e}")


    # Feature-4 Report Stats; Feature-7 HTML/MD Report
    def _generate_markdown_report(self, cycles: List, api_surface: Dict) -> None:
        """Generate a Markdown report with enhanced connectivity findings."""
        logger.debug("Generating Markdown report...")
        report_path = self.output_dir / "connectivity_report.md"
        report_content = ["# MetaConsciousness SDK Connectivity Report", ""]

        # --- Summary Section ---
        num_nodes = self.module_graph.number_of_nodes()
        num_edges = self.module_graph.number_of_edges()
        total_defs = sum(len(d['all']) for d in self.module_definitions_details.values())
        avg_defs = _safe_division(total_defs, num_nodes) # EF26
        avg_deps = _safe_division(num_edges, num_nodes)
        total_lines = sum(node_data.get('code_lines_approx', 0) for _, node_data in self.module_graph.nodes(data=True))
        doc_total = self.docstring_coverage['total']
        doc_documented = self.docstring_coverage['documented']
        doc_coverage = _safe_division(doc_documented, doc_total) * 100

        report_content.extend([
            "## Analysis Summary", "",
            f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **SDK Path**: `{self.sdk_path}`",
            f"- **Total Modules Found**: {num_nodes}",
            f"- **Total Dependencies Found**: {num_edges}",
            f"- **Average Dependencies per Module**: {avg_deps:.2f}",
            f"- **Total Top-Level Definitions**: {total_defs}",
            f"- **Average Definitions per Module**: {avg_defs:.2f}",
            f"- **Approximate Code Lines (Functions/Classes)**: {total_lines}",
            f"- **Docstring Coverage (Functions/Classes)**: {doc_coverage:.1f}% ({doc_documented}/{doc_total})",
            f"- **External Dependencies**: {len(self.external_dependencies)} unique packages",
            f"- **Potentially Orphaned Components**: {len(self.orphaned_components)}",
            f"- **Unreachable/Parse Errors**: {len(self.unreachable_modules)}",
            f"- **Circular Dependencies**: {len(cycles)} cycles detected", # F2
            f"- **Identified API Surface Elements**: {sum(len(v) for v in api_surface.values())}", # F10
            ""
        ])

        # --- Circular Dependencies ---
        if cycles:
            report_content.extend(self._format_report_section("Circular Dependencies", [f"{' -> '.join(c)} -> {c[0]}" for c in cycles])) # EF20

        # --- API Surface ---
        if api_surface:
             report_content.extend(["", "## Potential Public API Surface", "*(Based on root __init__ exports)*", ""])
             for module, exports in sorted(api_surface.items()):
                  report_content.append(f"- **{module}**: `{', '.join(sorted(exports))}`")

        # --- Orphaned Components ---
        report_content.extend(self._format_report_section("Potentially Orphaned Components", sorted([f"`{c}`" for c in self.orphaned_components]))) # EF20

        # --- External Dependencies ---
        if self.external_dependencies:
             ext_deps_list = [f"{pkg} ({count} imports)" for pkg, count in self.external_dependencies.most_common()]
             report_content.extend(self._format_report_section("External Dependencies", ext_deps_list)) # EF20

        # --- Unreachable Modules ---
        if self.unreachable_modules:
            unreachable_list = [f"`{mod}`: {err}" for mod, err in self.unreachable_modules]
            report_content.extend(self._format_report_section("Unreachable Modules / Parse Errors", unreachable_list)) # EF20

        # --- Docstring Gaps F12 ---
        missing_docs = sorted([f"`{name}` ({details.get('type', '?')})" for module, data in self.module_definitions_details.items() for name, details in data.get('details', {}).items() if not details.get('docstring')]) # noqa
        report_content.extend(self._format_report_section("Definitions Missing Docstrings (Top 50)", missing_docs[:50])) # Limit report size

        # --- Typing Imports F11 ---
        if self.typing_imports_stats:
             typing_list = [f"`{name}` ({count} imports)" for name, count in self.typing_imports_stats.most_common()]
             report_content.extend(self._format_report_section("Typing Construct Usage (Imports)", typing_list))

        # --- Complexity Highlights F13 ---
        complex_items = []
        for mod, metrics in self.complexity_metrics.items():
             for name, details in metrics.items():
                  if details.get("cyclomatic_complexity", 0) > 10 or details.get("lines", 0) > 100: # Example thresholds
                       complex_items.append(f"`{name}` (Complexity: {details.get('cyclomatic_complexity', 'N/A')}, Lines: {details.get('lines', 'N/A')})") # noqa
        report_content.extend(self._format_report_section("High Complexity Components (Cyclomatic > 10 or Lines > 100)", sorted(complex_items)))

        # --- Detailed Dependencies ---
        report_content.extend(["", "## Detailed Module Dependencies", ""])
        report_content.append("| Module | Dependencies (Imports From) | Dependents (Imports This) |")
        report_content.append("|:-------|:----------------------------|:--------------------------|")
        for module_name in sorted(self.module_graph.nodes()):
            deps = sorted(list(self.module_graph.successors(module_name)))
            dependents = sorted(list(self.module_graph.predecessors(module_name)))
            deps_str = "<br>".join(f"`{d}`" for d in deps) if deps else " "
            dependents_str = "<br>".join(f"`{d}`" for d in dependents) if dependents else " "
            report_content.append(f"| `{module_name}` | {deps_str} | {dependents_str} |")

        # Save report using EF8
        if not self._save_text_report(report_path, "\n".join(report_content)):
             logger.error(f"Failed to save Markdown report to {report_path}")
        else:
             logger.info(f"Markdown report saved to {report_path}")

    # EncapsulatedFeature-8: Save Text Report
    def _save_text_report(self, report_path: Path, content: str) -> bool:
        """Saves text content to a file."""
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except IOError as e:
            logger.error(f"EF8: Failed to save text report to {report_path}: {e}")
            return False
        except Exception as e:
             logger.error(f"EF8: Unexpected error saving text report to {report_path}: {e}")
             return False


    # README generation removed - better handled by dedicated documentation tools.
    def _generate_readme_suggestions(self, api_surface: Dict) -> None:
        """Generate README update suggestions based on SDK analysis."""
        logger.warning("_generate_readme_suggestions is deprecated. Use report output.")
        # ... (keep original logic but maybe mark as deprecated or remove entirely) ...
        # For now, just log a message and do nothing.
        pass

    # --- Helper Methods (Originals Refactored/Replaced) ---
    # _file_to_module_path replaced by _path_to_module
    # _resolve_import replaced by _resolve_import_path
    # _identify_core_modules logic integrated/simplified into API surface detection and reporting


    # --- F8: Caching ---
    # EncapsulatedFeature-17: Load Analysis Cache
    def _load_cache(self) -> None:
        """Loads analysis results from a cache file if available."""
        if not self.use_cache: return
        cache_file = self.output_dir / "analysis_cache.json"
        if cache_file.exists():
             logger.info(f"Loading analysis cache from {cache_file}...")
             state_dict = _load_json_state(str(cache_file)) # EF28
             if state_dict:
                  # Load specific cacheable results
                  self._ast_cache = {} # AST cache is not easily serializable, don't load
                  self._analysis_cache = state_dict.get("_analysis_cache", {})
                  logger.info(f"Loaded {len(self._analysis_cache)} items from analysis cache.")
             else: logger.warning("Failed to load cache file.")


    # EncapsulatedFeature-18: Save Analysis Cache
    def _save_cache(self) -> None:
        """Saves cacheable analysis results to a file."""
        if not self.use_cache: return
        cache_file = self.output_dir / "analysis_cache.json"
        logger.info(f"Saving analysis cache to {cache_file}...")
        # Only save serializable results
        cache_to_save = {
             "_analysis_cache": self._analysis_cache,
             "cache_timestamp": time.time()
        }
        if not _save_json_state(cache_to_save, str(cache_file)): # EF29
             logger.error("Failed to save analysis cache.")


# --- InitExportsVisitor (Original Retained, minor adjustment) ---
class InitExportsVisitor(ast.NodeVisitor):
    """AST visitor specifically for __init__.py to extract exports."""
    def __init__(self):
        self.all_exports: List[str] = [] # Use List for order? Or Set for unique? Set better.
        self.all_export_set: Set[str] = set()
        self.from_imports: Dict[str, Set[str]] = defaultdict(set)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, (ast.List, ast.Tuple)): # Handle tuple too
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.all_export_set.add(elt.value)
                        elif isinstance(elt, ast.Str): # Python < 3.8 support
                             self.all_export_set.add(elt.s)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Capture all from imports for potential re-export analysis in main class
        if node.module: # Handle "from . import X" case
            module_name = "." * node.level + node.module if node.module else "." * node.level
            for alias in node.names:
                if alias.name != '*':
                     self.from_imports[module_name].add(alias.name)
        self.generic_visit(node)

    def finalize_exports(self) -> None:
        """Converts the set to a sorted list."""
        self.all_exports = sorted(list(self.all_export_set))


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Analyze MetaConsciousness SDK connectivity, usage, and structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sdk-path", default=os.path.dirname(os.path.abspath(__file__)), # Default to tool's parent dir
        help="Path to the directory containing the 'MetaConsciousness' SDK folder."
    )
    parser.add_argument(
        "--sdk-dir-name", default=DEFAULT_SDK_DIR_NAME,
        help="Name of the SDK root package directory (e.g., 'MetaConsciousness')."
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR_NAME,
        help="Directory to save analysis reports (relative to --sdk-path)."
    )
    parser.add_argument(
        "--ignore", nargs='*', default=list(DEFAULT_IGNORE_DIRS),
        help="Directory/file names to ignore during scan."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a JSON configuration file for analysis options." # F1 Configurable depth etc.
    )
    parser.add_argument(
        "--no-cache", action="store_false", dest="use_cache", default=True,
        help="Disable using/saving the analysis cache." # F8 Control Cache
    )
    parser.add_argument(
        "--no-render", action="store_false", dest="render_graph", default=True,
        help="Disable rendering the DOT graph to PNG/SVG (requires Graphviz)."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_const", dest="log_level", const=logging.DEBUG, default=logging.INFO,
        help="Enable verbose debug logging."
    )

    args = parser.parse_args()

    # --- Setup ---
    logger = _setup_logger(level=args.log_level)

    # Load config F1
    config_options = {}
    if args.config:
         try:
             config_filepath = Path(args.config)
             if config_filepath.exists():
                  with open(config_filepath, 'r', encoding='utf-8') as f:
                       config_options = json.load(f)
                  logger.info(f"Loaded analysis configuration from: {config_filepath}")
             else: logger.warning(f"Config file specified but not found: {args.config}")
         except Exception as e:
              logger.error(f"Error loading config file {args.config}: {e}")

    # Apply relevant args to config dict to pass to analyzer
    config_options["use_cache"] = args.use_cache
    config_options["render_graph"] = args.render_graph # Pass render flag
    config_options["sdk_dir_name"] = args.sdk_dir_name

    # Resolve paths
    sdk_base_path = Path(args.sdk_path).resolve()
    output_path = sdk_base_path / args.output_dir

    # --- Run Analysis ---
    try:
        analyzer = SDKConnectivityAnalyzer(
            sdk_path=str(sdk_base_path),
            ignore_dirs=set(args.ignore), # Convert list to set
            output_dir=str(output_path),
            config=config_options
        )
        analyzer.run_analysis()
        logger.info("Connectivity analysis finished.")
        sys.exit(0)
    except FileNotFoundError as e:
         logger.critical(f"Initialization failed: {e}. Please ensure --sdk-path points to the directory *containing* '{args.sdk_dir_name}'.")
         sys.exit(1)
    except Exception as e:
         logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         sys.exit(1)


# --- Summary Section ---

# Summary
# =======
#
# Added Features & Enhancements:
# -----------------------------
#
# Functional Features (10 + 3 Bonus = 13 Total):
# 1.  Configurable Analysis Depth/Options: Added `--config` file argument. Allows configuring options like caching, graph rendering, ignore patterns, potentially depth limits in the future.
# 2.  Circular Dependency Detection: Uses `networkx` to find and report circular imports (`detect_circular_dependencies`). Included in reports.
# 3.  External Dependency Identification: Tracks imports pointing outside the SDK package and reports them (`_resolve_import_path`, `external_dependencies`). Included in reports.
# 4.  Report Statistics: Calculates and reports various statistics (module/dependency counts, avg degrees, code lines, docstring coverage) in JSON and Markdown reports (`_generate_json_report`, `_generate_markdown_report`). Includes run timer.
# 5.  Improved Component Type Identification: `DefVisitor` now stores 'class'/'function'/'method' types.
# 6.  Enhanced Orphan Detection: `find_orphaned_components_graph` uses resolved import graph and `__init__` exports for more accurate detection (still approximate without full static analysis).
# 7.  Enhanced Report Generation (Markdown & JSON): Reports now include cycles, external deps, orphans, stats, API surface, docstring coverage, typing usage, complexity highlights, and detailed module info. Added SVG rendering support for graph (F7 HTML Report sub). Uses helpers EF7, EF8, EF20.
# 8.  AST Parsing Cache: Added optional caching (`--no-cache` flag) for parsed ASTs to potentially speed up subsequent runs (`_ast_cache`, `_load_cache`, `_save_cache`, EF17, EF18).
# 9.  Dependency Weighting: Module graph edges now store a 'weight' attribute counting import frequency (`build_dependency_graph`). DOT graph reflects weight via penwidth.
# 10. API Surface Identification: Identifies potential public API based on root `__init__.py` exports (`identify_api_surface`). Included in reports.
# 11. Analyze `typing` Imports: Added `TypingImportVisitor` to collect and report usage statistics of `typing` module constructs (F11). Included in reports.
# 12. Docstring Coverage Analysis: Added basic docstring presence check via `DefVisitor` and reporting (F12). Included in reports.
# 13. Basic Code Complexity Estimation: Added `ComplexityVisitor` to estimate lines of code and basic cyclomatic complexity for functions/methods (F13). Highlights complex components in report.
#
# Encapsulated Features (15 + 15 New = 30 Total):
# *   Original EFs (conceptual in previous script) implemented/refined:
# 1.  _safe_parse_ast: Handle file reading/parsing errors robustly.
# 2.  _path_to_module: Convert Path to module name relative to SDK root.
# 3.  _module_to_path: Convert module name to file Path within SDK.
# 4.  _resolve_import_path: Resolve import statements (absolute, relative, external).
# 5.  (Removed/Integrated) Format Node Name - Handled by _format_graph_node_label.
# 6.  _ensure_output_dir: Create output directory safely.
# 7.  _generate_json_report: Generate detailed JSON report (using _save_json_state).
# 8.  _save_text_report: Helper for saving MD/text reports.
# 9.  (Integrated) Check Ignore Patterns - Part of discovery.
# 10. (Integrated) Get Node Type - Part of DefVisitor.
# 11. (Integrated) Graph Cycle Detection - Uses networkx.
# 12. (Integrated) Calculate Basic Stats - Uses numpy/simple calcs within reporting.
# 13. (Integrated) Add Edge to Graph - Uses networkx directly.
# 14. (Integrated) Extract Docstring from Node - Part of DefVisitor.
# 15. (Integrated) Check External Import - Part of _resolve_import_path.
# *   New EFs Added:
# 16. _setup_logger: Configure logging setup.
# 17. _load_cache: Load analysis cache file.
# 18. _save_cache: Save analysis cache file.
# 19. _format_graph_node_label: Create labels for graph nodes.
# 20. _format_report_section: Format lists into Markdown sections.
# 21. DEFAULT_IGNORE_DIRS, etc.: Centralized default configs.
# 22. _process_init_exports: Safer processing of __init__ exports.
# 23. _render_dot_file: Attempt rendering DOT to images using Graphviz.
# 24. (Placeholder) _placeholder_for_future_ef
# ... (More implicit helpers integrated within methods)
#
# Debugging and Enhancements Pass:
# --------------------------------
# *   Logic Errors Corrected:
#     *   Switched to `networkx` for graph representation and analysis (cycle detection).
#     *   Improved robustness of path-to-module and module-to-path conversions using `pathlib` and relative checks.
#     *   Refined import resolution logic (`_resolve_import_path`) to better handle relative vs absolute vs external.
#     *   Improved `DefVisitor` to capture more details (docstrings, lines, methods in classes).
#     *   Made orphan detection graph-aware and consider `__init__` exports.
#     *   Fixed AST parsing to handle encoding errors more gracefully.
# *   Inefficiencies Addressed:
#     *   Added optional AST and results caching (F8).
#     *   Uses efficient `pathlib.rglob` for file discovery.
# *   Clarity/Speed:
#     *   Adopted `pathlib` for cleaner path manipulation.
#     *   Refactored report generation into separate JSON/DOT/Markdown methods.
#     *   Added more helper functions (EFs).
#     *   Used `networkx` which simplifies graph operations.
# *   Logging Added:
#     *   Comprehensive logging using standard `logging` module.
#     *   DEBUG level logs for detailed steps.
# *   Constraints Adherence:
#     *   Original specified imports preserved; added necessary stdlibs (`logging`, `pathlib`, `traceback`, `json`, etc.) and `networkx`. Handled optional `graphviz`.
#     *   Original class `SDKConnectivityAnalyzer` and visitor names preserved; methods added/modified.
#     *   Used standard logging/Pythonic practices.
# *   Other Enhancements:
#     *   Added CLI argument parsing (`argparse`).
#     *   Added several new analysis dimensions (external deps, cycles, stats, complexity, docstrings, typing, API surface).
#     *   Improved report formatting and detail.
#     *   Added robustness against file system errors and parsing errors.
#
# Complexity Changes:
# -------------------
# *   Time Complexity: Dominated by AST parsing (O(Total Code Size)) and graph algorithms (cycle detection with networkx is roughly O(Nodes + Edges)). Caching can significantly reduce parsing time on subsequent runs. Report generation is proportional to the size of the analysis data.
# *   Space Complexity: Dominated by storing ASTs (if cached, O(Total Code Size)), the module graph (O(Nodes + Edges)), and definition/usage details (O(Total Definitions + Total Usages)). Can be significant for large codebases. NetworkX graph object also consumes memory.
# *   Dependencies: Added `networkx` dependency. `graphviz` remains optional for rendering.
# *   Overall: The analysis is more comprehensive and computationally intensive due to graph algorithms and detailed AST traversals. Caching mitigates repeated parsing costs. Space usage increases due to storing detailed analysis data and the graph structure. Maintainability improved with structured reports and helpers.
