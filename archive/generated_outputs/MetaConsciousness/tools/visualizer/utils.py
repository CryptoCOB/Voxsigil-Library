import os
import ast
from typing import List, Set, Dict, Any, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

def parse_python_file(filepath: str) -> Dict[str, Any]:
    """Parse Python file and extract important information."""
    stats = {
        'imports': [],
        'classes': [],
        'functions': [],
        'lines': 0,
        'docstring': None
    }
    
    try:
        # Read file with multiple encoding attempts
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except PermissionError:
                logger.warning(f"Permission denied when trying to read {filepath}")
                return stats
            
        if content is None:
            logger.warning(f"Could not read {filepath} with any common encoding")
            return stats
            
        # Try to parse the file, but handle syntax errors gracefully
        try:
            tree = ast.parse(content, filename=filepath)
            
            # Extract docstring
            stats['docstring'] = ast.get_docstring(tree)
            
            # Extract imports, classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    stats['imports'].extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:  # Ensure module is not None
                        stats['imports'].append(node.module)
                elif isinstance(node, ast.ClassDef):
                    stats['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    stats['functions'].append(node.name)
                    
            stats['lines'] = len(content.splitlines())
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {filepath}: {e}")
            # Count lines even if parsing fails
            stats['lines'] = len(content.splitlines())
            # Try to extract imports using regex as a fallback
            stats['imports'] = _extract_imports_with_regex(content)
            
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
        
    return stats

def _extract_imports_with_regex(content: str) -> List[str]:
    """Extract imports using regex as a fallback when AST parsing fails."""
    import re
    imports = []
    
    # Match import statements
    import_pattern = re.compile(r'^import\s+([\w\.]+)', re.MULTILINE)
    for match in import_pattern.finditer(content):
        module = match.group(1).split('.')[0]
        imports.append(module)
    
    # Match from ... import statements
    from_import_pattern = re.compile(r'^from\s+([\w\.]+)\s+import', re.MULTILINE)
    for match in from_import_pattern.finditer(content):
        module = match.group(1).split('.')[0]
        if module:  # Skip relative imports that start with .
            imports.append(module)
    
    return imports

def calculate_module_complexity(stats: Dict[str, Any]) -> float:
    """Calculate module complexity score."""
    return (
        len(stats.get('imports', [])) * 0.5 +
        len(stats.get('classes', [])) * 2.0 +
        len(stats.get('functions', [])) * 1.0 +
        stats.get('lines', 0) * 0.1
    )

def is_excluded(path: str, patterns: List[str]) -> bool:
    """Check if path matches any exclude pattern."""
    return any(pattern in path for pattern in patterns)

def _get_module_name_from_path(file_path: str, root_dir: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    if os.path.basename(rel_path) == "__init__.py":
        module_path = os.path.dirname(rel_path).replace(os.path.sep, ".")
    else:
        module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
    if module_path == ".":
        module_path = os.path.basename(root_dir).replace(".py", "")
    if module_path.startswith('.'):
        module_path = module_path[1:]
    return module_path

def _parse_import_line(line: str) -> List[Tuple[str, Optional[str]]]:
    imports = []
    line = line.strip()
    if line.startswith('#'):
        return imports

    parts = line.split('#', 1)
    line = parts[0].strip()

    if line.startswith('import '):
        modules_str = line[7:].strip()
        modules = [m.strip() for m in modules_str.split(',')]
        for module in modules:
            if module:
                base_module = module.split('.')[0]
                imports.append((base_module, None))

    elif line.startswith('from '):
        parts = line.split(' import ')
        if len(parts) >= 2:
            module_str = parts[0][5:].strip()
            base_module = module_str.lstrip('.').split('.')[0]
            if base_module:
                imported_items = [item.strip() for item in parts[1].split(',')]
                for item in imported_items:
                    if item:
                        imports.append((base_module, item))

    return imports

def _save_json_data(data: Any, filepath: str) -> bool:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Successfully saved JSON data to {filepath}")
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Error saving JSON data to {filepath}: {e}")
        return False

def _load_json_data(filepath: str) -> Optional[Any]:
    if not os.path.exists(filepath):
        logger.debug(f"JSON file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON data from {filepath}")
        return data
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON data from {filepath}: {e}")
        return None

def find_python_files(directory: str, root_dir: str, exclude_patterns: List[str]) -> List[str]:
    """Find all Python files in directory that aren't excluded."""
    python_files = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d), exclude_patterns)]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if not is_excluded(file_path, exclude_patterns):
                    python_files.append(file_path)
                else:
                    logger.debug(f"Skipping excluded file: {file_path}")

    return python_files
