#!/usr/bin/env python
"""
Extract information from .pyc files to help reconstruct source code
"""
import dis
import marshal
import sys
from pathlib import Path

def extract_pyc_info(pyc_file):
    """Extract useful information from a .pyc file"""
    with open(pyc_file, 'rb') as f:
        # Skip magic number (4 bytes) and flags (4 bytes for 3.7+)
        f.read(4)  # Magic
        f.read(4)  # Flags/timestamp
        f.read(4)  # Size (Python 3.3+)
        f.read(4)  # Additional field (Python 3.7+)
        
        # Read code object
        try:
            code = marshal.load(f)
        except Exception as e:
            print(f"Error loading code: {e}")
            return None
    
    return code

def print_code_info(code, indent=0):
    """Print information about a code object"""
    prefix = "  " * indent
    print(f"{prefix}Code object: {code.co_name}")
    print(f"{prefix}  Filename: {code.co_filename}")
    print(f"{prefix}  Args: {code.co_argcount} positional, {code.co_kwonlyargcount} keyword-only")
    print(f"{prefix}  Varnames: {code.co_varnames}")
    print(f"{prefix}  Names: {code.co_names}")
    print(f"{prefix}  Constants: {[c for c in code.co_consts if not isinstance(c, type(code))]}")
    print(f"{prefix}  Free vars: {code.co_freevars}")
    print(f"{prefix}  Cell vars: {code.co_cellvars}")
    
    # Print disassembly
    print(f"{prefix}  Bytecode:")
    dis.dis(code, file=sys.stdout)
    print()
    
    # Recursively print nested code objects
    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            print_code_info(const, indent + 1)

if __name__ == "__main__":
    pyc_files = [
        r"D:\01_1\Nebula\modules\__pycache__\blt.cpython-313.pyc",
        r"D:\01_1\Nebula\modules\__pycache__\mesh.cpython-313.pyc",
        r"D:\01_1\Nebula\modules\__pycache__\__init__.cpython-313.pyc",
    ]
    
    for pyc_file in pyc_files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {pyc_file}")
        print('='*80)
        
        if Path(pyc_file).exists():
            code = extract_pyc_info(pyc_file)
            if code:
                print_code_info(code)
        else:
            print(f"File not found: {pyc_file}")
