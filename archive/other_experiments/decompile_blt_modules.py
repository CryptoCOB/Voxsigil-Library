#!/usr/bin/env python3
"""
Decompile critical BLT bytecode files to recover Python source code
Priority: BLT core modules, consciousness managers, and critical components
"""

import os
import sys
from pathlib import Path
import subprocess
from collections import defaultdict

# Critical files to decompile (in priority order)
CRITICAL_MODULES = {
    # BLT Core System (HIGHEST PRIORITY)
    'modules/__pycache__': [
        'blt.cpython-313.pyc',
        'blt_encoder.cpython-313.pyc',
        'blt_encoder_interface.cpython-313.pyc',
        'bidirectional_blt_system.cpython-313.pyc',
        'hybrid_blt.cpython-313.pyc',
        'mesh.cpython-313.pyc',
        'eon.cpython-313.pyc',
        'recap.cpython-313.pyc',
        'art.cpython-313.pyc',
    ],
    # Consciousness/Core BLT Modules (HIGH PRIORITY)
    'modules/blt_modules/__pycache__': [
        'consciousness_manager.cpython-313.pyc',
        'consciousness_scaffold.cpython-313.pyc',
        'core_processor.cpython-313.pyc',
        'memory_reflector.cpython-313.pyc',
        'mesh_coordinator.cpython-313.pyc',
        'semantic_engine.cpython-313.pyc',
    ],
    # Orchestration (MEDIUM PRIORITY)
    'orchestration/__pycache__': [
        'task_reward_tracker.cpython-313.pyc',
        'task_fabric_engine.cpython-313.pyc',
        'ghost_detection_protocol.cpython-313.pyc',
        'ghost_detection_models.cpython-313.pyc',
    ],
    # Input Modules (MEDIUM PRIORITY)
    'input_modules/__pycache__': [
        'AuditoryInputModule.cpython-313.pyc',
        'ImageInputModuleV1.cpython-313.pyc',
        'TabularInputModuleV1.cpython-313.pyc',
        'TextInputModuleV1.cpython-313.pyc',
        'VideoInputModule.cpython-313.pyc',
        'ProprioceptiveInputModule.cpython-313.pyc',
        'OlfactoryInputModule.cpython-313.pyc',
        'TactileInputModule.cpython-313.pyc',
    ],
}

def decompile_file(pyc_path, output_dir):
    """Decompile a single .pyc file using uncompyle6"""
    try:
        # Get the module name from the .pyc file
        pyc_name = Path(pyc_path).stem  # Remove .cpython-313.pyc extension
        base_name = pyc_name.split('.')[0]
        
        # Output path
        output_file = os.path.join(output_dir, f"{base_name}.py")
        
        # Run uncompyle6 command directly
        result = subprocess.run(
            ['uncompyle6', pyc_path, '-o', output_file],
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✓ Decompiled: {base_name}.py")
            return True
        else:
            print(f"✗ Failed: {base_name} - {result.stderr[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {base_name}")
        return False
    except Exception as e:
        print(f"✗ Error: {base_name} - {str(e)[:100]}")
        return False

def main():
    base_path = Path(os.getcwd())
    
    if base_path.name != 'UBLT':
        base_path = Path('C:\\UBLT')
    
    print("=" * 70)
    print("BLT MODULE DECOMPILATION - CRITICAL FILES RECOVERY")
    print("=" * 70)
    print()
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    decompiled_files = defaultdict(list)
    
    for module_dir, pyc_files in CRITICAL_MODULES.items():
        print(f"\n[{module_dir}]")
        print("-" * 70)
        
        pycache_path = base_path / module_dir
        parent_dir = pycache_path.parent  # Get parent of __pycache__
        
        if not pycache_path.exists():
            print(f"  Skipped: Directory not found")
            stats['skipped'] += len(pyc_files)
            continue
        
        for pyc_file in pyc_files:
            full_pyc_path = pycache_path / pyc_file
            
            if not full_pyc_path.exists():
                print(f"  ! {pyc_file.split('.')[0]}.py - File not found")
                stats['skipped'] += 1
                continue
            
            if decompile_file(str(full_pyc_path), str(parent_dir)):
                stats['success'] += 1
                decompiled_files[module_dir].append(pyc_file.split('.')[0])
            else:
                stats['failed'] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("DECOMPILATION SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully decompiled: {stats['success']} files")
    print(f"✗ Failed: {stats['failed']} files")
    print(f"⊘ Skipped: {stats['skipped']} files")
    print()
    
    if decompiled_files:
        print("RECOVERED MODULES BY CATEGORY:")
        for module_dir in CRITICAL_MODULES.keys():
            if decompiled_files[module_dir]:
                print(f"\n  {module_dir}:")
                for module in decompiled_files[module_dir]:
                    print(f"    ✓ {module}.py")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Review decompiled files for quality and correctness")
    print("2. Test imports: python -c 'import modules.blt'")
    print("3. Update __init__.py files if needed")
    print("4. Run test suite to verify functionality")
    print("=" * 70)

if __name__ == '__main__':
    main()
