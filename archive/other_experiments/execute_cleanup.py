#!/usr/bin/env python3
"""
Git Cleanup & Push Preparation
Removes temp/deprecated files and prepares for push to CryptoCOB/Voxsigil.Predict
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Files to KEEP (from cleanup analysis)
FILES_TO_KEEP = {
    "phase6": [
        "phase6_parallel_benchmarking.py",
        "phase6_comprehensive_report_generator.py", 
        "phase6_discover_models.py"
    ],
    "phase5": [
        "phase5_attribution_reward_integration.py"
    ],
    "phase4b": [
        "phase4b1_student_embedder.py",
        "phase4b1_student_embedder_optimized.py",
        "phase4b2_semantic_space.py",
        "phase4b2_semantic_space_v2.py",
        "phase4b3_sheaf_consolidation.py",
        "phase_4b1_advanced_benchmarks.py",
        "phase_4b1_comprehensive_benchmarks.py",
        "phase_4b1_optimization_and_4b2_schema.py",
        "phase_4b1_optimize_inference.py",
        "phase_4b1_production_benchmarks.py",
        "phase_4b1_student_embedder.py",
        "phase_4b2_hierarchical_voxsigil_training.py",
        "phase_4b2_real_rebuild.py",
        "phase_4b2_real_voxsigil_training.py",
        "phase_4b2_schema_grounded_semantic.py"
    ],
    "phased": [
        "phase_d_attribution_system.py"
    ],
    "benchmarks": [
        "model_benchmark_orchestrator.py",
        "sigil_benchmark_sequential.py",
        "sigil_generation_benchmark.py"
    ]
}

# Files to REMOVE (old, temp, deprecated)
FILES_TO_REMOVE = [
    # Temporary/recovery files
    "temp_recovered_*.py",
    "recover*.py",
    "restore*.py",
    "decompile*.py",
    "diagnostic*.py",
    "extract_pyc*.py",
    "pyc_inspect*.py",
    
    # Old versions
    "*_v2.py",
    "*_v3.py", 
    "*_v4.py",
    "*_v5.py",
    
    # Deprecated infrastructure
    "async_processing_engine.py",
    "behavioral_nas_nextgen.py",
    "blt_student_interface.py",
    "cat_engine.py",
    "complete_blt_student_implementation.py",
    "energy_adaptation_system.py",
    "enforce_bonding_access_control.py",
    "evo_nas.py",
    "generate_synthetic_voxsigil.py",
    "ghost_detection_protocol.py",
    "high_speed_teacher_generator.py",
    "hf_dataset_training_system.py",
    "holistic_perception.py",
    "hybrid_cognition_engine.py",
    "hybrid_multimodal_student.py",
    "knowledge_distillation_system.py",
    "meta_consciousness.py",
    "neural_architecture_search.py",
    "proof_of_bandwidth.py",
    "proof_of_useful_work.py",
    "quantum_behavioral_nas.py",
    "quantum_lineage.py",
    "reasoning_engine.py",
    "tot_engine.py",
    "test_quantum_nas.py",
    "test_blt_*.py",
    "system_wide_blt_integration.py",
    "start_blt_integrated_training.py",
    
    # VoxSigil generators (not core VME)
    "voxsigil_*.py",
    "analyze_real_voxsigil_schema.py",
    "regenerate_sigils*.py",
    
    # Precompute training (old)
    "precompute_*.py",
    
    # Review items that are old/not production
    "demo_phase1.py",
    "gate_6_final_fix.py",
    "phase_4a_validate.py",
    "quick_speed_test.py",
    "test_hnswlib_api.py",
    "blt_training_generator.py",
    "convergence_training_system.py"
]

def should_remove(filename):
    """Check if file matches removal patterns"""
    for pattern in FILES_TO_REMOVE:
        if "*" in pattern:
            # Glob pattern
            prefix = pattern.replace("*.py", "").rstrip("_")
            if filename.startswith(prefix):
                return True
        elif filename == pattern:
            return True
    return False

def cleanup():
    """Remove deprecated and temporary files"""
    
    ublt_path = Path("c:\\UBLT")
    removed_files = []
    kept_files = []
    
    print("\n" + "="*80)
    print("🧹 STARTING CLEANUP - REMOVING DEPRECATED/TEMP FILES")
    print("="*80)
    
    py_files = list(ublt_path.glob("*.py"))
    
    for py_file in py_files:
        if should_remove(py_file.name):
            try:
                # Create backup first
                backup_dir = ublt_path / ".cleanup_backup"
                backup_dir.mkdir(exist_ok=True)
                shutil.copy2(py_file, backup_dir / py_file.name)
                
                # Remove original
                py_file.unlink()
                removed_files.append(py_file.name)
                print(f"  ✗ Removed: {py_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error removing {py_file.name}: {e}")
        else:
            kept_files.append(py_file.name)
    
    print(f"\n✓ Cleanup complete!")
    print(f"  Removed: {len(removed_files)} files")
    print(f"  Kept: {len(kept_files)} files")
    print(f"  Backup: {ublt_path}/.cleanup_backup/ (optional)")
    
    return {
        "removed": removed_files,
        "kept": kept_files
    }

def list_kept_files():
    """List all files being kept"""
    
    print("\n" + "="*80)
    print("✅ PRODUCTION FILES BEING PUSHED")
    print("="*80)
    
    ublt_path = Path("c:\\UBLT")
    
    # Production Python files
    print("\nPython Source Code:")
    print("-" * 80)
    all_kept = []
    for category, files in FILES_TO_KEEP.items():
        if files:
            print(f"\n{category.upper()}:")
            for f in files:
                py_file = ublt_path / f
                if py_file.exists():
                    size_kb = py_file.stat().st_size / 1024
                    print(f"  ✓ {f:<50} ({size_kb:>6.1f} KB)")
                    all_kept.append(f)
                else:
                    print(f"  ? {f:<50} (NOT FOUND)")
    
    # Documentation files
    print("\n" + "-" * 80)
    print("\nDocumentation Files:")
    print("-" * 80)
    md_files = list(ublt_path.glob("PHASE_*.md")) + list(ublt_path.glob("PROJECT_*.md")) + [ublt_path / "README.md"]
    for md_file in sorted(set(md_files)):
        if md_file.exists():
            size_kb = md_file.stat().st_size / 1024
            print(f"  ✓ {md_file.name:<50} ({size_kb:>6.1f} KB)")
            all_kept.append(md_file.name)
    
    # JSON reports
    print("\nJSON Reports:")
    print("-" * 80)
    json_files = []
    for json_file in ublt_path.glob("*.json"):
        if "phase6_" in json_file.name or "phase5_" in json_file.name:
            size_kb = json_file.stat().st_size / 1024
            print(f"  ✓ {json_file.name:<50} ({size_kb:>6.1f} KB)")
            json_files.append(json_file.name)
    
    print("\n" + "="*80)
    print(f"Total files to push: {len(all_kept) + len(json_files)}")
    print("="*80 + "\n")
    
    return all_kept + json_files

def save_cleanup_report(cleanup_result, files_kept):
    """Save cleanup report"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "cleanup": {
            "files_removed": len(cleanup_result["removed"]),
            "files_kept": len(cleanup_result["kept"]),
            "removed_files": cleanup_result["removed"]
        },
        "push_manifest": {
            "total_files": len(files_kept),
            "files": sorted(files_kept)
        }
    }
    
    output_file = Path("c:\\UBLT\\cleanup_report.json")
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Cleanup report saved to: {output_file}")

def main():
    """Execute cleanup"""
    
    print("\n" + "="*80)
    print("🚀 VOXSIGIL.PREDICT REPOSITORY CLEANUP & PREPARATION")
    print("="*80)
    
    # Run cleanup
    cleanup_result = cleanup()
    
    # List kept files
    files_kept = list_kept_files()
    
    # Save report
    save_cleanup_report(cleanup_result, files_kept)
    
    print("\n" + "="*80)
    print("✅ CLEANUP COMPLETE - READY FOR GIT PUSH")
    print("="*80)
    print("\nNext steps:")
    print("1. Review cleanup_report.json")
    print("2. Run: git status (to verify only production files)")
    print("3. Run: git add . && git commit -m 'Phase 6: Multi-model orchestration complete'")
    print("4. Run: git push origin main")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
