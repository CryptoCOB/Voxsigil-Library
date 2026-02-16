#!/usr/bin/env python3
"""
Cleanup & Repo Preparation Script
Organizes Phase 4-6 production code for pushing to CryptoCOB/Voxsigil.Predict
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def analyze_files():
    """Categorize all .py files in UBLT"""
    
    ublt_path = Path("c:\\UBLT")
    files = list(ublt_path.glob("*.py"))
    
    categories = {
        "KEEP_PHASE_6": [],  # Latest, production-ready
        "KEEP_PHASE_5": [],  # Attribution & rewards
        "KEEP_PHASE_4B": [], # Cognitive optimization
        "KEEP_PHASE_D": [],  # Attribution system
        "KEEP_PHASE_BENCHMARKS": [],  # Benchmarking infrastructure
        "KEEP_DOCS": [],  # Documentation (markdown)
        "REMOVE_TEMP": [],  # Temporary/test files
        "REMOVE_OLD_VERSIONS": [],  # Superseded files
        "REMOVE_DEPRECATED": [],  # Old infrastructure
    }
    
    # Phase 6 (latest, to keep)
    phase6_keywords = ["phase6_", "parallel_benchmark"]
    
    # Phase 5 (to keep)
    phase5_keywords = ["phase5_", "attribution_reward"]
    
    # Phase 4-B (to keep)
    phase4b_keywords = ["phase4b", "4b1_", "4b2_", "4b3_", "student_embedder", "semantic_space", "sheaf"]
    
    # Phase D (to keep)
    phased_keywords = ["phase_d", "attribution"]
    
    # Benchmarking (to keep)
    benchmark_keywords = ["bench", "sigil_generation_benchmark", "model_benchmark"]
    
    # Temporary/old (to remove)
    temp_keywords = ["temp_", "_v2", "_v3", "_v4", "_v5", "diagnostic", "test_", "recover", "restore", "decompile", "extract_pyc", "pyc_inspect"]
    
    # Old infrastructure (to remove)
    old_keywords = ["precompute_", "behavioral_nas", "knowledge_distillation", "hybrid_", "quantum_", "reasoning_engine", "tot_engine", "energy_adaptation", "evo_nas", "ghost_detection", "high_speed", "hf_dataset", "holistic_perception", "async_processing", "blt_student", "complete_blt", "cat_engine", "proof_of_", "neural_arch", "meta_consciousness"]
    
    # VoxSigil generators (to remove - not core VME)
    voxsigil_keywords = ["voxsigil_", "generate", "voxsigil_enhance", "enforce_bonding", "regenerate_sigils"]
    
    for file in files:
        name = file.name.lower()
        
        # Categorize
        if any(kw in name for kw in phase6_keywords):
            categories["KEEP_PHASE_6"].append(name)
        elif any(kw in name for kw in phase5_keywords):
            categories["KEEP_PHASE_5"].append(name)
        elif any(kw in name for kw in phase4b_keywords):
            categories["KEEP_PHASE_4B"].append(name)
        elif any(kw in name for kw in phased_keywords):
            categories["KEEP_PHASE_D"].append(name)
        elif any(kw in name for kw in benchmark_keywords):
            categories["KEEP_PHASE_BENCHMARKS"].append(name)
        elif any(kw in name for kw in temp_keywords):
            categories["REMOVE_TEMP"].append(name)
        elif any(kw in name for kw in old_keywords):
            categories["REMOVE_DEPRECATED"].append(name)
        elif any(kw in name for kw in voxsigil_keywords):
            categories["REMOVE_DEPRECATED"].append(name)
        elif "_v" in name and any(c.isdigit() for c in name.split("_v")[-1][:2]):
            categories["REMOVE_OLD_VERSIONS"].append(name)
        else:
            # Default to review
            print(f"? REVIEW: {name}")
    
    return categories

def print_cleanup_summary(categories):
    """Print cleanup analysis"""
    
    print("\n" + "="*80)
    print("🧹 CLEANUP ANALYSIS FOR VOXSIGIL.PREDICT PUSH")
    print("="*80)
    
    print("\n✅ FILES TO KEEP (Production Code):")
    print("-" * 80)
    
    for category in ["KEEP_PHASE_6", "KEEP_PHASE_5", "KEEP_PHASE_4B", "KEEP_PHASE_D", "KEEP_PHASE_BENCHMARKS"]:
        if categories[category]:
            label = category.replace("KEEP_", "")
            print(f"\n{label}: ({len(categories[category])} files)")
            for file in sorted(categories[category]):
                print(f"  ✓ {file}")
    
    total_keep = sum(len(v) for k, v in categories.items() if k.startswith("KEEP_"))
    print(f"\nTOTAL TO KEEP: {total_keep} files")
    
    print("\n" + "="*80)
    print("🗑️  FILES TO REMOVE (Cleanup):")
    print("-" * 80)
    
    temp_count = len(categories["REMOVE_TEMP"])
    old_count = len(categories["REMOVE_OLD_VERSIONS"])
    depr_count = len(categories["REMOVE_DEPRECATED"])
    total_remove = temp_count + old_count + depr_count
    
    if categories["REMOVE_TEMP"]:
        print(f"\nTemporary/Test Files: ({temp_count} files)")
        for file in sorted(categories["REMOVE_TEMP"])[:10]:
            print(f"  ✗ {file}")
        if len(categories["REMOVE_TEMP"]) > 10:
            print(f"  ... and {len(categories['REMOVE_TEMP']) - 10} more")
    
    if categories["REMOVE_OLD_VERSIONS"]:
        print(f"\nOld Versions (superseded): ({old_count} files)")
        for file in sorted(categories["REMOVE_OLD_VERSIONS"])[:10]:
            print(f"  ✗ {file}")
        if len(categories["REMOVE_OLD_VERSIONS"]) > 10:
            print(f"  ... and {len(categories['REMOVE_OLD_VERSIONS']) - 10} more")
    
    if categories["REMOVE_DEPRECATED"]:
        print(f"\nDeprecated Infrastructure: ({depr_count} files)")
        for file in sorted(categories["REMOVE_DEPRECATED"])[:10]:
            print(f"  ✗ {file}")
        if len(categories["REMOVE_DEPRECATED"]) > 10:
            print(f"  ... and {len(categories['REMOVE_DEPRECATED']) - 10} more")
    
    print(f"\nTOTAL TO REMOVE: {total_remove} files")
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Production Code to Keep:  {total_keep} files")
    print(f"Cleanup/Remove:           {total_remove} files")
    print(f"Total Files:              {total_keep + total_remove} files")
    print("\nCleanup will save space and make repo cleaner for push")
    print("="*80 + "\n")
    
    return {
        "keep_count": total_keep,
        "remove_count": total_remove,
        "categories": categories
    }

def save_analysis(analysis):
    """Save cleanup analysis to file"""
    
    output = Path("c:\\UBLT\\cleanup_analysis.json")
    
    with open(output, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis
        }, f, indent=2)
    
    print(f"📋 Analysis saved to: {output}")

if __name__ == "__main__":
    categories = analyze_files()
    analysis = print_cleanup_summary(categories)
    save_analysis(analysis)
