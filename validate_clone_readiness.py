#!/usr/bin/env python3
"""
VoxSigil Library - Final Validation Script
Verifies all components are ready for local clone and testing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def validate_critical_files():
    """Validate all critical files exist"""
    critical_files = [
        # Core system files
        "quick_setup.py",
        "launch_gui.bat", 
        "launch_gui.sh",
        "CLONE_AND_TEST_INSTRUCTIONS.md",
        "COMPLETE_9_PHASE_BUG_FIXING_ROADMAP.md",
        
        # Test files
        "test_pyqt5_gui.py",
        "test_complete_registration.py", 
        "validate_production_readiness.py",
        "setup_phase0_infrastructure.py",
        
        # Demo files
        "demo_novel_paradigms.py",
        "demo_api_driven_learning.py",
        
        # Core modules
        "core/__init__.py",
        "Vanta/__init__.py",
        "gui/__init__.py",
        
        # GUI bridge
        "Vanta/gui/bridge.py",
        "gui/components/dynamic_gridformer_gui.py",
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in critical_files:
        if Path(file_path).exists():
            present_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return present_files, missing_files

def validate_registration_files():
    """Validate all modules have vanta_registration.py files"""
    required_modules = [
        "agents", "engines", "core", "memory", "handlers", "config",
        "scripts", "strategies", "utils", "tags", "schema", "scaffolds", 
        "sigils", "ARC", "ART", "BLT", "gui", "llm", "vmb", "training",
        "middleware", "services", "interfaces", "integration",
        "VoxSigilRag", "voxsigil_supervisor", "Vanta"
    ]
    
    registered_modules = []
    missing_registration = []
    
    for module in required_modules:
        reg_file = Path(f"{module}/vanta_registration.py")
        if reg_file.exists():
            registered_modules.append(module)
        else:
            missing_registration.append(module)
    
    return registered_modules, missing_registration

def check_gui_components():
    """Check GUI component files"""
    gui_files = [
        "gui/components/dynamic_gridformer_gui.py",
        "gui/components/gui_styles.py", 
        "gui/components/__init__.py",
        "Vanta/gui/bridge.py",
    ]
    
    gui_present = []
    gui_missing = []
    
    for gui_file in gui_files:
        if Path(gui_file).exists():
            gui_present.append(gui_file)
        else:
            gui_missing.append(gui_file)
    
    return gui_present, gui_missing

def generate_validation_report():
    """Generate comprehensive validation report"""
    print("🔍 VoxSigil Library - Final Validation Report")
    print("=" * 60)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Validate critical files
    present_files, missing_files = validate_critical_files()
    print(f"📁 CRITICAL FILES:")
    print(f"   ✅ Present: {len(present_files)}")
    print(f"   ❌ Missing: {len(missing_files)}")
    if missing_files:
        print(f"   Missing files: {missing_files}")
    print()
    
    # Validate registrations
    registered_modules, missing_registration = validate_registration_files()
    print(f"📋 MODULE REGISTRATIONS:")
    print(f"   ✅ Registered: {len(registered_modules)}")
    print(f"   ❌ Missing: {len(missing_registration)}")
    if missing_registration:
        print(f"   Missing modules: {missing_registration}")
    print()
    
    # Validate GUI
    gui_present, gui_missing = check_gui_components()
    print(f"🖥️ GUI COMPONENTS:")
    print(f"   ✅ Present: {len(gui_present)}")
    print(f"   ❌ Missing: {len(gui_missing)}")
    if gui_missing:
        print(f"   Missing GUI files: {gui_missing}")
    print()
    
    # Calculate overall readiness
    total_critical = len(present_files) + len(missing_files)
    total_modules = len(registered_modules) + len(missing_registration)
    total_gui = len(gui_present) + len(gui_missing)
    
    files_percent = (len(present_files) / total_critical) * 100 if total_critical > 0 else 0
    modules_percent = (len(registered_modules) / total_modules) * 100 if total_modules > 0 else 0
    gui_percent = (len(gui_present) / total_gui) * 100 if total_gui > 0 else 0
    
    overall_percent = (files_percent + modules_percent + gui_percent) / 3
    
    print(f"📊 READINESS SUMMARY:")
    print(f"   📁 Critical Files: {files_percent:.1f}%")
    print(f"   📋 Module Registration: {modules_percent:.1f}%")
    print(f"   🖥️ GUI System: {gui_percent:.1f}%")
    print(f"   🎯 OVERALL READINESS: {overall_percent:.1f}%")
    print()
    
    # Readiness status
    if overall_percent >= 95:
        status = "🚀 READY FOR CLONE & TESTING"
        color = "GREEN"
    elif overall_percent >= 85:
        status = "⚠️ MOSTLY READY - Minor Issues"
        color = "YELLOW"
    else:
        status = "❌ NOT READY - Major Issues"
        color = "RED"
    
    print("🎯 CLONE READINESS STATUS:")
    print(f"   {status}")
    print()
    
    # Instructions
    if overall_percent >= 95:
        print("✅ READY TO CLONE!")
        print("   👉 Users can now:")
        print("      1. git clone https://github.com/CryptoCOB/Voxsigil-Library.git")
        print("      2. cd Voxsigil-Library")
        print("      3. python quick_setup.py")
        print("      4. python -m gui.components.dynamic_gridformer_gui")
        print()
        print("   📚 Documentation available:")
        print("      - CLONE_AND_TEST_INSTRUCTIONS.md")
        print("      - COMPLETE_9_PHASE_BUG_FIXING_ROADMAP.md")
        print("      - Launch scripts: launch_gui.bat / launch_gui.sh")
    else:
        print("❌ NOT READY YET")
        print("   🔧 Complete these tasks first:")
        if missing_files:
            print(f"      - Create missing files: {missing_files}")
        if missing_registration:
            print(f"      - Create registration files for: {missing_registration}")
        if gui_missing:
            print(f"      - Create missing GUI files: {gui_missing}")
    
    print()
    print("=" * 60)
    
    return overall_percent >= 95

def main():
    """Main validation function"""
    ready = generate_validation_report()
    
    if ready:
        print("🎉 VoxSigil Library is READY for local clone and testing!")
        return 0
    else:
        print("⚠️ VoxSigil Library needs more work before clone readiness.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
