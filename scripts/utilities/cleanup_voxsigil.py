#!/usr/bin/env python3
"""
VoxSigil Library Cleanup Script
Organizes files and removes duplicates for a clean project structure.
"""

import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def cleanup_voxsigil_library():
    """Clean up and organize the VoxSigil library"""

    base_path = Path(".")
    logger.info("🧹 Starting VoxSigil Library Cleanup")

    # Create organized directory structure
    directories_to_create = [
        "launchers",
        "demos",
        "tests_archive",
        "diagnostic_tools",
        "legacy_reports",
        "working_gui",
    ]

    for dir_name in directories_to_create:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Created/verified directory: {dir_name}")

    # File organization rules
    file_moves = {
        # Launchers
        "launchers": [
            "Launch_*.bat",
            "launch_*.py",
        ],
        # Demo files
        "demos": [
            "demo_*.py",
            "quick_*.py",
            "simple_*.py",
            "absolute_minimal_gui_test.py",
            "minimal_*.py",
            "ultra_minimal_gui.py",
        ],
        # Test files (archive old ones)
        "tests_archive": [
            "test_*.py",
            "verify_*.py",
            "validate_*.py",
            "final_*.py",
            "gradual_*.py",
            "comprehensive_*.py",
        ],
        # Diagnostic tools
        "diagnostic_tools": [
            "diagnose_*.py",
            "analyze_*.py",
            "check_*.py",
            "Debug_*.bat",
            "detailed_*.py",
            "direct_*.py",
        ],
        # Legacy reports and status files
        "legacy_reports": [
            "*_COMPLETE.py",
            "*_COMPLETION*.md",
            "*_COMPLETION*.py",
            "*_STATUS.md",
            "*_REPORT*.md",
            "*_FIXES*.md",
            "*_RESOLUTION*.md",
            "ALL_ISSUES_RESOLVED.md",
            "DEPLOYMENT_READY.md",
            "CONFIGURATION_ISSUE_RESOLVED.md",
            "EVENT_LOOP_ERROR_RESOLUTION_REPORT.md",
            "GUI_HANG_SOLUTION_COMPLETE.md",
            "NEURAL_TTS_PRODUCTION_COMPLETE.md",
            "OPTIMIZATION_IMPLEMENTATION_COMPLETE.md",
            "PROJECT_STRUCTURE_FINAL.md",
            "PROJECT_SUMMARY.md",
            "TRAINING_ACCURACY_FIX_COMPLETION.md",
            "VANTACORE_INTEGRATION_FIXES.md",
            "VANTACORE_INTEGRATION_FIXES_REPORT.md",
            "WARNING_FIXES_REPORT.md",
            "VOICE_PROCESSING_COMPLETE_GUIDE.md",
            "ADVANCED_TTS_TECHNIQUES_GUIDE.md",
            "GUI_CONVERSION_SUMMARY.md",
        ],
        # Working GUI files (keep the good ones)
        "working_gui": [
            "crash_proof_enhanced_gui.py",
            "optimized_enhanced_gui.py",
            "standalone_enhanced_gui.py",
        ],
    }

    # Move files according to rules
    for target_dir, patterns in file_moves.items():
        target_path = base_path / target_dir
        moved_count = 0

        for pattern in patterns:
            for file_path in base_path.glob(pattern):
                if file_path.is_file() and file_path.parent == base_path:
                    try:
                        dest_path = target_path / file_path.name
                        if not dest_path.exists():
                            shutil.move(str(file_path), str(dest_path))
                            moved_count += 1
                        else:
                            logger.warning(f"⚠️ File already exists in target: {dest_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to move {file_path}: {e}")

        if moved_count > 0:
            logger.info(f"📁 Moved {moved_count} files to {target_dir}/")

    # Files to remove (duplicates and obsolete)
    files_to_remove = [
        "proper_enhanced_gui.py",  # Superseded by crash_proof version
        "fixed_complete_enhanced_gui.py",  # Superseded by crash_proof version
        "gui_crash_debug.py",
        "gui_crash_debug.log",
        "gui_completion_status_report.py",
        "music_agent_diagnostic.py",
        "clean_encoding.py",
        "safe_syntax_check.py",
        "simple_syntax_check.py",
        "tts_stt_status.py",
        "vantacore_grid_former_integration.log",
        "agent_status.log",
        "test_output.log",
    ]

    removed_count = 0
    for file_name in files_to_remove:
        file_path = base_path / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                removed_count += 1
                logger.info(f"🗑️ Removed: {file_name}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {file_name}: {e}")

    if removed_count > 0:
        logger.info(f"🗑️ Removed {removed_count} obsolete files")

    # Create a main launcher that points to the best working version
    create_main_launcher()

    # Create README for each organized directory
    create_directory_readmes()

    logger.info("✅ VoxSigil Library cleanup completed!")

    # Show final structure
    show_final_structure()


def create_main_launcher():
    """Create the main launcher pointing to the best working GUI"""

    main_launcher_content = """@echo off
echo ========================================
echo VoxSigil Enhanced GUI - Main Launcher
echo ========================================
echo.
echo 🎯 This is the MAIN launcher for VoxSigil GUI
echo.
echo Available options:
echo 1. Crash-Proof GUI (Recommended)
echo 2. Optimized GUI
echo 3. Standalone GUI
echo.
set /p choice="Choose option (1-3) or press Enter for default: "

if "%choice%"=="2" (
    echo.
    echo 🚀 Launching Optimized Enhanced GUI...
    python working_gui/optimized_enhanced_gui.py
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Launching Standalone GUI...
    python working_gui/standalone_enhanced_gui.py
) else (
    echo.
    echo 🛡️ Launching Crash-Proof GUI (Default)...
    python working_gui/crash_proof_enhanced_gui.py
)

echo.
echo GUI session ended.
pause
"""

    with open("Launch_VoxSigil_Main.bat", "w") as f:
        f.write(main_launcher_content)

    logger.info("🚀 Created main launcher: Launch_VoxSigil_Main.bat")


def create_directory_readmes():
    """Create README files for each organized directory"""

    readmes = {
        "launchers/README.md": """# Launchers Directory

This directory contains all the GUI launcher files.

## Files:
- `Launch_*.bat` - Windows batch launchers
- `launch_*.py` - Python launcher scripts

## Usage:
Most launchers are legacy. Use the main launcher in the root directory:
`Launch_VoxSigil_Main.bat`
""",
        "demos/README.md": """# Demos Directory

This directory contains demonstration and testing scripts.

## Files:
- `demo_*.py` - Feature demonstration scripts
- `quick_*.py` - Quick test scripts
- `simple_*.py` - Simple test implementations
- `minimal_*.py` - Minimal test cases

## Usage:
These are for development and testing. Not needed for normal operation.
""",
        "tests_archive/README.md": """# Tests Archive

This directory contains archived test files.

## Files:
- `test_*.py` - Unit and integration tests
- `validate_*.py` - Validation scripts
- `verify_*.py` - Verification tools
- `final_*.py` - Final validation tests

## Usage:
These are archived for reference. Active tests are in the `tests/` directory.
""",
        "diagnostic_tools/README.md": """# Diagnostic Tools

This directory contains tools for diagnosing issues.

## Files:
- `diagnose_*.py` - Diagnostic scripts
- `analyze_*.py` - Analysis tools
- `check_*.py` - Health check scripts
- `Debug_*.bat` - Debug batch files

## Usage:
Use these tools when troubleshooting issues with VoxSigil.
""",
        "legacy_reports/README.md": """# Legacy Reports

This directory contains completion reports and status documents.

## Files:
- `*_COMPLETE.py` - Completion status scripts
- `*_REPORT.md` - Development reports  
- `*_STATUS.md` - Status documents
- `*_FIXES.md` - Fix documentation
- `*_GUIDE.md` - Implementation guides

## Usage:
These are historical documents showing the development process.
""",
        "working_gui/README.md": """# Working GUI Directory

This directory contains the main working GUI implementations.

## Files:
- `crash_proof_enhanced_gui.py` - **RECOMMENDED** - Guaranteed no crashes
- `optimized_enhanced_gui.py` - Performance optimized version
- `standalone_enhanced_gui.py` - No external dependencies version

## Usage:
Use `crash_proof_enhanced_gui.py` for the most stable experience.
All versions include the full feature set with different optimization approaches.
""",
    }

    for file_path, content in readmes.items():
        Path(file_path).parent.mkdir(exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"📝 Created README: {file_path}")


def show_final_structure():
    """Show the final organized structure"""
    logger.info("\n" + "=" * 60)
    logger.info("📁 FINAL VOXSIGIL LIBRARY STRUCTURE")
    logger.info("=" * 60)

    structure = {
        "🚀 Main Launchers": ["Launch_VoxSigil_Main.bat (👑 MAIN LAUNCHER)"],
        "🎯 Working GUI": [
            "working_gui/crash_proof_enhanced_gui.py (👑 RECOMMENDED)",
            "working_gui/optimized_enhanced_gui.py",
            "working_gui/standalone_enhanced_gui.py",
        ],
        "📁 Organized Directories": [
            "launchers/ (Legacy launchers)",
            "demos/ (Demo and test scripts)",
            "tests_archive/ (Archived tests)",
            "diagnostic_tools/ (Debug tools)",
            "legacy_reports/ (Development reports)",
            "working_gui/ (Main GUI files)",
        ],
        "🏗️ Core Directories": [
            "core/ (Core VoxSigil engine)",
            "gui/ (GUI components)",
            "engines/ (Processing engines)",
            "agents/ (AI agents)",
            "training/ (ML training)",
            "models/ (AI models)",
            "tests/ (Active tests)",
        ],
    }

    for category, items in structure.items():
        logger.info(f"\n{category}:")
        for item in items:
            logger.info(f"  ✅ {item}")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 CLEANUP COMPLETE!")
    logger.info("🚀 Use: Launch_VoxSigil_Main.bat")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        cleanup_voxsigil_library()
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        import traceback

        traceback.print_exc()
