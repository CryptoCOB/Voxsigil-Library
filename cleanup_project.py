#!/usr/bin/env python3
"""
Project Cleanup Script for Voxsigil-Library
Organizes the messy project structure into a clean, maintainable layout.
"""

import shutil
from pathlib import Path


def create_directory_structure():
    """Create the proper directory structure for the project."""
    base_path = Path(".")

    directories = [
        # Archive directories
        "archive/old_status_reports",
        "archive/old_tests",
        "archive/old_demos",
        "archive/old_launchers",
        "archive/legacy_files",
        # Main application directories
        "src/voxsigil",
        "src/voxsigil/gui",
        "src/voxsigil/core",
        "src/voxsigil/engines",
        "src/voxsigil/agents",
        "src/voxsigil/tts",
        "src/voxsigil/utils",
        # Development directories
        "dev/demos",
        "dev/diagnostics",
        "dev/test_scripts",
        # Documentation
        "documentation/guides",
        "documentation/reports",
        # Resources
        "resources/audio_samples",
        "resources/configs",
        "resources/data",
        # Scripts
        "scripts/launchers",
        "scripts/utilities",
        # Build and deployment
        "build",
        "dist",
    ]

    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def move_files_by_pattern():
    """Move files to appropriate directories based on patterns."""
    base_path = Path(".")

    # File movement mappings
    file_moves = {
        # Status reports and documentation
        "archive/old_status_reports": [
            "*COMPLETE*.md",
            "*COMPLETE*.py",
            "*COMPLETION*.md",
            "*COMPLETION*.py",
            "*RESOLVED*.md",
            "*REPORT*.md",
            "*STATUS*.md",
            "*SUCCESS*.md",
            "*FINAL*.md",
            "*DEPLOYMENT*.md",
            "*VALIDATION*.md",
        ],
        # Test files
        "dev/test_scripts": [
            "test_*.py",
            "*test*.py",
            "*Test*.py",
            "check_*.py",
            "verify_*.py",
            "validate_*.py",
            "quick_test*.py",
            "*minimal*test*.py",
        ],
        # Demo files
        "dev/demos": ["demo_*.py", "*demo*.py", "simple_*.py"],
        # Diagnostic files
        "dev/diagnostics": [
            "*diagnostic*.py",
            "*diagnose*.py",
            "*debug*.py",
            "*crash*.py",
            "*hang*.py",
            "analyze_*.py",
        ],
        # Launcher files
        "scripts/launchers": ["launch_*.py", "Launch_*.bat", "*launcher*.py"],
        # Documentation guides
        "documentation/guides": ["*GUIDE*.md", "*TECHNIQUES*.md", "README.md"],
        # Audio samples
        "resources/audio_samples": ["*.wav", "*.mp3", "*.ogg", "*voice_sample*"],
        # Configuration files
        "resources/configs": ["*.json", "*.toml", "*.ini", "*.cfg", "*config*"],
        # Cleanup and utility scripts
        "scripts/utilities": ["cleanup_*.py", "clean_*.py", "*syntax_check*.py"],
        # Legacy reports
        "archive/legacy_files": [
            "*FIXES*.py",
            "*ERROR*.py",
            "*ATTRIBUTE*.py",
            "*ENCODING*.py",
            "*CIRCULAR*.py",
            "*EVENT_LOOP*.py",
        ],
    }

    # Move files based on patterns
    for target_dir, patterns in file_moves.items():
        target_path = base_path / target_dir
        for pattern in patterns:
            for file_path in base_path.glob(pattern):
                if file_path.is_file() and file_path.parent == base_path:
                    try:
                        destination = target_path / file_path.name
                        shutil.move(str(file_path), str(destination))
                        print(f"Moved {file_path.name} -> {target_dir}/")
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")


def move_specific_files():
    """Move specific important files to their correct locations."""
    base_path = Path(".")

    specific_moves = {
        # Main application files
        "src/voxsigil/": ["__init__.py"],
        # Keep current optimized GUI in main src
        "src/voxsigil/gui/": [
            "optimized_enhanced_gui.py",
            "proper_enhanced_gui.py",
            "standalone_enhanced_gui.py",
        ],
        # Build files
        "build/": ["requirements.in", "requirements.lock", "pyproject.toml"],
    }

    for target_dir, files in specific_moves.items():
        target_path = base_path / target_dir
        for file_name in files:
            file_path = base_path / file_name
            if file_path.exists():
                try:
                    destination = target_path / file_name
                    shutil.move(str(file_path), str(destination))
                    print(f"Moved {file_name} -> {target_dir}")
                except Exception as e:
                    print(f"Error moving {file_name}: {e}")


def create_main_launcher():
    """Create a main launcher script."""
    launcher_content = '''#!/usr/bin/env python3
"""
Main VoxSigil Application Launcher
Launches the optimized VoxSigil GUI application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Launch the VoxSigil application."""
    try:
        from voxsigil.gui.optimized_enhanced_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    with open("voxsigil_launcher.py", "w") as f:
        f.write(launcher_content)
    print("Created main launcher: voxsigil_launcher.py")


def create_project_readme():
    """Create an updated README file."""
    readme_content = """# VoxSigil Library

A comprehensive voice processing and TTS (Text-to-Speech) library with advanced GUI capabilities.

## Project Structure

```
├── src/voxsigil/          # Main application source code
│   ├── gui/               # GUI components
│   ├── core/              # Core functionality
│   ├── engines/           # TTS/STT engines
│   ├── agents/            # Voice agents
│   └── utils/             # Utility functions
├── dev/                   # Development files
│   ├── demos/             # Demo scripts
│   ├── diagnostics/       # Diagnostic tools
│   └── test_scripts/      # Test files
├── documentation/         # Project documentation
├── resources/             # Audio samples, configs, data
├── scripts/               # Utility and launcher scripts
├── archive/               # Old files and reports
└── build/                 # Build configuration files
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r build/requirements.lock
   ```

2. Launch the application:
   ```bash
   python voxsigil_launcher.py
   ```

## Development

- Demo scripts are in `dev/demos/`
- Test scripts are in `dev/test_scripts/`
- Diagnostic tools are in `dev/diagnostics/`

## Documentation

See the `documentation/` directory for guides and reports.
"""

    with open("README_NEW.md", "w") as f:
        f.write(readme_content)
    print("Created updated README: README_NEW.md")


def cleanup_empty_directories():
    """Remove empty directories after file moves."""
    base_path = Path(".")

    # Get all directories and sort by depth (deepest first)
    all_dirs = []
    for item in base_path.rglob("*"):
        if item.is_dir() and item != base_path:
            all_dirs.append(item)

    all_dirs.sort(key=lambda x: len(x.parts), reverse=True)

    for directory in all_dirs:
        try:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                print(f"Removed empty directory: {directory}")
        except OSError:
            # Directory not empty or permission error
            pass


def main():
    """Main cleanup function."""
    print("Starting VoxSigil Library cleanup...")
    print("=" * 50)

    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()

    # Move files by patterns
    print("\n2. Moving files by patterns...")
    move_files_by_pattern()

    # Move specific files
    print("\n3. Moving specific important files...")
    move_specific_files()

    # Create main launcher
    print("\n4. Creating main launcher...")
    create_main_launcher()

    # Create updated README
    print("\n5. Creating updated documentation...")
    create_project_readme()

    # Cleanup empty directories
    print("\n6. Cleaning up empty directories...")
    cleanup_empty_directories()

    print("\n" + "=" * 50)
    print("Cleanup completed successfully!")
    print("\nNext steps:")
    print("1. Review the new structure")
    print("2. Test the main launcher: python voxsigil_launcher.py")
    print("3. Update any import paths if needed")
    print("4. Remove old README and rename README_NEW.md to README.md")


if __name__ == "__main__":
    main()
