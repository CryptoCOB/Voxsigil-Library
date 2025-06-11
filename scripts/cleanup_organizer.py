#!/usr/bin/env python3
"""
Voxsigil Library Cleanup and Organization Script
This script moves files from the root directory to their proper organized folders.
"""

import logging
import os
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for organization."""
    directories = [
        "vmb",
        "engines", 
        "config",
        "logs",
        "docs/debug",
        "scripts/production"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created/verified directory: {directory}")

def move_files():
    """Move files to their proper directories."""
    
    file_moves = {
        # VMB files to vmb/
        "vmb": [
            "vmb_config_status.py",
            "vmb_status.py", 
            "vmb_advanced_demo.py",
            "vmb_completion_report.py",
            "vmb_final_status.py",
            "vmb_import_test.py",
            "vmb_production_executor.py"
        ],
        
        # Engine files to engines/
        "engines": [
            "async_processing_engine.py",
            "async_stt_engine.py", 
            "async_training_engine.py",
            "async_tts_engine.py",
            "cat_engine.py",
            "hybrid_cognition_engine.py",
            "rag_compression_engine.py",
            "tot_engine.py"
        ],
        
        # VANTA files to Vanta/
        "Vanta": [
            "vanta_integration.py",
            "vanta_mesh_graph.py",
            "vanta_orchestrator.py", 
            "vanta_runner.py",
            "vantacore_grid_connector.py",
            "vantacore_grid_former_integration.py",
            "vantacore_integration_starter.py"
        ],
        
        # Interface files to interfaces/
        "interfaces": [
            "art_interface.py",
            "memory_interface.py",
            "neural_interface.py",
            "rag_interface.py",
            "supervisor_connector_interface.py",
            "hybrid_middleware_interface.py"
        ],
        
        # Training files to training/
        "training": [
            "async_training_engine.py",
            "finetune_pipeline.py",
            "hyperparameter_search.py",
            "minimal_training_example.py",
            "mistral_finetune.py",
            "phi2_finetune.py",
            "tinyllama_voxsigil_finetune.py",
            "optimize_tinyllama_memory.py"
        ],
        
        # Test files to test/
        "test": [
            "test_integration.py",
            "test_step4.py",
            "quick_step4_test.py",
            "agent_validation.py",
            "gui_validation.py",
            "validate.py"
        ],
        
        # Config files to config/
        "config": [
            "production_config.py",
            "imports.py"
        ],
        
        # Documentation to docs/
        "docs": [
            "README.md",
            "AGENTS.md",
            "BUG_HUNT_VANTA.md"
        ],
        
        # Debug files to docs/debug/
        "docs/debug": [
            "debug_log_voxsigil.md",
            "debug_log_voxsigil_phase_2.md", 
            "debug_fix_phase_2.patch",
            "debug_summary_phase_2.json"
        ],
        
        # Log files to logs/
        "logs": [
            "agent_status.log",
            "gui_validation.log",
            "vantacore_grid_former_integration.log",
            "migrated_gui_status.md"
        ],
        
        # Utils 
        "utils": [
            "visualization_utils.py",
            "path_helper.py",
            "numpy_resolver.py",
            "sleep_time_compute.py",
            "data_loader.py"
        ],
        
        # Scripts to scripts/
        "scripts": [
            "create_all_components.py",
            "run_vantacore_grid_connector.py"
        ]
    }
    
    for target_dir, files in file_moves.items():
        for filename in files:
            source_path = Path(filename)
            if source_path.exists():
                target_path = Path(target_dir) / filename
                
                try:
                    # Copy instead of move to preserve original during development
                    shutil.copy2(source_path, target_path)
                    logger.info(f"✓ Copied {filename} → {target_dir}/")
                except Exception as e:
                    logger.error(f"✗ Failed to copy {filename}: {e}")
            else:
                logger.warning(f"⚠ File not found: {filename}")

def update_imports():
    """Update import statements in moved files."""
    logger.info("📝 Import updates would be handled in a separate phase")
    
    # This would involve:
    # 1. Scanning moved files for import statements
    # 2. Updating relative imports based on new locations  
    # 3. Adding __init__.py files where needed
    # 4. Updating any hardcoded paths

def main():
    """Main cleanup execution."""
    logger.info("🧹 Starting Voxsigil Library Cleanup")
    
    # Create directory structure
    create_directories()
    
    # Move files to organized structure
    move_files()
    
    # Note about import updates
    update_imports()
    
    logger.info("✅ Cleanup organization complete!")
    logger.info("📋 Next steps:")
    logger.info("   1. Verify moved files work correctly")
    logger.info("   2. Update import statements as needed")
    logger.info("   3. Remove original files after verification")
    logger.info("   4. Update documentation and README files")

if __name__ == "__main__":
    main()
