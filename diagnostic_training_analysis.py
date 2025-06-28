#!/usr/bin/env python
"""
Comprehensive Training Pipeline Diagnostic Analysis
Verifies ARC data usage, VantaCore data generation, component training, and GPU utilization
"""

import logging
import os
import sys
from pathlib import Path

# Add the library path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainingDiagnostic")


def check_arc_data_availability():
    """Check if real ARC data is available"""
    logger.info("=== ARC Data Availability Check ===")

    possible_paths = [
        "./arc_data",
        "../arc_data",
        "./data/arc",
        "../data/arc",
        "./ARC/data",
        "./training/arc_data",
    ]

    found_data = False
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found potential ARC data directory: {path}")
            try:
                contents = os.listdir(path)
                logger.info(f"   Contents: {contents}")

                # Check for typical ARC files
                arc_files = ["training.json", "evaluation.json", "test.json"]
                for arc_file in arc_files:
                    file_path = os.path.join(path, arc_file)
                    if os.path.exists(file_path):
                        logger.info(f"   ✅ Found {arc_file}")
                        found_data = True
                    else:
                        logger.info(f"   ❌ Missing {arc_file}")
            except Exception as e:
                logger.error(f"   Error accessing {path}: {e}")
        else:
            logger.info(f"❌ No data at: {path}")

    if not found_data:
        logger.warning("⚠️ No real ARC dataset found - training will use mock data")

    return found_data


def check_vantacore_data_generation():
    """Check VantaCore data generation capabilities"""
    logger.info("\n=== VantaCore Data Generation Check ===")

    try:
        from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore

        logger.info("✅ VantaCore imported successfully")

        # Initialize VantaCore
        vanta_core = VantaCore(config={}, device="cpu", initialize_subsystems=False)
        logger.info("✅ VantaCore initialized")

        # Check if VantaCore has data generation methods
        data_gen_methods = [
            "generate_training_data",
            "create_augmented_data",
            "generate_synthetic_samples",
            "enhance_dataset",
            "generate_data",
        ]

        found_methods = []
        for method in data_gen_methods:
            if hasattr(vanta_core, method):
                found_methods.append(method)
                logger.info(f"✅ Found data generation method: {method}")

        if not found_methods:
            logger.warning("⚠️ No explicit data generation methods found in VantaCore")
            # Check for general methods that might handle data generation
            general_methods = ["process", "enhance", "transform", "augment"]
            for method in general_methods:
                if hasattr(vanta_core, method):
                    logger.info(f"✅ Found general processing method: {method}")

        return len(found_methods) > 0

    except Exception as e:
        logger.error(f"❌ Error checking VantaCore: {e}")
        return False


def check_component_training_status():
    """Check which components are available for training"""
    logger.info("\n=== Component Training Status Check ===")

    components = {
        "ARCGridTrainer": "training.arc_grid_trainer",
        "GRID_Former": "core.grid_former",
        "BLTEncoder": "BLT.blt_encoder",
        "NovelReasoning": "core.novel_reasoning",
        "HOLOMesh": "agents.holo_mesh",
        "ART": "ART.art_trainer",
        "VoxSigilRAG": "VoxSigilRag.voxsigil_blt_rag",
        "AdaptiveMemory": "core.novel_efficiency.adaptive_memory",
    }

    available_components = []

    for component_name, module_path in components.items():
        try:
            module = __import__(module_path, fromlist=[""])
            logger.info(f"✅ {component_name} available")
            available_components.append(component_name)

            # Check for training methods
            training_methods = ["train", "fit", "update", "learn"]
            has_training = False
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "__call__"):
                    for method in training_methods:
                        if hasattr(attr, method):
                            has_training = True
                            logger.info(f"   ✅ Has training method: {method}")
                            break

            if not has_training:
                logger.warning("   ⚠️ No obvious training methods found")

        except Exception as e:
            logger.warning(f"❌ {component_name} not available: {e}")

    logger.info(f"\nAvailable components for training: {len(available_components)}/8")
    return available_components


def check_gpu_utilization():
    """Check GPU availability and utilization setup"""
    logger.info("\n=== GPU Utilization Check ===")

    try:
        import torch

        logger.info("✅ PyTorch available")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available with {device_count} GPU(s)")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (
                    1024**3
                )
                logger.info(f"   GPU {i}: {device_name}")
                logger.info(
                    f"     Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB"
                )

            return True
        else:
            logger.warning("⚠️ CUDA not available - training will use CPU")
            return False

    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False


def check_training_pipeline():
    """Check the complete training pipeline"""
    logger.info("\n=== Training Pipeline Check ===")

    try:
        from training.arc_grid_trainer import ARCGridTrainer

        logger.info("✅ ARCGridTrainer imported")

        # Check if trainer has all necessary methods
        trainer_methods = [
            "start_coordinated_training",
            "train",
            "_initialize_vanta_core",
            "_initialize_novel_paradigms",
            "_initialize_holo_mesh",
            "_initialize_art",
        ]

        missing_methods = []
        for method in trainer_methods:
            if not hasattr(ARCGridTrainer, method):
                missing_methods.append(method)

        if missing_methods:
            logger.warning(f"⚠️ Missing methods in ARCGridTrainer: {missing_methods}")
        else:
            logger.info("✅ All required training methods available")

        return len(missing_methods) == 0

    except Exception as e:
        logger.error(f"❌ Error checking training pipeline: {e}")
        return False


def analyze_mock_vs_real_data_flow():
    """Analyze whether the system is using mock or real data"""
    logger.info("\n=== Data Flow Analysis ===")

    try:
        from ARC.arc_data_processor import ARCGridDataProcessor

        logger.info("✅ ARC data processor available")

        # Try to create real data loaders
        try:
            processor = ARCGridDataProcessor()
            logger.info("✅ ARCGridDataProcessor created")

            # Check for real data paths
            test_paths = ["./arc_data/training", "./data/arc/training.json"]
            for path in test_paths:
                if os.path.exists(path):
                    logger.info(f"✅ Real data path exists: {path}")
                    return "REAL_DATA"
                else:
                    logger.info(f"❌ Real data path missing: {path}")

            logger.warning("⚠️ No real data paths found - will use mock data")
            return "MOCK_DATA"

        except Exception as e:
            logger.error(f"❌ Error creating data loaders: {e}")
            return "ERROR"

    except Exception as e:
        logger.error(f"❌ ARC data processor not available: {e}")
        return "UNAVAILABLE"


def main():
    """Run comprehensive diagnostic analysis"""
    logger.info("🔍 Starting Comprehensive Training Pipeline Diagnostic")
    logger.info("=" * 60)

    # Run all checks
    results = {
        "arc_data_available": check_arc_data_availability(),
        "vantacore_data_generation": check_vantacore_data_generation(),
        "available_components": check_component_training_status(),
        "gpu_available": check_gpu_utilization(),
        "training_pipeline_ready": check_training_pipeline(),
        "data_flow_status": analyze_mock_vs_real_data_flow(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"🗃️  ARC Data Available: {'✅ YES' if results['arc_data_available'] else '❌ NO'}"
    )
    logger.info(
        f"🏭 VantaCore Data Gen: {'✅ YES' if results['vantacore_data_generation'] else '❌ NO'}"
    )
    logger.info(f"🧩 Components Ready: {len(results['available_components'])}/8")
    logger.info(
        f"🚀 GPU Available: {'✅ YES' if results['gpu_available'] else '❌ NO'}"
    )
    logger.info(
        f"🔧 Pipeline Ready: {'✅ YES' if results['training_pipeline_ready'] else '❌ NO'}"
    )
    logger.info(f"📊 Data Flow: {results['data_flow_status']}")

    # Recommendations
    logger.info("\n📋 RECOMMENDATIONS:")

    if not results["arc_data_available"]:
        logger.info(
            "📥 Download real ARC dataset using: python core/download_arc_data.py"
        )

    if not results["vantacore_data_generation"]:
        logger.info("🏭 Verify VantaCore data generation methods are implemented")

    if len(results["available_components"]) < 6:
        logger.info("🧩 Some training components are missing - verify imports")

    if not results["gpu_available"]:
        logger.info("🚀 Install CUDA-enabled PyTorch for GPU acceleration")

    if results["data_flow_status"] == "MOCK_DATA":
        logger.info(
            "⚠️  System will use mock data - consider downloading real ARC dataset"
        )

    logger.info("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()
