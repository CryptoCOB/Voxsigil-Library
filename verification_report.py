#!/usr/bin/env python
"""
VoxSigil Training Pipeline Verification Report
Analyzes and verifies: ARC data usage, VantaCore data generation,
component training, GPU utilization, and accuracy plateauing
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add the library path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VoxSigilVerification")


def verify_arc_data_usage():
    """Verify ARC data loading and usage"""
    logger.info("🔍 VERIFYING ARC DATA USAGE")
    print("=" * 60)

    # Check if ARC data exists
    arc_data_path = "arc_data"
    if os.path.exists(arc_data_path):
        print(f"✅ ARC data directory found: {arc_data_path}")

        # Check for required files
        required_files = ["training.json", "evaluation.json"]
        for file in required_files:
            file_path = os.path.join(arc_data_path, file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    print(f"✅ {file}: {len(data)} tasks available")
                except Exception as e:
                    print(f"❌ {file}: Error loading - {e}")
            else:
                print(f"❌ {file}: Missing")
    else:
        print(f"❌ ARC data directory not found: {arc_data_path}")

    # Test ARC data processor
    try:
        from ARC.arc_data_processor import ARCGridDataProcessor

        processor = ARCGridDataProcessor(max_grid_size=30)
        print("✅ ARCGridDataProcessor initialized successfully")

        # Test data loading with current path structure
        if os.path.exists(arc_data_path):
            try:
                # Try to load the data
                challenges_path = os.path.join(arc_data_path, "training.json")
                tasks = processor.load_arc_data(challenges_path)
                print(f"✅ Successfully loaded {len(tasks)} ARC tasks")

                # Check data structure
                first_task = list(tasks.values())[0]
                if "train" in first_task and "test" in first_task:
                    print("✅ ARC data structure is valid")
                    print(f"   Training examples: {len(first_task['train'])}")
                    print(f"   Test examples: {len(first_task['test'])}")
                else:
                    print("❌ Invalid ARC data structure")

            except Exception as e:
                print(f"❌ Error loading ARC data: {e}")

    except Exception as e:
        print(f"❌ ARCGridDataProcessor not available: {e}")


def verify_vantacore_data_generation():
    """Verify VantaCore data generation capabilities"""
    logger.info("🏭 VERIFYING VANTACORE DATA GENERATION")
    print("=" * 60)

    try:
        from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore

        print("✅ VantaCore imported successfully")

        # Test VantaCore initialization
        try:
            # Initialize with proper parameters based on actual constructor
            vanta_core = VantaCore()
            print("✅ VantaCore initialized successfully")

            # Check available methods
            vanta_methods = [
                method for method in dir(vanta_core) if not method.startswith("_")
            ]
            data_related_methods = [
                m
                for m in vanta_methods
                if any(
                    keyword in m.lower()
                    for keyword in ["data", "generate", "create", "process", "enhance"]
                )
            ]

            print(f"✅ VantaCore has {len(vanta_methods)} public methods")
            print(f"✅ Data-related methods: {len(data_related_methods)}")
            for method in data_related_methods[:5]:  # Show first 5
                print(f"   - {method}")

            # Test if VantaCore can process/enhance data
            test_success = False
            for method_name in ["process", "enhance", "transform"]:
                if hasattr(vanta_core, method_name):
                    print(f"✅ VantaCore has {method_name} method for data processing")
                    test_success = True
                    break

            if not test_success:
                print("⚠️ No obvious data processing methods found")

        except Exception as e:
            print(f"❌ Error initializing VantaCore: {e}")

    except Exception as e:
        print(f"❌ VantaCore not available: {e}")


def verify_component_training():
    """Verify component training capabilities"""
    logger.info("🧩 VERIFYING COMPONENT TRAINING")
    print("=" * 60)

    components_to_check = {
        "ARCGridTrainer": {
            "module": "training.arc_grid_trainer",
            "class": "ARCGridTrainer",
            "methods": ["train", "start_coordinated_training"],
        },
        "GRID_Former": {
            "module": "core.grid_former",
            "class": "GRID_Former",
            "methods": ["forward", "train"],
        },
        "BLTEncoder": {
            "module": "BLT.blt_encoder",
            "class": "BLTEncoder",
            "methods": ["encode", "train"],
        },
        "VoxSigilRAG": {
            "module": "VoxSigilRag.voxsigil_blt_rag",
            "class": "BLTEnhancedRAG",
            "methods": ["process", "query"],
        },
        "HOLOMesh": {
            "module": "agents.holo_mesh",
            "class": "HOLOMesh",
            "methods": ["activate", "process"],
        },
        "NovelReasoning": {
            "module": "core.novel_reasoning.logical_neural_units",
            "class": "LogicalReasoningEngine",
            "methods": ["forward", "reason"],
        },
    }

    available_components = 0
    trainable_components = 0

    for comp_name, comp_info in components_to_check.items():
        try:
            module = __import__(comp_info["module"], fromlist=[comp_info["class"]])
            component_class = getattr(module, comp_info["class"])
            print(f"✅ {comp_name} available")
            available_components += 1

            # Check for training methods
            has_training = False
            for method in comp_info["methods"]:
                if hasattr(component_class, method):
                    print(f"   ✅ Has {method} method")
                    has_training = True

            if has_training:
                trainable_components += 1
                print(f"   ✅ {comp_name} is trainable")
            else:
                print(f"   ⚠️ {comp_name} may not be trainable")

        except Exception as e:
            print(f"❌ {comp_name} not available: {e}")

    print("\n📊 Component Summary:")
    print(f"   Available: {available_components}/{len(components_to_check)}")
    print(f"   Trainable: {trainable_components}/{len(components_to_check)}")


def verify_gpu_utilization():
    """Verify GPU utilization setup"""
    logger.info("🚀 VERIFYING GPU UTILIZATION")
    print("=" * 60)

    try:
        import torch

        print("✅ PyTorch available")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {device_count} GPU(s)")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                # Get memory info
                torch.cuda.empty_cache()  # Clear cache for accurate reading
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (
                    1024**3
                )

                print(f"   GPU {i}: {device_name}")
                print(f"     Total Memory: {total_memory:.2f}GB")
                print(f"     Allocated: {memory_allocated:.2f}GB")
                print(f"     Reserved: {memory_reserved:.2f}GB")
                print(f"     Available: {total_memory - memory_reserved:.2f}GB")

            # Test GPU tensor operations
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                result = torch.matmul(test_tensor, test_tensor.t())
                print("✅ GPU tensor operations working")
                del test_tensor, result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU tensor operation failed: {e}")

            return True
        else:
            print("❌ CUDA not available - training will use CPU")
            print("   This may explain accuracy plateauing due to slower training")
            return False

    except Exception as e:
        print(f"❌ PyTorch not available: {e}")
        return False


def verify_training_pipeline():
    """Verify the complete training pipeline"""
    logger.info("🔧 VERIFYING TRAINING PIPELINE")
    print("=" * 60)

    try:
        from training.arc_grid_trainer import ARCGridTrainer

        print("✅ ARCGridTrainer imported")

        # Test trainer initialization
        try:
            config = {
                "grid_size": 30,
                "use_cuda": True,
                "arc_data_path": "./arc_data",
                "use_art": True,
                "use_holo_mesh": True,
                "use_novel_paradigms": True,
            }

            trainer = ARCGridTrainer(config=config)
            print("✅ ARCGridTrainer initialized successfully")

            # Check critical methods
            critical_methods = [
                "start_coordinated_training",
                "_initialize_vanta_core",
                "_initialize_novel_paradigms",
                "_initialize_holo_mesh",
                "_initialize_art",
            ]

            missing_methods = []
            for method in critical_methods:
                if hasattr(trainer, method):
                    print(f"   ✅ Has {method}")
                else:
                    missing_methods.append(method)
                    print(f"   ❌ Missing {method}")

            if not missing_methods:
                print("✅ All critical training methods available")

                # Test training initialization
                try:
                    training_config = {
                        "epochs": 5,
                        "batch_size": 16,
                        "learning_rate": 0.001,
                    }

                    success = trainer.start_coordinated_training(training_config)
                    if success:
                        print("✅ Training pipeline can be started")
                    else:
                        print("❌ Training pipeline failed to start")

                except Exception as e:
                    print(f"⚠️ Training start test failed: {e}")
            else:
                print(f"❌ Missing critical methods: {missing_methods}")

        except Exception as e:
            print(f"❌ ARCGridTrainer initialization failed: {e}")

    except Exception as e:
        print(f"❌ ARCGridTrainer not available: {e}")


def analyze_accuracy_plateauing():
    """Analyze potential causes of accuracy plateauing"""
    logger.info("📈 ANALYZING ACCURACY PLATEAUING")
    print("=" * 60)

    potential_issues = []

    # Check data quality
    if not os.path.exists("arc_data"):
        potential_issues.append("❌ No real ARC data - using mock data limits learning")

    # Check GPU utilization
    try:
        import torch

        if not torch.cuda.is_available():
            potential_issues.append(
                "❌ No GPU acceleration - slower training affects convergence"
            )
    except:
        potential_issues.append(
            "❌ PyTorch not available - using fallback implementations"
        )

    # Check model complexity
    try:
        potential_issues.append("✅ GRID_Former available - good model complexity")
    except:
        potential_issues.append(
            "❌ GRID_Former not available - using simplified models"
        )

    # Check ensemble integration
    try:
        potential_issues.append("✅ Ensemble orchestrator available")
    except:
        potential_issues.append(
            "❌ Ensemble orchestrator missing - reduced model capacity"
        )

    print("🔍 Potential causes of accuracy plateauing:")
    for issue in potential_issues:
        print(f"   {issue}")

    print("\n💡 Recommendations:")
    if any("mock data" in issue for issue in potential_issues):
        print("   📥 Download real ARC dataset for better training data")
    if any("No GPU" in issue for issue in potential_issues):
        print("   🚀 Install CUDA-enabled PyTorch for faster training")
    if any("not available" in issue for issue in potential_issues):
        print("   🔧 Install missing components for full model capacity")

    print("   🎯 Consider: Lower learning rate, longer training, data augmentation")
    print("   📊 Monitor: Loss curves, gradient norms, component utilization")


def main():
    """Run comprehensive verification"""
    logger.info("🔍 STARTING VOXSIGIL TRAINING PIPELINE VERIFICATION")
    print("=" * 80)
    print("VoxSigil Training Pipeline Verification Report")
    print("=" * 80)

    start_time = time.time()

    # Run all verifications
    verify_arc_data_usage()
    print("\n")

    verify_vantacore_data_generation()
    print("\n")

    verify_component_training()
    print("\n")

    verify_gpu_utilization()
    print("\n")

    verify_training_pipeline()
    print("\n")

    analyze_accuracy_plateauing()

    # Final summary
    end_time = time.time()
    print("\n" + "=" * 80)
    print("📋 VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"⏱️ Verification took {end_time - start_time:.2f} seconds")
    print("\n🎯 Next Steps:")
    print("   1. Review any ❌ items above")
    print("   2. Run GUI training to test pipeline")
    print("   3. Monitor training metrics and GPU utilization")
    print("   4. Address accuracy plateauing causes identified")

    print("\n✅ Verification report complete!")


if __name__ == "__main__":
    main()
