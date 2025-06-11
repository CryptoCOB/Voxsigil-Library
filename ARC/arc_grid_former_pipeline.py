#!/usr/bin/env python
"""
arc_grid_former_pipeline.py - Comprehensive training and optimization pipeline for GRID-Former

This script provides a complete pipeline for training, optimizing, and evaluating
GRID-Former models on ARC tasks, integrating all components of the system.
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path

import torch

# Add project root to path for easier imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import GRID-Former modules
from Gridformer.training.grid_model_trainer import GridFormerTrainer
from Gridformer.training.grid_sigil_handler import GridSigilHandler
from Gridformer.core.grid_former import GRID_Former
from Gridformer.core.hyperparameter_search import HyperparameterSearch
from Gridformer.core.grid_distillation import DistillationTrainer
from Gridformer.core.vantacore_grid_connector import GridFormerConnector


# Define a placeholder if quantizer not available
class GridFormerQuantizer:
    def __init__(self, *args, **kwargs):
        logger.warning("GridFormerQuantizer not available - using placeholder")

    def quantize_and_evaluate(self, *args, **kwargs):
        logger.warning("quantize_and_evaluate not available")
        return None, {"accuracy": 0}


from ARC.core.arc_data_processor import create_arc_dataloaders


# Import ARC LLM handler for distillation
from ARC.llm.arc_llm_interface import ARCAwareLLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("arc_grid_former_pipeline.log"),
    ],
)

logger = logging.getLogger("GRID-Former.Pipeline")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GRID-Former Complete Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--challenges",
        type=str,
        default="arc-agi_training_challenges.json",
        help="Path to ARC challenges JSON file",
    )

    parser.add_argument(
        "--solutions",
        type=str,
        default="arc-agi_training_solutions.json",
        help="Path to ARC solutions JSON file",
    )

    parser.add_argument(
        "--test-challenges",
        type=str,
        default="arc-agi_test_challenges.json",
        help="Path to ARC test challenges JSON file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./grid_former_pipeline",
        help="Directory for output storage",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for training"
    )

    parser.add_argument(
        "--hyperopt-trials",
        type=int,
        default=20,
        help="Number of hyperparameter optimization trials",
    )

    parser.add_argument(
        "--hyperopt-method",
        type=str,
        default="bayesian",
        choices=["grid", "random", "bayesian", "ray"],
        help="Hyperparameter optimization method",
    )

    parser.add_argument(
        "--skip-hyperopt", action="store_true", help="Skip hyperparameter optimization"
    )

    parser.add_argument(
        "--skip-distillation",
        action="store_true",
        help="Skip teacher-student distillation",
    )

    parser.add_argument(
        "--skip-quantization", action="store_true", help="Skip model quantization"
    )

    parser.add_argument(
        "--llm-name", type=str, default="mistral", help="LLM to use for distillation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu, etc.)",
    )

    parser.add_argument(
        "--output-sigil",
        type=str,
        default="grid_former_model.sigil",
        help="Path to save output model sigil",
    )

    parser.add_argument(
        "--create-vantacore-connector",
        action="store_true",
        help="Create VantaCore connector for the model",
    )

    return parser.parse_args()


def main():
    """Main pipeline execution"""
    start_time = time.time()
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_str = device  # Keep string version
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Save configuration
    config_path = output_dir / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Phase 1: Hyperparameter Optimization
    best_params = {}
    if not args.skip_hyperopt:
        logger.info("Phase 1: Hyperparameter Optimization")
        hyperopt_dir = output_dir / "hyperopt"
        hyperopt_dir.mkdir(exist_ok=True)
        optimizer = HyperparameterSearch(model_type="grid_former")

        # Define parameter search space
        optimizer.search_space = {
            "epochs": [min(args.epochs // 5, 10)],  # Short epochs for hyperopt
            "batch_size": [args.batch_size],
            "hidden_dim": [128, 256, 512],
            "num_layers": [4, 6, 8],
            "num_heads": [4, 8],
            "dropout": [0.1, 0.2],
            "learning_rate": [0.0001, 0.0005, 0.001],
            "weight_decay": [0.01, 0.001],
        }

        # Run hyperparameter search
        search_results = optimizer.run_search(num_experiments=args.hyperopt_trials)

        # Extract best parameters
        if (
            search_results
            and "best_experiment" in search_results
            and search_results["best_experiment"]
        ):
            best_exp_id = search_results["best_experiment"]
            for exp in search_results["experiments"]:
                if exp["experiment_id"] == best_exp_id:
                    best_params = exp["hyperparams"]
                    best_score = search_results.get("best_accuracy", 0)
                    break
        else:
            logger.warning(
                "Hyperparameter search did not return valid results. Using default parameters."
            )
            best_params = {
                "hidden_dim": 256,
                "num_layers": 6,
                "num_heads": 8,
                "dropout": 0.1,
                "learning_rate": 0.0005,
                "weight_decay": 0.01,
            }
            best_score = 0

        logger.info(
            f"Hyperparameter optimization complete. Best score: {best_score:.4f}"
        )
        logger.info(f"Best parameters: {best_params}")
    else:
        logger.info("Skipping hyperparameter optimization")
        # Default parameters
        best_params = {
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
        }

    # Phase 2: Full Model Training
    logger.info("Phase 2: Full Model Training")
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)

    # Load data for training
    logger.info(f"Loading ARC data from {args.challenges} and {args.solutions}")
    train_loader, val_loader = create_arc_dataloaders(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        batch_size=args.batch_size,
    )

    # Create model with optimized hyperparameters
    model = GRID_Former(
        hidden_dim=best_params.get("hidden_dim", 256),
        num_layers=best_params.get("num_layers", 6),
        num_heads=best_params.get("num_heads", 8),
        dropout=best_params.get("dropout", 0.1),
    ).to(device)
    trainer = GridFormerTrainer(
        model=model,
        learning_rate=best_params.get("learning_rate", 0.0005),
        weight_decay=best_params.get("weight_decay", 0.01),
        device=device_str,
        output_dir=str(model_dir),
    )
    # Train model
    logger.info(f"Training model for {args.epochs} epochs")
    history = trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=args.epochs
    )

    # Save trained model
    base_model_path = model_dir / "grid_former_base.pt"
    trainer._save_checkpoint(Path(base_model_path))
    logger.info(f"Saved base model to {base_model_path}")

    # Phase 3: Teacher-Student Distillation (if enabled)
    if not args.skip_distillation:
        logger.info("Phase 3: Teacher-Student Distillation")
        distill_dir = output_dir / "distillation"
        distill_dir.mkdir(exist_ok=True)  # Initialize LLM interface
        logger.info(f"Initializing LLM interface with model: {args.llm_name}")
        llm_interface = ARCAwareLLMInterface(model_path=f"models/{args.llm_name}")

        # Initialize distillation trainer
        distill_trainer = DistillationTrainer(
            student_model=model,  # Use the already trained model as starting point
            llm_interface=llm_interface,
            device=device_str,
            alpha=0.5,  # Weight for distillation loss
            temperature=2.0,  # Temperature for softening
            output_dir=str(distill_dir),
        )

        # Train with distillation
        logger.info("Running distillation training")
        distill_history = distill_trainer.train_with_distillation(
            challenges_path=args.challenges,
            solutions_path=args.solutions,
            batch_size=args.batch_size,
            num_epochs=args.epochs // 2,  # Fewer epochs for fine-tuning
            lr=best_params.get("learning_rate", 0.0005)
            / 5,  # Lower learning rate for fine-tuning
        )

        # Update model reference to the distilled one
        model = distill_trainer.student_model

        # Save distilled model
        distill_model_path = distill_dir / "grid_former_distilled.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "distillation_params": {
                    "alpha": distill_trainer.alpha,
                    "temperature": distill_trainer.temperature,
                },
            },
            distill_model_path,
        )
        logger.info(f"Saved distilled model to {distill_model_path}")
    else:
        logger.info("Skipping teacher-student distillation")

    # Phase 4: Model Quantization (if enabled)
    if not args.skip_quantization:
        logger.info("Phase 4: Model Quantization")
        quant_dir = output_dir / "quantized"
        quant_dir.mkdir(exist_ok=True)
        # Initialize quantizer
        quantizer = GridFormerQuantizer(
            model=model, device=device_str, quantized_model_dir=str(quant_dir)
        )

        # Apply dynamic quantization (faster inference, lower memory)
        logger.info("Applying dynamic quantization")
        dynamic_model, dynamic_metrics = quantizer.quantize_and_evaluate(
            eval_dataloader=val_loader,
            quantization_type="dynamic",
            model_name="grid_former_dynamic",
        )

        # Apply static quantization (more precise, requires calibration)
        logger.info("Applying static quantization")
        static_model, static_metrics = quantizer.quantize_and_evaluate(
            eval_dataloader=val_loader,
            quantization_type="static",
            model_name="grid_former_static",
        )

        # Compare quantization results
        logger.info(f"Dynamic quantization metrics: {dynamic_metrics}")
        logger.info(f"Static quantization metrics: {static_metrics}")

        # Choose best quantized model based on accuracy
        if dynamic_metrics.get("accuracy", 0) >= static_metrics.get("accuracy", 0):
            quantized_model = dynamic_model
            quant_type = "dynamic"
            logger.info("Selected dynamic quantization model")
        else:
            quantized_model = static_model
            quant_type = "static"
            logger.info("Selected static quantization model")

        # Save selected quantized model
        selected_model_path = quant_dir / f"grid_former_{quant_type}_selected.pt"
        torch.save(quantized_model, selected_model_path)
        logger.info(f"Saved selected quantized model to {selected_model_path}")

        # Update model reference
        model = quantized_model
    else:
        logger.info("Skipping model quantization")

    # Phase 5: Create Sigil and VantaCore Connector
    logger.info("Phase 5: Create Sigil and VantaCore Connector")

    # Generate sigil for the final model
    sigil_path = output_dir / args.output_sigil
    logger.info(f"Creating model sigil at {sigil_path}")

    # Create training summary
    training_summary = {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "train_accuracy": history["train_accuracy"][-1]
        if history["train_accuracy"]
        else None,
        "val_accuracy": history["val_accuracy"][-1]
        if history["val_accuracy"]
        else None,
        "epochs_trained": len(history["train_loss"]) if history["train_loss"] else 0,
        "hyperparameters": best_params,
        "distillation_applied": not args.skip_distillation,
        "quantization_applied": not args.skip_quantization,
    }  # Generate the model sigil
    sigil_handler = GridSigilHandler()
    if model is not None:
        sigil = sigil_handler.save_model_to_sigil(
            model=model,
            training_info=training_summary,
            metadata={"model_id": f"grid_former_pipeline_{int(time.time())}"},
        )
    else:
        logger.warning("Model is None, creating empty sigil")
        sigil = {
            "SigilType": "GRID-Former-Model",
            "Version": "1.0",
            "Content": {
                "metadata": {"model_id": f"grid_former_pipeline_{int(time.time())}"},
                "training_info": training_summary,
            },
        }

    # Save sigil to file
    with open(sigil_path, "w") as f:
        json.dump(sigil, f, indent=2)
    logger.info(f"Saved model sigil to {sigil_path}")

    # Create VantaCore connector if requested
    if args.create_vantacore_connector:
        logger.info("Creating VantaCore connector")
        connector_dir = output_dir / "vantacore"
        connector_dir.mkdir(exist_ok=True)
        connector = GridFormerConnector(
            default_model_path=str(base_model_path),
            device=device_str,
            hidden_dim=best_params.get("hidden_dim", 256),
            num_layers=best_params.get("num_layers", 6),
            num_heads=best_params.get("num_heads", 8),
        )

        # Test connector on a sample ARC task
        logger.info("Testing connector on sample task")
        with open(args.test_challenges, "r") as f:
            test_challenges = json.load(f)

        # Select first test task
        sample_task_id = list(test_challenges.keys())[0]
        sample_task = test_challenges[sample_task_id]
        # Predict solution
        prediction_results = connector.handle_arc_task(sample_task)
        prediction = prediction_results.get("predictions", [{}])[0].get(
            "predicted_grid", []
        )

        # Save prediction
        prediction_path = connector_dir / f"sample_prediction_{sample_task_id}.json"
        with open(prediction_path, "w") as f:
            json.dump(
                {"task_id": sample_task_id, "prediction": prediction}, f, indent=2
            )
        logger.info(f"Saved sample prediction to {prediction_path}")

        # Save connector configuration
        connector_config = {
            "model_path": str(base_model_path),
            "parameters": best_params,
            "created_at": time.time(),
            "sigil_path": str(sigil_path),
        }

        connector_config_path = connector_dir / "connector_config.json"
        with open(connector_config_path, "w") as f:
            json.dump(connector_config, f, indent=2)
        logger.info(f"Saved connector configuration to {connector_config_path}")

    # Complete
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline complete! Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
