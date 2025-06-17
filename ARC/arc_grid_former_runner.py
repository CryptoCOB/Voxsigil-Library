#!/usr/bin/env python
"""
arc_grid_former_runner.py - Main entry point for GRID-Former ARC training and evaluation

This script provides a command-line interface for training and evaluating
GRID-Former models on ARC tasks, with integration into the VoxSigil ecosystem.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from core.grid_former import GRID_Former
from core.grid_sigil_handler import GridSigilHandler
from core.vantacore_grid_connector import GridFormerConnector
from training.gridformer_training import GridFormerTrainer

# Import GRID-Former modules
from .core.arc_data_processor import create_arc_dataloaders

# Add project root to path for easier imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("arc_grid_former.log")],
)

logger = logging.getLogger("GRID-Former.Runner")


def parse_arguments():
    parser = argparse.ArgumentParser(description="GRID-Former ARC Training and Evaluation")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict"],
        help="Operation mode (train, evaluate, or predict)",
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
        help="Path to ARC test challenges JSON file for evaluation",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./grid_former_models",
        help="Directory for model storage",
    )

    parser.add_argument("--load-model", type=str, help="Path to model checkpoint to load")

    parser.add_argument("--task-id", type=str, help="Specific ARC task ID to process")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")

    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for model")

    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")

    parser.add_argument(
        "--learning-rate", type=float, default=0.0005, help="Learning rate for training"
    )

    parser.add_argument(
        "--vanta-integration", action="store_true", help="Enable VantaCore integration"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu, etc.)",
    )

    parser.add_argument("--output-sigil", type=str, help="Path to save output model sigil")

    parser.add_argument(
        "--input-grid",
        type=str,
        help="Path to JSON file containing input grid for prediction",
    )

    parser.add_argument("--output-grid", type=str, help="Path to save prediction output grid")

    return parser.parse_args()


def load_task_data(
    challenges_path: str,
    solutions_path: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load ARC task data from JSON files.

    Args:
        challenges_path: Path to challenges JSON file
        solutions_path: Path to solutions JSON file
        task_id: Optional specific task ID to load

    Returns:
        Dictionary with task data
    """
    # Load challenges
    with open(challenges_path, "r") as f:
        challenges = json.load(f)

    # Load solutions if provided
    solutions = None
    if solutions_path and os.path.exists(solutions_path):
        with open(solutions_path, "r") as f:
            solutions = json.load(f)

    # Filter to specific task if requested
    if task_id and task_id in challenges:
        filtered_challenges = {task_id: challenges[task_id]}
        filtered_solutions = (
            {task_id: solutions[task_id]} if solutions and task_id in solutions else None
        )
        return {
            "challenges": filtered_challenges,
            "solutions": filtered_solutions,
            "task_id": task_id,
        }

    return {"challenges": challenges, "solutions": solutions}


def train_model(args):
    """Train a GRID-Former model on ARC data"""
    logger.info(f"Starting training with args: {args}")

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading ARC data from {args.challenges} and {args.solutions}")
    train_loader, val_loader = create_arc_dataloaders(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        batch_size=args.batch_size,
    )
    # Initialize model and trainer
    if args.load_model and os.path.exists(args.load_model):
        logger.info(f"Loading model from {args.load_model}")
        model = GRID_Former.load_from_file(args.load_model, device=str(device))
    else:
        logger.info("Creating new model")
        model = GRID_Former(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    trainer = GridFormerTrainer(
        model=model,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=str(device),
        output_dir=args.model_dir,
    )

    # Train model
    history = trainer.train(
        train_loader=train_loader, val_loader=val_loader, num_epochs=args.epochs
    )

    # Save final model
    model_path = os.path.join(args.model_dir, "final_grid_former.pt")
    trainer._save_checkpoint(Path(model_path))
    logger.info(f"Saved final model to {model_path}")
    # Create sigil if requested
    if args.output_sigil:
        logger.info(f"Creating model sigil at {args.output_sigil}")
        # Create sigil handler
        sigil_handler = GridSigilHandler()

        # Create sigil
        run_id = f"grid_former_{int(time.time())}"  # Generate a unique run ID

        sigil_data = sigil_handler.save_model_to_sigil(
            model=model,
            training_info={
                "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "train_accuracy": history["train_accuracy"][-1]
                if history["train_accuracy"]
                else None,
                "val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
                "epochs_trained": len(history["train_loss"]) if history["train_loss"] else 0,
            },
            metadata={"model_id": run_id},
            save_path=args.output_sigil,
        )

    logger.info("Training completed successfully")
    logger.info("sigil_data: %s", sigil_data if 'sigil_data' in locals() else "No sigil created")


def evaluate_model(args):
    """Evaluate a GRID-Former model on ARC data"""
    logger.info(f"Starting evaluation with args: {args}")

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load model
    if not args.load_model or not os.path.exists(args.load_model):
        logger.error("Must provide --load-model for evaluation")
        return

    # Initialize connector with model
    connector = GridFormerConnector(
        model_dir=args.model_dir, default_model_path=args.load_model, device=str(device)
    )

    # Load evaluation data
    task_data = load_task_data(
        challenges_path=args.test_challenges or args.challenges,
        solutions_path=args.solutions,
        task_id=args.task_id,
    )

    challenges = task_data["challenges"]
    solutions = task_data["solutions"]

    # Initialize metrics
    total_tasks = 0
    correct_tasks = 0

    # Process each task
    for task_id, challenge in challenges.items():
        logger.info(f"Evaluating task {task_id}")

        # Get train and test pairs
        train_pairs = challenge.get("train", [])
        test_pairs = challenge.get("test", [])

        # Skip tasks without test pairs or solutions
        if not test_pairs or not solutions or task_id not in solutions:
            logger.warning(f"Skipping task {task_id}: missing test pairs or solutions")
            continue

        # Prepare task data
        task = {"train": train_pairs, "test": test_pairs}

        # Process the task
        result = connector.handle_arc_task(task, task_id=task_id)

        # Calculate accuracy
        task_correct = 0
        task_total = 0

        for i, prediction_data in enumerate(result["predictions"]):
            if i >= len(solutions[task_id]):
                continue

            predicted_grid = prediction_data["predicted_grid"]
            solution_grid = solutions[task_id][i]

            # Check if prediction matches solution
            correct = np.array_equal(np.array(predicted_grid), np.array(solution_grid))
            if correct:
                task_correct += 1
            task_total += 1

            logger.info(f"Task {task_id}, Test {i}: {'✓' if correct else '✗'}")

        # Log task results
        task_accuracy = task_correct / task_total if task_total > 0 else 0
        logger.info(f"Task {task_id} accuracy: {task_accuracy:.4f} ({task_correct}/{task_total})")

        # Update global metrics
        if task_correct == task_total and task_total > 0:
            correct_tasks += 1
        total_tasks += 1

    # Log overall results
    overall_accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
    logger.info(f"Overall task accuracy: {overall_accuracy:.4f} ({correct_tasks}/{total_tasks})")


def predict_with_model(args):
    """Make predictions with a GRID-Former model"""
    logger.info(f"Starting prediction with args: {args}")

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load model
    if not args.load_model or not os.path.exists(args.load_model):
        logger.error("Must provide --load-model for prediction")
        return

    # Initialize connector with model
    connector = GridFormerConnector(
        model_dir=args.model_dir, default_model_path=args.load_model, device=str(device)
    )

    # Load input grid
    if not args.input_grid:
        logger.error("Must provide --input-grid for prediction")
        return

    with open(args.input_grid, "r") as f:
        input_grid = json.load(f)

    # Make prediction
    output_grid = connector.predict(input_grid)

    # Save output grid if requested
    if args.output_grid:
        with open(args.output_grid, "w") as f:
            json.dump(output_grid.tolist(), f, indent=2)
        logger.info(f"Saved prediction to {args.output_grid}")

    # Print prediction
    logger.info(f"Input grid shape: {np.array(input_grid).shape}")
    logger.info(f"Predicted grid shape: {output_grid.shape}")
    logger.info(f"Prediction: {output_grid.tolist()}")


def main():
    args = parse_arguments()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "predict":
        predict_with_model(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
