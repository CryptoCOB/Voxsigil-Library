#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete ARC fine-tuning pipeline

This script runs the complete fine-tuning pipeline for a model on the ARC dataset.
"""

import argparse
import logging
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and return the result."""
    logger.info(f"{description}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        logger.info(f"{description} completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"{description} failed: {result.stderr}")
    
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": end_time - start_time,
    }

def setup_environment():
    """Set up the Python environment with required packages."""
    cmd = ["python", "setup_environment.py"]
    return run_command(cmd, "Setting up environment")

def finetune_model(model_type, epochs, batch_size, learning_rate, run_id):
    """Fine-tune a model on the ARC dataset."""
    # Determine script and dataset based on model type
    if model_type == "phi-2":
        script = "phi2_finetune.py"
        dataset = "voxsigil_finetune/data/phi-2/arc_training_phi-2.jsonl"
    elif model_type == "mistral-7b":
        script = "mistral_finetune.py"
        dataset = "voxsigil_finetune/data/mistral-7b/arc_training_mistral-7b.jsonl"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create output directory
    output_dir = f"models/{model_type}_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python",
        script,
        f"--dataset={dataset}",
        f"--output_dir={output_dir}",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--learning_rate={learning_rate}",
    ]
    
    result = run_command(cmd, f"Fine-tuning {model_type}")
    result["output_dir"] = output_dir
    
    return result

def evaluate_model(model_path, model_type, run_id):
    """Evaluate a fine-tuned model on the ARC evaluation dataset."""
    # Determine evaluation dataset based on model type
    if model_type == "phi-2":
        eval_dataset = "voxsigil_finetune/data/phi-2/arc_evaluation_phi-2.jsonl"
    elif model_type == "mistral-7b":
        eval_dataset = "voxsigil_finetune/data/mistral-7b/arc_evaluation_mistral-7b.jsonl"
    else:
        eval_dataset = "voxsigil_finetune/data/arc_evaluation_dataset_fixed.jsonl"
    
    # Create results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Build command
    results_file = f"{results_dir}/{model_type}_{run_id}_results.json"
    cmd = [
        "python",
        "evaluate_model.py",
        f"--model_path={model_path}",
        f"--dataset={eval_dataset}",
        f"--output={results_file}",
    ]
    
    result = run_command(cmd, f"Evaluating {model_type}")
    result["results_file"] = results_file
    
    # Extract accuracy from results file if successful
    if result["success"]:
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                result["accuracy"] = eval_data.get("accuracy", 0)
                logger.info(f"Evaluation accuracy: {result['accuracy']:.2%}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read evaluation results: {e}")
            result["accuracy"] = 0
    
    return result

def test_sample_problem(model_path, model_type, run_id):
    """Test the fine-tuned model on a sample problem."""
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Build command
    results_file = f"{results_dir}/{model_type}_{run_id}_sample_test.json"
    cmd = [
        "python",
        "test_model.py",
        f"--model={model_path}",
        "--problem_file=sample_problem.txt",
        f"--output={results_file}",
    ]
    
    result = run_command(cmd, f"Testing {model_type} on sample problem")
    result["results_file"] = results_file
    
    return result

def visualize_results(results_file, model_type, run_id):
    """Visualize the evaluation results."""
    cmd = [
        "python",
        "visualize_results.py",
        f"--results={results_file}",
    ]
    
    return run_command(cmd, f"Visualizing results for {model_type}")

def run_pipeline(model_type, epochs, batch_size, learning_rate):
    """Run the complete fine-tuning pipeline."""
    logger.info(f"Starting fine-tuning pipeline for {model_type}")
    
    # Create timestamp for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create pipeline directory
    pipeline_dir = f"pipeline_runs/{model_type}_{run_id}"
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # Step 1: Setup environment
    setup_result = setup_environment()
    
    # Step 2: Fine-tune model
    if setup_result["success"]:
        finetune_result = finetune_model(model_type, epochs, batch_size, learning_rate, run_id)
    else:
        logger.error("Environment setup failed, skipping fine-tuning")
        finetune_result = {"success": False, "error": "Environment setup failed"}
    
    # Step 3: Evaluate model
    if finetune_result["success"]:
        model_path = finetune_result["output_dir"]
        evaluate_result = evaluate_model(model_path, model_type, run_id)
    else:
        logger.error("Fine-tuning failed, skipping evaluation")
        evaluate_result = {"success": False, "error": "Fine-tuning failed"}
    
    # Step 4: Test on sample problem
    if finetune_result["success"]:
        model_path = finetune_result["output_dir"]
        test_result = test_sample_problem(model_path, model_type, run_id)
    else:
        logger.error("Fine-tuning failed, skipping test")
        test_result = {"success": False, "error": "Fine-tuning failed"}
    
    # Step 5: Visualize results
    if evaluate_result["success"]:
        results_file = evaluate_result["results_file"]
        visualize_result = visualize_results(results_file, model_type, run_id)
    else:
        logger.error("Evaluation failed, skipping visualization")
        visualize_result = {"success": False, "error": "Evaluation failed"}
    
    # Save pipeline results
    pipeline_results = {
        "run_id": run_id,
        "model_type": model_type,
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "steps": {
            "setup": setup_result,
            "finetune": finetune_result,
            "evaluate": evaluate_result,
            "test_sample": test_result,
            "visualize": visualize_result,
        },
        "success": (
            setup_result["success"] and
            finetune_result["success"] and
            evaluate_result["success"]
        ),
        "model_path": finetune_result.get("output_dir") if finetune_result.get("success") else None,
        "accuracy": evaluate_result.get("accuracy", 0) if evaluate_result.get("success") else 0,
    }
    
    # Save results to file
    results_file = f"{pipeline_dir}/pipeline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_results, f, indent=2)
    
    logger.info(f"Pipeline completed. Results saved to {results_file}")
    
    if pipeline_results["success"]:
        logger.info(f"Pipeline completed successfully with accuracy: {pipeline_results['accuracy']:.2%}")
    else:
        logger.error("Pipeline failed")
    
    return pipeline_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run complete ARC fine-tuning pipeline")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["phi-2", "mistral-7b"],
        required=True,
        help="Model type to fine-tune",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training",
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(args.model, args.epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
