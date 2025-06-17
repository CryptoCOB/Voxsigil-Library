#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter search for ARC fine-tuning

This script runs a hyperparameter search for fine-tuning models on the ARC dataset.
"""

import argparse
import json
import logging
import os
import itertools
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperparameter_search.log"),
    ],
)
logger = logging.getLogger(__name__)

def run_experiment(model_type, hyperparams, experiment_id):
    """Run a single fine-tuning experiment with the given hyperparameters."""
    # Create output directory
    output_dir = f"models/{model_type}_hp_search_{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare command based on model type
    if model_type == "phi-2":
        script = "phi2_finetune.py"
        dataset = "voxsigil_finetune/data/phi-2/arc_training_phi-2.jsonl"
    elif model_type == "mistral-7b":
        script = "mistral_finetune.py"
        dataset = "voxsigil_finetune/data/mistral-7b/arc_training_mistral-7b.jsonl"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Build command with hyperparameters
    cmd = [
        "python",
        script,
        f"--dataset={dataset}",
        f"--output_dir={output_dir}",
        f"--epochs={hyperparams['epochs']}",
        f"--batch_size={hyperparams['batch_size']}",
        f"--learning_rate={hyperparams['learning_rate']}",
    ]
    
    # Run the command
    logger.info(f"Running experiment {experiment_id} with hyperparams: {hyperparams}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        # Extract information from the output
        output = result.stdout
        
        # Save experiment details
        experiment_details = {
            "experiment_id": experiment_id,
            "model_type": model_type,
            "hyperparams": hyperparams,
            "output_dir": output_dir,
            "success": True,
            "duration_seconds": end_time - start_time,
            "stdout": output,
            "stderr": result.stderr,
        }
        
        # Save details to file
        details_file = f"{output_dir}/experiment_details.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(experiment_details, f, indent=2)
        
        logger.info(f"Experiment {experiment_id} completed successfully in {end_time - start_time:.2f} seconds")
        return experiment_details
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        
        # Save experiment details even if it failed
        experiment_details = {
            "experiment_id": experiment_id,
            "model_type": model_type,
            "hyperparams": hyperparams,
            "output_dir": output_dir,
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
        
        # Save details to file
        details_file = f"{output_dir}/experiment_details.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(experiment_details, f, indent=2)
        
        return experiment_details

def evaluate_experiment(experiment_details):
    """Evaluate the experiment using the evaluation script."""
    if not experiment_details["success"]:
        logger.warning(f"Skipping evaluation for failed experiment {experiment_details['experiment_id']}")
        return None
    
    # Run evaluation
    output_dir = experiment_details["output_dir"]
    model_type = experiment_details["model_type"]
    
    if model_type == "phi-2":
        eval_dataset = "voxsigil_finetune/data/phi-2/arc_evaluation_phi-2.jsonl"
    elif model_type == "mistral-7b":
        eval_dataset = "voxsigil_finetune/data/mistral-7b/arc_evaluation_mistral-7b.jsonl"
    else:
        eval_dataset = "voxsigil_finetune/data/arc_evaluation_dataset_fixed.jsonl"
    
    # Create results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run evaluation command
    results_file = f"{results_dir}/{model_type}_exp_{experiment_details['experiment_id']}_results.json"
    
    cmd = [
        "python",
        "evaluate_model.py",
        f"--model_path={output_dir}",
        f"--dataset={eval_dataset}",
        f"--output={results_file}",
        "--num_samples=50",  # Use a subset for faster evaluation
    ]
    
    logger.info(f"Evaluating experiment {experiment_details['experiment_id']}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load evaluation results
        with open(results_file, "r", encoding="utf-8") as f:
            eval_results = json.load(f)
        
        # Add evaluation results to experiment details
        experiment_details["evaluation"] = {
            "results_file": results_file,
            "accuracy": eval_results.get("accuracy", 0),
        }
        
        # Update experiment details file
        details_file = f"{output_dir}/experiment_details.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(experiment_details, f, indent=2)
        
        logger.info(f"Evaluation for experiment {experiment_details['experiment_id']} completed with accuracy: {eval_results.get('accuracy', 0):.2%}")
        return eval_results
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Evaluation failed for experiment {experiment_details['experiment_id']}: {e}")
        return None

def generate_hyperparameters():
    """Generate hyperparameter combinations to try."""
    # Define hyperparameter ranges
    hyperparams = {
        "epochs": [1, 2, 3],
        "batch_size": [2, 4, 8],
        "learning_rate": [1e-5, 2e-5, 5e-5],
    }
    
    # Generate all combinations
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(itertools.product(*values))
    
    # Convert to list of dictionaries
    hyperparams_list = [dict(zip(keys, combo)) for combo in combinations]
    
    return hyperparams_list

def run_hyperparameter_search(model_type, num_experiments=None):
    """Run hyperparameter search for the specified model."""
    # Generate hyperparameter combinations
    hyperparams_list = generate_hyperparameters()
    
    # Limit number of experiments if specified
    if num_experiments is not None:
        hyperparams_list = hyperparams_list[:num_experiments]
    
    logger.info(f"Running hyperparameter search for {model_type} with {len(hyperparams_list)} combinations")
    
    # Create timestamp for this search
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_id = f"{model_type}_{timestamp}"
    
    # Create results directory
    results_dir = f"hp_search_results/{search_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run each experiment
    experiments = []
    best_accuracy = 0
    best_experiment = None
    
    for i, hyperparams in enumerate(hyperparams_list):
        # Run the experiment
        experiment_id = f"{i+1:02d}_{timestamp}"
        experiment_details = run_experiment(model_type, hyperparams, experiment_id)
        
        # Evaluate the experiment
        eval_results = evaluate_experiment(experiment_details)
        
        # Check if this is the best so far
        if eval_results and eval_results.get("accuracy", 0) > best_accuracy:
            best_accuracy = eval_results.get("accuracy", 0)
            best_experiment = experiment_details
        
        experiments.append(experiment_details)
    
    # Save overall results
    results = {
        "search_id": search_id,
        "model_type": model_type,
        "num_experiments": len(experiments),
        "best_experiment": best_experiment["experiment_id"] if best_experiment else None,
        "best_accuracy": best_accuracy,
        "experiments": experiments,
    }
    
    results_file = f"{results_dir}/search_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Hyperparameter search completed. Best accuracy: {best_accuracy:.2%}")
    logger.info(f"Results saved to {results_file}")
    
    # Print best hyperparameters
    if best_experiment:
        logger.info(f"Best hyperparameters: {best_experiment['hyperparams']}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run hyperparameter search for ARC fine-tuning")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["phi-2", "mistral-7b"],
        required=True,
        help="Model type to fine-tune",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=None,
        help="Number of experiments to run (default: all combinations)",
    )
    
    args = parser.parse_args()
    
    # Run hyperparameter search
    run_hyperparameter_search(args.model, args.experiments)

if __name__ == "__main__":
    main()
