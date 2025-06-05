#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import itertools
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperparameter_search.log"),
    ],
)
logger = logging.getLogger(__name__)


class HyperparameterSearch:
    def __init__(
        self, model_type: str, search_space: Optional[Dict[str, List[Any]]] = None
    ):
        self.model_type = model_type
        self.search_space = search_space or {
            "epochs": [1, 2, 3],
            "batch_size": [2, 4, 8],
            "learning_rate": [1e-5, 2e-5, 5e-5],
        }

    def run_experiment(
        self, hyperparams: Dict[str, Any], experiment_id: str
    ) -> Dict[str, Any]:
        output_dir = f"models/{self.model_type}_hp_search_{experiment_id}"
        os.makedirs(output_dir, exist_ok=True)

        if self.model_type == "phi-2":
            script = "phi2_finetune.py"
            dataset = "voxsigil_finetune/data/phi-2/arc_training_phi-2.jsonl"
        elif self.model_type == "mistral-7b":
            script = "mistral_finetune.py"
            dataset = "voxsigil_finetune/data/mistral-7b/arc_training_mistral-7b.jsonl"
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        cmd = [
            "python",
            script,
            f"--dataset={dataset}",
            f"--output_dir={output_dir}",
            f"--epochs={hyperparams['epochs']}",
            f"--batch_size={hyperparams['batch_size']}",
            f"--learning_rate={hyperparams['learning_rate']}",
        ]

        logger.info(
            f"Running experiment {experiment_id} with hyperparams: {hyperparams}"
        )
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            output = result.stdout

            experiment_details = {
                "experiment_id": experiment_id,
                "model_type": self.model_type,
                "hyperparams": hyperparams,
                "output_dir": output_dir,
                "success": True,
                "duration_seconds": end_time - start_time,
                "stdout": output,
                "stderr": result.stderr,
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            logger.info(
                f"Experiment {experiment_id} completed successfully in {end_time - start_time:.2f} seconds"
            )
            return experiment_details

        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")

            experiment_details = {
                "experiment_id": experiment_id,
                "model_type": self.model_type,
                "hyperparams": hyperparams,
                "output_dir": output_dir,
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            return experiment_details

    def evaluate_experiment(
        self, experiment_details: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not experiment_details["success"]:
            logger.warning(
                f"Skipping evaluation for failed experiment {experiment_details['experiment_id']}"
            )
            return None

        output_dir = experiment_details["output_dir"]

        if self.model_type == "phi-2":
            eval_dataset = "voxsigil_finetune/data/phi-2/arc_evaluation_phi-2.jsonl"
        elif self.model_type == "mistral-7b":
            eval_dataset = (
                "voxsigil_finetune/data/mistral-7b/arc_evaluation_mistral-7b.jsonl"
            )
        else:
            eval_dataset = "voxsigil_finetune/data/arc_evaluation_dataset_fixed.jsonl"

        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        results_file = f"{results_dir}/{self.model_type}_exp_{experiment_details['experiment_id']}_results.json"

        cmd = [
            "python",
            "evaluate_model.py",
            f"--model_path={output_dir}",
            f"--dataset={eval_dataset}",
            f"--output={results_file}",
            "--num_samples=50",
        ]

        logger.info(f"Evaluating experiment {experiment_details['experiment_id']}...")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            with open(results_file, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

            experiment_details["evaluation"] = {
                "results_file": results_file,
                "accuracy": eval_results.get("accuracy", 0),
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            logger.info(
                f"Evaluation for experiment {experiment_details['experiment_id']} completed with accuracy: {eval_results.get('accuracy', 0):.2%}"
            )
            return eval_results

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(
                f"Evaluation failed for experiment {experiment_details['experiment_id']}: {e}"
            )
            return None

    def generate_hyperparameters(self) -> List[Dict[str, Any]]:
        keys = self.search_space.keys()
        values = self.search_space.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def run_search(self, num_experiments: Optional[int] = None) -> Dict[str, Any]:
        hyperparams_list = self.generate_hyperparameters()
        if num_experiments is not None:
            hyperparams_list = hyperparams_list[:num_experiments]

        logger.info(
            f"Running hyperparameter search for {self.model_type} with {len(hyperparams_list)} combinations"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_id = f"{self.model_type}_{timestamp}"
        results_dir = f"hp_search_results/{search_id}"
        os.makedirs(results_dir, exist_ok=True)

        experiments = []
        best_accuracy = 0
        best_experiment = None

        for i, hyperparams in enumerate(hyperparams_list):
            experiment_id = f"{i + 1:02d}_{timestamp}"
            experiment_details = self.run_experiment(hyperparams, experiment_id)
            eval_results = self.evaluate_experiment(experiment_details)
            if eval_results and eval_results.get("accuracy", 0) > best_accuracy:
                best_accuracy = eval_results.get("accuracy", 0)
                best_experiment = experiment_details
            experiments.append(experiment_details)

        results = {
            "search_id": search_id,
            "model_type": self.model_type,
            "num_experiments": len(experiments),
            "best_experiment": best_experiment["experiment_id"]
            if best_experiment
            else None,
            "best_accuracy": best_accuracy,
            "experiments": experiments,
        }

        results_file = f"{results_dir}/search_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Hyperparameter search completed. Best accuracy: {best_accuracy:.2%}"
        )
        logger.info(f"Results saved to {results_file}")

        if best_experiment:
            logger.info(f"Best hyperparameters: {best_experiment['hyperparams']}")

        return results

    def get_best_hyperparameters(
        self, results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        best_exp_id = results.get("best_experiment")
        if not best_exp_id:
            return None
        for exp in results.get("experiments", []):
            if exp["experiment_id"] == best_exp_id:
                return exp.get("hyperparams")
        return None

    @classmethod
    def from_config(cls, config_path: str) -> "HyperparameterSearch":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            model_type=config["model_type"],
            search_space=config.get("search_space"),
        )

    @property
    def search_space_keys(self) -> List[str]:
        return list(self.search_space.keys())

    @property
    def search_space_values(self) -> List[List[Any]]:
        return list(self.search_space.values())


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search for ARC fine-tuning"
    )
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file for search space and model type",
    )
    args = parser.parse_args()

    if args.config:
        searcher = HyperparameterSearch.from_config(args.config)
    else:
        searcher = HyperparameterSearch(args.model)
    searcher.run_search(args.experiments)


if __name__ == "__main__":
    main()
