#!/usr/bin/env python3
"""
🔮 VoxSigil GridFormer Evaluator - Complete Evaluation Tool v1.0
Integrates visualization and inference capabilities for ARC tasks

Features:
- Comprehensive model evaluation
- Visual side-by-side comparison (input, expected output, prediction)
- Multi-strategy inference testing
- Performance metrics with detailed reports
- Batch testing across multiple tasks
"""

import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
import logging
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/grid_former_eval.log", mode="a"),
    ],
)
logger = logging.getLogger("GridFormer-Evaluator")

# Ensure project root is in the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import VoxSigil modules with graceful fallbacks
try:
    from ARC.data_loader import ARCDataLoader
    from Gridformer.inference import GridFormerInference, InferenceStrategy
    from tools.utilities.model_utils import ModelLoader, discover_models, get_latest_models

    logger.info("✅ Successfully imported VoxSigil utility modules")
except ImportError as e:
    logger.error(f"❌ Error importing VoxSigil modules: {e}")
    logger.info("⚠️ Attempting to use fallback imports...")

    # Try alternative import paths
    try:
        sys.path.insert(0, str(project_root / "utils"))
        from ARC.data_loader import ARCDataLoader
        from inference import GridFormerInference, InferenceStrategy
        from model_utils import ModelLoader, discover_models, get_latest_models

        logger.info("✅ Successfully imported from utils directory")
    except ImportError as e:
        logger.error(f"❌ Critical import failure: {e}")
        logger.error("Cannot proceed without core modules")
        sys.exit(1)

# ARC color palette for visualization
ARC_COLORS = [
    "#000000",  # 0: Black (background)
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Sky blue
    "#870C25",  # 9: Brown
]


class EvaluationMode(Enum):
    """Evaluation modes for GridFormer evaluator"""

    SINGLE_TASK = "single_task"
    BATCH = "batch"
    INTERACTIVE = "interactive"
    BENCHMARK = "benchmark"


class GridFormerEvaluator:
    """Complete evaluation system for GridFormer models"""

    def __init__(
        self, model_path: Optional[str] = None, output_dir: str = "results/evaluation"
    ):
        """Initialize the evaluator with optional model path"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = ARCDataLoader()
        self.model_loader = ModelLoader()
        self.inference_engine = GridFormerInference(
            model_loader=self.model_loader, data_loader=self.data_loader
        )

        # Load model if provided
        if model_path:
            try:
                self.model_loader.load_model(model_path)
                logger.info(f"✅ Model loaded from: {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")

    def grid_to_image(self, grid: List[List[int]]) -> np.ndarray:
        """Convert ARC grid to colored image array"""
        if not grid or not grid[0]:
            return np.zeros((1, 1, 3))

        grid_array = np.array(grid)
        h, w = grid_array.shape

        # Create RGB image
        image = np.zeros((h, w, 3))

        for i in range(h):
            for j in range(w):
                color_idx = grid_array[i, j]
                if 0 <= color_idx < len(ARC_COLORS):
                    # Convert hex to RGB
                    hex_color = ARC_COLORS[color_idx]
                    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
                    image[i, j] = [c / 255.0 for c in rgb]

        return image

    def visualize_task(
        self,
        task: Dict[str, Any],
        predictions: List[List[List[int]]],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize a complete ARC task with side-by-side comparison"""

        train_examples = task.get("train", [])
        test_examples = task.get("test", [])

        # Calculate layout
        num_train = len(train_examples)
        num_test = len(test_examples)

        # Calculate total rows and columns
        if num_train > 0:
            total_rows = max(num_train, num_test)
            cols = 4  # Train input, train output, test input, prediction
        else:
            total_rows = num_test
            cols = 3  # Test input, prediction, (empty for alignment)

        # Create figure
        fig, axes = plt.subplots(total_rows, cols, figsize=(5 * cols, 4 * total_rows))

        # Handle single row case
        if total_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot training examples
        for i in range(total_rows):
            # Initialize row
            if i < num_train:
                # Plot training example
                train_input = train_examples[i]["input"]
                train_output = train_examples[i]["output"]

                # Display train input
                ax = axes[i, 0]
                ax.imshow(self.grid_to_image(train_input))
                ax.set_title(f"Train {i + 1}: Input")
                ax.axis("off")

                # Display train output
                ax = axes[i, 1]
                ax.imshow(self.grid_to_image(train_output))
                ax.set_title(f"Train {i + 1}: Output")
                ax.axis("off")
            else:
                # Empty training slots
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")

            # Plot test examples and predictions
            if i < num_test:
                test_input = test_examples[i]["input"]

                # Display test input
                ax = axes[i, 2]
                ax.imshow(self.grid_to_image(test_input))
                ax.set_title(f"Test {i + 1}: Input")
                ax.axis("off")

                # Display prediction if available
                ax = axes[i, 3]
                if i < len(predictions):
                    ax.imshow(self.grid_to_image(predictions[i]))
                    ax.set_title(f"Test {i + 1}: Prediction")
                else:
                    ax.set_title("No prediction")
                ax.axis("off")
            else:
                # Empty test slots
                if i < total_rows:
                    axes[i, 2].axis("off")
                    axes[i, 3].axis("off")

        # Add expected output if present in test
        for i, test in enumerate(test_examples):
            if "output" in test and i < total_rows:
                # Only add if we have the expected output
                expected_output = test["output"]
                # Add text annotation about expected output
                ax = axes[i, 3]
                if i < len(predictions):
                    match = predictions[i] == expected_output
                    match_text = "✓ MATCH" if match else "✗ MISMATCH"
                    ax.set_title(f"Prediction ({match_text})")

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"📊 Visualization saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def evaluate_task(
        self,
        task_id: str,
        strategy: InferenceStrategy = InferenceStrategy.ITERATIVE,
        visualize: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single task with the specified strategy"""

        start_time = time.time()

        # Load the specified task
        task = self.data_loader.load_arc_task(task_id)

        if not task:
            logger.error(f"❌ Failed to load task: {task_id}")
            return {"error": f"Task {task_id} not found"}

        # Generate predictions
        try:
            predictions = self.inference_engine.predict(task, strategy=strategy)

            # Calculate metrics
            test_examples = task.get("test", [])
            metrics = {
                "task_id": task_id,
                "strategy": strategy.value,
                "num_test_examples": len(test_examples),
                "num_predictions": len(predictions) if predictions else 0,
                "time_taken": time.time() - start_time,
            }

            # Check for ground truth if available
            correct_predictions = 0
            for i, test in enumerate(test_examples):
                if "output" in test and i < len(predictions):
                    if predictions[i] == test["output"]:
                        correct_predictions += 1

            if correct_predictions > 0:
                metrics["correct_predictions"] = correct_predictions
                metrics["accuracy"] = correct_predictions / len(test_examples)

            # Visualize if requested
            if visualize:
                save_path = None
                if save_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = (
                        Path(save_dir)
                        / f"task_{task_id}_{strategy.value}_{timestamp}.png"
                    )

                self.visualize_task(task, predictions, save_path=save_path)

            return {
                "success": True,
                "task": task,
                "predictions": predictions,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            return {"error": str(e)}

    def batch_evaluate(
        self,
        task_ids: List[str] = None,
        max_tasks: int = 10,
        strategies: List[InferenceStrategy] = None,
        visualize: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of tasks with multiple strategies"""

        # Default to all strategies if none specified
        if strategies is None:
            strategies = list(InferenceStrategy)

        # Get task IDs if not provided
        if task_ids is None:
            # Load all available tasks
            all_tasks = self.data_loader.load_training_data()
            if not all_tasks:
                logger.error("❌ No tasks available for evaluation")
                return {"error": "No tasks available"}

            # Limit to max_tasks
            task_ids = [task["id"] for task in all_tasks[:max_tasks] if "id" in task]

        batch_results = {}
        overall_metrics = {
            "total_tasks": len(task_ids),
            "total_strategies": len(strategies),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results_by_strategy": {},
        }

        # Process each strategy
        for strategy in strategies:
            strategy_results = []
            correct_predictions = 0
            total_predictions = 0

            logger.info(
                f"🧠 Evaluating {len(task_ids)} tasks with strategy: {strategy.value}"
            )

            # Process each task
            for i, task_id in enumerate(task_ids):
                logger.info(f"  📋 Task {i + 1}/{len(task_ids)}: {task_id}")

                result = self.evaluate_task(
                    task_id,
                    strategy=strategy,
                    visualize=visualize
                    and (i < 5),  # Only visualize first 5 to avoid clutter
                    save_dir=save_dir,
                )

                strategy_results.append(result)

                # Update metrics
                if result.get("success") and "metrics" in result:
                    metrics = result["metrics"]
                    if "correct_predictions" in metrics:
                        correct_predictions += metrics["correct_predictions"]
                    total_predictions += metrics.get("num_test_examples", 0)

            # Calculate strategy metrics
            accuracy = (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            )

            overall_metrics["results_by_strategy"][strategy.value] = {
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
                "accuracy": accuracy,
            }

            batch_results[strategy.value] = strategy_results

            logger.info(
                f"✅ {strategy.value} accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
            )

        # Add overall metrics
        overall_metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate summary report
        if save_dir:
            report_path = (
                Path(save_dir)
                / f"batch_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(overall_metrics, f, indent=2)
            logger.info(f"📊 Evaluation report saved to: {report_path}")

        return {"results": batch_results, "metrics": overall_metrics}

    def find_available_models(self) -> List[Dict[str, Any]]:
        """Find all available models for evaluation"""
        try:
            models = discover_models()
            logger.info(f"✅ Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"❌ Error discovering models: {e}")
            return []

    def interactive_mode(self):
        """Start interactive evaluation mode"""
        print("\n" + "=" * 60)
        print("🔮 VoxSigil GridFormer Interactive Evaluator")
        print("=" * 60 + "\n")

        # Find available models
        models = self.find_available_models()
        if not models:
            print("❌ No models available for evaluation")
            return

        # Display available models
        print(f"📚 Found {len(models)} available models:")
        for i, model in enumerate(models):
            print(
                f"  [{i + 1}] {model.get('name', 'Unknown')} - {model.get('path', 'Unknown path')}"
            )

        # Select model
        selected_idx = input(
            "\n📋 Select model by number (or press Enter for latest): "
        )
        if selected_idx.strip():
            try:
                idx = int(selected_idx) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                else:
                    print("❌ Invalid selection, using latest model")
                    selected_model = get_latest_models(models, limit=1)[0]
            except ValueError:
                print("❌ Invalid input, using latest model")
                selected_model = get_latest_models(models, limit=1)[0]
        else:
            selected_model = get_latest_models(models, limit=1)[0]

        # Load selected model
        try:
            self.model_loader.load_model(selected_model["path"])
            print(f"✅ Loaded model: {selected_model.get('name', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return

        # Load available tasks
        tasks = self.data_loader.load_training_data()
        if not tasks:
            print("❌ No tasks available for evaluation")
            return

        # Display task options
        print(f"\n📚 Found {len(tasks)} available tasks")
        print("  [1] Evaluate single task")
        print("  [2] Batch evaluate multiple tasks")
        print("  [3] Benchmark all strategies on selected tasks")

        # Select mode
        mode_choice = input("\n📋 Select evaluation mode: ")

        if mode_choice == "1":
            # Single task mode
            task_id = input("\n📋 Enter task ID or index (1-based): ")
            try:
                if task_id.isdigit():
                    idx = int(task_id) - 1
                    if 0 <= idx < len(tasks):
                        task = tasks[idx]
                        task_id = task.get("id", f"task_{idx}")
                    else:
                        print("❌ Invalid task index")
                        return

                # Select strategy
                print("\n📚 Available strategies:")
                for i, strategy in enumerate(InferenceStrategy):
                    print(f"  [{i + 1}] {strategy.value}")

                strategy_choice = input("📋 Select strategy (or Enter for iterative): ")
                if strategy_choice.strip() and strategy_choice.isdigit():
                    idx = int(strategy_choice) - 1
                    if 0 <= idx < len(InferenceStrategy):
                        strategy = list(InferenceStrategy)[idx]
                    else:
                        strategy = InferenceStrategy.ITERATIVE
                else:
                    strategy = InferenceStrategy.ITERATIVE

                # Run evaluation
                print(f"\n🧠 Evaluating task {task_id} with strategy: {strategy.value}")
                result = self.evaluate_task(task_id, strategy=strategy, visualize=True)

                if result.get("success"):
                    print("✅ Evaluation complete!")
                    metrics = result.get("metrics", {})
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                else:
                    print(
                        f"❌ Evaluation failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"❌ Error: {e}")

        elif mode_choice == "2":
            # Batch mode
            num_tasks = input("\n📋 Number of tasks to evaluate (default: 5): ")
            try:
                max_tasks = int(num_tasks) if num_tasks.strip() else 5
                max_tasks = min(max_tasks, len(tasks))

                # Select strategies
                print(
                    "\n📚 Select strategies (comma-separated numbers, or Enter for all):"
                )
                for i, strategy in enumerate(InferenceStrategy):
                    print(f"  [{i + 1}] {strategy.value}")

                strategy_choices = input("📋 Strategies: ")
                if strategy_choices.strip():
                    indices = [
                        int(idx.strip()) - 1 for idx in strategy_choices.split(",")
                    ]
                    strategies = [
                        list(InferenceStrategy)[idx]
                        for idx in indices
                        if 0 <= idx < len(InferenceStrategy)
                    ]
                else:
                    strategies = list(InferenceStrategy)

                # Run batch evaluation
                print(
                    f"\n🧠 Batch evaluating {max_tasks} tasks with {len(strategies)} strategies"
                )
                result = self.batch_evaluate(
                    task_ids=[
                        task.get("id", f"task_{i}")
                        for i, task in enumerate(tasks[:max_tasks])
                    ],
                    strategies=strategies,
                    visualize=True,
                    save_dir="results/batch_evaluation",
                )

                print("\n📊 Evaluation Results:")
                metrics = result.get("metrics", {})
                for strategy, stats in metrics.get("results_by_strategy", {}).items():
                    accuracy = stats.get("accuracy", 0)
                    correct = stats.get("correct_predictions", 0)
                    total = stats.get("total_predictions", 0)
                    print(f"  {strategy}: {accuracy:.2%} ({correct}/{total})")

            except Exception as e:
                print(f"❌ Error: {e}")

        elif mode_choice == "3":
            # Benchmark mode
            print("\n🧪 Benchmark Mode - Running all strategies on selected tasks")

            num_tasks = input("📋 Number of tasks to benchmark (default: 10): ")
            try:
                max_tasks = int(num_tasks) if num_tasks.strip() else 10
                max_tasks = min(max_tasks, len(tasks))

                # Run benchmark
                print(f"\n🧠 Benchmarking {max_tasks} tasks with all strategies")
                result = self.batch_evaluate(
                    task_ids=[
                        task.get("id", f"task_{i}")
                        for i, task in enumerate(tasks[:max_tasks])
                    ],
                    strategies=list(InferenceStrategy),
                    visualize=False,
                    save_dir="results/benchmark",
                )

                print("\n📊 Benchmark Results:")
                metrics = result.get("metrics", {})

                # Print results in table format
                print("\n" + "-" * 60)
                print(
                    f"{'Strategy':<20} | {'Accuracy':<10} | {'Correct':<10} | {'Total':<10}"
                )
                print("-" * 60)

                for strategy, stats in metrics.get("results_by_strategy", {}).items():
                    accuracy = stats.get("accuracy", 0)
                    correct = stats.get("correct_predictions", 0)
                    total = stats.get("total_predictions", 0)
                    print(
                        f"{strategy:<20} | {accuracy:.2%:<10} | {correct:<10} | {total:<10}"
                    )

                print("-" * 60)

            except Exception as e:
                print(f"❌ Error: {e}")

        else:
            print("❌ Invalid mode selection")


def main():
    """Main entry point for the evaluator"""
    parser = argparse.ArgumentParser(description="VoxSigil GridFormer Evaluator")

    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["single", "batch", "interactive", "benchmark"],
        help="Evaluation mode",
    )

    parser.add_argument("--model", type=str, help="Path to model file")

    parser.add_argument("--task", type=str, help="Task ID for single evaluation")

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["rule_based", "pattern_matching", "neural_network", "iterative"],
        default="iterative",
        help="Inference strategy",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks for batch evaluation",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )

    parser.add_argument(
        "--visualize", action="store_true", default=True, help="Generate visualizations"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = GridFormerEvaluator(model_path=args.model, output_dir=args.output_dir)

    # Map strategy string to enum
    strategy_map = {
        "rule_based": InferenceStrategy.RULE_BASED,
        "pattern_matching": InferenceStrategy.PATTERN_MATCHING,
        "neural_network": InferenceStrategy.NEURAL_NETWORK,
        "iterative": InferenceStrategy.ITERATIVE,
    }

    # Run appropriate mode
    if args.mode == "interactive":
        evaluator.interactive_mode()

    elif args.mode == "single":
        if not args.task:
            logger.error("❌ Task ID required for single evaluation mode")
            return 1

        strategy = strategy_map.get(args.strategy, InferenceStrategy.ITERATIVE)
        result = evaluator.evaluate_task(
            args.task,
            strategy=strategy,
            visualize=args.visualize,
            save_dir=args.output_dir,
        )

        if result.get("success"):
            logger.info("✅ Evaluation complete!")
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            return 0
        else:
            logger.error(
                f"❌ Evaluation failed: {result.get('error', 'Unknown error')}"
            )
            return 1

    elif args.mode == "batch":
        result = evaluator.batch_evaluate(
            max_tasks=args.max_tasks,
            strategies=[strategy_map.get(args.strategy, InferenceStrategy.ITERATIVE)],
            visualize=args.visualize,
            save_dir=args.output_dir,
        )

        logger.info("📊 Batch Evaluation Results:")
        metrics = result.get("metrics", {})
        for strategy, stats in metrics.get("results_by_strategy", {}).items():
            accuracy = stats.get("accuracy", 0)
            correct = stats.get("correct_predictions", 0)
            total = stats.get("total_predictions", 0)
            logger.info(f"  {strategy}: {accuracy:.2%} ({correct}/{total})")

        return 0

    elif args.mode == "benchmark":
        result = evaluator.batch_evaluate(
            max_tasks=args.max_tasks,
            strategies=list(InferenceStrategy),
            visualize=args.visualize,
            save_dir=args.output_dir,
        )

        logger.info("📊 Benchmark Results:")
        metrics = result.get("metrics", {})

        # Print results in table format
        logger.info("\n" + "-" * 60)
        logger.info(
            f"{'Strategy':<20} | {'Accuracy':<10} | {'Correct':<10} | {'Total':<10}"
        )
        logger.info("-" * 60)

        for strategy, stats in metrics.get("results_by_strategy", {}).items():
            accuracy = stats.get("accuracy", 0)
            correct = stats.get("correct_predictions", 0)
            total = stats.get("total_predictions", 0)
            logger.info(
                f"{strategy:<20} | {accuracy:.2%:<10} | {correct:<10} | {total:<10}"
            )

        logger.info("-" * 60)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
