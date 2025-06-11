#!/usr/bin/env python
"""
arc_dataset_generator.py - Create synthetic ARC training data

This script generates a simplified set of ARC dataset examples for testing the GRID-Former model.
Since we're having issues downloading the real dataset, this will allow development to continue.
"""

import json
import logging
import argparse
import random
import sys
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ARC-Data-Generator")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate synthetic ARC dataset")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/arc",
        help="Output directory for generated ARC dataset",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=20, help="Number of tasks to generate"
    )
    parser.add_argument(
        "--examples_per_task", type=int, default=3, help="Number of examples per task"
    )
    parser.add_argument(
        "--max_grid_size", type=int, default=10, help="Maximum grid size"
    )

    return parser.parse_args()


def generate_grid(min_size=2, max_size=10, n_colors=10, complexity=0.3):
    """Generate a random grid"""
    h = random.randint(min_size, max_size)
    w = random.randint(min_size, max_size)

    # Generate grid with some structure
    if random.random() < complexity:
        # Create some patterns
        pattern_type = random.choice(["checkerboard", "gradient", "border", "random"])

        if pattern_type == "checkerboard":
            grid = np.zeros((h, w), dtype=int)
            for i in range(h):
                for j in range(w):
                    if (i + j) % 2 == 0:
                        grid[i, j] = random.randint(1, n_colors - 1)
                    else:
                        grid[i, j] = 0

        elif pattern_type == "gradient":
            grid = np.zeros((h, w), dtype=int)
            start_color = random.randint(0, n_colors - 1)
            end_color = random.randint(0, n_colors - 1)
            for i in range(h):
                color = (
                    start_color + (end_color - start_color) * i // (h - 1)
                    if h > 1
                    else start_color
                )
                grid[i, :] = color

        elif pattern_type == "border":
            grid = np.zeros((h, w), dtype=int)
            border_color = random.randint(1, n_colors - 1)
            grid[0, :] = border_color
            grid[-1, :] = border_color
            grid[:, 0] = border_color
            grid[:, -1] = border_color

        else:  # random
            grid = np.random.randint(0, n_colors, (h, w))
    else:
        # Completely random grid
        grid = np.random.randint(0, n_colors, (h, w))

    return grid.tolist()


def apply_transformation(input_grid, transformation_type=None):
    """Apply a transformation to the input grid"""
    if transformation_type is None:
        transformation_type = random.choice(
            [
                "flip_horizontal",
                "flip_vertical",
                "rotate",
                "shift_colors",
                "invert_colors",
                "extract_region",
            ]
        )

    input_array = np.array(input_grid)

    if transformation_type == "flip_horizontal":
        output_array = np.flip(input_array, axis=1)

    elif transformation_type == "flip_vertical":
        output_array = np.flip(input_array, axis=0)

    elif transformation_type == "rotate":
        k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        output_array = np.rot90(input_array, k=k)

    elif transformation_type == "shift_colors":
        shift = random.randint(1, 5)
        output_array = (input_array + shift) % 10

    elif transformation_type == "invert_colors":
        max_color = np.max(input_array)
        output_array = max_color - input_array

    elif transformation_type == "extract_region":
        h, w = input_array.shape
        if h <= 2 or w <= 2:
            # Grid too small, use another transformation
            return apply_transformation(input_grid, "flip_horizontal")

        start_h = random.randint(0, h // 2)
        start_w = random.randint(0, w // 2)
        end_h = random.randint(start_h + 1, h)
        end_w = random.randint(start_w + 1, w)

        output_array = input_array[start_h:end_h, start_w:end_w]

    else:
        # Default to identity transformation
        output_array = input_array.copy()

    return output_array.tolist()


def generate_task(task_id, num_examples=3, max_grid_size=10):
    """Generate a consistent ARC task with examples"""
    # Choose a consistent transformation for this task
    transformation_type = random.choice(
        ["flip_horizontal", "flip_vertical", "rotate", "shift_colors", "invert_colors"]
    )

    task = {"train": [], "test": []}

    # Generate training examples
    for _ in range(num_examples):
        input_grid = generate_grid(min_size=2, max_size=max_grid_size)
        output_grid = apply_transformation(input_grid, transformation_type)

        task["train"].append({"input": input_grid, "output": output_grid})

    # Generate test examples
    for _ in range(1):  # Usually 1 test example per task
        input_grid = generate_grid(min_size=2, max_size=max_grid_size)
        output_grid = apply_transformation(input_grid, transformation_type)

        task["test"].append({"input": input_grid, "output": output_grid})

    return task


def generate_arc_dataset(
    output_dir, num_tasks=20, examples_per_task=3, max_grid_size=10
):
    """Generate a full ARC dataset with training, evaluation, and test tasks"""
    output_dir = Path(output_dir)

    # Create directories
    training_dir = output_dir / "training"
    evaluation_dir = output_dir / "evaluation"
    test_dir = output_dir / "test"

    for d in [training_dir, evaluation_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate training tasks
    training_tasks = {}
    logger.info(f"Generating {num_tasks} training tasks...")
    for i in range(num_tasks):
        task_id = f"training_{i:03d}"
        training_tasks[task_id] = generate_task(
            task_id, num_examples=examples_per_task, max_grid_size=max_grid_size
        )

    # Generate evaluation tasks (fewer)
    eval_tasks = {}
    logger.info(f"Generating {num_tasks // 4} evaluation tasks...")
    for i in range(num_tasks // 4):
        task_id = f"evaluation_{i:03d}"
        eval_tasks[task_id] = generate_task(
            task_id, num_examples=examples_per_task, max_grid_size=max_grid_size
        )

    # Generate test tasks (just to have the structure; no solutions)
    test_tasks = {}
    logger.info(f"Generating {num_tasks // 5} test tasks...")
    for i in range(num_tasks // 5):
        task_id = f"test_{i:03d}"
        task = generate_task(
            task_id,
            num_examples=1,  # Test tasks have fewer examples
            max_grid_size=max_grid_size,
        )
        # Remove outputs from test examples to simulate the real test set
        for example in task["test"]:
            if "output" in example:
                del example["output"]
        test_tasks[task_id] = task

    # Write datasets to files
    logger.info("Writing datasets to files...")
    with open(training_dir / "training.json", "w") as f:
        json.dump(training_tasks, f, indent=2)

    with open(evaluation_dir / "evaluation.json", "w") as f:
        json.dump(eval_tasks, f, indent=2)

    with open(test_dir / "test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)

    logger.info(f"ARC dataset generated and saved to {output_dir}")
    return True


def main():
    """Main function"""
    args = parse_arguments()

    logger.info("Generating synthetic ARC dataset for training...")
    success = generate_arc_dataset(
        output_dir=args.output_dir,
        num_tasks=args.num_tasks,
        examples_per_task=args.examples_per_task,
        max_grid_size=args.max_grid_size,
    )

    if success:
        logger.info("Dataset generation completed successfully!")

        # Generate a readme file with dataset information
        readme_path = Path(args.output_dir) / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Synthetic ARC Dataset\n\n")
            f.write(
                "This is a synthetic dataset created for training and testing the GRID-Former model.\n\n"
            )
            f.write("## Dataset Statistics\n")
            f.write(f"- Number of training tasks: {args.num_tasks}\n")
            f.write(f"- Number of evaluation tasks: {args.num_tasks // 4}\n")
            f.write(f"- Number of test tasks: {args.num_tasks // 5}\n")
            f.write(f"- Examples per task: {args.examples_per_task}\n")
            f.write(f"- Maximum grid size: {args.max_grid_size}\n\n")
            f.write(
                "**Note:** This is a synthetic dataset created for development purposes when the original ARC dataset is unavailable.\n"
            )

        logger.info(f"Dataset information written to {readme_path}")
    else:
        logger.error("Dataset generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
