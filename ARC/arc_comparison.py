"""
ARC Solver Performance Comparison Script

This script runs the ARC solver with different configurations to compare
the performance of different strategies, particularly the effectiveness
of the CATENGINE for categorization-based reasoning.
"""

import os
import sys
import json
import random
import time
import argparse
from pathlib import Path
import logging

# Setup a basic logger if not provided by arc_config
logger = logging.getLogger("arc_comparison")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Ensure the ARC.py script can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Import only after adding to path
try:
    from ARC.arc_config import (
        load_llm_response_cache,
        save_llm_response_cache,
        initialize_and_validate_models_config,
        load_voxsigil_entries,
        VoxSigilComponent,
        USE_LLM_CACHE,
        # logger  # Removed because it's not provided by arc_config
        analyze_task_for_categorization_needs,
        OLLAMA_API_BASE_URL,
        LMSTUDIO_API_BASE_URL,
        RESULTS_OUTPUT_DIR,
        ARC_DATA_DIR,
    )
except ImportError as e:
    print(f"Error importing ARC.py: {e}")
    sys.exit(1)


def run_comparison_test(
    tasks_file="arc-agi_training_challenges.json",
    solutions_file="arc-agi_training_solutions.json",
    max_tasks=10,
    force_catengine=False,
    force_arc_solver=False,
):
    """Run a comparison test between different ARC solving strategies."""
    # Setup
    run_timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    load_llm_response_cache()

    logger.info("🧪 Initializing ARC Solver Performance Comparison Test")

    # Initialize models
    if not initialize_and_validate_models_config():
        logger.critical("Model initialization failed. Cannot proceed.")
        return False

    # Load VoxSigil components
    voxsigil_entries = load_voxsigil_entries(Path(SCRIPT_DIR) / "VoxSigil-Library")
    voxsigil_components = [
        VoxSigilComponent(entry, entry.get("capabilities", []))
        for entry in voxsigil_entries
    ]
    logger.info(f"Loaded {len(voxsigil_components)} VoxSigil components")

    # Load ARC tasks and solutions
    try:
        arc_tasks_path = ARC_DATA_DIR / tasks_file
        arc_solutions_path = ARC_DATA_DIR / solutions_file

        with open(arc_tasks_path, encoding="utf-8") as f_tasks:
            tasks = json.load(f_tasks)
        with open(arc_solutions_path, encoding="utf-8") as f_solutions:
            solutions = json.load(f_solutions)

        logger.info(f"Loaded {len(tasks)} tasks and {len(solutions)} solutions")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.critical(f"Failed to load ARC data: {e}")
        return False

    # Select tasks for comparison
    available_task_ids = list(tasks.keys())
    if max_tasks is None or max_tasks <= 0 or max_tasks >= len(available_task_ids):
        selected_tasks = available_task_ids
    else:
        selected_tasks = random.sample(available_task_ids, max_tasks)

    logger.info(f"Selected {len(selected_tasks)} tasks for comparison")

    # Analyze tasks for categorization needs
    categorization_results = {}
    for task_id in selected_tasks:
        task_data = tasks.get(task_id)
        if task_data:
            needs_categorization = analyze_task_for_categorization_needs(task_data)
            categorization_results[task_id] = {
                "task_id": task_id,
                "needs_categorization": needs_categorization,
            }

    # Group tasks by categorization need
    catengine_candidates = [
        tid
        for tid, result in categorization_results.items()
        if result["needs_categorization"]
    ]
    arc_solver_candidates = [
        tid
        for tid, result in categorization_results.items()
        if not result["needs_categorization"]
    ]

    logger.info(
        f"Categorization analysis: {len(catengine_candidates)} tasks need categorization, "
        f"{len(arc_solver_candidates)} tasks don't need categorization"
    )

    # Save categorization analysis to a file for reference
    analysis_file = (
        RESULTS_OUTPUT_DIR / f"categorization_analysis_{run_timestamp_str}.json"
    )
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(categorization_results, f, indent=2)
    logger.info(f"Saved categorization analysis to {analysis_file}")

    # Prepare parameters for ARC.py
    if force_catengine:
        logger.info("Force using CATENGINE for all tasks")
        # Modify environment variables or set flags to force CATENGINE
        os.environ["FORCE_CATENGINE"] = "True"
    elif force_arc_solver:
        logger.info("Force using ARC_SOLVER for all tasks")
        # Modify environment variables or set flags to force ARC_SOLVER
        os.environ["FORCE_ARC_SOLVER"] = "True"
    else:
        logger.info("Using automatic strategy selection based on task properties")

    # Generate command line suggestions
    suggestion_file = RESULTS_OUTPUT_DIR / f"test_commands_{run_timestamp_str}.txt"
    with open(suggestion_file, "w", encoding="utf-8") as f:
        f.write("# ARC Solver Comparison Test Commands\n\n")
        f.write("## Commands to test different strategies\n\n")
        f.write("# Run with CATENGINE for all tasks:\n")
        f.write(
            f'$env:FORCE_CATENGINE="True"; $env:MAX_ARC_TASKS="{max_tasks}"; python ARC.py\n\n'
        )
        f.write("# Run with ARC_SOLVER for all tasks:\n")
        f.write(
            f'$env:FORCE_ARC_SOLVER="True"; $env:MAX_ARC_TASKS="{max_tasks}"; python ARC.py\n\n'
        )
        f.write("# Run with automatic strategy selection:\n")
        f.write(f'$env:MAX_ARC_TASKS="{max_tasks}"; python ARC.py\n\n')
        f.write("## Tasks that need categorization:\n")
        for task_id in catengine_candidates:
            f.write(f"# {task_id}\n")
        f.write("\n## Tasks that don't need categorization:\n")
        for task_id in arc_solver_candidates:
            f.write(f"# {task_id}\n")

    logger.info(f"Generated test command suggestions in {suggestion_file}")

    # Return without running the tests - we'll just provide the setup
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ARC solver strategies")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to test (0 for all)",
    )
    parser.add_argument(
        "--force-catengine",
        action="store_true",
        help="Force using CATENGINE for all tasks",
    )
    parser.add_argument(
        "--force-arc-solver",
        action="store_true",
        help="Force using ARC_SOLVER for all tasks",
    )
    args = parser.parse_args()

    run_comparison_test(
        max_tasks=args.max_tasks,
        force_catengine=args.force_catengine,
        force_arc_solver=args.force_arc_solver,
    )
