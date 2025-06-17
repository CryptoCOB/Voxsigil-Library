# voxsigil_supervisor/examples/arc_task_example.py
"""
Example of using the ARCAwareLLMInterface for ARC-style reasoning tasks.

This example demonstrates how to use the ARC-aware LLM interface both directly
and through the VoxSigil VANTA supervisor for symbolic reasoning tasks.
"""

import json
import sys
from pathlib import Path

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import VoxSigil components
from .llm.arc_llm_interface import ARCAwareLLMInterface

try:
    from Vanta.integration.vanta_integration import create_vanta_supervisor
    from Vanta.integration.vanta_supervisor import (
        VantaSigilSupervisor as VANTASupervisor,
    )

    HAS_VANTA = True
except ImportError:
    HAS_VANTA = False


def run_direct_arc_example(model_path, task_data):
    """Run an ARC task directly with the ARC-aware LLM interface."""
    print("🧠 Running ARC task directly with ARCAwareLLMInterface...")

    # Create the interface
    llm = ARCAwareLLMInterface(model_path, temperature=0.2)

    # Enhanced symbolic context with more specific pattern guidance
    symbolic_context = (
        "Apply recursive reasoning and category alignment. "
        "Look for patterns that repeat or alternate across rows and columns. "
        "Check if the output grid is a scaled or tiled version of the input grid. "
        "Identify transformation patterns like repetition, reflection, or rotation. "
        "Pay attention to the relationship between input and output dimensions."
    )  # Solve the task with increased retries
    prediction = llm.solve_arc_task(
        train_pairs=task_data["train"],
        test_input=task_data["test"][0]["input"],
        base_symbolic_context_query=symbolic_context,
        num_retries=3,  # Increased from default for better chance of success
        verbose=True,  # Enable verbose logging to see pattern analysis
    )

    print("\n✅ Prediction:")
    for row in prediction["predicted_grid"]:
        print(row)

    # Compare with ground truth if available
    if "output" in task_data["test"][0]:
        ground_truth = task_data["test"][0]["output"]
        print("\n🎯 Ground Truth:")
        for row in ground_truth:
            print(row)

        if prediction == ground_truth:
            print("\n✨ SUCCESS: Prediction matches ground truth!")
        else:
            print("\n❌ FAILURE: Prediction does not match ground truth.")

    return prediction


def run_vanta_arc_example(model_path, task_data):
    """Run an ARC task through the VANTA supervisor."""
    if not HAS_VANTA:
        print("❌ VANTA supervisor not available. Skipping VANTA example.")
        return None

    print("🧠 Running ARC task through VANTA supervisor...")

    # Create the LLM interface with slightly higher temperature for creative thinking
    llm = ARCAwareLLMInterface(model_path, temperature=0.3)

    # Create a more robust RAG interface for this example
    class EnhancedRagInterface:
        def retrieve_context(self, query, context=None, k=5):
            return [
                {
                    "content": "Apply pattern recognition and systematic reasoning to ARC tasks.",
                    "source": "sigils/critical_lens.voxsigil",
                    "_similarity_score": 0.90,
                },
                {
                    "content": "Look for transformation rules between input and output grids. Pay special attention to grid repetition patterns.",
                    "source": "sigils/insight_nucleator.voxsigil",
                    "_similarity_score": 0.88,
                },
                {
                    "content": "Check if the output grid dimensions are multiples of the input grid dimensions. This often indicates tiling or repetition.",
                    "source": "sigils/pattern_detector.voxsigil",
                    "_similarity_score": 0.86,
                },
                {
                    "content": "Look for alternating patterns in the output grid, where values may switch in a regular sequence.",
                    "source": "sigils/alternation_detector.voxsigil",
                    "_similarity_score": 0.82,
                },
                {
                    "content": "Verify your solution by checking that the pattern is consistent across the entire grid.",
                    "source": "sigils/consistency_checker.voxsigil",
                    "_similarity_score": 0.80,
                },
            ]

    # Create the VANTA supervisor
    vanta = create_vanta_supervisor(rag_interface=EnhancedRagInterface(), llm_interface=llm)

    # Check if VANTA supervisor is available
    if vanta is None:
        print("❌ VANTA supervisor not available. Skipping VANTA example.")
        return None

    # Format the task as a query with enhanced instructions
    train_examples = json.dumps(task_data["train"], indent=2)
    test_input = json.dumps(task_data["test"][0]["input"], indent=2)

    # Calculate dimensions for guidance
    test_rows = len(task_data["test"][0]["input"])
    test_cols = len(task_data["test"][0]["input"][0]) if test_rows > 0 else 0

    # Analyze train examples for dimension scaling
    scales_row = []
    scales_col = []
    for pair in task_data["train"]:
        in_rows = len(pair["input"])
        in_cols = len(pair["input"][0]) if in_rows > 0 else 0
        out_rows = len(pair["output"])
        out_cols = len(pair["output"][0]) if out_rows > 0 else 0

        if in_rows > 0 and out_rows > 0:
            scales_row.append(out_rows / in_rows)
        if in_cols > 0 and out_cols > 0:
            scales_col.append(out_cols / in_cols)

    dimension_guidance = ""
    if scales_row and scales_col:
        avg_scale_row = sum(scales_row) / len(scales_row)
        avg_scale_col = sum(scales_col) / len(scales_col)

        # Round if close to integer
        if abs(round(avg_scale_row) - avg_scale_row) < 0.05:
            avg_scale_row = round(avg_scale_row)
        if abs(round(avg_scale_col) - avg_scale_col) < 0.05:
            avg_scale_col = round(avg_scale_col)

        predicted_rows = int(test_rows * avg_scale_row)
        predicted_cols = int(test_cols * avg_scale_col)

        dimension_guidance = f"""
Based on the training examples, I observe a pattern where:
- Input rows are scaled by approximately {avg_scale_row}x in the output
- Input columns are scaled by approximately {avg_scale_col}x in the output
- The test input is {test_rows}x{test_cols}
- Therefore, the expected output dimensions are approximately {predicted_rows}x{predicted_cols}
"""

    query = f"""Solve this ARC reasoning task:
    
Training examples:
{train_examples}

Test input:
{test_input}

{dimension_guidance}

When solving this task:
1. Identify the transformation rule from input to output in the training examples
2. Pay special attention to repetition patterns and dimension scaling
3. Look for alternating values in rows, columns, or blocks
4. Check if the output is created by tiling the input with transformations
5. Apply the transformation rule consistently to the test input
6. Verify your solution has the correct dimensions and pattern consistency

Return the output grid for the test input as a JSON array.
"""

    # Run the task through VANTA
    response = vanta.orchestrate_thought_cycle(query)

    print("\n✅ VANTA Response:")
    print(response)

    # Try to extract grid from response
    try:
        # Find the first occurrence of '[' and last occurrence of ']'
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            prediction = json.loads(json_str)

            print("\n✅ Extracted Prediction:")
            for row in prediction:
                print(row)

            # Compare with ground truth if available
            if "output" in task_data["test"][0]:
                ground_truth = task_data["test"][0]["output"]
                print("\n🎯 Ground Truth:")
                for row in ground_truth:
                    print(row)

                if prediction == ground_truth:
                    print("\n✨ SUCCESS: Prediction matches ground truth!")
                else:
                    print("\n❌ FAILURE: Prediction does not match ground truth.")

            return prediction
    except:
        print("\n❌ Could not extract a valid grid from VANTA response.")
        return None


def main():
    """Main entry point for the example."""
    # Check if model path is provided as command-line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default model path - adjust as needed
        model_path = "llm/models/Qwen/8B"
        print(f"No model path provided. Using default: {model_path}")
        print(
            "You can specify a model path as an argument: python arc_task_example.py path/to/model"
        )

    # Simple example ARC task
    task_data = {
        "train": [
            {
                "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            },
            {
                "input": [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                "output": [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            },
        ],
        "test": [
            {
                "input": [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
                "output": [[0, 1, 1], [0, 1, 0], [1, 1, 0]],
            }
        ],
    }

    # Run direct ARC example
    prediction_direct = run_direct_arc_example(model_path, task_data)

    print("\n" + "-" * 50 + "\n")

    # Run VANTA ARC example
    prediction_vanta = run_vanta_arc_example(model_path, task_data)


if __name__ == "__main__":
    main()
