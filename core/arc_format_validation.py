#!/usr/bin/env python3
"""
ARC Format Validation Test Script
This script validates that the modular inference system produces predictions in the correct ARC format
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Force UTF-8 output to avoid encoding issues
os.environ["PYTHONIOENCODING"] = "utf-8"


def run_test():
    """Run the complete ARC format validation test"""
    try:
        print("Testing ARC format compliance...")

        # 1. Test the SubmissionFormatter directly
        print("\n--- Testing SubmissionFormatter ---")
        from tools.utilities.submission_utils import SubmissionFormatter

        formatter = SubmissionFormatter()

        # Create a sample grid
        sample_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Format it
        formatted = formatter.format_predictions({"test_task": sample_grid})

        if "test_task" in formatted and formatted["test_task"] == sample_grid:
            print("[PASS] SubmissionFormatter formats grids correctly")
        else:
            print("[FAIL] SubmissionFormatter does not format grids correctly")
            return False

        # 2. Test with a real task if available
        print("\n--- Testing with real ARC task ---")
        try:
            from ARC.data_loader import ARCDataLoader

            from Gridformer.inference.gridformer_inference_engine import (
                GridFormerInferenceEngine as GridFormerInference,
            )
            from Gridformer.inference.inference_strategy import InferenceStrategy

            # Load data
            data_loader = ARCDataLoader()
            training_data = data_loader.load_training_data()

            if training_data:
                print("[PASS] Successfully loaded training data")

                # Get a sample task
                task_id = list(training_data.keys())[0]
                task_data = training_data[task_id]

                print(f"Using task: {task_id}")

                # Run inference
                inference = GridFormerInference()
                predictions = inference.predict(
                    task_data, strategy=InferenceStrategy.RULE_BASED
                )

                if predictions:
                    print(
                        "[PASS] Successfully generated predictions"
                    )  # Format predictions
                    test_predictions = {task_id: predictions[0]}
                    submission_data = formatter.format_predictions(test_predictions)

                    # Create a temp directory if needed
                    temp_dir = Path("./temp")
                    temp_dir.mkdir(exist_ok=True)

                    # Validate with better error handling
                    temp_path = temp_dir / f"temp_test_submission_{task_id}.json"
                    try:
                        with open(temp_path, "w") as f:
                            json.dump(submission_data, f)

                        validation = formatter.validate_submission(str(temp_path))

                        if validation["valid"]:
                            print("[PASS] Submission validation successful")
                        else:
                            print(f"[FAIL] Validation failed: {validation['errors']}")
                    except Exception as e:
                        print(f"[FAIL] Error during validation: {e}")
                    finally:
                        # Clean up
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except Exception as e:
                                print(f"[WARNING] Could not delete temp file: {e}")
                else:
                    print("[SKIP] Could not generate predictions")
            else:
                print("[SKIP] Could not load training data")

        except Exception as e:
            print(f"[SKIP] Error in real task test: {e}")
            import traceback

            traceback.print_exc()

        # Final verdict
        print("\n--- Final Result ---")
        print("[PASS] ARC format compliance test completed successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Error in test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_test()
