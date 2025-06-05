#!/usr/bin/env python3
"""
🎨 ARC Task Visualizer - Side-by-side Input/Output/Prediction Comparison
Creates visual grids to see what the model is actually doing
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, cast

# ARC color palette for consistent visualization
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


def grid_to_image(grid: List[List[int]], title: str = "") -> np.ndarray:
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


def visualize_arc_task(
    task: Dict[str, Any],
    predictions: Optional[List[List[List[int]]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Visualize complete ARC task with side-by-side comparison"""

    train_examples = task.get("train", [])
    test_examples = task.get("test", [])

    # Count total plots needed
    num_train = len(train_examples)
    num_test = len(test_examples)

    if predictions:
        # Training examples (input + output) + Test examples (input + prediction)
        total_plots = num_train * 2 + num_test * 2
        cols = 4  # input, output, test_input, prediction
        rows = max(num_train, num_test)
    else:
        # Just training examples
        total_plots = num_train * 2
        cols = 2  # input, output
        rows = num_train

    if total_plots == 0:
        print("❌ No data to visualize")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # Ensure axes is always 2D for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "🎯 ARC Task Analysis: Input → Output → Prediction",
        fontsize=16,
        fontweight="bold",
    )

    # Training examples
    for i, example in enumerate(train_examples):
        input_grid = example["input"]
        output_grid = example["output"]

        # Input
        input_img = grid_to_image(input_grid)
        axes[i][0].imshow(input_img, interpolation="nearest")
        axes[i][0].set_title(
            f"📥 Train Input {i + 1}\n{len(input_grid)}×{len(input_grid[0]) if input_grid else 0}",
            fontweight="bold",
        )
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])

        # Output
        output_img = grid_to_image(output_grid)
        axes[i][1].imshow(output_img, interpolation="nearest")
        axes[i][1].set_title(
            f"📤 Train Output {i + 1}\n{len(output_grid)}×{len(output_grid[0]) if output_grid else 0}",
            fontweight="bold",
        )
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])

        # Test input and predictions (if available)
        if predictions and i < len(test_examples):
            test_input = test_examples[i]["input"]
            prediction = predictions[i] if i < len(predictions) else test_input

            # Test input
            test_input_img = grid_to_image(test_input)
            axes[i][2].imshow(test_input_img, interpolation="nearest")
            axes[i][2].set_title(
                f"🧪 Test Input {i + 1}\n{len(test_input)}×{len(test_input[0]) if test_input else 0}",
                fontweight="bold",
                color="blue",
            )
            axes[i][2].set_xticks([])
            axes[i][2].set_yticks([])

            # Prediction
            prediction_img = grid_to_image(prediction)
            axes[i][3].imshow(prediction_img, interpolation="nearest")
            axes[i][3].set_title(
                f"🔮 Prediction {i + 1}\n{len(prediction)}×{len(prediction[0]) if prediction else 0}",
                fontweight="bold",
                color="red",
            )
            axes[i][3].set_xticks([])
            axes[i][3].set_yticks([])

    # Hide unused subplots
    for i in range(rows):
        for j in range(cols):
            if i >= num_train or (not predictions and j >= 2):
                axes[i][j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Visualization saved to: {save_path}")

    if show:
        plt.show()

    return fig


def test_visualization_with_real_data():
    """Test visualization with actual ARC data and predictions"""
    print("🎨 Testing ARC Task Visualization...")

    try:
        # Try to load real data
        from ARC.data_loader import ARCDataLoader
        from Gridformer.inference.gridformer_inference_engine import (
            GridFormerInferenceEngine as GridFormerInference,
        )
        from Gridformer.inference.inference_strategy import InferenceStrategy

        loader = ARCDataLoader()
        # Cast the loader result to the expected list type so type-checkers accept indexing
        training_data = cast(List[Dict[str, Any]], loader.load_training_data())

        if not training_data:
            print("❌ No training data available")
            return False

        # Use first task
        task = training_data[0]
        print(
            f"✅ Loaded task with {len(task.get('train', []))} training examples, {len(task.get('test', []))} test examples"
        )

        # Create inference engine and make predictions
        inference = GridFormerInference()
        predictions = inference.predict(task, strategy=InferenceStrategy.RULE_BASED)

        print(f"✅ Generated {len(predictions)} predictions")

        # Visualize
        save_path = "visualization_results/arc_task_comparison.png"
        Path("visualization_results").mkdir(exist_ok=True)

        fig = visualize_arc_task(task, predictions, save_path=save_path, show=False)

        print("✅ Visualization complete!")
        print(f"📁 Saved to: {save_path}")

        return True

    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_simple_demo():
    """Create a simple demo visualization"""
    print("🎨 Creating simple demo visualization...")

    # Create simple demo task
    demo_task = {
        "train": [
            {
                "input": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                "output": [[1, 1, 2], [1, 2, 1], [2, 1, 1]],
            },
            {
                "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "output": [[2, 1, 2], [1, 2, 1], [2, 1, 2]],
            },
        ],
        "test": [{"input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}],
    }

    # Simple prediction (just modify the input slightly)
    demo_predictions = [
        [[2, 1, 2], [1, 2, 1], [2, 1, 2]]  # Prediction for test
    ]

    save_path = "visualization_results/demo_comparison.png"
    Path("visualization_results").mkdir(exist_ok=True)

    fig = visualize_arc_task(
        demo_task, demo_predictions, save_path=save_path, show=False
    )

    print("✅ Demo visualization created!")
    print(f"📁 Saved to: {save_path}")

    return True


def analyze_model_results(results_file: str = "test_results/batch_test_results.json"):
    """Analyze and visualize model results from a results file"""
    print(f"🔍 Analyzing model results from: {results_file}")

    try:
        if not Path(results_file).exists():
            print(f"❌ Results file not found: {results_file}")
            return False

        with open(results_file, "r") as f:
            results = json.load(f)

        print(
            f"✅ Loaded results with {len(results.get('individual_results', []))} samples"
        )

        # Visualize first few successful predictions
        individual_results = results.get("individual_results", [])
        successful_results = [r for r in individual_results if r.get("success", False)]

        print(f"📊 Found {len(successful_results)} successful predictions")

        if successful_results:
            # Visualize first successful result
            first_result = successful_results[0]
            task_data = first_result.get("task_data", {})
            predictions = first_result.get("predictions", [])

            if task_data and predictions:
                save_path = "visualization_results/model_results_analysis.png"
                Path("visualization_results").mkdir(exist_ok=True)

                fig = visualize_arc_task(
                    task_data, predictions, save_path=save_path, show=False
                )

                print("✅ Model results visualization complete!")
                print(f"📁 Saved to: {save_path}")
                return True

        print("❌ No suitable data found for visualization")
        return False

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main visualization testing function"""
    print("🎨 ARC Task Visualizer - Reality Check")
    print("=" * 50)

    # Create visualization results directory
    Path("visualization_results").mkdir(exist_ok=True)

    tests = [
        ("Simple Demo", create_simple_demo),
        ("Real Data Test", test_visualization_with_real_data),
        ("Model Results Analysis", analyze_model_results),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name} completed successfully")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")

    print("\n📁 Check 'visualization_results/' folder for generated images")


if __name__ == "__main__":
    main()
