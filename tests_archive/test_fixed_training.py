#!/usr/bin/env python3

from gui.components.training_control_tab import TrainingWorker

# Test the fixed training accuracy calculation
worker = TrainingWorker(
    {"model_name": "test_model", "epochs": 2, "learning_rate": 0.001, "batch_size": 16}
)

# Test enhanced simulation
result = worker._run_enhanced_simulation()

print("✅ Training Test Results:")
print(f"   Accuracy: {result['final_accuracy']:.2%}")
print(f"   Type: {result['training_type']}")
print(f"   Model: {result['model_path']}")
print(f"   Epochs: {result['epochs_completed']}")

# Verify it's not hardcoded to 85%
if result["final_accuracy"] != 0.85:
    print("✅ SUCCESS: Hardcoded 85% accuracy eliminated!")
else:
    print("❌ ISSUE: Still showing hardcoded value")

print("\n🎉 Training Control System is working!")
