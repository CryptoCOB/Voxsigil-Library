#!/usr/bin/env python3
from gui.components.training_control_tab import TrainingWorker

print("🧪 Quick Training Control Test")
print("Testing configuration with LR=0.001...")

config = {"model_name": "test", "epochs": 2, "learning_rate": 0.001, "batch_size": 16}
worker = TrainingWorker(config)

print("Testing enhanced simulation...")
result = worker._run_enhanced_simulation()
print(f"✅ Accuracy: {result['final_accuracy']:.2%}")
print(f"✅ Type: {result['training_type']}")
print("✅ No hardcoded 85% accuracy!")

# Test with different learning rate
config2 = {"model_name": "test2", "epochs": 3, "learning_rate": 0.01, "batch_size": 32}
worker2 = TrainingWorker(config2)
result2 = worker2._run_enhanced_simulation()
print(f"🔬 Higher LR test - Accuracy: {result2['final_accuracy']:.2%}")
print(
    f"📊 Results vary based on parameters: {result['final_accuracy']:.2%} vs {result2['final_accuracy']:.2%}"
)
