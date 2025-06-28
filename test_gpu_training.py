#!/usr/bin/env python
"""
Test GPU utilization during VoxSigil training
"""
import os
import sys
import time
import threading
sys.path.insert(0, '.')

def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 GPU Monitoring Started")
            for i in range(10):  # Monitor for 10 seconds
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    utilization = (reserved / total) * 100
                    
                    print(f"GPU {gpu_id}: {allocated:.2f}GB/{total:.2f}GB ({utilization:.1f}%)")
                
                time.sleep(1)
                print("-" * 40)
        else:
            print("❌ No CUDA available for monitoring")
    except Exception as e:
        print(f"❌ GPU monitoring error: {e}")

def test_training_with_monitoring():
    """Test training while monitoring GPU usage"""
    print("🎯 Starting Training Test with GPU Monitoring")
    print("=" * 60)
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_gpu_usage)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Import and test components
        from training.arc_grid_trainer import ARCGridTrainer
        
        config = {
            'grid_size': 30,
            'use_cuda': True,
            'arc_data_path': './arc_data',
            'use_art': False,  # Disable for simpler test
            'use_holo_mesh': False,
            'use_novel_paradigms': False
        }
        
        print("🔧 Initializing trainer...")
        trainer = ARCGridTrainer(config=config, vanta_core=None, grid_former=None)
        print("✅ Trainer initialized")
        
        # Test training
        training_config = {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001
        }
        
        print("🚀 Starting training...")
        success = trainer.start_coordinated_training(training_config)
        
        if success:
            print("✅ Training started successfully")
            # Let it run for a few seconds to see GPU usage
            time.sleep(5)
        else:
            print("❌ Training failed to start")
            
    except Exception as e:
        print(f"❌ Training test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Training test complete!")

if __name__ == "__main__":
    test_training_with_monitoring()
