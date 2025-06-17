"""
Profiling tool for VoxSigil-Library application.
This script profiles the application startup and runtime performance to identify bottlenecks.
"""
import cProfile
import pstats
import io
import sys
import os
import time
from pathlib import Path

# Add parent directory to path so we can import from the library
sys.path.append(str(Path(__file__).parent.parent))

def profile_startup():
    """Profile the application startup sequence."""
    print("Profiling application startup...")
    
    # Create a profile object
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    # Import and run the startup sequence
    try:
        from launch_enhanced_gui import main
        main()
    except KeyboardInterrupt:
        print("Profiling stopped by user")
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        # Stop profiling
        pr.disable()
        
        # Print stats sorted by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Print top 50 time-consuming functions
        print(s.getvalue())
        
        # Save stats to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"profile_results_{timestamp}.txt"
        output_path = Path(__file__).parent.parent / "logs" / output_file
        os.makedirs(output_path.parent, exist_ok=True)
        
        with open(output_path, 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()
            
        print(f"Full profile results saved to {output_path}")

def profile_specific_component(component_import_path, function_name, *args, **kwargs):
    """
    Profile a specific component or function.
    
    Args:
        component_import_path: Import path (e.g., 'gui.components.heartbeat_monitor_tab')
        function_name: Function or method to profile
        *args, **kwargs: Arguments to pass to the function
    """
    print(f"Profiling {component_import_path}.{function_name}...")
    
    # Import the component
    import importlib
    module = importlib.import_module(component_import_path)
    
    # Get the function or class
    components = function_name.split('.')
    obj = module
    for comp in components:
        obj = getattr(obj, comp)
    
    # Create profile object
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    # Call the function
    try:
        result = obj(*args, **kwargs)
    except Exception as e:
        print(f"Error during profiling: {e}")
        result = None
    finally:
        # Stop profiling
        pr.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)
        print(s.getvalue())
        
        # Save stats to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        component_name = component_import_path.split('.')[-1]
        output_file = f"profile_{component_name}_{function_name.split('.')[-1]}_{timestamp}.txt"
        output_path = Path(__file__).parent.parent / "logs" / output_file
        os.makedirs(output_path.parent, exist_ok=True)
        
        with open(output_path, 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()
            
        print(f"Profile results saved to {output_path}")
    
    return result

def memory_usage_monitoring():
    """Monitor memory usage of the application over time."""
    try:
        import psutil
        import matplotlib.pyplot as plt
        import numpy as np
        from threading import Thread
        import time
        
        process = psutil.Process(os.getpid())
        
        # Initialize data storage
        timestamps = []
        memory_usage = []
        
        # Start the application in a separate thread
        def run_app():
            from launch_enhanced_gui import main
            main()
            
        app_thread = Thread(target=run_app)
        app_thread.daemon = True
        app_thread.start()
        
        print("Monitoring memory usage... Press Ctrl+C to stop")
        
        # Monitor memory usage
        start_time = time.time()
        try:
            while app_thread.is_alive():
                current_time = time.time() - start_time
                mem = process.memory_info().rss / (1024 * 1024)  # MB
                
                timestamps.append(current_time)
                memory_usage.append(mem)
                
                print(f"Time: {current_time:.2f}s, Memory: {mem:.2f} MB")
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_usage)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True)
        
        # Save plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"memory_profile_{timestamp}.png"
        output_path = Path(__file__).parent.parent / "logs" / output_file
        plt.savefig(output_path)
        
        print(f"Memory profile saved to {output_path}")
        
    except ImportError:
        print("Error: Required packages not found. Install with: pip install psutil matplotlib")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile VoxSigil-Library application")
    parser.add_argument("--startup", action="store_true", help="Profile application startup")
    parser.add_argument("--component", help="Component import path to profile (e.g., 'gui.components.heartbeat_monitor_tab')")
    parser.add_argument("--function", help="Function name to profile")
    parser.add_argument("--memory", action="store_true", help="Monitor memory usage over time")
    
    args = parser.parse_args()
    
    if args.startup:
        profile_startup()
    elif args.component and args.function:
        profile_specific_component(args.component, args.function)
    elif args.memory:
        memory_usage_monitoring()
    else:
        parser.print_help()
