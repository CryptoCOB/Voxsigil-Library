#!/usr/bin/env python3
"""
VoxSigil Performance Comparison Tool
Compares startup performance between different GUI versions
"""

import subprocess
import time
import os
import sys
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Results storage
RESULTS_DIR = "performance_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Available GUI launchers to test
LAUNCHERS = {
    "original": "launch_enhanced_gui.py",
    "profiling": "launch_enhanced_gui_with_profiling.py",
    "optimized": "launch_optimized_gui.py"
}

def run_performance_test(launcher_name, max_runtime=30):
    """
    Run a performance test on the specified launcher
    
    Args:
        launcher_name: Name of the launcher to test (key from LAUNCHERS dict)
        max_runtime: Maximum runtime in seconds before terminating
        
    Returns:
        Dictionary with performance metrics
    """
    if launcher_name not in LAUNCHERS:
        print(f"Error: Unknown launcher '{launcher_name}'")
        return None
        
    launcher_path = LAUNCHERS[launcher_name]
    
    print(f"Running performance test for {launcher_name} ({launcher_path})...")
    print(f"Maximum runtime: {max_runtime} seconds")
    
    start_time = time.time()
    
    # Start the process
    process = subprocess.Popen(
        [sys.executable, launcher_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Collect output and parse performance data
    stdout_lines = []
    performance_data = {
        "total_startup_time": None,
        "components": {},
        "phases": {}
    }
    
    try:
        # Read output until timeout or process ends
        while time.time() - start_time < max_runtime and process.poll() is None:
            output = process.stdout.readline()
            if output:
                stdout_lines.append(output.strip())
                print(f"  > {output.strip()}")
                
                # Parse performance data
                # Check for performance logger output
                if "PERF:" in output:
                    perf_match = re.search(r"PERF: ([^:]+): (\d+\.\d+) seconds", output)
                    if perf_match:
                        phase = perf_match.group(1).strip()
                        seconds = float(perf_match.group(2))
                        performance_data["phases"][phase] = seconds
                
                # Check for total initialization time
                total_time_match = re.search(r"Total initialization time: (\d+\.\d+) seconds", output)
                if total_time_match:
                    performance_data["total_startup_time"] = float(total_time_match.group(1))
                    
                # Check for component initialization
                component_match = re.search(r"✅ ([^:]+) initialized in (\d+\.\d+)s", output)
                if component_match:
                    component = component_match.group(1).strip()
                    seconds = float(component_match.group(2))
                    performance_data["components"][component] = seconds
                    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Kill the process if it's still running
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        # Get remaining output
        stdout, stderr = process.communicate()
        if stdout:
            for line in stdout.splitlines():
                stdout_lines.append(line.strip())
                print(f"  > {line.strip()}")
                
        if stderr:
            for line in stderr.splitlines():
                print(f"  ! {line.strip()}")
    
    # If we didn't get a total time, calculate it
    if performance_data["total_startup_time"] is None:
        performance_data["total_startup_time"] = time.time() - start_time
    
    print(f"Test completed in {performance_data['total_startup_time']:.2f} seconds")
    
    # Add metadata
    performance_data["launcher"] = launcher_name
    performance_data["launcher_path"] = launcher_path
    performance_data["timestamp"] = datetime.now().isoformat()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"{launcher_name}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
        
    print(f"Results saved to {results_file}")
    
    return performance_data

def compare_results(results):
    """
    Compare and visualize results from multiple tests
    
    Args:
        results: List of result dictionaries from run_performance_test
    """
    if not results:
        print("No results to compare")
        return
        
    # Extract and organize data for comparison
    launchers = [r["launcher"] for r in results]
    total_times = [r["total_startup_time"] for r in results]
    
    # Create phase comparison
    all_phases = set()
    for r in results:
        all_phases.update(r.get("phases", {}).keys())
    
    phase_data = {}
    for phase in all_phases:
        phase_data[phase] = [r.get("phases", {}).get(phase, 0) for r in results]
    
    # Create component comparison
    all_components = set()
    for r in results:
        all_components.update(r.get("components", {}).keys())
    
    component_data = {}
    for comp in all_components:
        component_data[comp] = [r.get("components", {}).get(comp, 0) for r in results]
    
    # Create figures
    plt.figure(figsize=(10, 6))
    plt.bar(launchers, total_times)
    plt.title('Total Startup Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.ylim(bottom=0)
    
    for i, v in enumerate(total_times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.png")
    plt.savefig(chart_file)
    
    print(f"Comparison chart saved to {chart_file}")
    
    # Create detailed phase comparison if we have phase data
    if phase_data:
        plt.figure(figsize=(12, 8))
        x = range(len(launchers))
        width = 0.8 / len(phase_data)
        
        i = 0
        for phase, times in phase_data.items():
            plt.bar([p + (i * width) for p in x], times, width=width, label=phase)
            i += 1
        
        plt.title('Startup Phases Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks([p + 0.4 for p in x], launchers)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        # Save chart
        phases_chart_file = os.path.join(RESULTS_DIR, f"phases_comparison_{timestamp}.png")
        plt.savefig(phases_chart_file)
        
        print(f"Phases comparison chart saved to {phases_chart_file}")
    
    # Print textual comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY:")
    print("=" * 60)
    
    for i, launcher in enumerate(launchers):
        print(f"{launcher}: {total_times[i]:.2f}s total startup time")
    
    if len(launchers) > 1:
        baseline = total_times[0]
        for i in range(1, len(launchers)):
            improvement = baseline - total_times[i]
            percent = (improvement / baseline) * 100
            print(f"{launchers[i]} vs {launchers[0]}: {improvement:.2f}s faster ({percent:.1f}% improvement)")
    
    print("\nPhase-by-phase comparison:")
    for phase, times in phase_data.items():
        print(f"  {phase}:")
        for i, launcher in enumerate(launchers):
            if times[i] > 0:
                print(f"    {launcher}: {times[i]:.2f}s")

def main():
    """Run performance comparison tests"""
    print("VoxSigil Performance Comparison Tool")
    print("=" * 60)
    
    # List available launchers
    print("Available GUI versions to test:")
    for key, path in LAUNCHERS.items():
        print(f"  - {key}: {path}")
    
    # Ask which launchers to test
    print("\nWhich launchers do you want to test? (comma-separated list, e.g. 'original,optimized')")
    print("Press Enter for all launchers.")
    
    choice = input("> ").strip()
    
    if choice:
        launchers_to_test = [l.strip() for l in choice.split(",")]
    else:
        launchers_to_test = list(LAUNCHERS.keys())
    
    # Validate choices
    valid_launchers = []
    for launcher in launchers_to_test:
        if launcher in LAUNCHERS:
            valid_launchers.append(launcher)
        else:
            print(f"Warning: Unknown launcher '{launcher}' - skipping")
    
    if not valid_launchers:
        print("No valid launchers selected. Exiting.")
        return
    
    # Ask about max runtime
    print("\nMaximum runtime for each test in seconds (default: 30):")
    max_runtime_input = input("> ").strip()
    max_runtime = int(max_runtime_input) if max_runtime_input else 30
    
    # Run tests
    results = []
    for launcher in valid_launchers:
        result = run_performance_test(launcher, max_runtime)
        if result:
            results.append(result)
        print("\n")
    
    # Compare results
    if len(results) > 1:
        compare_results(results)
    
    print("\nPerformance testing complete.")

if __name__ == "__main__":
    main()
