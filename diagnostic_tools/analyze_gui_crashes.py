#!/usr/bin/env python3
"""
GUI Crash Analysis Tool
Monitors and reports on GUI crashes with detailed logging.
"""

import sys
import os
import logging
import time
from pathlib import Path

def setup_crash_logging():
    """Set up comprehensive crash logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create crash log with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    crash_log_file = log_dir / f"gui_crash_{timestamp}.log"
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(crash_log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return crash_log_file

def analyze_recent_logs():
    """Analyze recent log files for crash patterns."""
    print("📊 Analyzing recent GUI logs...")
    
    log_files = []
    for pattern in ["*.log", "logs/*.log"]:
        log_files.extend(Path(".").glob(pattern))
    
    # Sort by modification time
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    crash_indicators = [
        "error", "crash", "exception", "traceback", 
        "failed", "abort", "segmentation fault",
        "access violation", "memory error"
    ]
    
    for log_file in log_files[:5]:  # Check last 5 log files
        print(f"\n🔍 Checking: {log_file}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                
                for indicator in crash_indicators:
                    if indicator in content:
                        print(f"⚠️  Found '{indicator}' in {log_file}")
                        
                        # Show context around the error
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if indicator in line:
                                start = max(0, i-2)
                                end = min(len(lines), i+3)
                                print(f"   Context (lines {start}-{end}):")
                                for j in range(start, end):
                                    marker = ">>> " if j == i else "    "
                                    print(f"   {marker}{lines[j]}")
                                break
                        break
        except Exception as e:
            print(f"   Error reading {log_file}: {e}")

def main():
    print("🔧 VoxSigil GUI Crash Analysis Tool")
    print("=" * 50)
    
    # Analyze existing logs
    analyze_recent_logs()
    
    print("\n💡 CRASH DEBUGGING TIPS:")
    print("1. Run 'python gradual_gui_test.py' to test step-by-step")
    print("2. Check the gradual test window - it will show where it crashes")
    print("3. Look for error messages in the step-by-step loading")
    print("4. Common crash points:")
    print("   - Real-time data provider initialization")
    print("   - Enhanced tab component imports")
    print("   - VantaCore integration attempts")
    print("   - PyQt5 widget creation with heavy components")
    
    print(f"\n📝 New crash logs will be saved to: logs/gui_crash_*.log")
    
    # Set up for future crash logging
    crash_log_file = setup_crash_logging()
    print(f"📝 Crash logging enabled: {crash_log_file}")
    
    return 0

if __name__ == "__main__":
    main()
