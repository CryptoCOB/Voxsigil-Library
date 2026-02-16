"""
Test Debug Utility for MetaConsciousness SDK.

This script helps diagnose test failures by running specific tests
with enhanced logging and state tracing.
"""
import os
import sys
import unittest
import json
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_debug.log", mode='w')
    ]
)

logger = logging.getLogger("test_debug")

def run_specific_test(test_path, test_name=None, verbose=True):
    """
    Run a specific test with enhanced debugging.
    
    Args:
        test_path: Path to the test file
        test_name: Specific test method to run (optional)
        verbose: Whether to print verbose output
    """
    # Add root directory to path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Clear environment
    logger.info(f"Running test: {test_path}")
    
    # Record start time
    start_time = time.time()
    
    # Run test with unittest
    if test_name:
        test_spec = f"{test_path}:{test_name}"
        logger.info(f"Running specific test: {test_spec}")
        result = os.system(f"{sys.executable} -m unittest {test_spec}")
    else:
        logger.info(f"Running all tests in: {test_path}")
        result = os.system(f"{sys.executable} -m unittest {test_path}")
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    # Report result
    logger.info(f"Test completed in {duration:.2f} seconds with result code: {result}")
    
    # Check for test event logs
    test_events_dir = root_dir
    event_files = list(Path(test_events_dir).glob("test_events_*.json"))
    
    if event_files:
        logger.info(f"Found {len(event_files)} test event logs:")
        for event_file in event_files:
            logger.info(f"  {event_file}")
            
            # Analyze the event file
            with open(event_file) as f:
                events = json.load(f)
                logger.info(f"  Contains {len(events)} events")
                
                # Look for specific event types
                llm_events = [e for e in events if "LLM" in e.get("message", "")]
                omega3_events = [e for e in events if "Omega-3" in e.get("message", "")]
                error_events = [e for e in events if e.get("level", "") == "ERROR"]
                
                logger.info(f"  LLM events: {len(llm_events)}")
                logger.info(f"  Omega-3 events: {len(omega3_events)}")
                logger.info(f"  Error events: {len(error_events)}")
                
                # Print error events
                if error_events:
                    logger.error("  Error events found:")
                    for event in error_events:
                        logger.error(f"    {event['message']}")
    
    # Check for inference logs
    inference_logs = list(Path(root_dir).glob("sdk_inference*.txt"))
    if inference_logs:
        logger.info(f"Found {len(inference_logs)} inference logs:")
        for log_file in inference_logs:
            logger.info(f"  {log_file}")
            # Check file size
            size = os.path.getsize(log_file)
            logger.info(f"  Size: {size} bytes")
            
            # Check content if not too large
            if size < 10000:
                with open(log_file) as f:
                    content = f.read()
                    logger.info(f"  Content excerpt: {content[:200]}...")
    
    return result == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Debug Utility")
    parser.add_argument("test_path", help="Path to test file or module")
    parser.add_argument("--test", "-t", help="Specific test method to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Run the test
    success = run_specific_test(args.test_path, args.test, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
