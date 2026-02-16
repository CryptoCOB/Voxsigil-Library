#!/usr/bin/env python
"""
LM Studio Configuration Launcher

This script launches the LM Studio Configuration UI,
which allows you to manage OpenAI-compatible endpoints and model preferences.
"""

import os
import sys
import logging

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from MetaConsciousness.interface.lmstudio_config_ui import run
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the UI
    if __name__ == "__main__":
        print("Starting LM Studio Configuration UI...")
        run()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the MetaConsciousness package is installed correctly.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)