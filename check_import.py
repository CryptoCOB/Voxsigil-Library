#!/usr/bin/env python3
import sys
import importlib
import traceback

# Add the current directory to the path
sys.path.append('.')

# Open a file for writing output
with open('import_check_results.txt', 'w') as f:
    try:
        # Import the module
        f.write("Attempting to import working_gui.complete_live_gui...\n")
        module = importlib.import_module('working_gui.complete_live_gui')
        f.write("Module imported successfully\n")
        
        # Print the module attributes
        f.write(f"Module attributes: {dir(module)}\n")
        
        # Try to get the CompleteVoxSigilGUI class
        if hasattr(module, 'CompleteVoxSigilGUI'):
            f.write("CompleteVoxSigilGUI class found in module\n")
        else:
            f.write("CompleteVoxSigilGUI class NOT found in module\n")
            
    except ImportError as e:
        f.write(f"Import error: {e}\n")
        f.write(traceback.format_exc())
    except Exception as e:
        f.write(f"Other error: {e}\n")
        f.write(traceback.format_exc())
