#!/usr/bin/env python3
"""
System Launch Script for MetaConsciousness

This script provides a unified entry point for launching different components
of the MetaConsciousness system with proper initialization.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment (paths, etc.)."""
    # Add project root to path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added project root to sys.path: {project_root}")

    # Set environment variables if needed
    # os.environ["PYTHONIOENCODING"] = "utf-8"
    
    return True

def bootstrap_system():
    """Run system bootstrap to register components."""
    try:
        logger.info("Running system bootstrap...")
        
        # Import and run bootstrap
        from bootstrap import run_bootstrap, mark_bootstrap_completed
        
        success = run_bootstrap()
        
        if success:
            logger.info("Bootstrap successful")
            # Mark bootstrap as completed for downstream components
            mark_bootstrap_completed()
            return True
        else:
            logger.critical("Bootstrap failed!")
            return False
    except ImportError as e:
        logger.critical(f"Failed to import bootstrap module: {e}")
        return False
    except Exception as e:
        logger.critical(f"Error during bootstrap: {e}", exc_info=True)
        return False

def initialize_system():
    """Initialize the system and get SDK context."""
    try:
        logger.info("Initializing system components...")
        
        # Import system initialization module
        from MetaConsciousness.core.system_init import initialize_system as init_system
        
        # Run initialization
        sdk_context = init_system({})
        
        if sdk_context is None:
            logger.critical("System initialization failed! SDK context is None.")
            return None
            
        # Verify meta_core is present in context
        if not sdk_context.get('meta_core'):
            logger.critical("CRITICAL: 'meta_core' component not found in SDK context after initialization!")
            return None
            
        logger.info(f"System initialized successfully with meta_core of type: {type(sdk_context.get('meta_core')).__name__}")
        return sdk_context
        
    except ImportError as e:
        logger.critical(f"Failed to import system initialization module: {e}")
        return None
    except Exception as e:
        logger.critical(f"Error during system initialization: {e}", exc_info=True)
        return None

def launch_gui(context=None):
    """Launch the GUI interface with provided context."""
    try:
        logger.info("Launching GUI...")
        
        if context is None:
            logger.critical("Cannot launch GUI: No valid context provided")
            return False
            
        # Import the launch_application function
        from MetaConsciousness.interface.launch_gui import launch_application
        
        # Launch the GUI with the provided context
        launch_application(context=context)
        
        return True
    except ImportError as e:
        logger.critical(f"Failed to import GUI components: {e}")
        return False
    except Exception as e:
        logger.critical(f"Error launching GUI: {e}", exc_info=True)
        return False

def main():
    """Main entry point with strict execution order and error handling."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch the MetaConsciousness system")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap phase")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Step 1: Setup environment
        if not setup_environment():
            logger.critical("Environment setup failed!")
            return 1
        
        # Step 2: Run bootstrap unless skipped
        if not args.skip_bootstrap:
            if not bootstrap_system():
                logger.critical("System bootstrap failed!")
                return 1
        else:
            logger.warning("Bootstrap skipped as requested (--skip-bootstrap)")
        
        # Step 3: Initialize system and get SDK context
        sdk_context = initialize_system()
        if sdk_context is None:
            logger.critical("Failed to get valid SDK context. Cannot continue.")
            return 1
        
        # Step 4: Verify meta_core is present in context
        meta_core = sdk_context.get('meta_core')
        if meta_core is None:
            logger.critical("CRITICAL: 'meta_core' component not found in context after initialization!")
            return 1
        
        logger.info(f"Successfully initialized system with meta_core of type: {type(meta_core).__name__}")
        
        # Step 5: Launch GUI with the initialized context
        if not launch_gui(context=sdk_context):
            logger.critical("GUI launch failed!")
            return 1
        
        logger.info("MetaConsciousness system terminated normally")
        return 0
        
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
