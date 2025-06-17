#!/usr/bin/env python3
"""
VoxSigil Optimized GUI Launcher
Launch the fully optimized enhanced GUI with all performance improvements.
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

def main():
    """Launch the optimized GUI"""
    try:
        logger.info("🚀 Launching VoxSigil Optimized Enhanced GUI")
        logger.info("=" * 60)
        logger.info("Features:")
        logger.info("• Timeout protection (5-20s per tab)")
        logger.info("• Automatic retry (2-3 attempts)")
        logger.info("• Circuit breaker for failed tabs")
        logger.info("• Memory leak detection")
        logger.info("• Resource monitoring (CPU/RAM)")
        logger.info("• Background loading")
        logger.info("• Keyboard shortcuts (Ctrl+R/T/G)")
        logger.info("• Splash screen")
        logger.info("=" * 60)
        
        # Import and run the optimized GUI
        from optimized_enhanced_gui import main as run_optimized_gui
        return run_optimized_gui()
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure optimized_enhanced_gui.py is in the current directory")
        return 1
    except Exception as e:
        logger.error(f"❌ Launch error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
