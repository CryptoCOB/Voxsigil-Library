"""
Main entry point for executing the visualizer module directly with:
python -m tools.visualizer [args]
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
