#!/bin/bash

echo "================================"
echo "VoxSigil Library - GUI Launcher"
echo "================================"
echo

echo "Starting VoxSigil GUI..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    echo "Please install Python 3.8+ and add it to your PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python command: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Launch the GUI
echo "Launching Dynamic GridFormer GUI..."
$PYTHON_CMD -m gui.components.dynamic_gridformer_gui

# If that fails, try alternative launch method
if [ $? -ne 0 ]; then
    echo
    echo "Primary launch failed, trying alternative method..."
    $PYTHON_CMD test_pyqt5_gui.py
fi

# If still failing, provide diagnostics
if [ $? -ne 0 ]; then
    echo
    echo "GUI launch failed. Running diagnostics..."
    echo
    
    echo "Checking dependencies..."
    $PYTHON_CMD -c "import PyQt5; print('PyQt5: OK')" 2>/dev/null || echo "PyQt5: MISSING - Install with: pip install PyQt5"
    $PYTHON_CMD -c "import torch; print('torch: OK')" 2>/dev/null || echo "torch: MISSING - Install with: pip install torch"
    $PYTHON_CMD -c "import numpy; print('numpy: OK')" 2>/dev/null || echo "numpy: MISSING - Install with: pip install numpy"
    
    echo
    echo "To fix missing dependencies, run:"
    echo "pip install PyQt5 torch numpy transformers"
    echo
    echo "Or run the automated setup:"
    echo "$PYTHON_CMD quick_setup.py"
    echo
fi

echo
read -p "Press Enter to continue..."
