@echo off
echo ================================
echo VoxSigil Library - GUI Launcher
echo ================================
echo.
echo Starting VoxSigil GUI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Launch the GUI
echo Launching Dynamic GridFormer GUI...
python -m gui.components.dynamic_gridformer_gui

REM If that fails, try alternative launch method
if errorlevel 1 (
    echo.
    echo Primary launch failed, trying alternative method...
    python test_pyqt5_gui.py
)

REM If still failing, provide diagnostics
if errorlevel 1 (
    echo.
    echo GUI launch failed. Running diagnostics...
    echo.
    echo Python version:
    python --version
    echo.
    echo Checking dependencies...
    python -c "import PyQt5; print('PyQt5: OK')" 2>nul || echo "PyQt5: MISSING - Install with: pip install PyQt5"
    python -c "import torch; print('torch: OK')" 2>nul || echo "torch: MISSING - Install with: pip install torch"
    python -c "import numpy; print('numpy: OK')" 2>nul || echo "numpy: MISSING - Install with: pip install numpy"
    echo.
    echo To fix missing dependencies, run:
    echo pip install PyQt5 torch numpy transformers
    echo.
    echo Or run the automated setup:
    echo python quick_setup.py
)

echo.
pause
