@echo off
title VoxSigil Ultra-Stable GUI Test
echo.
echo ========================================
echo  Testing VoxSigil Ultra-Stable GUI
echo ========================================
echo.

cd /d "d:\Vox\Voxsigil-Library"

echo Attempting to launch ultra-stable GUI...
python "working_gui\ultra_stable_gui.py"

if errorlevel 1 (
    echo.
    echo ❌ Ultra-stable GUI failed. Trying standalone...
    python "working_gui\standalone_enhanced_gui.py"
)

if errorlevel 1 (
    echo.
    echo ❌ Both GUIs failed. Check Python installation and dependencies.
    echo.
    echo Checking PyQt5...
    python -c "import PyQt5; print('PyQt5 OK')"
    echo.
    pause
)
