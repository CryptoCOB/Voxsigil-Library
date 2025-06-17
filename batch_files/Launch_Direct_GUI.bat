@echo off
title VoxSigil Direct GUI Launcher
echo.
echo ========================================
echo  VoxSigil Direct GUI (No Placeholders)
echo ========================================
echo.
echo Starting Direct GUI with real tabs...
echo.

cd /d "d:\Vox\Voxsigil-Library"
python "working_gui\direct_gui.py"

if errorlevel 1 (
    echo.
    echo ❌ Direct GUI failed. Check Python installation.
    pause
) else (
    echo.
    echo ✅ Direct GUI closed normally.
)
