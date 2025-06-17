@echo off
title VoxSigil Enhanced GUI Launcher
color 0A
echo.
echo ================================================================================
echo                        VoxSigil Enhanced GUI Launcher
echo                    Now with Fully Interactive Tabs & Controls!
echo ================================================================================
echo.

echo 🚀 Starting VoxSigil Enhanced GUI with Interactive Features...
echo.
echo ✨ New Features in This Version:
echo    • 📊 Interactive control panels with working buttons
echo    • 🎛️ Live system metrics and progress bars  
echo    • ⚙️ Real configuration settings that respond
echo    • 📋 Activity logs with real-time event tracking
echo    • 🔄 Auto-refresh and export capabilities
echo    • 🎯 Start/Stop/Restart system controls
echo.

echo 🔄 Launching Enhanced GUI...
cd /d "d:\Vox\Voxsigil-Library"

REM Try the enhanced launcher first
python launch_enhanced_gui.py

if errorlevel 1 (
    echo.
    echo ⚠️  Enhanced launcher failed, trying direct GUI launch...
    python "working_gui\complete_live_gui.py"
)

if errorlevel 1 (
    echo.
    echo ❌ Complete GUI failed. Trying fallback options...
    python "working_gui\direct_gui.py"
)

if errorlevel 1 (
    echo.
    echo ❌ All GUI options failed. Please check dependencies.
    echo.
    echo 🔧 Troubleshooting steps:
    echo    1. Ensure Python is installed
    echo    2. Install PyQt5: pip install PyQt5
    echo    3. Check the project directory path
    pause
)

echo.
echo ================================================================================
echo                              Launch Complete
echo ================================================================================

pause
