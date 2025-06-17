@echo off
echo 🚀 Launching VoxSigil Enhanced GUI...
echo.
echo This will open the enhanced GUI with real-time streaming data
echo and no VantaCore event loop errors.
echo.
cd /d "%~dp0"
python direct_gui_launcher.py
pause
