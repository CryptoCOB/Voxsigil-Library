@echo off
echo ========================================
echo VoxSigil Enhanced GUI - No Hang Launcher  
echo ========================================
echo.
echo This launcher avoids startup hangs by using
echo progressive loading of GUI components.
echo.
echo Starting GUI...
echo.

cd /d "%~dp0"
python launch_no_hang_gui.py

echo.
echo GUI session ended.
pause
