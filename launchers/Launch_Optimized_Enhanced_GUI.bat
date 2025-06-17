@echo off
echo ========================================
echo VoxSigil Optimized Enhanced GUI Launcher
echo ========================================
echo.
echo Features:
echo - Timeout protection (5-20s per tab)
echo - Automatic retry (2-3 attempts) 
echo - Circuit breaker for failed tabs
echo - Memory leak detection
echo - Resource monitoring (CPU/RAM)
echo - Background loading
echo - Keyboard shortcuts (Ctrl+R/T/G)
echo - Splash screen
echo.
echo ========================================
echo.

cd /d "%~dp0"
python launch_optimized_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: GUI failed to launch
    echo Check the console output above for details
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo GUI session ended successfully
    echo ========================================
)
