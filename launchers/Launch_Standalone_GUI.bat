@echo off
echo ========================================
echo VoxSigil Standalone Enhanced GUI
echo ========================================
echo.
echo This version removes gui.components imports
echo to prevent hanging issues.
echo.
echo Features:
echo - Timeout protection (4-10s per tab)
echo - Automatic retry (2-3 attempts)
echo - Circuit breaker for failed tabs
echo - Memory leak detection
echo - Resource monitoring
echo - Background loading
echo - No external dependencies
echo.
echo ========================================
echo.

cd /d "%~dp0"
python standalone_enhanced_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Standalone GUI failed to launch
    echo Check the console output above for details
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo Standalone GUI session ended successfully
    echo ========================================
)
