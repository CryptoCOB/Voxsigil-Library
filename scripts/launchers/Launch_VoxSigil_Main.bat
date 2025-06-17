@echo off
echo ========================================
echo VoxSigil Enhanced GUI - Main Launcher
echo ========================================
echo.
echo 🎯 This is the MAIN launcher for VoxSigil GUI
echo.
echo Available options:
echo 1. Crash-Proof GUI (Recommended)
echo 2. Optimized GUI
echo 3. Standalone GUI
echo.
set /p choice="Choose option (1-3) or press Enter for default: "

if "%choice%"=="2" (
    echo.
    echo 🚀 Launching Optimized Enhanced GUI...
    python working_gui/optimized_enhanced_gui.py
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Launching Standalone GUI...
    python working_gui/standalone_enhanced_gui.py
) else (
    echo.
    echo 🛡️ Launching Crash-Proof GUI (Default)...
    python working_gui/crash_proof_enhanced_gui.py
)

echo.
echo GUI session ended.
pause
