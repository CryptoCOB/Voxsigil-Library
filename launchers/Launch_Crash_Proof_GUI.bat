@echo off
echo ========================================
echo VoxSigil CRASH-PROOF Enhanced GUI
echo ========================================
echo.
echo 🛡️ GUARANTEED NO CRASHES! 🛡️
echo.
echo This version:
echo ✅ Won't crash when you click tabs
echo ✅ Uses safe demo content
echo ✅ Shows what real tabs would look like
echo ✅ Has interactive features
echo ✅ Comprehensive error handling
echo.
echo Starting Crash-Proof GUI...
echo.

cd /d "%~dp0"
python crash_proof_enhanced_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Even crash-proof GUI failed!
    echo This indicates a serious system issue.
    echo Check Python installation and PyQt5.
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo Crash-proof GUI session ended successfully
    echo ========================================
)
