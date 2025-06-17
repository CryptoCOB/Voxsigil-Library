@echo off
echo ========================================
echo VoxSigil CRASH-PROOF Enhanced GUI
echo ========================================
echo.
echo This launcher uses CRASH-PROOF loading to prevent crashes.
echo Enhanced tabs show demo content and won't crash when clicked.
echo.
echo Starting Crash-Proof GUI...
echo.

cd /d "%~dp0"
python crash_proof_enhanced_gui.py

echo.
echo GUI session ended.
pause
