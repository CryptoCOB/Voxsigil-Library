@echo off
echo ========================================
echo VoxSigil Library Cleanup
echo ========================================
echo.
echo This will organize your VoxSigil library:
echo.
echo ✅ Move files to organized directories
echo ✅ Remove duplicate and obsolete files  
echo ✅ Create main launcher
echo ✅ Generate directory documentation
echo.
echo WARNING: This will reorganize your files!
echo Make sure you have a backup if needed.
echo.
set /p confirm="Continue? (y/N): "

if /I "%confirm%" NEQ "y" (
    echo Cleanup cancelled.
    pause
    exit /b
)

echo.
echo 🧹 Starting cleanup...
echo.

python cleanup_voxsigil.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✅ CLEANUP COMPLETED SUCCESSFULLY!
    echo ========================================
    echo.
    echo 🚀 Your VoxSigil library is now organized!
    echo.
    echo Next steps:
    echo 1. Use Launch_VoxSigil_Main.bat as your main launcher
    echo 2. Crash-proof GUI is recommended for stability
    echo 3. Check the README files in each directory
    echo.
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ❌ CLEANUP FAILED
    echo Check the output above for errors
    echo ========================================
)

pause
