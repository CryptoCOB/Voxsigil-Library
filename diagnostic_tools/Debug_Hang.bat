@echo off
echo ========================================
echo VoxSigil GUI Hang Diagnostic
echo ========================================
echo.
echo Current directory: %CD%
echo Python version:
python --version
echo.
echo Testing basic Python execution...
python -c "print('Python execution works')"
echo.
echo Testing PyQt5 availability...
python -c "import PyQt5; print('PyQt5 available')"
echo.
echo Testing if proper_enhanced_gui.py can be compiled...
python -m py_compile proper_enhanced_gui.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Syntax is valid
) else (
    echo ❌ Syntax errors found
)
echo.
echo Attempting to import proper_enhanced_gui...
python -c "import proper_enhanced_gui; print('Import successful')"
echo.
echo ========================================
echo Diagnostic complete
echo ========================================
pause
