# UV Installation Issue and Fix

> **Update**: A new comprehensive installation script (`install_enhanced.py`) has been created that addresses all the issues below and provides a reliable installation process for all Python versions. Please use this new script instead of the older installation methods.

## Problem Identified

The UV installation script was failing due to several critical issues:

1. **Broken Virtual Environment**: The `.venv` directory existed but was missing the critical `pyvenv.cfg` file, causing Python to report "No pyvenv.cfg file" errors.

2. **Permission Issues**: There were permission problems when trying to create or remove the virtual environment.

3. **Python 3.13 Compatibility**: Many packages, including PyTorch and NumPy, don't yet have binary wheels for Python 3.13, causing installation failures.

4. **Dependency Chain Issues**: The installation order of dependencies was causing conflicts, especially with NumPy and PyTorch.

5. **Ninja Build Tool**: The ninja build tool wasn't being installed correctly with UV, causing NumPy build failures.

## Solutions Implemented

1. **Clean Virtual Environment**: Created a proper virtual environment from scratch using `python -m venv .venv`.

2. **Fixed Permission Issues**: The new installation scripts handle permission issues properly.

3. **Python Version Handling**: Added special handling for Python 3.13 to use a different installation approach.

4. **Improved Dependency Management**: 
   - Install Ninja with pip instead of UV (more reliable)
   - Install a specific NumPy version compatible with Python 3.13
   - Handle PyTorch with flexible version constraints
   - Filter requirements.txt to avoid conflicts

5. **Multiple Installation Options**: Created three different installation scripts for different scenarios.

## How to Install VoxSigil Library

### Option 1: Using UV (For Python 3.10-3.12)

Best for Python 3.10-3.12 environments:

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Run the fixed UV installation script:
   ```powershell
   python install_uv_fixed.py
   ```

### Option 2: Using Pip (For Python 3.13+)

Better for Python 3.13 or when UV fails:

```powershell
python install_pip_fallback.py
```

### Option 3: General Purpose Installation

For environments where other methods fail:

```powershell
python install_fixed.py
```

## Specific Issues Fixed

1. **Missing pyvenv.cfg**: Fixed by creating a clean virtual environment
2. **Ninja installation failures**: Now using pip instead of UV for ninja
3. **NumPy build errors**: Using compatible NumPy versions and setting proper environment variables
4. **PyTorch compatibility**: Using version-specific installation for different Python versions
5. **Requirements conflicts**: Filtering requirements.txt to remove conflicts

## Troubleshooting

### If you encounter "No pyvenv.cfg file" errors:
1. Remove the broken virtual environment: `Remove-Item -Recurse -Force -Path ".venv"`
2. Create a new one: `python -m venv .venv`
3. Activate it: `.venv\Scripts\Activate.ps1`
4. Try the installation again

### For Python 3.13 users:
The best option is to use the `install_pip_fallback.py` script, as many packages don't yet have binary wheels for Python 3.13.

### If you can't install NumPy:
Try installing it manually first:
```powershell
python -m pip install numpy==1.24.4
```

### If you can't install PyTorch:
Try installing without version constraints:
```powershell
python -m pip install torch torchvision torchaudio
```
