# VoxSigil Library Installation Guide

This guide explains how to install the VoxSigil Library with the new enhanced installation script that fixes the import issues and provides reliable installation across different Python versions.

## Installation Methods

### Option 1: Enhanced Installer (Recommended)

The new `install_enhanced.py` script detects your environment and automatically chooses the best installation approach. It includes:

- Automatic virtual environment validation and repair
- Python version-specific installation paths
- NumPy and PyTorch compatibility handling
- Multiple installation methods with fallbacks

```bash
# Windows
python install_enhanced.py

# Linux/macOS
python3 install_enhanced.py
```

### Option 2: UV-Based Installation (Python 3.10-3.12)

For systems using Python 3.10-3.12, you can use the UV-specific installer:

```bash
python install_uv_fixed.py
```

### Option 3: Pip-Based Installation (Python 3.13+)

For Python 3.13+ or when UV fails:

```bash
python install_pip_fallback.py
```

## Python Version Compatibility

| Python Version | Installation Method | NumPy Version    | PyTorch Version    | Notes                                |
|----------------|---------------------|------------------|--------------------|------------------------------------- |
| 3.10           | UV (preferred)      | 1.21.6 - 1.26.x  | 1.13.0 - 2.2.x     | Ideal, fully supported               |
| 3.11           | UV (preferred)      | 1.21.6 - 1.26.x  | 1.13.0 - 2.2.x     | Fully supported                      |
| 3.12           | UV (preferred)      | 1.23.0 - 1.26.x  | 1.13.0 - 2.2.x     | Fully supported                      |
| 3.13           | Pip (only)          | 1.24.4 (specific)| Latest available   | Limited wheel availability           |

## Virtual Environment Setup

All installation scripts will automatically set up a virtual environment if you're not already in one. We recommend using the virtual environment to keep your dependencies isolated:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/macOS
source .venv/bin/activate

# Then install
python install_enhanced.py
```

## Troubleshooting Common Issues

### 1. "No pyvenv.cfg file" Error

This happens when the virtual environment is corrupted:

```bash
# Remove the broken environment
rm -rf .venv  # Linux/macOS
rmdir /s /q .venv  # Windows

# Run the enhanced installer to create a new one
python install_enhanced.py
```

### 2. NumPy Import Errors

If you see errors like `ImportError: numpy.core.multiarray failed to import`:

```bash
# Clear Python cache and reinstall numpy
python -c "import sys; [sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('numpy')]"
pip uninstall numpy -y
pip install numpy==1.24.4
```

### 3. PyTorch Installation Issues on Python 3.13+

PyTorch doesn't yet have wheels for Python 3.13. Try:

```bash
# Install without version constraints
pip install torch torchvision torchaudio
```

### 4. Installation Permission Errors

If you encounter permission errors creating or removing the virtual environment:

```bash
# Run with administrator privileges on Windows
# Or use sudo on Linux/macOS
sudo python install_enhanced.py
```

### 5. UV Installation Failures

If UV installation fails:

```bash
# Install UV manually
pip install uv

# Or use the pip fallback installer
python install_pip_fallback.py
```

## Advanced Configuration

### CUDA Support

To enable CUDA support, set the CUDA_HOME environment variable before installation:

```bash
# Windows
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

# Linux/macOS
export CUDA_HOME=/usr/local/cuda-12.1
```

### NumPy Build Configuration

For optimized NumPy builds:

```bash
# Windows
set NPY_NUM_BUILD_JOBS=4
set OPENBLAS_NUM_THREADS=1

# Linux/macOS
export NPY_NUM_BUILD_JOBS=4
export OPENBLAS_NUM_THREADS=1
```

## Verifying Installation

After installation, you can verify the setup:

```python
# Run this script to test critical imports
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
import transformers

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print("All imports successful!")
```

## Known Working Combinations

### Windows 10/11 + Python 3.10.x
```
numpy==1.24.3
torch==2.0.1
PyQt5==5.15.9
transformers==4.30.2
```

### Python 3.13.x
```
numpy==1.24.4
torch==2.2.0 (CPU only typically)
PyQt5==5.15.10
transformers==4.38.0
```

## Installation Logs

The installation process creates a detailed log file named `installation_log.txt`. If you encounter issues, please include this file when asking for help.

---

**Remember**: The `install_enhanced.py` script handles most issues automatically!
