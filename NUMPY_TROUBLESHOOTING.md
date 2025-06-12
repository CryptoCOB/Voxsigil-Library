# 🔧 NumPy Troubleshooting Guide for VoxSigil Library

## 🚨 **Common NumPy Issues & Solutions**

### **Problem 1: NumPy Import Errors**
```
ImportError: numpy.core.multiarray failed to import
AttributeError: module 'numpy' has no attribute 'array'
```

**Solution:**
```bash
# Clear Python cache and reinstall
python -c "import sys; [sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('numpy')]"
pip uninstall numpy -y
pip install "numpy>=1.21.6,<1.27.0"
```

### **Problem 2: Version Conflicts**
```
ERROR: Cannot install torch and numpy because these package versions have conflicting dependencies
```

**Solution:**
```bash
# Install in correct order with our script
python install_with_uv.py
```

### **Problem 3: Circular Import Issues**
```
Circular import detected: numpy trying to import from itself
```

**Solution:**
```bash
# Use our numpy resolver
python -c "from numpy_resolver import safe_import_numpy; numpy, np, have_numpy = safe_import_numpy(); print(f'NumPy OK: {have_numpy}')"
```

## ✅ **Quick Fix Commands**

### **Complete NumPy Reset**
```bash
# Windows PowerShell
pip uninstall numpy scipy torch transformers -y
python install_with_uv.py

# Linux/Mac
pip uninstall numpy scipy torch transformers -y && python install_with_uv.py
```

### **Verify Installation**
```bash
python -c "
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
print('✅ All critical imports successful')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
"
```

## 🎯 **Python 3.10 Specific Fixes**

### **If Using Python 3.10.x:**
```bash
# Ensure compatible versions
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.21.6,<1.27.0"  # Tested stable range
pip install "torch>=1.13.0,<2.3.0"   # Compatible with numpy above
```

### **Environment Variables (if needed):**
```bash
# Set these before installation
export NPY_NUM_BUILD_JOBS=4
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Then install
python install_with_uv.py
```

## 🔍 **Diagnostic Commands**

### **Check Current State**
```python
# Run this in Python to diagnose issues
import sys
print("Python version:", sys.version)
print("Modules loaded:", [m for m in sys.modules.keys() if 'numpy' in m])

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} - OK")
    test_array = np.array([1, 2, 3])
    print(f"✅ Array creation - OK: {test_array}")
except Exception as e:
    print(f"❌ NumPy issue: {e}")
```

### **Check Installation Paths**
```bash
python -c "
import numpy
import torch
print('NumPy path:', numpy.__file__)
print('PyTorch path:', torch.__file__)
"
```

## 🚀 **Emergency Fallback Installation**

If everything fails, use this step-by-step approach:

```bash
# 1. Clean environment
pip freeze | grep -E "(numpy|torch|scipy)" | xargs pip uninstall -y

# 2. Install build tools
pip install --upgrade pip setuptools wheel

# 3. Install NumPy alone first
pip install "numpy==1.24.3"  # Known stable version

# 4. Install PyTorch alone
pip install "torch==2.0.1"   # Compatible version

# 5. Install GUI framework
pip install "PyQt5==5.15.9"

# 6. Test core functionality
python -c "import numpy, torch; from PyQt5.QtWidgets import QApplication; print('✅ Core imports OK')"

# 7. Install remaining requirements
pip install -r requirements-fixed.txt
```

## 📞 **Getting Help**

If you're still having issues:

1. **Check your exact Python version**: `python --version`
2. **Check your OS**: `python -c "import platform; print(platform.platform())"`
3. **Run diagnostics**: `python install_with_uv.py`
4. **Check the logs**: Look for specific error messages in the output

## 🎯 **Known Working Combinations**

### **Windows 10/11 + Python 3.10.x**
```
numpy==1.24.3
torch==2.0.1
PyQt5==5.15.9
transformers==4.30.2
```

### **Ubuntu 20.04+ + Python 3.10.x**
```
numpy==1.24.3
torch==2.0.1+cpu
PyQt5==5.15.9
transformers==4.30.2
```

### **macOS + Python 3.10.x**
```
numpy==1.24.3
torch==2.0.1
PyQt5==5.15.9
transformers==4.30.2
```

---

**Remember**: The `install_with_uv.py` script handles most of these issues automatically!
