#!/usr/bin/env python3
"""
VoxSigil Library - UV Installation Script with NumPy Fix
Handles numpy compatibility issues for Python 3.10
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ UV found: {result.stdout.strip()}")
            return True
        else:
            print("❌ UV not found")
            return False
    except FileNotFoundError:
        print("❌ UV not found")
        return False

def install_uv():
    """Install uv package manager"""
    print("📦 Installing UV package manager...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['powershell', '-Command', 'irm https://astral.sh/uv/install.ps1 | iex'], check=True)
        else:  # Unix-like
            subprocess.run(['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'], shell=True, check=True)
        print("✅ UV installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install UV: {e}")
        return False

def fix_numpy_environment():
    """Fix numpy environment issues before installation"""
    print("🔧 Preparing environment for NumPy...")
    
    # Clear any problematic numpy cache
    try:
        import sys
        numpy_modules = [mod for mod in sys.modules.keys() if mod.startswith('numpy')]
        for mod in numpy_modules:
            del sys.modules[mod]
        print("✅ Cleared numpy from module cache")
    except:
        pass
    
    # Set environment variables for better numpy compilation
    env_vars = {
        'NPY_NUM_BUILD_JOBS': '4',
        'OPENBLAS_NUM_THREADS': '1',  # Prevent threading conflicts
        'OMP_NUM_THREADS': '1'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")

def install_with_uv():
    """Install dependencies using UV"""
    print("📦 Installing dependencies with UV...")
    
    # Fix numpy environment first
    fix_numpy_environment()
    
    # Install core dependencies first (numpy, torch separately to avoid conflicts)
    core_deps = [
        'numpy>=1.21.6,<1.27.0',
        'setuptools>=65.0.0',
        'wheel>=0.38.0',
    ]
    
    print("🔧 Installing core dependencies first...")
    for dep in core_deps:
        try:
            subprocess.run(['uv', 'pip', 'install', dep], check=True)
            print(f"✅ Installed: {dep}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {dep}: {e}")
    
    # Install PyTorch separately (often has specific requirements)
    print("🔧 Installing PyTorch...")
    try:
        subprocess.run(['uv', 'pip', 'install', 'torch>=1.13.0,<2.3.0'], check=True)
        print("✅ Installed PyTorch")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: PyTorch installation issue: {e}")
    
    # Install all other requirements
    print("📦 Installing remaining requirements...")
    try:
        subprocess.run(['uv', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_with_pip():
    """Fallback installation using regular pip"""
    print("📦 Installing dependencies with pip (fallback)...")
    
    # Fix numpy environment first
    fix_numpy_environment()
    
    # Upgrade pip first
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        print("✅ Upgraded pip")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Could not upgrade pip: {e}")
    
    # Install core dependencies first
    core_deps = [
        'numpy>=1.21.6,<1.27.0',
        'setuptools>=65.0.0',
        'wheel>=0.38.0',
    ]
    
    print("🔧 Installing core dependencies first...")
    for dep in core_deps:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
            print(f"✅ Installed: {dep}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {dep}: {e}")
    
    # Install PyTorch separately
    print("🔧 Installing PyTorch...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch>=1.13.0,<2.3.0'], check=True)
        print("✅ Installed PyTorch")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: PyTorch installation issue: {e}")
    
    # Install all requirements
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify critical imports work"""
    print("🔍 Verifying installation...")
    
    critical_imports = [
        ('numpy', 'import numpy as np; print(f"NumPy {np.__version__}")'),
        ('torch', 'import torch; print(f"PyTorch {torch.__version__}")'),
        ('PyQt5', 'from PyQt5.QtWidgets import QApplication; print("PyQt5 OK")'),
        ('transformers', 'import transformers; print(f"Transformers {transformers.__version__}")'),
    ]
    
    success_count = 0
    for name, test_import in critical_imports:
        try:
            subprocess.run([sys.executable, '-c', test_import], 
                         check=True, capture_output=True, text=True)
            print(f"✅ {name}: OK")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {name}: Failed")
            print(f"   Error: {e.stderr}")
    
    print(f"\n📊 Verification: {success_count}/{len(critical_imports)} imports successful")
    return success_count == len(critical_imports)

def main():
    """Main installation function"""
    print("🚀 VoxSigil Library - UV Installation with NumPy Fix")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required for optimal compatibility")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Try UV installation first
    if check_uv_installed() or install_uv():
        print("\n🎯 Attempting installation with UV...")
        if install_with_uv():
            if verify_installation():
                print("\n🎉 Installation completed successfully with UV!")
                return True
            else:
                print("\n⚠️ UV installation completed but verification failed")
    
    # Fallback to pip
    print("\n🎯 Falling back to pip installation...")
    if install_with_pip():
        if verify_installation():
            print("\n🎉 Installation completed successfully with pip!")
            return True
        else:
            print("\n⚠️ Pip installation completed but verification failed")
    
    print("\n❌ Installation failed. Please check error messages above.")
    print("\n🛠️ Manual installation suggestions:")
    print("1. Update pip: python -m pip install --upgrade pip")
    print("2. Install numpy separately: pip install 'numpy>=1.21.6,<1.27.0'")
    print("3. Install torch separately: pip install 'torch>=1.13.0,<2.3.0'")
    print("4. Then install requirements: pip install -r requirements.txt")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
