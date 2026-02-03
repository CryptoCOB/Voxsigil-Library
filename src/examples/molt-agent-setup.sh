#!/bin/bash
# molt-agent-setup.sh
# Setup script for VoxSigil molt agent integration

set -e

echo "========================================================================"
echo "VoxSigil Library - Molt Agent Setup"
echo "========================================================================"
echo ""

# Check if running in the right directory
if [ ! -f "package.json" ] || [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the Voxsigil-Library root directory"
    exit 1
fi

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
echo ""

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✓ Node.js installed: $NODE_VERSION"
else
    echo "✗ Node.js not found. Please install Node.js >= 14.0.0"
    echo "  Visit: https://nodejs.org/"
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python installed: $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python >= 3.8"
    echo "  Visit: https://www.python.org/"
fi

echo ""

# Step 2: Install JavaScript dependencies
echo "Step 2: Installing JavaScript dependencies..."
if command -v npm &> /dev/null; then
    npm install
    echo "✓ npm packages installed"
else
    echo "⚠ npm not found, skipping JavaScript setup"
fi
echo ""

# Step 3: Install Python dependencies
echo "Step 3: Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -e .
    echo "✓ Python package installed"
else
    echo "⚠ pip3 not found, skipping Python setup"
fi
echo ""

# Step 4: Verify agent files
echo "Step 4: Verifying agent files..."
AGENTS_DIR="src/agents"
FILES=("boot.md" "agents.md" "memory.md" "hooks-config.json")

for file in "${FILES[@]}"; do
    if [ -f "$AGENTS_DIR/$file" ]; then
        echo "✓ $file found"
    else
        echo "✗ $file missing"
    fi
done
echo ""

# Step 5: Compute checksums
echo "Step 5: Computing file checksums..."
if command -v sha256sum &> /dev/null; then
    for file in "${FILES[@]}"; do
        if [ -f "$AGENTS_DIR/$file" ]; then
            CHECKSUM=$(sha256sum "$AGENTS_DIR/$file" | cut -d' ' -f1)
            echo "  $file: ${CHECKSUM:0:16}..."
        fi
    done
elif command -v shasum &> /dev/null; then
    for file in "${FILES[@]}"; do
        if [ -f "$AGENTS_DIR/$file" ]; then
            CHECKSUM=$(shasum -a 256 "$AGENTS_DIR/$file" | cut -d' ' -f1)
            echo "  $file: ${CHECKSUM:0:16}..."
        fi
    done
else
    echo "⚠ SHA256 tool not found, skipping checksums"
fi
echo ""

# Step 6: Test integration
echo "Step 6: Testing integration..."

# Test Python integration
if command -v python3 &> /dev/null; then
    echo "Running Python integration test..."
    python3 src/examples/python-integration.py
    echo ""
fi

# Test JavaScript integration
if command -v node &> /dev/null; then
    echo "Running JavaScript integration test..."
    node src/examples/javascript-integration.js
    echo ""
fi

# Step 7: Setup complete
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Your VoxSigil molt agent integration is ready."
echo ""
echo "Next steps:"
echo "1. Set your API key: export VOXSIGIL_API_KEY='your-key-here'"
echo "2. Review documentation: docs/MOLT_INTEGRATION.md"
echo "3. Run example scripts:"
echo "   - Python: python3 src/examples/python-integration.py"
echo "   - JavaScript: node src/examples/javascript-integration.js"
echo ""
echo "For support, visit: https://github.com/CryptoCOB/Voxsigil-Library/issues"
echo ""
