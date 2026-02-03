# Installation Guide - VoxSigil Library

## Overview

This guide covers installation and setup of the VoxSigil Library for molt agent integration.

## Prerequisites

### JavaScript/Node.js
- **Node.js**: Version 14.0.0 or higher
- **npm**: Version 6.0.0 or higher

### Python
- **Python**: Version 3.8 or higher
- **pip**: Version 21.0 or higher

### Optional
- **Git**: For cloning the repository
- **curl**: For downloading files directly

## Installation Methods

### Method 1: Package Managers (Recommended)

#### JavaScript/Node.js
```bash
# Install from npm
npm install @voxsigil/library

# Or add to package.json
npm install --save @voxsigil/library
```

#### Python
```bash
# Install from PyPI
pip install voxsigil-library

# Or with pip3
pip3 install voxsigil-library

# Install with development dependencies
pip install voxsigil-library[dev]
```

### Method 2: From Source

#### Clone Repository
```bash
# Clone from GitHub
git clone https://github.com/CryptoCOB/Voxsigil-Library.git
cd Voxsigil-Library

# Or download specific version
git clone --branch v1.0.0 https://github.com/CryptoCOB/Voxsigil-Library.git
```

#### Install Dependencies
```bash
# JavaScript
npm install

# Python
pip install -e .

# Or both
npm install && pip install -e .
```

### Method 3: Direct File Download

Download agent files directly:

```bash
# Create directory
mkdir -p voxsigil-agents

# Download agent files
curl -o voxsigil-agents/boot.md \
  https://raw.githubusercontent.com/CryptoCOB/Voxsigil-Library/main/src/agents/boot.md

curl -o voxsigil-agents/agents.md \
  https://raw.githubusercontent.com/CryptoCOB/Voxsigil-Library/main/src/agents/agents.md

curl -o voxsigil-agents/memory.md \
  https://raw.githubusercontent.com/CryptoCOB/Voxsigil-Library/main/src/agents/memory.md

curl -o voxsigil-agents/hooks-config.json \
  https://raw.githubusercontent.com/CryptoCOB/Voxsigil-Library/main/src/agents/hooks-config.json
```

## Configuration

### Environment Variables

Set required environment variables:

```bash
# VoxSigil API Key
export VOXSIGIL_API_KEY='your-api-key-here'

# Optional: Custom API endpoint
export VOXSIGIL_API_URL='https://voxsigil.online/api'

# Optional: Session directory
export VOXSIGIL_SESSION_DIR='.voxsigil/sessions'
```

Add to your shell profile for persistence:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export VOXSIGIL_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Configuration Files

Create a configuration file (optional):

**config.json:**
```json
{
  "api_endpoint": "https://voxsigil.online/api",
  "api_key": "${VOXSIGIL_API_KEY}",
  "checkpoint_interval_minutes": 30,
  "max_active_markets": 50,
  "confidence_threshold": 0.70,
  "log_level": "info"
}
```

## Verification

### Verify Installation

#### JavaScript
```bash
# Check if package is installed
npm list @voxsigil/library

# Run example
node src/examples/javascript-integration.js
```

#### Python
```bash
# Check if package is installed
pip show voxsigil-library

# Run example
python3 src/examples/python-integration.py

# Or import in Python
python3 -c "from voxsigil import VoxSigilAgent; print(VoxSigilAgent.get_metadata())"
```

### Verify Agent Files

Check that all agent files are present:

```bash
ls -lh src/agents/
# Should see:
# - boot.md
# - agents.md
# - memory.md
# - hooks-config.json
```

### Compute Checksums

Verify file integrity:

```bash
# Linux/macOS
sha256sum src/agents/*.md src/agents/*.json

# macOS only
shasum -a 256 src/agents/*.md src/agents/*.json
```

## Auto Setup Script

Use the automated setup script:

```bash
# Run setup script
bash src/examples/molt-agent-setup.sh

# The script will:
# 1. Check dependencies
# 2. Install packages
# 3. Verify agent files
# 4. Compute checksums
# 5. Run integration tests
```

## Quick Start Examples

### JavaScript Example

Create **test.js**:
```javascript
const voxsigil = require('@voxsigil/library');

// Get metadata
const metadata = voxsigil.getMetadata();
console.log('VoxSigil Library:', metadata.version);

// Load configuration
const config = voxsigil.loadAgentConfig();
console.log('Agent files loaded successfully!');
```

Run:
```bash
node test.js
```

### Python Example

Create **test.py**:
```python
from voxsigil import VoxSigilAgent

# Initialize agent
agent = VoxSigilAgent()

# Get metadata
metadata = agent.get_metadata()
print(f"VoxSigil Library: {metadata['version']}")

# Load configuration
config = agent.load_agent_config()
print("Agent files loaded successfully!")
```

Run:
```bash
python3 test.py
```

## Troubleshooting

### Common Issues

#### Issue: Module not found

**JavaScript:**
```bash
# Ensure you're in the right directory
npm install @voxsigil/library

# Or check NODE_PATH
export NODE_PATH=$(npm root -g)
```

**Python:**
```bash
# Ensure package is installed
pip install voxsigil-library

# Or check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue: Agent files not found

```bash
# Verify file paths
ls src/agents/

# If missing, re-download from GitHub
git pull origin main

# Or clone fresh
git clone https://github.com/CryptoCOB/Voxsigil-Library.git
```

#### Issue: Permission denied

```bash
# Make setup script executable
chmod +x src/examples/molt-agent-setup.sh

# Run with explicit interpreter
bash src/examples/molt-agent-setup.sh
```

#### Issue: API connection failed

```bash
# Check API key is set
echo $VOXSIGIL_API_KEY

# Test connection
curl -H "Authorization: Bearer $VOXSIGIL_API_KEY" \
  https://voxsigil.online/api/status
```

## Next Steps

After installation:

1. **Read the API Reference**: [docs/API.md](API.md)
2. **Follow Molt Integration Guide**: [docs/MOLT_INTEGRATION.md](MOLT_INTEGRATION.md)
3. **Run Examples**: Try the integration examples in `src/examples/`
4. **Join Community**: Visit https://voxsigil.online for support

## Updates

### Checking for Updates

```bash
# JavaScript
npm outdated @voxsigil/library

# Python
pip list --outdated | grep voxsigil-library
```

### Updating

```bash
# JavaScript
npm update @voxsigil/library

# Python
pip install --upgrade voxsigil-library
```

## Uninstallation

### Remove Package

```bash
# JavaScript
npm uninstall @voxsigil/library

# Python
pip uninstall voxsigil-library
```

### Clean Up

```bash
# Remove session data
rm -rf .voxsigil/

# Remove environment variables
unset VOXSIGIL_API_KEY
```

## Support

For installation issues:
- **GitHub Issues**: https://github.com/CryptoCOB/Voxsigil-Library/issues
- **Documentation**: https://voxsigil.online/docs
- **Email**: support@voxsigil.online
