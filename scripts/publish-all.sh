#!/bin/bash
#
# Voxsigil Library - Master Publishing Script
#
# Publishes to both npm and PyPI simultaneously
#
# Usage: ./publish-all.sh [version]
#
# Examples:
#   ./publish-all.sh patch
#   ./publish-all.sh 1.2.3
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════╗"
echo "║  🚀 Voxsigil Master Publishing              ║"
echo "║  Publishing to npm AND PyPI                 ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

VERSION=${1:-patch}

# Check prerequisites
if [ ! -f "package.json" ] || [ ! -f "setup.py" ]; then
    echo -e "${RED}❌ Not in Voxsigil-Library root directory${NC}"
    exit 1
fi

# Verify we're on main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo -e "${YELLOW}⚠️  Not on main branch (current: $BRANCH)${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}❌ Uncommitted changes detected${NC}"
    git status
    exit 1
fi

echo -e "${BLUE}Publishing version: ${VERSION}${NC}\n"

# Step 1: npm publish
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1/2: Publishing to npm${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"

if bash scripts/publish-npm.sh "$VERSION"; then
    echo -e "${GREEN}✅ npm publish successful${NC}\n"
else
    echo -e "${RED}❌ npm publish failed${NC}"
    exit 1
fi

# Step 2: PyPI publish
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2/2: Publishing to PyPI${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"

if bash scripts/publish-pypi.sh "$VERSION"; then
    echo -e "${GREEN}✅ PyPI publish successful${NC}\n"
else
    echo -e "${RED}❌ PyPI publish failed${NC}"
    exit 1
fi

# Summary
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════╗"
echo "║  ✅ PUBLISHING COMPLETE                    ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

# Get new version
NEW_VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' setup.py)

echo -e "${GREEN}✅ npm: @voxsigil/library@${NEW_VERSION}${NC}"
echo -e "${GREEN}✅ PyPI: voxsigil-library==${NEW_VERSION}${NC}\n"

echo "Links:"
echo "  npm:  https://www.npmjs.com/package/@voxsigil/library"
echo "  PyPI: https://pypi.org/project/voxsigil-library/\n"

echo -e "${BLUE}Next steps:${NC}"
echo "  1. Verify packages are live on registries (may take 1-2 minutes)"
echo "  2. Push tags: git push origin main --tags"
echo "  3. Create GitHub release with version ${NEW_VERSION}"
echo "  4. Announce in community channels\n"

echo -e "${YELLOW}Pro Tips:${NC}"
echo "  • Test with: npm install @voxsigil/library / pip install voxsigil-library"
echo "  • Check package size: npm pack"
echo "  • View dependencies: npm ls (npm) / pip freeze (PyPI)"
