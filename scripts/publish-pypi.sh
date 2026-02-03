#!/bin/bash
#
# Voxsigil Library - PyPI Publishing Script
#
# Usage: ./publish-pypi.sh [version]
#
# Examples:
#   ./publish-pypi.sh patch       # Bump patch version (1.0.0 → 1.0.1)
#   ./publish-pypi.sh minor       # Bump minor version (1.0.0 → 1.1.0)
#   ./publish-pypi.sh major       # Bump major version (1.0.0 → 2.0.0)
#   ./publish-pypi.sh 1.2.3       # Set specific version
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Voxsigil PyPI Publishing${NC}"
echo "=================================="

# Check prerequisites
for cmd in python pip twine; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}❌ $cmd is not installed${NC}"
        exit 1
    fi
done

# Get current version from setup.py
CURRENT_VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' setup.py)
echo -e "${BLUE}Current version: ${CURRENT_VERSION}${NC}"

# Update version in setup.py
VERSION_TYPE=${1:-patch}
case $VERSION_TYPE in
    patch)
        # Parse version and increment patch
        IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
        PATCH=$((PATCH + 1))
        NEW_VERSION="$MAJOR.$MINOR.$PATCH"
        ;;
    minor)
        IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
        MINOR=$((MINOR + 1))
        PATCH=0
        NEW_VERSION="$MAJOR.$MINOR.$PATCH"
        ;;
    major)
        IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        NEW_VERSION="$MAJOR.$MINOR.$PATCH"
        ;;
    *)
        # Assume it's a specific version number
        NEW_VERSION=$VERSION_TYPE
        ;;
esac

echo -e "${BLUE}Updating version to: ${NEW_VERSION}${NC}"

# Update setup.py
sed -i "s/version=\"$CURRENT_VERSION\"/version=\"$NEW_VERSION\"/" setup.py

# Update CHANGELOG.md
echo "" >> CHANGELOG.md
echo "## [$NEW_VERSION] - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "" >> CHANGELOG.md
echo "### Added" >> CHANGELOG.md
echo "- Release version $NEW_VERSION" >> CHANGELOG.md
echo "" >> CHANGELOG.md

echo -e "${GREEN}✅ Version updated: ${CURRENT_VERSION} → ${NEW_VERSION}${NC}"

# Run tests
echo -e "${BLUE}Running tests...${NC}"
python -m pytest tests/ -v || {
    echo -e "${RED}❌ Tests failed. Aborting publish.${NC}"
    exit 1
}

# Run security scan
echo -e "${BLUE}Running security checks...${NC}"
python -m pip install bandit safety 2>/dev/null || true
bandit -r src/ -ll || true

# Build distribution
echo -e "${BLUE}Building distribution...${NC}"
rm -rf build dist *.egg-info
python -m pip install --upgrade build
python -m build

if [ ! -d dist ] || [ -z "$(ls -A dist)" ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Distribution built${NC}"

# Check if pypi token is set
if [ -z "$PYPI_API_TOKEN" ] && [ ! -f ~/.pypirc ]; then
    echo -e "${YELLOW}⚠️  PyPI credentials not found${NC}"
    echo "Set PYPI_API_TOKEN environment variable or create ~/.pypirc"
    exit 1
fi

# Upload to PyPI
echo -e "${BLUE}Uploading to PyPI...${NC}"
python -m twine upload dist/* --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Published successfully!${NC}"
    echo -e "${GREEN}✅ Package: voxsigil-library===${NEW_VERSION}${NC}"
    echo -e "${GREEN}✅ View: https://pypi.org/project/voxsigil-library/${NC}"
    
    # Commit changes
    echo -e "${BLUE}Committing changes...${NC}"
    git add setup.py CHANGELOG.md
    git commit -m "chore: Bump PyPI version to ${NEW_VERSION}"
    git tag -a "pypi-v${NEW_VERSION}" -m "PyPI Release v${NEW_VERSION}"
    
    echo -e "${GREEN}✅ Git tag created: pypi-v${NEW_VERSION}${NC}"
    echo -e "${BLUE}Push with: git push origin main --tags${NC}"
    
    # Cleanup
    echo -e "${BLUE}Cleaning up...${NC}"
    rm -rf build dist *.egg-info
else
    echo -e "${RED}❌ Failed to publish to PyPI${NC}"
    exit 1
fi
