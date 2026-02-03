#!/bin/bash
#
# Voxsigil Library - npm Publishing Script
#
# Usage: ./publish-npm.sh [version]
#
# Examples:
#   ./publish-npm.sh patch       # Bump patch version (1.0.0 → 1.0.1)
#   ./publish-npm.sh minor       # Bump minor version (1.0.0 → 1.1.0)
#   ./publish-npm.sh major       # Bump major version (1.0.0 → 2.0.0)
#   ./publish-npm.sh 1.2.3       # Set specific version
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Voxsigil npm Publishing${NC}"
echo "=================================="

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

# Check if logged in to npm
npm whoami > /dev/null 2>&1 || {
    echo -e "${YELLOW}⚠️  Not logged in to npm${NC}"
    echo "Running: npm login"
    npm login
}

# Get current version
CURRENT_VERSION=$(npm pkg get version | tr -d '"')
echo -e "${BLUE}Current version: ${CURRENT_VERSION}${NC}"

# Determine new version
VERSION_TYPE=${1:-patch}
case $VERSION_TYPE in
    patch|minor|major)
        # Let npm handle semantic versioning
        npm version $VERSION_TYPE --no-git-tag-version
        ;;
    *)
        # Assume it's a specific version number
        npm version $VERSION_TYPE --no-git-tag-version
        ;;
esac

NEW_VERSION=$(npm pkg get version | tr -d '"')
echo -e "${GREEN}✅ Version updated: ${CURRENT_VERSION} → ${NEW_VERSION}${NC}"

# Run tests before publishing
echo -e "${BLUE}Running tests...${NC}"
npm test || {
    echo -e "${RED}❌ Tests failed. Aborting publish.${NC}"
    exit 1
}

# Run security check
echo -e "${BLUE}Running security scan...${NC}"
npm audit --audit-level=moderate || {
    echo -e "${YELLOW}⚠️  Security warnings found${NC}"
}

# Verify package setup
echo -e "${BLUE}Verifying package...${NC}"
npm pack --dry-run

# Publish to npm
echo -e "${BLUE}Publishing to npm...${NC}"
npm publish --access public

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Published successfully!${NC}"
    echo -e "${GREEN}✅ Package: @voxsigil/library@${NEW_VERSION}${NC}"
    echo -e "${GREEN}✅ View: https://www.npmjs.com/package/@voxsigil/library${NC}"
    
    # Commit version bump
    echo -e "${BLUE}Committing version update...${NC}"
    git add package.json
    git commit -m "chore: Bump npm version to ${NEW_VERSION}"
    git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
    
    echo -e "${GREEN}✅ Git tag created: v${NEW_VERSION}${NC}"
    echo -e "${BLUE}Push with: git push origin main --tags${NC}"
else
    echo -e "${RED}❌ Failed to publish to npm${NC}"
    exit 1
fi
