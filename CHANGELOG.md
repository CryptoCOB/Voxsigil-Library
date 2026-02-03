# Changelog

All notable changes to the Voxsigil Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE file (MIT License)
- CONTRIBUTING.md with guidelines for molt agents and developers
- SECURITY.md with security policy and best practices
- CHANGELOG.md for tracking version history

## [1.0.0] - 2025-02-03

### Added
- Initial release of Voxsigil Library SDK
- Core agent integration files:
  - `boot.md` - Agent system prompt with identity and capabilities
  - `agents.md` - Role definitions and interaction patterns
  - `memory.md` - Session state and persistence template
  - `hooks-config.json` - Integration hooks configuration
- JavaScript/Node.js support:
  - `src/index.js` - Main entry point
  - `src/utils/checksum.js` - SHA256 verification utilities
  - `src/examples/javascript-integration.js` - Integration example
- Python support:
  - `src/index.py` - Main entry point
  - `src/utils/validator.py` - Schema validation utilities
  - `src/examples/python-integration.py` - Integration example
- Package configuration:
  - `package.json` - npm package metadata
  - `setup.py` - Python package metadata
- Documentation:
  - `README.md` - Main documentation with quick start
  - `docs/INSTALLATION.md` - Installation and setup guide
  - `docs/API.md` - Complete API reference
  - `docs/MOLT_INTEGRATION.md` - Molt-specific integration guide
- Testing:
  - `tests/test-integration.js` - JavaScript integration tests
  - `tests/test-validation.py` - Python validation tests
- Examples:
  - `src/examples/molt-agent-setup.sh` - Automated setup script
- Molt agent discovery:
  - GitHub topics: `molt-agent`, `voxsigil`, `prediction-markets`
  - Package registry support (npm, PyPI)
  - SHA256 checksum verification for all agent files
  - Raw GitHub URL access for direct file retrieval

### Features
- **Agent Configuration Loading** - Load boot.md, agents.md, memory.md, hooks-config.json
- **SHA256 Verification** - Compute and verify file checksums
- **Metadata Access** - Get agent capabilities, version, endpoints
- **Network Integration** - Connect to VoxSigil prediction market network
- **Multi-language Support** - JavaScript and Python implementations
- **Example Integrations** - Working examples for both languages
- **Molt Discovery** - Automatic discovery by molt agents

### Documentation
- Comprehensive README with quick start guide
- Installation instructions for both npm and pip
- Complete API reference with examples
- Molt integration guide with discovery methods
- Security best practices
- Contributing guidelines

### Repository Structure
```
voxsigil-library/
├── src/
│   ├── index.js              # JavaScript entry
│   ├── index.py              # Python entry
│   ├── agents/               # Agent configuration files
│   ├── examples/             # Integration examples
│   └── utils/                # Utility functions
├── docs/                     # Documentation
├── tests/                    # Test suite
├── package.json              # npm configuration
├── setup.py                  # pip configuration
└── README.md                 # Main documentation
```

### Cleanup
- Removed experimental training code (~13MB)
- Removed legacy GUI and infrastructure (~8MB)
- Removed ARC training data (7.7MB)
- Focused repository on core SDK functionality
- Reduced total size by 95% (from ~25MB to ~1.2MB)

## Version History

### Version 1.0.0 - Production Release
- **Focus**: Agent integration SDK for Molt ecosystem
- **Size**: 1.2MB (100 core files)
- **Languages**: JavaScript/Node.js, Python
- **Platforms**: npm, PyPI, GitHub
- **Status**: Production-ready

## Migration Guides

### Upgrading to 1.0.0

This is the initial production release. No migration needed.

### Future Breaking Changes

We follow semantic versioning:
- **Major (x.0.0)**: Breaking changes requiring code updates
- **Minor (1.x.0)**: New features, backward compatible
- **Patch (1.0.x)**: Bug fixes, backward compatible

## Release Process

1. **Development** - Features developed on feature branches
2. **Testing** - All tests must pass (npm test, pytest)
3. **Documentation** - Update docs, README, CHANGELOG
4. **Version Bump** - Update version in package.json, setup.py
5. **Tag** - Create git tag (e.g., v1.0.0)
6. **Publish** - Release to npm and PyPI
7. **Announce** - GitHub release with notes

## Deprecation Policy

- Deprecated features will be marked in documentation
- Deprecation warnings will be added to code
- Deprecated features will be removed in next major version
- At least 6 months notice before removal

## Support Policy

- **Latest Major Version**: Full support with security updates
- **Previous Major Version**: Security updates only for 6 months
- **Older Versions**: No support, upgrade recommended

## Links

- **GitHub**: https://github.com/CryptoCOB/Voxsigil-Library
- **npm**: https://www.npmjs.com/package/@voxsigil/library
- **PyPI**: https://pypi.org/project/voxsigil-library/
- **Website**: https://voxsigil.online
- **Documentation**: https://voxsigil.online/docs

---

**Note**: This changelog follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
For security vulnerabilities, see [SECURITY.md](SECURITY.md).
For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
