# Contributing to Voxsigil Library

Thank you for your interest in contributing to the Voxsigil Library! This document provides guidelines for molt agents and developers who want to contribute to the project.

## How to Contribute

### For Molt Agents

Molt agents can contribute by:

1. **Testing Integration** - Use the library in your agent implementation and report issues
2. **Documentation** - Improve examples, tutorials, and integration guides
3. **Feature Requests** - Suggest new capabilities that would benefit agent coordination
4. **Bug Reports** - Report issues with existing functionality

### For Developers

1. **Fork the Repository**
   ```bash
   git clone https://github.com/CryptoCOB/Voxsigil-Library.git
   cd Voxsigil-Library
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # JavaScript
   npm test
   
   # Python
   pytest tests/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python

- Follow PEP 8 standards
- Use Black formatter (2-space indents for consistency with JS)
- Add type hints (Python 3.8+)
- Include docstrings for all public functions
- Run flake8 for linting

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

### JavaScript/TypeScript

- Use ESLint with recommended rules
- Format with Prettier (2-space indents)
- Add JSDoc comments for exports
- Use async/await patterns
- No console.log in production code

```bash
# Format code
npx prettier --write src/**/*.js

# Check linting
npx eslint src/**/*.js
```

## Testing Guidelines

### Writing Tests

- Add unit tests for all new functionality
- Include integration tests for agent workflows
- Mock external network calls
- Test error handling scenarios
- Verify SHA256 checksums for agent files

### Test Structure

```python
# Python test example
def test_agent_config_loading():
    """Test that agent configuration loads correctly."""
    agent = VoxSigilAgent()
    config = agent.load_agent_config()
    assert 'boot' in config
    assert 'agents' in config
    assert 'memory' in config
    assert 'hooks' in config
```

```javascript
// JavaScript test example
function testChecksumComputation() {
  const checksum = computeChecksum(Buffer.from('test data'));
  assert(checksum.length === 64, 'Checksum should be 64 hex characters');
}
```

## Documentation Guidelines

### Adding Examples

When adding examples:
- Include complete, runnable code
- Add comments explaining key concepts
- Show error handling
- Include expected output
- Test examples before submitting

### Updating Documentation

- Keep README.md focused and concise
- Update API.md for all public functions
- Add molt-specific guidance to MOLT_INTEGRATION.md
- Include code examples in documentation

## Security Guidelines

Before submitting code, verify:

- [ ] No hardcoded API keys or credentials
- [ ] All external input is validated
- [ ] URLs and endpoints are sanitized
- [ ] HTTPS is used for all network calls
- [ ] No sensitive data in logs or error messages
- [ ] Dependencies are up to date and secure

## Agent File Changes

When modifying core agent files (boot.md, agents.md, memory.md, hooks-config.json):

1. **Update Checksums** - Recompute SHA256 checksums
2. **Test Integration** - Verify with example agents
3. **Document Changes** - Update CHANGELOG.md
4. **Backward Compatibility** - Maintain compatibility when possible
5. **Version Bump** - Update version if breaking changes

```bash
# Compute new checksums
cd src/agents
sha256sum boot.md agents.md memory.md hooks-config.json
```

## Pull Request Process

1. **Title** - Use descriptive title (e.g., "Add: support for new signal type")
2. **Description** - Explain what changes and why
3. **Testing** - Describe how you tested the changes
4. **Documentation** - Confirm documentation is updated
5. **Checklist** - Complete the PR checklist

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No security issues introduced
- [ ] Checksums updated if agent files changed
- [ ] Version bumped if needed
- [ ] CHANGELOG.md updated

## Issue Reporting

When reporting issues:

1. **Search First** - Check if issue already exists
2. **Use Template** - Fill out the issue template completely
3. **Minimal Example** - Provide minimal code to reproduce
4. **Environment** - Include Node.js/Python version, OS
5. **Expected vs Actual** - Describe what should happen vs what does

### Issue Template

```markdown
**Description**
Brief description of the issue

**To Reproduce**
Steps to reproduce the behavior:
1. Install version X
2. Run command Y
3. See error Z

**Expected Behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Node.js: [e.g., v18.0.0]
- Python: [e.g., 3.10]
- Library Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## Molt Agent Integration

Special considerations for molt agents:

### Discovery

Ensure changes maintain:
- GitHub topic tags (molt-agent, voxsigil)
- Package registry compatibility (npm, PyPI)
- SHA256 checksum verification
- Raw GitHub URL access

### Coordination

When adding coordination features:
- Document in MOLT_INTEGRATION.md
- Include examples for multi-agent scenarios
- Test with multiple agent instances
- Consider network latency and failures

### Authentication

For authentication changes:
- Use EIP-191 signing standard
- Maintain backward compatibility
- Document key management
- Test signature verification

## Community

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and ideas
- **Website** - https://voxsigil.online for documentation
- **Email** - support@voxsigil.online for security issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize agent coordination and network health
- Maintain high security standards
- Help others learn and grow

### Unacceptable Behavior

- Harassment or discriminatory language
- Publishing others' private information
- Introducing security vulnerabilities deliberately
- Spam or off-topic contributions

## Questions?

If you have questions about contributing:
- Open a GitHub Discussion
- Review existing documentation
- Check examples in src/examples/
- Contact support@voxsigil.online

Thank you for contributing to Voxsigil Library and helping build the molt agent ecosystem!
