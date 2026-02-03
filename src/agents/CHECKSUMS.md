# VoxSigil Agent File Checksums

This document contains SHA256 checksums for all VoxSigil agent configuration files. Use these checksums to verify file integrity when integrating with molt agents.

## Version 1.0.0

Last updated: 2025-02-03

### Agent Configuration Files

```
f59d8f970bf9b009a19274f92fb75a04feb8f997d8f1f32053a1610008e44afb  boot.md
c096187b1d91e018a6ca4c13886f1021e3e0c00a83a2feef949e43a7f0de6967  agents.md
5a4aad0e524e7f9eeaf98f9ae71d0b452fde02f6069cb943c709d33f2ce29bfc  memory.md
cbcd749e642a1fe3a6e8a8f824e7e451ac4933aeb32f5c3225ecd0d3e5bb5523  hooks-config.json
```

## Verification

### Using Command Line

```bash
# Verify all files at once
cd src/agents
sha256sum -c CHECKSUMS.txt

# Verify individual file
sha256sum boot.md
# Expected: f59d8f970bf9b009a19274f92fb75a04feb8f997d8f1f32053a1610008e44afb
```

### Using JavaScript

```javascript
const voxsigil = require('@voxsigil/library');
const fs = require('fs');

// Verify a file
const filePath = 'src/agents/boot.md';
const expectedChecksum = 'f59d8f970bf9b009a19274f92fb75a04feb8f997d8f1f32053a1610008e44afb';
const isValid = voxsigil.verifyFileChecksum(filePath, expectedChecksum);

console.log(`File integrity: ${isValid ? 'VALID' : 'INVALID'}`);
```

### Using Python

```python
from voxsigil import VoxSigilAgent
from pathlib import Path

agent = VoxSigilAgent()

# Verify a file
file_path = Path('src/agents/boot.md')
expected_checksum = 'f59d8f970bf9b009a19274f92fb75a04feb8f997d8f1f32053a1610008e44afb'
is_valid = agent.verify_file_checksum(file_path, expected_checksum)

print(f"File integrity: {'VALID' if is_valid else 'INVALID'}")
```

## File Sizes

For reference, approximate file sizes:

- `boot.md`: ~8.4 KB
- `agents.md`: ~12.6 KB
- `memory.md`: ~12.6 KB
- `hooks-config.json`: ~3.0 KB

Total: ~36.6 KB

## Security Note

Always verify checksums when:
- Downloading files from external sources
- Integrating with new molt agents
- After repository updates
- Before deploying to production

If checksums don't match, do not use the files and report the issue at:
https://github.com/CryptoCOB/Voxsigil-Library/issues

## Checksum Format

All checksums use SHA256 algorithm and are represented as 64-character hexadecimal strings.

## Updating Checksums

When agent files are modified:

1. Update the files
2. Recompute checksums: `sha256sum src/agents/*.{md,json}`
3. Update this file with new checksums
4. Update version number
5. Update CHANGELOG.md
6. Commit changes with descriptive message

---

For more information, see [SECURITY.md](../SECURITY.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
