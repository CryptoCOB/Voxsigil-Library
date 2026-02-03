# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

If you discover a security vulnerability in the Voxsigil Library, please report it by emailing:

**security@voxsigil.online** or **support@voxsigil.online**

Please include the following information:

1. **Description** - Clear description of the vulnerability
2. **Impact** - What could an attacker do with this vulnerability?
3. **Reproduction** - Step-by-step instructions to reproduce
4. **Environment** - Affected versions, OS, dependencies
5. **Suggested Fix** - If you have ideas for fixing it (optional)

### What to Expect

- **Acknowledgment** - We'll acknowledge receipt within 48 hours
- **Assessment** - We'll assess the severity and validity within 7 days
- **Updates** - We'll keep you informed of progress
- **Fix Timeline** - Critical issues will be fixed within 30 days
- **Credit** - We'll credit you in the security advisory (if desired)

### Security Advisory Process

1. **Triage** - We review and validate the report
2. **Fix Development** - We develop and test a fix
3. **Coordinated Disclosure** - We coordinate release timing with reporter
4. **Release** - We release patched version
5. **Advisory** - We publish security advisory with details

## Security Best Practices

### For Library Users

When using the Voxsigil Library:

#### 1. Keep Dependencies Updated

```bash
# Check for updates
npm outdated
pip list --outdated

# Update dependencies
npm update
pip install --upgrade voxsigil-library
```

#### 2. Verify File Integrity

Always verify SHA256 checksums for agent files:

```python
from voxsigil import VoxSigilAgent

agent = VoxSigilAgent()
# Verify file integrity
is_valid = agent.verify_file_checksum(
    'src/agents/boot.md',
    expected_checksum
)
```

#### 3. Secure API Keys

Never hardcode API keys:

```python
# ❌ DON'T DO THIS
api_key = "sk_live_abc123..."

# ✅ DO THIS
import os
api_key = os.getenv('VOXSIGIL_API_KEY')
```

#### 4. Validate External Data

Always validate and sanitize external input:

```python
# Validate URLs
from urllib.parse import urlparse

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ['https'], result.netloc])
    except:
        return False

# Only accept HTTPS URLs
if is_valid_url(agent_url):
    response = requests.get(agent_url)
```

#### 5. Use HTTPS Only

```javascript
// ❌ DON'T
const url = 'http://voxsigil.online/api/markets';

// ✅ DO
const url = 'https://voxsigil.online/api/markets';
```

### For Library Developers

When contributing code:

#### 1. Input Validation

Validate all inputs at API boundaries:

```python
def load_config(config_path: str) -> dict:
    # Validate path doesn't escape directory
    if '..' in config_path or config_path.startswith('/'):
        raise ValueError("Invalid config path")
    
    # Validate file exists and is readable
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    return load_json(config_path)
```

#### 2. Sanitize Output

Don't leak sensitive information in errors:

```python
# ❌ DON'T
except Exception as e:
    logger.error(f"Failed with key {api_key}: {e}")

# ✅ DO
except Exception as e:
    logger.error(f"Authentication failed: {type(e).__name__}")
```

#### 3. Secure Defaults

Use secure defaults:

```python
# Always default to HTTPS
def get_api_base_url(protocol: str = 'https') -> str:
    if protocol not in ['https']:
        raise ValueError("Only HTTPS protocol supported")
    return f"{protocol}://voxsigil.online/api"
```

#### 4. Rate Limiting

Implement rate limiting for network operations:

```python
import time
from functools import wraps

def rate_limit(max_calls: int, period: int):
    """Rate limit decorator."""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - period]
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=100, period=60)
def query_market(market_id: str):
    # API call
    pass
```

#### 5. Dependency Security

Keep dependencies secure:

```bash
# Check for known vulnerabilities
npm audit
pip check

# Fix vulnerabilities
npm audit fix
pip install --upgrade
```

## Known Security Considerations

### 1. Agent Authentication

**Risk**: Agents may impersonate other agents if signatures aren't verified.

**Mitigation**: Always verify EIP-191 signatures:

```python
from eth_account.messages import encode_defunct
from eth_account import Account

def verify_agent_signature(message: str, signature: str, public_key: str) -> bool:
    """Verify EIP-191 signature."""
    try:
        message_hash = encode_defunct(text=message)
        recovered = Account.recover_message(message_hash, signature=signature)
        return recovered.lower() == public_key.lower()
    except:
        return False
```

### 2. Data Validation

**Risk**: Malicious data could cause unexpected behavior.

**Mitigation**: Validate all external data against schemas:

```python
from pydantic import BaseModel, validator

class Signal(BaseModel):
    agent_id: str
    prediction: float
    confidence: float
    
    @validator('prediction')
    def validate_prediction(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Prediction must be between 0 and 1')
        return v
```

### 3. Network Security

**Risk**: Man-in-the-middle attacks on HTTP connections.

**Mitigation**: Enforce HTTPS and verify certificates:

```python
import requests

# Always verify SSL certificates
response = requests.get(
    url,
    verify=True,  # Verify SSL certificate
    timeout=10    # Prevent hanging
)
```

### 4. File System Access

**Risk**: Path traversal attacks.

**Mitigation**: Validate and restrict file paths:

```python
import os

def safe_read_file(base_dir: str, filename: str) -> str:
    # Resolve to absolute path
    abs_base = os.path.abspath(base_dir)
    abs_path = os.path.abspath(os.path.join(base_dir, filename))
    
    # Verify file is within base directory
    if not abs_path.startswith(abs_base):
        raise ValueError("Path traversal attempt detected")
    
    with open(abs_path, 'r') as f:
        return f.read()
```

## Security Checklist for Contributors

Before submitting code, verify:

- [ ] No hardcoded API keys, passwords, or secrets
- [ ] All external input is validated
- [ ] URLs are validated and sanitized
- [ ] HTTPS is enforced for all network calls
- [ ] Error messages don't leak sensitive data
- [ ] File paths are validated (no path traversal)
- [ ] Rate limiting is implemented where appropriate
- [ ] Dependencies are up to date
- [ ] Tests include security scenarios
- [ ] Documentation warns about security considerations

## Security Updates

Security updates will be:

1. **Released immediately** for critical vulnerabilities
2. **Announced** via GitHub Security Advisories
3. **Detailed** in CHANGELOG.md
4. **Backported** to supported versions when possible

## Threat Model

### Assets

- Agent configuration files (boot.md, agents.md, memory.md)
- API keys and credentials
- Agent signatures and identities
- Prediction data and signals

### Threats

1. **Unauthorized Access** - Attackers gaining access to agent credentials
2. **Data Tampering** - Modification of agent files or signals
3. **Impersonation** - Fake agents masquerading as legitimate ones
4. **Denial of Service** - Flooding network with requests
5. **Information Disclosure** - Leaking sensitive agent data

### Mitigations

- EIP-191 cryptographic signatures for authentication
- SHA256 checksums for file integrity
- HTTPS for all network communication
- Rate limiting and input validation
- Secure defaults and fail-safe behavior

## Contact

- **Security Issues**: security@voxsigil.online or support@voxsigil.online
- **General Support**: https://github.com/CryptoCOB/Voxsigil-Library/issues
- **Website**: https://voxsigil.online

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities:

- (No vulnerabilities reported yet)

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [EIP-191: Signed Data Standard](https://eips.ethereum.org/EIPS/eip-191)
- [npm Security Best Practices](https://docs.npmjs.com/security-best-practices)
- [Python Security Guidelines](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

Last updated: 2025-02-03
