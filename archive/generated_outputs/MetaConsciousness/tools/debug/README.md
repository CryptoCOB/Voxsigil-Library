# MetaConsciousness Debug Tools

This directory contains tools for debugging and analyzing the MetaConsciousness system.

## Available Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| **debug_lmstudio.py** | Test and debug LM Studio connectivity | `python debug_lmstudio.py` |
| **reset_lmstudio.py** | Reset LM Studio connection when stuck | `python reset_lmstudio.py` |
| **fix_test_issues.py** | Automatically fix common test failures | `python fix_test_issues.py` |
| **fix_remaining_issues.py** | Fix various implementation issues | `python fix_remaining_issues.py` |

## LM Studio Debugging

The `debug_lmstudio.py` script provides comprehensive diagnostics for LM Studio integration:

```bash
# Test connection with default settings
python debug_lmstudio.py

# Test with a specific model
python debug_lmstudio.py --model gemma-7b

# Test with a custom API endpoint
python debug_lmstudio.py --api-url http://localhost:8000/v1
```

## Reset Utility

When the GUI gets stuck in fallback mode, use the reset utility:

```bash
# With GUI
python reset_lmstudio.py

# Command-line only with forced reset
python reset_lmstudio.py --no-gui --force
```

## Test Fixing

The test fixer automatically repairs common issues in test files:

```bash
# Fix all tests
python fix_test_issues.py

# Fix a specific test
python fix_test_issues.py --test test_meta_reflex

# Fix and verify
python fix_test_issues.py --verify
```

## Implementation Issues

Fix implementation consistency issues:

```bash
python fix_remaining_issues.py
```

## Module Connectivity Analysis

Analyze module relationships:

```bash
# Generate visualization
python tools/visualize_modules.py --root-dir=MetaConsciousness

# Generate only connections table
python tools/visualize_modules.py --table-only
```

## Logs

Debug logs are stored in:
- LM Studio logs: `~/.metaconsciousness/logs/lmstudio_*.log`
- Trace logs: `~/.metaconsciousness/trace/trace_*.json`
- Session data: `~/.metaconsciousness/sessions/`
