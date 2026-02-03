"""
Integration Tests for VoxSigil Library (Python)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.index import VoxSigilAgent, get_metadata, compute_checksum
from src.utils.validator import (
    AgentValidator, ValidationError,
    validate_signal, validate_agent_config
)

# Test counter
tests_run = 0
tests_passed = 0
tests_failed = 0


def test(name, fn):
    """Run a test."""
    global tests_run, tests_passed, tests_failed
    tests_run += 1
    try:
        fn()
        tests_passed += 1
        print(f"✓ {name}")
    except AssertionError as e:
        tests_failed += 1
        print(f"✗ {name}")
        print(f"  Error: {e}")
    except Exception as e:
        tests_failed += 1
        print(f"✗ {name}")
        print(f"  Unexpected error: {e}")


print("Running VoxSigil Library Integration Tests (Python)")
print("=" * 70)
print()

# Test 1: Module imports
def test_module_imports():
    """Test module imports successfully."""
    agent = VoxSigilAgent()
    assert agent is not None, "Agent should instantiate"
    assert hasattr(agent, 'load_agent_config'), "Should have load_agent_config method"
    assert hasattr(agent, 'compute_checksum'), "Should have compute_checksum method"

test("Module imports successfully", test_module_imports)

# Test 2: Get metadata
def test_get_metadata():
    """Test get_metadata returns valid data."""
    metadata = get_metadata()
    
    assert metadata['name'] == 'voxsigil-library', "Name should match"
    assert metadata['version'] == '1.0.0', "Version should be 1.0.0"
    assert isinstance(metadata['capabilities'], list), "Capabilities should be a list"
    assert len(metadata['capabilities']) > 0, "Should have capabilities"

test("get_metadata returns valid data", test_get_metadata)

# Test 3: Compute checksum
def test_compute_checksum():
    """Test compute_checksum works correctly."""
    test_data = b"Hello, VoxSigil!"
    checksum = compute_checksum(test_data)
    
    assert isinstance(checksum, str), "Checksum should be a string"
    assert len(checksum) == 64, "SHA256 should be 64 hex characters"
    assert all(c in '0123456789abcdef' for c in checksum), "Checksum should be hex"

test("compute_checksum works correctly", test_compute_checksum)

# Test 4: Agent files exist
def test_agent_files_exist():
    """Test agent files exist."""
    agents_dir = Path(__file__).parent.parent / 'src' / 'agents'
    files = ['boot.md', 'agents.md', 'memory.md', 'hooks-config.json']
    
    for filename in files:
        filepath = agents_dir / filename
        assert filepath.exists(), f"{filename} should exist"

test("Agent files exist", test_agent_files_exist)

# Test 5: Load agent config
def test_load_agent_config():
    """Test load_agent_config loads all files."""
    agent = VoxSigilAgent()
    config = agent.load_agent_config()
    
    assert 'boot' in config, "Should have boot content"
    assert 'agents' in config, "Should have agents content"
    assert 'memory' in config, "Should have memory content"
    assert 'hooks' in config, "Should have hooks config"
    
    assert isinstance(config['boot'], str), "boot should be a string"
    assert isinstance(config['agents'], str), "agents should be a string"
    assert isinstance(config['memory'], str), "memory should be a string"
    assert isinstance(config['hooks'], dict), "hooks should be a dict"

test("load_agent_config loads all files", test_load_agent_config)

# Test 6: Hooks config is valid
def test_hooks_config_valid():
    """Test hooks-config.json is valid."""
    agent = VoxSigilAgent()
    config = agent.load_agent_config()
    
    assert 'hooks' in config['hooks'], "Should have hooks section"
    
    # Check a specific hook
    boot_hook = config['hooks']['hooks']['boot-md']
    assert boot_hook, "boot-md hook should exist"
    assert boot_hook['enabled'] is True, "boot-md should be enabled"
    assert boot_hook['trigger'] == 'on_startup', "boot-md should trigger on startup"

test("hooks-config.json is valid", test_hooks_config_valid)

# Test 7: Validate agent config
def test_validate_agent_config():
    """Test agent config validation."""
    agent = VoxSigilAgent()
    config = agent.load_agent_config()
    
    # Should validate successfully
    is_valid = validate_agent_config(config)
    assert is_valid, "Agent config should validate"

test("Agent config validates successfully", test_validate_agent_config)

# Test 8: Validate signal
def test_validate_signal():
    """Test signal validation."""
    valid_signal = {
        'agent_id': 'voxsigil-001',
        'market_id': 'market-123',
        'prediction': 0.67,
        'confidence': 0.85,
        'timestamp': '2026-02-03T12:00:00Z'
    }
    
    # Should validate successfully
    is_valid = validate_signal(valid_signal)
    assert is_valid, "Valid signal should validate"
    
    # Invalid signal (prediction out of range)
    invalid_signal = valid_signal.copy()
    invalid_signal['prediction'] = 1.5
    
    try:
        validate_signal(invalid_signal)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected

test("Signal validation works", test_validate_signal)

# Test 9: Checksum verification
def test_checksum_verification():
    """Test checksum verification."""
    agent = VoxSigilAgent()
    agents_dir = Path(__file__).parent.parent / 'src' / 'agents'
    boot_path = agents_dir / 'boot.md'
    
    # Compute checksum
    with open(boot_path, 'rb') as f:
        data = f.read()
    expected_checksum = agent.compute_checksum(data)
    
    # Verify it matches
    is_valid = agent.verify_file_checksum(boot_path, expected_checksum)
    assert is_valid, "File checksum should verify correctly"
    
    # Verify it fails with wrong checksum
    is_invalid = agent.verify_file_checksum(boot_path, 'wrong_checksum')
    assert not is_invalid, "Should fail with wrong checksum"

test("Checksum verification works", test_checksum_verification)

# Test 10: Agent files have content
def test_agent_files_content():
    """Test agent files have substantial content."""
    agent = VoxSigilAgent()
    config = agent.load_agent_config()
    
    # BOOT.md should be 250-300+ lines
    assert len(config['boot']) > 5000, "BOOT.md should have substantial content"
    
    # AGENTS.md should be 300-350+ lines
    assert len(config['agents']) > 8000, "AGENTS.md should have substantial content"
    
    # MEMORY.md should be 250-300+ lines
    assert len(config['memory']) > 8000, "MEMORY.md should have substantial content"

test("Agent files have substantial content", test_agent_files_content)

# Test 11: setup.py exists and is valid
def test_setup_py_valid():
    """Test setup.py is valid."""
    setup_path = Path(__file__).parent.parent / 'setup.py'
    assert setup_path.exists(), "setup.py should exist"
    
    # Read and check contents
    content = setup_path.read_text()
    assert 'voxsigil-library' in content, "Should have package name"
    assert 'molt-agent' in content, "Should have molt-agent keyword"

test("setup.py is valid", test_setup_py_valid)

# Test 12: Validator handles errors
def test_validator_errors():
    """Test validator handles errors correctly."""
    # Missing required field
    invalid_signal = {
        'agent_id': 'voxsigil-001',
        # missing market_id
        'prediction': 0.67,
        'confidence': 0.85,
        'timestamp': '2026-02-03T12:00:00Z'
    }
    
    try:
        validate_signal(invalid_signal)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert 'market_id' in str(e), "Error should mention missing field"

test("Validator handles errors correctly", test_validator_errors)

# Summary
print()
print("=" * 70)
print("Test Summary")
print("=" * 70)
print(f"Total: {tests_run}")
print(f"Passed: {tests_passed}")
print(f"Failed: {tests_failed}")
print()

if tests_failed > 0:
    print("❌ Some tests failed")
    sys.exit(1)
else:
    print("✅ All tests passed")
    sys.exit(0)
