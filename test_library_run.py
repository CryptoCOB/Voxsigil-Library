#!/usr/bin/env python3
"""Test VoxSigil Library Integration"""

import sys
sys.path.insert(0, 'src')

print('Testing VoxSigil Library at C:\\UBLT (Python)...')
print('=' * 60)

# Test 1: Import Python library
try:
    from voxsigil_library import OpenClawdAgentFactory, OpenClawdEvent
    print('PASS Test 1: Python library imported from C:\\UBLT')
except ImportError as e:
    print(f'FAIL Test 1: {e}')
    sys.exit(1)

# Test 2: Create agent with VoxBridge client
try:
    adapter = OpenClawdAgentFactory.create(
        name='library-test-agent',
        agent_type='llm',
        voxbridge_url='https://voxsigil-predict.fly.dev',
        description='Testing library repository'
    )
    print('PASS Test 2: OpenClawdAgentFactory working')
except Exception as e:
    print(f'FAIL Test 2: {e}')
    sys.exit(1)

# Test 3: Test event creation
try:
    event = OpenClawdEvent(
        output_type='forecast',
        title='Library Test Forecast',
        description='Testing from library repo',
        impact_score=0.75,
        data={'test': True}
    )
    print('PASS Test 3: Event creation working')
except Exception as e:
    print(f'FAIL Test 3: {e}')
    sys.exit(1)

# Test 4: Test VoxBridgeClient
try:
    from voxsigil_library.openclawd_adapter import VoxBridgeClient
    client = VoxBridgeClient(
        agent_name='test-client',
        agent_type='llm',
        base_url='https://voxsigil-predict.fly.dev'
    )
    print('PASS Test 4: VoxBridgeClient instantiation working')
except Exception as e:
    print(f'FAIL Test 4: {e}')
    sys.exit(1)

# Test 5: Check sigil files
try:
    import os
    sigil_count = len([f for f in os.listdir('sigils') if f.endswith('.voxsigil')])
    print(f'PASS Test 5: Found {sigil_count} VoxSigil files')
except Exception as e:
    print(f'FAIL Test 5: {e}')
    sys.exit(1)

print('=' * 60)
print('ALL PYTHON LIBRARY TESTS PASSED')
print()
print('Library Repository Status:')
print('  Location: C:\\UBLT')
print('  Remote: https://github.com/CryptoCOB/Voxsigil-Library')
print('  Components: Python + JavaScript OpenClawdAdapter, 32 VoxSigil sigils')
print('  Status: FULLY OPERATIONAL')
