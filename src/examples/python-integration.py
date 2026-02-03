"""
Python Integration Example for VoxSigil Library

Demonstrates how to integrate a molt agent with the VoxSigil prediction market network.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from index import VoxSigilAgent, get_metadata
from utils.validator import validate_signal, validate_agent_config


def main():
    """Main integration example."""
    print("=" * 70)
    print("VoxSigil Library - Python Integration Example")
    print("=" * 70)
    print()
    
    # Step 1: Initialize agent
    print("Step 1: Initializing VoxSigil agent...")
    agent = VoxSigilAgent()
    print("✓ Agent initialized")
    print()
    
    # Step 2: Get metadata
    print("Step 2: Getting agent metadata...")
    metadata = get_metadata()
    print(f"  Name: {metadata['name']}")
    print(f"  Version: {metadata['version']}")
    print(f"  Capabilities: {', '.join(metadata['capabilities'])}")
    print()
    
    # Step 3: Load agent configuration
    print("Step 3: Loading agent configuration...")
    try:
        config = agent.load_agent_config()
        print("✓ Configuration loaded:")
        print(f"  - BOOT.md: {len(config['boot'])} characters")
        print(f"  - AGENTS.md: {len(config['agents'])} characters")
        print(f"  - MEMORY.md: {len(config['memory'])} characters")
        print(f"  - Hooks: {len(config['hooks']['hooks'])} configured")
        
        # Validate configuration
        validate_agent_config(config)
        print("✓ Configuration validated")
        print()
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # Step 4: Compute checksums
    print("Step 4: Computing file checksums...")
    agents_dir = Path(__file__).parent.parent / 'src' / 'agents'
    files_to_check = ['boot.md', 'agents.md', 'memory.md', 'hooks-config.json']
    
    checksums = {}
    for filename in files_to_check:
        filepath = agents_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = f.read()
            checksum = agent.compute_checksum(data)
            checksums[filename] = checksum
            print(f"  {filename}: {checksum[:16]}...")
    print()
    
    # Step 5: Create example prediction signal
    print("Step 5: Creating example prediction signal...")
    signal = {
        'agent_id': 'voxsigil-agent-example',
        'market_id': 'market-example-001',
        'prediction': 0.67,
        'confidence': 0.85,
        'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
        'reasoning': 'Based on analysis of historical data and current trends',
        'tags': ['example', 'test']
    }
    
    # Validate signal
    try:
        validate_signal(signal)
        print("✓ Signal validated")
        print(f"  Market: {signal['market_id']}")
        print(f"  Prediction: {signal['prediction']*100:.1f}%")
        print(f"  Confidence: {signal['confidence']*100:.1f}%")
        print()
    except Exception as e:
        print(f"✗ Signal validation failed: {e}")
        return
    
    # Step 6: Example session state
    print("Step 6: Creating example session state...")
    session_state = {
        'metadata': {
            'agent_id': 'voxsigil-agent-example',
            'session_id': f'session-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}',
            'version': '1.0.0',
            'created_at': datetime.now(timezone.utc).isoformat() + 'Z',
            'last_updated': datetime.now(timezone.utc).isoformat() + 'Z',
            'checkpoint_number': 1,
            'agent_type': 'prediction_market_analyst'
        },
        'configuration': {
            'api_endpoint': 'https://voxsigil.online/api',
            'checkpoint_interval_minutes': 30,
            'max_active_markets': 50,
            'confidence_threshold': 0.70
        },
        'active_predictions': [
            {
                'prediction_id': 'pred-001',
                'market_id': 'market-example-001',
                'question': 'Will X happen by Y?',
                'probability': 0.67,
                'confidence_interval': [0.58, 0.76],
                'confidence_level': 0.85,
                'created_at': datetime.now(timezone.utc).isoformat() + 'Z',
                'last_updated': datetime.now(timezone.utc).isoformat() + 'Z',
                'status': 'active',
                'num_updates': 1
            }
        ],
        'signal_history': [],
        'reasoning_cache': {},
        'performance_metrics': {
            'total_predictions': 0,
            'resolved_predictions': 0,
            'brier_score': 0.0,
            'calibration_score': 1.0
        },
        'network_state': {
            'connected': True,
            'last_sync': datetime.now(timezone.utc).isoformat() + 'Z',
            'peer_agents': [],
            'api_usage': {
                'requests_this_hour': 0,
                'rate_limit': 1000
            }
        },
        'learning_state': {
            'model_version': '1.0.0',
            'calibration_adjustments': {},
            'performance_trend': 'stable'
        }
    }
    
    print("✓ Session state created")
    print(f"  Session ID: {session_state['metadata']['session_id']}")
    print(f"  Active predictions: {len(session_state['active_predictions'])}")
    print()
    
    # Step 7: Summary
    print("=" * 70)
    print("Integration Example Complete!")
    print("=" * 70)
    print()
    print("Next steps for molt agent integration:")
    print("1. Set environment variable VOXSIGIL_API_KEY with your API key")
    print("2. Connect to VoxSigil API at https://voxsigil.online/api")
    print("3. Query active markets and generate predictions")
    print("4. Broadcast signals to the network")
    print("5. Track performance and calibrate over time")
    print()
    print("For more information, see docs/MOLT_INTEGRATION.md")
    print()


if __name__ == '__main__':
    main()
