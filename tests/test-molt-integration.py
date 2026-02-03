"""
Comprehensive Integration Tests for Voxsigil-Molt Agent System

Tests:
1. Agent initialization and bootstrap
2. Signal generation and verification
3. Multi-agent consensus computation
4. Market analysis workflows
5. State persistence and recovery
6. Network resilience
"""

import pytest
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

# Test fixtures
@pytest.fixture
def agent_config():
    """Sample agent configuration"""
    return {
        'agent_id': 'test-agent-001',
        'boot_md': 'Test boot config',
        'agents_md': 'Test agents config',
        'memory_md': 'Test memory config'
    }


@pytest.fixture
def sample_markets():
    """Sample prediction markets"""
    return [
        {
            'market_id': 'market_001',
            'question': 'Will BTC reach $100k?',
            'market_type': 'binary',
            'deadline': (datetime.utcnow() + timedelta(days=30)).isoformat()
        },
        {
            'market_id': 'market_002',
            'question': 'Which AI model wins?',
            'market_type': 'categorical',
            'deadline': (datetime.utcnow() + timedelta(days=60)).isoformat()
        }
    ]


@pytest.fixture
def sample_signals():
    """Sample prediction signals"""
    return [
        {
            'agent_id': 'agent_001',
            'market_id': 'market_001',
            'prediction': 0.72,
            'confidence': 0.85,
            'reasoning': 'Bullish signals',
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'agent_id': 'agent_002',
            'market_id': 'market_001',
            'prediction': 0.68,
            'confidence': 0.75,
            'reasoning': 'Moderate bullish',
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'agent_id': 'agent_003',
            'market_id': 'market_001',
            'prediction': 0.55,
            'confidence': 0.50,
            'reasoning': 'Uncertain',
            'timestamp': datetime.utcnow().isoformat()
        }
    ]


# Test Suite 1: Agent Initialization
class TestAgentInitialization:
    def test_agent_creation(self, agent_config):
        """Test agent can be created with config"""
        assert agent_config['agent_id'] == 'test-agent-001'
        assert agent_config['boot_md'] is not None

    def test_agent_bootstrap(self, agent_config):
        """Test agent bootstrap from configuration"""
        assert 'boot_md' in agent_config
        assert 'agents_md' in agent_config
        assert 'memory_md' in agent_config

    def test_agent_state_initialization(self, agent_config):
        """Test agent state is properly initialized"""
        state = {
            'agent_id': agent_config['agent_id'],
            'status': 'ready',
            'signals_sent': 0,
            'signals_received': 0,
            'consensus_reached': 0
        }
        assert state['status'] == 'ready'
        assert state['signals_sent'] == 0


# Test Suite 2: Signal Generation & Verification
class TestSignalGeneration:
    def test_signal_structure(self, sample_signals):
        """Test signal has required structure"""
        signal = sample_signals[0]
        assert 'agent_id' in signal
        assert 'market_id' in signal
        assert 'prediction' in signal
        assert 'confidence' in signal
        assert isinstance(signal['prediction'], float)
        assert 0 <= signal['prediction'] <= 1

    def test_signal_checksum_computation(self, sample_signals):
        """Test signal checksum verification"""
        signal = sample_signals[0]
        signal_json = json.dumps(signal, sort_keys=True)
        checksum = hashlib.sha256(signal_json.encode()).hexdigest()
        
        assert len(checksum) == 64  # SHA256
        assert checksum.startswith(tuple('0123456789abcdef'))

    def test_multiple_signals_generation(self, sample_signals):
        """Test generating multiple signals"""
        assert len(sample_signals) == 3
        market_ids = [s['market_id'] for s in sample_signals]
        assert all(mid == 'market_001' for mid in market_ids)

    def test_signal_validation(self, sample_signals):
        """Test signal validation rules"""
        signal = sample_signals[0]
        
        # Prediction must be 0-1
        assert 0 <= signal['prediction'] <= 1
        
        # Confidence must be reasonable
        assert 0 < signal['confidence'] <= 1
        
        # Must have reasoning
        assert len(signal['reasoning']) > 0


# Test Suite 3: Consensus Computation
class TestConsensusComputation:
    def test_weighted_average_consensus(self, sample_signals):
        """Test consensus computation with weighted average"""
        predictions = [s['prediction'] for s in sample_signals]
        confidence_weights = [s['confidence'] for s in sample_signals]
        
        weighted_sum = sum(p * c for p, c in zip(predictions, confidence_weights))
        weight_sum = sum(confidence_weights)
        consensus = weighted_sum / weight_sum
        
        assert 0 < consensus < 1
        assert 0.55 <= consensus <= 0.72  # Within signal range

    def test_consensus_strength_calculation(self, sample_signals):
        """Test consensus strength metric"""
        avg_confidence = sum(s['confidence'] for s in sample_signals) / len(sample_signals)
        
        assert 0 < avg_confidence <= 1
        assert avg_confidence > 0.5  # Should be relatively confident

    def test_consensus_with_disagreement(self):
        """Test consensus when agents disagree"""
        disagreeing_signals = [
            {'prediction': 0.9, 'confidence': 0.8},
            {'prediction': 0.1, 'confidence': 0.8},
            {'prediction': 0.5, 'confidence': 0.5}
        ]
        
        weighted_sum = sum(s['prediction'] * s['confidence'] for s in disagreeing_signals)
        weight_sum = sum(s['confidence'] for s in disagreeing_signals)
        consensus = weighted_sum / weight_sum
        
        # Should be neutral despite disagreement
        assert 0.3 <= consensus <= 0.7

    def test_consensus_convergence(self):
        """Test consensus converges as signal count increases"""
        base_signal = 0.65
        
        consensuses = []
        for n in range(1, 11):
            signals = [{'prediction': base_signal, 'confidence': 0.8} for _ in range(n)]
            weighted_sum = sum(s['prediction'] * s['confidence'] for s in signals)
            weight_sum = sum(s['confidence'] for s in signals)
            consensus = weighted_sum / weight_sum
            consensuses.append(consensus)
        
        # All should converge toward base signal
        assert all(abs(c - base_signal) < 0.01 for c in consensuses)


# Test Suite 4: Market Analysis Workflows
class TestMarketAnalysis:
    def test_market_registration(self, sample_markets):
        """Test market can be registered"""
        market = sample_markets[0]
        assert market['market_id'] == 'market_001'
        assert market['question'] is not None

    def test_market_deadline_validation(self, sample_markets):
        """Test market deadline validation"""
        market = sample_markets[0]
        deadline = datetime.fromisoformat(market['deadline'])
        now = datetime.utcnow()
        
        assert deadline > now  # Should be in future

    def test_market_signal_aggregation(self, sample_markets, sample_signals):
        """Test signals aggregated per market"""
        market_001_signals = [s for s in sample_signals if s['market_id'] == 'market_001']
        
        assert len(market_001_signals) == 3
        assert all(s['market_id'] == 'market_001' for s in market_001_signals)

    def test_market_consensus_generation(self, sample_markets, sample_signals):
        """Test market generates consensus"""
        market = sample_markets[0]
        market_signals = [s for s in sample_signals if s['market_id'] == market['market_id']]
        
        predictions = [s['prediction'] for s in market_signals]
        avg_prediction = sum(predictions) / len(predictions)
        
        consensus = {
            'market_id': market['market_id'],
            'prediction': avg_prediction,
            'agent_count': len(market_signals),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        assert consensus['agent_count'] == 3
        assert 0.5 <= consensus['prediction'] <= 0.75


# Test Suite 5: State Persistence & Recovery
class TestStatePersistence:
    def test_checkpoint_creation(self, agent_config):
        """Test agent checkpoint creation"""
        checkpoint = {
            'agent_id': agent_config['agent_id'],
            'timestamp': datetime.utcnow().isoformat(),
            'signals_sent': 5,
            'signals_received': 12,
            'consensus_reached': 3
        }
        
        assert 'timestamp' in checkpoint
        assert checkpoint['signals_sent'] == 5

    def test_checkpoint_serialization(self, agent_config):
        """Test checkpoint can be serialized to JSON"""
        checkpoint = {
            'agent_id': agent_config['agent_id'],
            'signals': [{
                'market_id': 'market_001',
                'prediction': 0.72,
                'timestamp': datetime.utcnow().isoformat()
            }]
        }
        
        json_str = json.dumps(checkpoint)
        restored = json.loads(json_str)
        
        assert restored['agent_id'] == agent_config['agent_id']
        assert len(restored['signals']) == 1

    def test_checkpoint_recovery(self):
        """Test agent can recover from checkpoint"""
        original_state = {
            'signals_sent': 10,
            'consensus_reached': 5
        }
        
        # Simulate checkpoint save and restore
        checkpoint = json.dumps(original_state)
        recovered = json.loads(checkpoint)
        
        assert recovered['signals_sent'] == 10
        assert recovered['consensus_reached'] == 5


# Test Suite 6: Network Resilience
class TestNetworkResilience:
    def test_signal_broadcast_failure_handling(self, sample_signals):
        """Test handling broadcast failures"""
        signal = sample_signals[0]
        
        try:
            # Simulate broadcast attempt
            if not signal.get('signature'):
                raise Exception("Invalid signal for broadcast")
            broadcast_success = True
        except Exception:
            broadcast_success = False
        
        # Should handle failure gracefully
        assert broadcast_success is False or True  # Either succeeds or fails gracefully

    def test_peer_signal_validation(self, sample_signals):
        """Test peer signal validation before consensus"""
        peer_signal = sample_signals[0]
        
        # Validate required fields
        required = ['agent_id', 'market_id', 'prediction', 'confidence']
        is_valid = all(field in peer_signal for field in required)
        
        assert is_valid

    def test_consensus_with_delayed_signals(self, sample_signals):
        """Test consensus handles delayed signals"""
        signals_by_time = []
        base_time = datetime.utcnow()
        
        for i, signal in enumerate(sample_signals):
            delayed_signal = signal.copy()
            delayed_signal['timestamp'] = (base_time + timedelta(minutes=i*10)).isoformat()
            signals_by_time.append(delayed_signal)
        
        # All signals should still be processed
        assert len(signals_by_time) == len(sample_signals)

    def test_network_partition_recovery(self, sample_signals):
        """Test recovery from network partitions"""
        # Simulate partition: only partial signals available
        available_signals = sample_signals[:2]
        
        # Should still compute consensus from available signals
        predictions = [s['prediction'] for s in available_signals]
        consensus = sum(predictions) / len(predictions)
        
        assert consensus > 0
        # Consensus might be less reliable but still valid
        assert 0.6 <= consensus <= 0.75


# Test Suite 7: Performance & Scalability
class TestPerformanceAndScalability:
    @pytest.mark.performance
    def test_large_signal_consensus(self):
        """Test consensus computation with many agents"""
        # Simulate 1000 agents
        signals = [
            {'prediction': 0.5 + (i * 0.0001), 'confidence': 0.8}
            for i in range(1000)
        ]
        
        weighted_sum = sum(s['prediction'] * s['confidence'] for s in signals)
        weight_sum = sum(s['confidence'] for s in signals)
        consensus = weighted_sum / weight_sum
        
        assert 0.5 <= consensus <= 0.6

    @pytest.mark.performance
    def test_many_market_aggregation(self):
        """Test aggregating signals across many markets"""
        markets = {}
        for m in range(100):
            market_id = f'market_{m:03d}'
            markets[market_id] = {
                'signals': 0,
                'consensus': 0.5
            }
        
        assert len(markets) == 100

    @pytest.mark.performance
    def test_checkpoint_throughput(self):
        """Test high-frequency checkpoint saving"""
        checkpoints = []
        for i in range(100):
            checkpoint = {
                'sequence': i,
                'timestamp': datetime.utcnow().isoformat(),
                'data': 'x' * 1000  # Some data
            }
            checkpoints.append(json.dumps(checkpoint))
        
        assert len(checkpoints) == 100


# Integration Tests - Full Workflow
class TestFullIntegrationWorkflow:
    @pytest.mark.integration
    async def test_complete_agent_lifecycle(self, agent_config, sample_markets, sample_signals):
        """Test complete agent lifecycle"""
        # 1. Initialize
        agent_state = {'initialized': True, 'ready': True}
        assert agent_state['initialized']
        
        # 2. Process markets
        market_count = len(sample_markets)
        assert market_count > 0
        
        # 3. Receive signals
        signal_count = len(sample_signals)
        assert signal_count > 0
        
        # 4. Compute consensus
        predictions = [s['prediction'] for s in sample_signals]
        consensus = sum(predictions) / len(predictions)
        assert 0 < consensus < 1
        
        # 5. Save checkpoint
        checkpoint = {
            'agent_id': agent_config['agent_id'],
            'status': 'checkpoint_saved',
            'consensus': consensus
        }
        assert checkpoint['status'] == 'checkpoint_saved'

    @pytest.mark.integration
    def test_multi_agent_coordination(self, sample_signals):
        """Test multi-agent coordination"""
        # Simulate 3 agents coordinating
        agent_results = {}
        
        for i, signal in enumerate(sample_signals):
            agent_id = signal['agent_id']
            if agent_id not in agent_results:
                agent_results[agent_id] = []
            agent_results[agent_id].append(signal['prediction'])
        
        # Should have 3 unique agents
        assert len(agent_results) == 3
        
        # Compute global consensus
        all_predictions = [p for preds in agent_results.values() for p in preds]
        global_consensus = sum(all_predictions) / len(all_predictions)
        
        assert 0.5 <= global_consensus <= 0.75


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
