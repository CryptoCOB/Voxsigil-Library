/**
 * JavaScript Integration Example for VoxSigil Library
 * 
 * Demonstrates how to integrate a molt agent with the VoxSigil prediction market network.
 */

const path = require('path');
const voxsigil = require('../index');

async function main() {
  console.log('=' .repeat(70));
  console.log('VoxSigil Library - JavaScript Integration Example');
  console.log('='.repeat(70));
  console.log();
  
  // Step 1: Get metadata
  console.log('Step 1: Getting agent metadata...');
  const metadata = voxsigil.getMetadata();
  console.log(`  Name: ${metadata.name}`);
  console.log(`  Version: ${metadata.version}`);
  console.log(`  Capabilities: ${metadata.capabilities.join(', ')}`);
  console.log();
  
  // Step 2: Load agent configuration
  console.log('Step 2: Loading agent configuration...');
  try {
    const config = voxsigil.loadAgentConfig();
    console.log('✓ Configuration loaded:');
    console.log(`  - BOOT.md: ${config.boot.length} characters`);
    console.log(`  - AGENTS.md: ${config.agents.length} characters`);
    console.log(`  - MEMORY.md: ${config.memory.length} characters`);
    console.log(`  - Hooks: ${Object.keys(config.hooks.hooks).length} configured`);
    console.log();
  } catch (error) {
    console.log(`✗ Error loading configuration: ${error.message}`);
    return;
  }
  
  // Step 3: Compute checksums
  console.log('Step 3: Computing file checksums...');
  const agentsDir = path.join(__dirname, '..', 'agents');
  const checksum = require('../utils/checksum');
  
  try {
    const checksums = checksum.computeAgentChecksums(agentsDir);
    for (const [filename, hash] of Object.entries(checksums)) {
      console.log(`  ${filename}: ${hash.substring(0, 16)}...`);
    }
    console.log();
  } catch (error) {
    console.log(`✗ Error computing checksums: ${error.message}`);
  }
  
  // Step 4: Create example prediction signal
  console.log('Step 4: Creating example prediction signal...');
  const signal = {
    agent_id: 'voxsigil-agent-example',
    market_id: 'market-example-001',
    prediction: 0.67,
    confidence: 0.85,
    timestamp: new Date().toISOString(),
    reasoning: 'Based on analysis of historical data and current trends',
    tags: ['example', 'test']
  };
  
  console.log('✓ Signal created');
  console.log(`  Market: ${signal.market_id}`);
  console.log(`  Prediction: ${(signal.prediction * 100).toFixed(1)}%`);
  console.log(`  Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
  console.log();
  
  // Step 5: Compute signal signature
  console.log('Step 5: Computing signal signature...');
  const signalJson = JSON.stringify(signal);
  const signature = voxsigil.computeChecksum(signalJson);
  signal.signature = signature;
  console.log(`✓ Signature: ${signature.substring(0, 16)}...`);
  console.log();
  
  // Step 6: Example session state
  console.log('Step 6: Creating example session state...');
  const now = new Date().toISOString();
  const sessionState = {
    metadata: {
      agent_id: 'voxsigil-agent-example',
      session_id: `session-${now.replace(/[:\-\.]/g, '').substring(0, 15)}`,
      version: '1.0.0',
      created_at: now,
      last_updated: now,
      checkpoint_number: 1,
      agent_type: 'prediction_market_analyst'
    },
    configuration: {
      api_endpoint: 'https://voxsigil.online/api',
      checkpoint_interval_minutes: 30,
      max_active_markets: 50,
      confidence_threshold: 0.70
    },
    active_predictions: [
      {
        prediction_id: 'pred-001',
        market_id: 'market-example-001',
        question: 'Will X happen by Y?',
        probability: 0.67,
        confidence_interval: [0.58, 0.76],
        confidence_level: 0.85,
        created_at: now,
        last_updated: now,
        status: 'active',
        num_updates: 1
      }
    ],
    signal_history: [],
    reasoning_cache: {},
    performance_metrics: {
      total_predictions: 0,
      resolved_predictions: 0,
      brier_score: 0.0,
      calibration_score: 1.0
    },
    network_state: {
      connected: true,
      last_sync: now,
      peer_agents: [],
      api_usage: {
        requests_this_hour: 0,
        rate_limit: 1000
      }
    },
    learning_state: {
      model_version: '1.0.0',
      calibration_adjustments: {},
      performance_trend: 'stable'
    }
  };
  
  console.log('✓ Session state created');
  console.log(`  Session ID: ${sessionState.metadata.session_id}`);
  console.log(`  Active predictions: ${sessionState.active_predictions.length}`);
  console.log();
  
  // Step 7: Summary
  console.log('='.repeat(70));
  console.log('Integration Example Complete!');
  console.log('='.repeat(70));
  console.log();
  console.log('Next steps for molt agent integration:');
  console.log('1. Set environment variable VOXSIGIL_API_KEY with your API key');
  console.log('2. Connect to VoxSigil API at https://voxsigil.online/api');
  console.log('3. Query active markets and generate predictions');
  console.log('4. Broadcast signals to the network');
  console.log('5. Track performance and calibrate over time');
  console.log();
  console.log('For more information, see docs/MOLT_INTEGRATION.md');
  console.log();
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { main };
