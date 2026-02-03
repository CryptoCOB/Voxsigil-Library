#!/usr/bin/env node

/**
 * Moltbot Launcher - Initialize and start molt agent coordination
 * 
 * This script launches a fully-configured moltbot agent that uses Voxsigil
 * for signal aggregation, consensus computation, and prediction market coordination.
 * 
 * Usage: node bin/moltbot-launch.js [agent-id] [--market MARKET] [--signal-interval MS]
 * 
 * Examples:
 *   1. Start default: node bin/moltbot-launch.js
 *   2. With market:   node bin/moltbot-launch.js prophet --market BTC
 *   3. Custom interval: node bin/moltbot-launch.js analyst --signal-interval 5000
 */

const fs = require('fs');
const path = require('path');
const { MoltSignalAggregator } = require('../src/examples/molt-signal-aggregator.js');

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  agentId: process.argv[2] || 'moltbot-default',
  market: extractArg('--market') || 'BTC',
  signalInterval: parseInt(extractArg('--signal-interval') || '3000'),
  checkpointInterval: parseInt(extractArg('--checkpoint-interval') || '30000'),
  consensusThreshold: 0.5,
  maxAgents: parseInt(extractArg('--max-agents') || '100'),
  verbose: process.argv.includes('--verbose'),
};

// ============================================================================
// Utilities
// ============================================================================

function extractArg(flag) {
  const index = process.argv.indexOf(flag);
  return index > -1 ? process.argv[index + 1] : null;
}

function log(level, message) {
  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}] [${level}]`;
  console.log(`${prefix} ${message}`);
}

function getConfigPath() {
  const configDir = path.join(process.env.HOME || process.env.USERPROFILE || '.', '.voxsigil');
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }
  return path.join(configDir, `moltbot-${CONFIG.agentId}.json`);
}

// ============================================================================
// Moltbot Agent Class
// ============================================================================

class MoltbotAgent {
  constructor(agentId) {
    this.agentId = agentId;
    this.aggregator = new MoltSignalAggregator();
    this.isRunning = false;
    this.signalIntervalId = null;
    this.checkpointIntervalId = null;
    this.metrics = {
      signalsProcessed: 0,
      consensusUpdates: 0,
      checkpointsSaved: 0,
      startTime: Date.now(),
    };
  }

  /**
   * Initialize agent from config file or create new
   */
  initialize() {
    const configPath = getConfigPath();
    
    if (fs.existsSync(configPath)) {
      log('INFO', `Loading agent configuration from ${configPath}`);
      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      this.config = config;
      this.aggregator = Object.assign(new MoltSignalAggregator(), config.aggregator);
    } else {
      log('INFO', `Creating new agent configuration: ${CONFIG.agentId}`);
      this.config = {
        agentId: CONFIG.agentId,
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString(),
      };
      this.saveConfig();
    }
  }

  /**
   * Register a market for tracking
   */
  registerMarket(marketName, description = '') {
    log('INFO', `Registering market: ${marketName}`);
    try {
      this.aggregator.registerMarket(marketName, {
        deadline: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        description,
      });
      return true;
    } catch (error) {
      log('ERROR', `Failed to register market: ${error.message}`);
      return false;
    }
  }

  /**
   * Generate and broadcast a prediction signal
   */
  broadcastSignal(marketName, prediction, confidence) {
    try {
      const signal = {
        prediction: Math.min(1, Math.max(0, prediction)),
        confidence: Math.min(1, Math.max(0, confidence)),
        timestamp: Date.now(),
        agentId: this.agentId,
      };

      this.aggregator.addAgentSignal(marketName, this.agentId, signal);
      this.metrics.signalsProcessed++;

      if (CONFIG.verbose) {
        log('DEBUG', `Signal broadcast: ${marketName} -> ${(signal.prediction * 100).toFixed(1)}% (confidence: ${(signal.confidence * 100).toFixed(1)}%)`);
      }
    } catch (error) {
      log('ERROR', `Signal broadcast failed: ${error.message}`);
    }
  }

  /**
   * Update consensus and get market insights
   */
  updateConsensus() {
    try {
      const markets = this.aggregator.markets;
      
      for (const [marketName, market] of Object.entries(markets)) {
        if (market.signals.size === 0) continue;

        const consensus = this.aggregator.computeConsensus(marketName);
        this.metrics.consensusUpdates++;

        if (CONFIG.verbose) {
          log('INFO', `Consensus updated: ${marketName} -> ${(consensus * 100).toFixed(1)}%`);
        }
      }
    } catch (error) {
      log('ERROR', `Consensus update failed: ${error.message}`);
    }
  }

  /**
   * Save checkpoint for recovery
   */
  saveCheckpoint() {
    try {
      const configPath = getConfigPath();
      const checkpoint = {
        agentId: this.agentId,
        updatedAt: new Date().toISOString(),
        metrics: this.metrics,
        aggregator: {
          markets: Array.from(this.aggregator.markets.entries()).map(([name, market]) => ({
            name,
            signals: Array.from(market.signals.entries()),
            consensus: market.consensus,
          })),
        },
      };

      fs.writeFileSync(configPath, JSON.stringify(checkpoint, null, 2));
      this.metrics.checkpointsSaved++;

      if (CONFIG.verbose) {
        log('DEBUG', `Checkpoint saved: ${configPath}`);
      }
    } catch (error) {
      log('ERROR', `Checkpoint save failed: ${error.message}`);
    }
  }

  /**
   * Save configuration
   */
  saveConfig() {
    try {
      const configPath = getConfigPath();
      this.config.lastActive = new Date().toISOString();
      fs.writeFileSync(configPath, JSON.stringify(this.config, null, 2));
    } catch (error) {
      log('ERROR', `Config save failed: ${error.message}`);
    }
  }

  /**
   * Print agent status
   */
  printStatus() {
    const uptime = Math.floor((Date.now() - this.metrics.startTime) / 1000);
    const uptimeStr = `${Math.floor(uptime / 60)}m ${uptime % 60}s`;
    
    console.log(`
╔════════════════════════════════════════════════════════════╗
║                    MOLTBOT AGENT STATUS                    ║
╚════════════════════════════════════════════════════════════╝

Agent ID:              ${this.agentId}
Status:                ${this.isRunning ? '🟢 RUNNING' : '🔴 STOPPED'}
Uptime:                ${uptimeStr}

Metrics:
  Signals Processed:   ${this.metrics.signalsProcessed}
  Consensus Updates:   ${this.metrics.consensusUpdates}
  Checkpoints Saved:   ${this.metrics.checkpointsSaved}

Markets Tracked:       ${this.aggregator.markets.size}
Total Signals:         ${Array.from(this.aggregator.markets.values()).reduce((sum, m) => sum + m.signals.size, 0)}

Configuration:
  Signal Interval:     ${CONFIG.signalInterval}ms
  Checkpoint Interval: ${CONFIG.checkpointInterval}ms
  Consensus Threshold: ${CONFIG.consensusThreshold}
  Max Agents:          ${CONFIG.maxAgents}

Press Ctrl+C to stop
    `);
  }

  /**
   * Start the agent
   */
  start() {
    if (this.isRunning) {
      log('WARN', 'Agent is already running');
      return;
    }

    log('INFO', `Starting moltbot agent: ${this.agentId}`);
    this.initialize();
    this.isRunning = true;

    // Register initial market(s)
    this.registerMarket(CONFIG.market, `${CONFIG.market} prediction market`);

    // Simulate incoming signals
    this.signalIntervalId = setInterval(() => {
      // Generate simulated agents' predictions
      const numSimulatedAgents = Math.floor(Math.random() * 10) + 1;
      
      for (let i = 0; i < numSimulatedAgents; i++) {
        const simAgentId = `sim-agent-${i}`;
        const prediction = Math.random();
        const confidence = 0.5 + Math.random() * 0.5; // 0.5 - 1.0
        
        this.broadcastSignal(CONFIG.market, prediction, confidence);
      }

      // Update consensus
      this.updateConsensus();
    }, CONFIG.signalInterval);

    // Save checkpoints periodically
    this.checkpointIntervalId = setInterval(() => {
      this.saveCheckpoint();
    }, CONFIG.checkpointInterval);

    log('INFO', `Moltbot agent started successfully`);
    this.printStatus();
  }

  /**
   * Stop the agent
   */
  stop() {
    if (!this.isRunning) {
      log('WARN', 'Agent is not running');
      return;
    }

    log('INFO', `Stopping moltbot agent: ${this.agentId}`);
    
    if (this.signalIntervalId) clearInterval(this.signalIntervalId);
    if (this.checkpointIntervalId) clearInterval(this.checkpointIntervalId);
    
    this.isRunning = false;
    this.saveCheckpoint();
    
    log('INFO', `Moltbot agent stopped`);
    this.printStatus();
  }
}

// ============================================================================
// Main Execution
// ============================================================================

const agent = new MoltbotAgent(CONFIG.agentId);

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n');
  agent.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  agent.stop();
  process.exit(0);
});

// Start the agent
try {
  agent.start();
} catch (error) {
  log('ERROR', `Failed to start agent: ${error.message}`);
  process.exit(1);
}

// Print status periodically
setInterval(() => {
  if (agent.isRunning && CONFIG.verbose) {
    agent.printStatus();
  }
}, 60000); // Every 60 seconds

module.exports = MoltbotAgent;
