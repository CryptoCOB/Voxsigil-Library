#!/usr/bin/env node
/**
 * Moltbot Launcher - Initialize and start molt agent coordination
 */

const fs = require('fs');
const path = require('path');
const { MoltSignalAggregator, MoltSignal } = require('../src/examples/molt-signal-aggregator.js');

// Configuration
const CONFIG = {
  agentId: process.argv[2] || 'moltbot-default',
  market: extractArg('--market') || 'BTC_PREDICTION',
  signalInterval: parseInt(extractArg('--signal-interval') || '3000'),
  checkpointInterval: parseInt(extractArg('--checkpoint-interval') || '30000'),
  maxAgents: parseInt(extractArg('--max-agents') || '100'),
  verbose: process.argv.includes('--verbose'),
};

function extractArg(flag) {
  const index = process.argv.indexOf(flag);
  return index > -1 ? process.argv[index + 1] : null;
}

function log(level, message) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] [${level}] ${message}`);
}

function getConfigPath() {
  const configDir = path.join(process.env.HOME || process.env.USERPROFILE || '.', '.voxsigil');
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }
  return path.join(configDir, `moltbot-${CONFIG.agentId}.json`);
}

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

  initialize() {
    const configPath = getConfigPath();
    if (fs.existsSync(configPath)) {
      log('INFO', `Loading agent configuration from ${configPath}`);
      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      this.config = config;
    } else {
      log('INFO', `Creating new agent configuration: ${this.agentId}`);
      this.config = {
        agentId: this.agentId,
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString(),
      };
      this.saveConfig();
    }
  }

  registerMarket(marketName, description = '') {
    log('INFO', `Registering market: ${marketName}`);
    try {
      this.aggregator.registerMarket({
        marketId: marketName,
        question: `${marketName} Prediction Market`,
        description: description || `Tracking ${marketName} predictions`,
        deadline: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
      });
      return true;
    } catch (error) {
      log('ERROR', `Failed to register market: ${error.message}`);
      return false;
    }
  }

  async broadcastSignal(marketName, prediction, confidence) {
    try {
      const signal = new MoltSignal({
        agentId: this.agentId,
        marketId: marketName,
        prediction: Math.min(1, Math.max(0, prediction)),
        confidence: Math.min(1, Math.max(0, confidence)),
        reasoning: `Agent ${this.agentId} prediction`,
        timestamp: new Date().toISOString(),
        signature: Math.random().toString(36).substring(2, 18),
      });

      await this.aggregator.addSignal(signal);
      this.metrics.signalsProcessed++;

      if (CONFIG.verbose) {
        log('DEBUG', `Signal: ${(signal.prediction * 100).toFixed(1)}% confidence ${(signal.confidence * 100).toFixed(1)}%`);
      }
    } catch (error) {
      if (CONFIG.verbose) {
        log('ERROR', `Signal broadcast failed: ${error.message}`);
      }
    }
  }

  updateConsensus() {
    try {
      for (const [marketId, market] of this.aggregator.markets) {
        if (market.signals.size === 0) continue;
        const consensus = market.computeConsensus();
        this.metrics.consensusUpdates++;
        if (CONFIG.verbose) {
          log('INFO', `Consensus ${marketId}: ${(consensus * 100).toFixed(1)}%`);
        }
      }
    } catch (error) {
      log('ERROR', `Consensus update failed: ${error.message}`);
    }
  }

  saveCheckpoint() {
    try {
      const configPath = getConfigPath();
      const checkpoint = {
        agentId: this.agentId,
        updatedAt: new Date().toISOString(),
        metrics: this.metrics,
        markets: Array.from(this.aggregator.markets.entries()).map(([id, market]) => ({
          marketId: id,
          question: market.question,
          signalCount: market.signals.size,
          consensus: market.computeConsensus(),
        })),
      };
      fs.writeFileSync(configPath, JSON.stringify(checkpoint, null, 2));
      this.metrics.checkpointsSaved++;
      if (CONFIG.verbose) {
        log('DEBUG', `Checkpoint saved`);
      }
    } catch (error) {
      log('ERROR', `Checkpoint save failed: ${error.message}`);
    }
  }

  saveConfig() {
    try {
      const configPath = getConfigPath();
      this.config.lastActive = new Date().toISOString();
      fs.writeFileSync(configPath, JSON.stringify(this.config, null, 2));
    } catch (error) {
      log('ERROR', `Config save failed: ${error.message}`);
    }
  }

  printStatus() {
    const uptime = Math.floor((Date.now() - this.metrics.startTime) / 1000);
    const uptimeStr = `${Math.floor(uptime / 60)}m ${uptime % 60}s`;
    const totalSignals = Array.from(this.aggregator.markets.values()).reduce((sum, m) => sum + m.signals.size, 0);
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

Markets:               ${this.aggregator.markets.size}
Total Signals:         ${totalSignals}

Config Path:           ${getConfigPath()}
    `);
  }

  async start() {
    if (this.isRunning) {
      log('WARN', 'Agent is already running');
      return;
    }

    log('INFO', `Starting moltbot agent: ${this.agentId}`);
    this.initialize();
    this.isRunning = true;

    this.registerMarket(CONFIG.market, `${CONFIG.market} prediction market`);

    this.signalIntervalId = setInterval(async () => {
      const numAgents = Math.floor(Math.random() * 10) + 1;
      for (let i = 0; i < numAgents; i++) {
        try {
          const signal = new MoltSignal({
            agentId: `agent-${i}`,
            marketId: CONFIG.market,
            prediction: Math.random(),
            confidence: 0.5 + Math.random() * 0.5,
            reasoning: `Prediction from agent-${i}`,
            timestamp: new Date().toISOString(),
            signature: Math.random().toString(36).substring(2, 18),
          });
          await this.aggregator.addSignal(signal);
          this.metrics.signalsProcessed++;
        } catch (error) {
          // silently continue
        }
      }
      this.updateConsensus();
    }, CONFIG.signalInterval);

    this.checkpointIntervalId = setInterval(() => {
      this.saveCheckpoint();
    }, CONFIG.checkpointInterval);

    log('INFO', `Moltbot agent started successfully`);
    this.printStatus();
  }

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

// Main execution
const agent = new MoltbotAgent(CONFIG.agentId);

process.on('SIGINT', () => {
  console.log('\n');
  agent.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  agent.stop();
  process.exit(0);
});

agent.start().catch(error => {
  log('ERROR', `Failed to start agent: ${error.message}`);
  process.exit(1);
});

setInterval(() => {
  if (agent.isRunning && CONFIG.verbose) {
    agent.printStatus();
  }
}, 60000);

module.exports = MoltbotAgent;
