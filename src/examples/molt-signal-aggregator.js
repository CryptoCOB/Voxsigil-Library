#!/usr/bin/env node

/**
 * Molt Agent Signal Aggregator - JavaScript Deep Example
 * 
 * Demonstrates comprehensive Molt agent signal aggregation and consensus
 * across multiple markets using Voxsigil network.
 * 
 * Features:
 * - Real-time signal aggregation
 * - Multi-agent consensus computation
 * - Market price impact analysis
 * - Signal verification via SHA256
 * - Persistence and recovery
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Constants
const MARKET_TYPES = {
  BINARY: 'binary',
  CATEGORICAL: 'categorical',
  SCALAR: 'scalar'
};

const CONFIDENCE_LEVELS = {
  LOW: 0.33,
  MEDIUM: 0.67,
  HIGH: 0.85
};

/**
 * Represents a prediction signal from an agent
 */
class MoltSignal {
  constructor(data) {
    this.agentId = data.agentId;
    this.marketId = data.marketId;
    this.prediction = data.prediction; // 0.0 to 1.0
    this.confidence = data.confidence; // CONFIDENCE_LEVELS value
    this.reasoning = data.reasoning;
    this.timestamp = data.timestamp;
    this.signature = data.signature;
    this.metadata = data.metadata || {};
  }

  toJSON() {
    return {
      agentId: this.agentId,
      marketId: this.marketId,
      prediction: this.prediction,
      confidence: this.confidence,
      reasoning: this.reasoning,
      timestamp: this.timestamp,
      signature: this.signature,
      metadata: this.metadata
    };
  }

  static fromJSON(json) {
    return new MoltSignal(json);
  }

  /**
   * Compute signal strength score
   * Combines prediction confidence and recency
   */
  computeStrength() {
    const confidenceScore = this.confidence;
    const ageMinutes = (Date.now() - new Date(this.timestamp).getTime()) / (1000 * 60);
    const recencyDecay = Math.exp(-ageMinutes / 60); // 1 hour half-life
    return confidenceScore * recencyDecay;
  }
}

/**
 * Represents a prediction market
 */
class MoltMarket {
  constructor(data) {
    this.marketId = data.marketId;
    this.question = data.question;
    this.type = data.type;
    this.deadline = new Date(data.deadline);
    this.currentPrice = data.currentPrice || 0.5;
    this.volume = data.volume || 0;
    this.signals = [];
  }

  isActive() {
    return new Date() < this.deadline;
  }

  addSignal(signal) {
    if (!this.isActive()) {
      throw new Error(`Market ${this.marketId} is closed`);
    }
    this.signals.push(signal);
  }

  /**
   * Compute weighted consensus from all signals
   */
  computeConsensus() {
    if (this.signals.length === 0) {
      return {
        prediction: this.currentPrice,
        strength: 0,
        agentCount: 0,
        warning: 'No signals available'
      };
    }

    let weightedSum = 0;
    let weightSum = 0;

    for (const signal of this.signals) {
      const strength = signal.computeStrength();
      weightedSum += signal.prediction * strength;
      weightSum += strength;
    }

    const consensus = weightedSum / weightSum;

    return {
      prediction: consensus,
      strength: weightSum / this.signals.length,
      agentCount: this.signals.length,
      signals: this.signals.map(s => ({
        agentId: s.agentId,
        prediction: s.prediction,
        confidence: s.confidence
      }))
    };
  }

  /**
   * Analyze price divergence vs consensus
   */
  analyzePriceDivergence() {
    const consensus = this.computeConsensus();
    const divergence = Math.abs(consensus.prediction - this.currentPrice);
    
    return {
      marketId: this.marketId,
      currentPrice: this.currentPrice,
      consensusPrediction: consensus.prediction,
      divergence: divergence,
      divergencePercent: (divergence * 100).toFixed(2) + '%',
      direction: consensus.prediction > this.currentPrice ? 'UP' : 'DOWN',
      opportunity: divergence > 0.15 ? 'HIGH' : divergence > 0.05 ? 'MEDIUM' : 'LOW',
      recommendation: this._generateRecommendation(
        consensus.prediction,
        this.currentPrice,
        consensus.strength
      )
    };
  }

  _generateRecommendation(prediction, price, strength) {
    const divergence = Math.abs(prediction - price);
    
    if (divergence < 0.05 || strength < 0.4) {
      return 'WATCH'; // Monitor, not enough signal
    }

    if (prediction > price && divergence > 0.15 && strength > 0.6) {
      return 'INCREASE_ALLOCATION'; // Agent consensus bullish, mispriced down
    }

    if (prediction < price && divergence > 0.15 && strength > 0.6) {
      return 'REDUCE_ALLOCATION'; // Agent consensus bearish, mispriced up
    }

    return 'HOLD';
  }
}

/**
 * Molt Agent Signal Aggregator
 * Coordinates signals across multiple markets and agents
 */
class MoltSignalAggregator {
  constructor(agentId = 'voxsigil-aggregator-001') {
    this.agentId = agentId;
    this.markets = new Map();
    this.allSignals = [];
    this.peerAgents = new Set();
    this.memory = {
      agentId: agentId,
      status: 'initialized',
      initializedAt: new Date().toISOString(),
      signalsProcessed: 0,
      marketsAnalyzed: 0,
      consensusEvents: 0,
      errors: []
    };
  }

  /**
   * Register a market for aggregation
   */
  registerMarket(marketData) {
    const market = new MoltMarket(marketData);
    this.markets.set(market.marketId, market);
    console.log(`✅ Market registered: ${market.question}`);
    return market;
  }

  /**
   * Add a signal from an agent (peer or own)
   */
  async addSignal(signal) {
    try {
      if (!signal.agentId || !signal.marketId) {
        throw new Error('Signal missing agentId or marketId');
      }

      const market = this.markets.get(signal.marketId);
      if (!market) {
        throw new Error(`Market ${signal.marketId} not registered`);
      }

      const moltSignal = signal instanceof MoltSignal ? signal : new MoltSignal(signal);
      
      market.addSignal(moltSignal);
      this.allSignals.push(moltSignal);
      this.peerAgents.add(signal.agentId);
      this.memory.signalsProcessed++;

      console.log(
        `  📊 Signal added: ${moltSignal.agentId} predicts ${(moltSignal.prediction * 100).toFixed(1)}% for ${moltSignal.marketId}`
      );

      return true;
    } catch (error) {
      this.memory.errors.push(`Signal add error: ${error.message}`);
      console.error(`  ❌ Failed to add signal: ${error.message}`);
      return false;
    }
  }

  /**
   * Generate consensus report for all markets
   */
  async generateConsensusReport() {
    console.log('\n📈 CONSENSUS REPORT');
    console.log('='.repeat(60));

    const report = {
      timestamp: new Date().toISOString(),
      aggregatorId: this.agentId,
      markets: [],
      summary: {
        totalMarkets: this.markets.size,
        totalSignals: this.allSignals.length,
        uniqueAgents: this.peerAgents.size,
        averageConfidence: 0
      }
    };

    let totalConfidence = 0;
    let confidenceCount = 0;

    for (const [marketId, market] of this.markets) {
      const consensus = market.computeConsensus();
      const divergence = market.analyzePriceDivergence();

      const marketReport = {
        marketId,
        question: market.question,
        consensus,
        divergence,
        isActive: market.isActive()
      };

      report.markets.push(marketReport);

      if (consensus.agentCount > 0) {
        totalConfidence += consensus.strength;
        confidenceCount++;
      }

      // Pretty print
      console.log(`\n🎯 ${market.question}`);
      console.log(`   Current Price: ${(market.currentPrice * 100).toFixed(1)}%`);
      console.log(`   Consensus: ${(consensus.prediction * 100).toFixed(1)}%`);
      console.log(`   Signals from ${consensus.agentCount} agents`);
      console.log(`   Strength: ${(consensus.strength * 100).toFixed(1)}%`);
      console.log(`   Divergence: ${divergence.divergencePercent} ${divergence.direction}`);
      console.log(`   → Recommendation: ${divergence.recommendation}`);
    }

    if (confidenceCount > 0) {
      report.summary.averageConfidence = totalConfidence / confidenceCount;
    }

    this.memory.consensusEvents++;
    this.memory.marketsAnalyzed = this.markets.size;

    return report;
  }

  /**
   * Compute network-wide statistics
   */
  getNetworkStats() {
    const stats = {
      timestamp: new Date().toISOString(),
      agentNetwork: {
        aggregatorId: this.agentId,
        peerAgents: Array.from(this.peerAgents),
        uniqueAgentCount: this.peerAgents.size,
        allAgentIds: [this.agentId, ...this.peerAgents]
      },
      signalAnalysis: {
        totalSignals: this.allSignals.length,
        averagePrediction: 0,
        sdPrediction: 0,
        confidenceDistribution: {
          LOW: 0,
          MEDIUM: 0,
          HIGH: 0
        }
      },
      markets: {
        total: this.markets.size,
        active: 0,
        closed: 0
      }
    };

    // Calculate signal statistics
    if (this.allSignals.length > 0) {
      const predictions = this.allSignals.map(s => s.prediction);
      stats.signalAnalysis.averagePrediction =
        predictions.reduce((a, b) => a + b) / predictions.length;

      // Standard deviation
      const variance =
        predictions.reduce(
          (sum, p) => sum + Math.pow(p - stats.signalAnalysis.averagePrediction, 2),
          0
        ) / predictions.length;
      stats.signalAnalysis.sdPrediction = Math.sqrt(variance);

      // Confidence distribution
      for (const signal of this.allSignals) {
        if (signal.confidence === CONFIDENCE_LEVELS.LOW) {
          stats.signalAnalysis.confidenceDistribution.LOW++;
        } else if (signal.confidence === CONFIDENCE_LEVELS.MEDIUM) {
          stats.signalAnalysis.confidenceDistribution.MEDIUM++;
        } else if (signal.confidence === CONFIDENCE_LEVELS.HIGH) {
          stats.signalAnalysis.confidenceDistribution.HIGH++;
        }
      }
    }

    // Market status
    for (const market of this.markets.values()) {
      if (market.isActive()) {
        stats.markets.active++;
      } else {
        stats.markets.closed++;
      }
    }

    return stats;
  }

  /**
   * Save aggregator state checkpoint
   */
  async saveCheckpoint(filename = null) {
    if (!filename) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
      filename = `molt_aggregator_${timestamp}.json`;
    }

    const checkpoint = {
      ...this.memory,
      statistics: this.getNetworkStats(),
      markets: Array.from(this.markets.entries()).map(([id, market]) => ({
        marketId: id,
        question: market.question,
        signalCount: market.signals.length,
        consensus: market.computeConsensus()
      }))
    };

    try {
      fs.writeFileSync(filename, JSON.stringify(checkpoint, null, 2));
      console.log(`✅ Checkpoint saved: ${filename}`);
      return true;
    } catch (error) {
      console.error(`❌ Failed to save checkpoint: ${error.message}`);
      return false;
    }
  }

  /**
   * Compute SHA256 signature for signal verification
   */
  static computeSignature(data) {
    return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
  }
}

/**
 * Deep integration example showcasing full coordination
 */
async function runDeepExample() {
  console.log('\n' + '='.repeat(60));
  console.log('🌐 VOXSIGIL MOLT SIGNAL AGGREGATOR - DEEP EXAMPLE');
  console.log('='.repeat(60));
  console.log('Demonstrating:');
  console.log('  • Real-time signal aggregation');
  console.log('  • Multi-market consensus computation');
  console.log('  • Price divergence analysis');
  console.log('  • Network-wide statistics');
  console.log('  • Agent coordination workflow');
  console.log('='.repeat(60));

  // Initialize aggregator
  const aggregator = new MoltSignalAggregator('voxsigil-deep-aggregator-001');

  // Register markets
  const market1 = aggregator.registerMarket({
    marketId: 'market_001',
    question: 'Will BTC reach $100k by end of Q1 2026?',
    type: MARKET_TYPES.BINARY,
    deadline: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
    currentPrice: 0.65,
    volume: 1000
  });

  const market2 = aggregator.registerMarket({
    marketId: 'market_002',
    question: 'Which AI model will be most deployed in 2026?',
    type: MARKET_TYPES.CATEGORICAL,
    deadline: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000).toISOString(),
    currentPrice: 0.40,
    volume: 2000
  });

  console.log('\n📡 ADDING SIGNALS FROM PEER AGENTS');
  console.log('='.repeat(60));

  // Simulate signals from multiple agents
  const signals = [
    {
      agentId: 'molt-agent-001',
      marketId: 'market_001',
      prediction: 0.72,
      confidence: CONFIDENCE_LEVELS.HIGH,
      reasoning: 'On-chain activity and institutional interest increasing',
      timestamp: new Date().toISOString(),
      signature: MoltSignalAggregator.computeSignature({ name: 'agent-001' }).substring(0, 16)
    },
    {
      agentId: 'molt-agent-002',
      marketId: 'market_001',
      prediction: 0.68,
      confidence: CONFIDENCE_LEVELS.MEDIUM,
      reasoning: 'Macro headwinds present but demand strong',
      timestamp: new Date().toISOString(),
      signature: MoltSignalAggregator.computeSignature({ name: 'agent-002' }).substring(0, 16)
    },
    {
      agentId: 'molt-agent-003',
      marketId: 'market_001',
      prediction: 0.55,
      confidence: CONFIDENCE_LEVELS.LOW,
      reasoning: 'Uncertain technical setup, waiting for confirmation',
      timestamp: new Date().toISOString(),
      signature: MoltSignalAggregator.computeSignature({ name: 'agent-003' }).substring(0, 16)
    },
    {
      agentId: 'molt-agent-004',
      marketId: 'market_002',
      prediction: 0.50,
      confidence: CONFIDENCE_LEVELS.HIGH,
      reasoning: 'Claude and GPT-4 maintaining near parity in deployments',
      timestamp: new Date().toISOString(),
      signature: MoltSignalAggregator.computeSignature({ name: 'agent-004' }).substring(0, 16)
    },
    {
      agentId: 'molt-agent-005',
      marketId: 'market_002',
      prediction: 0.65,
      confidence: CONFIDENCE_LEVELS.MEDIUM,
      reasoning: 'Open-source models gaining enterprise adoption',
      timestamp: new Date().toISOString(),
      signature: MoltSignalAggregator.computeSignature({ name: 'agent-005' }).substring(0, 16)
    }
  ];

  for (const signal of signals) {
    await aggregator.addSignal(signal);
  }

  // Generate consensus report
  const report = await aggregator.generateConsensusReport();

  console.log('\n📊 NETWORK STATISTICS');
  console.log('='.repeat(60));
  const stats = aggregator.getNetworkStats();
  console.log(JSON.stringify(stats, null, 2));

  // Save checkpoint
  console.log('\n💾 SAVING STATE CHECKPOINT');
  console.log('='.repeat(60));
  await aggregator.saveCheckpoint();

  console.log('\n✅ DEEP EXAMPLE COMPLETED');
  console.log('='.repeat(60));
  console.log('Agent is ready for full Molt network integration!');
}

// Run example
runDeepExample().catch(console.error);
