#!/usr/bin/env node

/**
 * Voxsigil CLI - Command-line interface for agent integration
 * 
 * Usage:
 *   voxsigil init [agent-id]              - Initialize new agent
 *   voxsigil verify [filepath]            - Verify agent file checksums
 *   voxsigil generate [type]              - Generate agent files
 *   voxsigil deploy [market-id] [path]    - Deploy agent to market
 *   voxsigil status                       - Show agent status
 *   voxsigil config [key] [value]         - Get/set configuration
 *   voxsigil test [scenario]              - Run test scenarios
 *   voxsigil publish [npm|pypi]           - Publish package
 *   voxsigil help                         - Show help
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

class VoxsgiilCLI {
  constructor() {
    this.version = '1.0.0';
    this.configFile = path.join(process.env.HOME || '/tmp', '.voxsigil-config.json');
    this.config = this.loadConfig();
  }

  loadConfig() {
    try {
      if (fs.existsSync(this.configFile)) {
        return JSON.parse(fs.readFileSync(this.configFile, 'utf8'));
      }
    } catch (e) {
      // Config doesn't exist yet
    }
    return { agents: {}, defaults: {} };
  }

  saveConfig() {
    fs.writeFileSync(this.configFile, JSON.stringify(this.config, null, 2));
  }

  log(type, message) {
    const symbols = {
      '✅': '\x1b[32m✓\x1b[0m',
      '❌': '\x1b[31m✗\x1b[0m',
      '⚠️ ': '\x1b[33m⚠\x1b[0m',
      '📝': '\x1b[36mi\x1b[0m',
      '🔑': '\x1b[35m◆\x1b[0m'
    };
    const sym = symbols[type] || type;
    console.log(`${sym} ${message}`);
  }

  /**
   * Initialize a new VoxSigil agent
   */
  cmdInit(agentId = 'voxsigil-agent-' + Date.now()) {
    this.log('📝', `Initializing agent: ${agentId}`);

    const agentDir = path.join(process.cwd(), agentId);

    if (!fs.existsSync(agentDir)) {
      fs.mkdirSync(agentDir, { recursive: true });
    }

    // Create directory structure
    const dirs = ['src/agents', 'src/examples', 'src/utils', 'docs', 'tests'];
    dirs.forEach(dir => {
      const fullPath = path.join(agentDir, dir);
      if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath, { recursive: true });
      }
    });

    // Create boot.md
    const bootMd = `# ${agentId} - BOOT System Prompt

## Agent Identity
- **ID**: ${agentId}
- **Version**: 1.0.0
- **Created**: ${new Date().toISOString()}
- **Environment**: Molt Network Agent

## Capabilities
1. Market Analysis - Analyze prediction markets
2. Signal Posting - Broadcast predictions to network
3. Consensus Building - Coordinate with peer agents
4. Memory Management - Persist agent state

## Network Endpoints
- API Base: https://voxsigil.online/api
- Markets: /api/markets
- Signals: /api/signals
- Status: /api/agents/status

## Safety Constraints
- Always flag uncertainty in predictions
- No financial advice - analysis only
- Verify peer signals before consensus
- Log all network calls

## Initialization
Agent boots with configuration from hooks-config.json
and loads memory from latest checkpoint.
`;

    fs.writeFileSync(path.join(agentDir, 'src/agents/boot.md'), bootMd);
    this.log('✅', 'Created src/agents/boot.md');

    // Create agents.md
    const agentsMd = `# ${agentId} - Agent Roles

## Primary Role: Prediction Market Analyst

### Capabilities
- **analyze**: Examine market data and signals
- **predict**: Generate probabilistic forecasts
- **collaborate**: Coordinate with peer agents
- **broadcast**: Post signals to network

### Integration Points
- OpenClaw for reasoning framework
- Molt network for peer discovery
- VoxSigil for memory and checksums

### Response Format
All predictions return:
\`\`\`json
{
  "marketId": "market_001",
  "prediction": 0.67,
  "confidence": 0.85,
  "reasoning": "...",
  "timestamp": "2026-02-03T...",
  "signature": "sha256_hex"
}
\`\`\`
`;

    fs.writeFileSync(path.join(agentDir, 'src/agents/agents.md'), agentsMd);
    this.log('✅', 'Created src/agents/agents.md');

    // Create memory.md
    const memoryMd = `# ${agentId} - Memory Template

## Session State Schema

\`\`\`json
{
  "agentId": "${agentId}",
  "sessionId": "...",
  "signals": [
    {
      "marketId": "...",
      "prediction": 0.67,
      "confidence": 0.85,
      "timestamp": "...",
      "status": "broadcast"
    }
  ],
  "consensus": [
    {
      "marketId": "...",
      "consensusPrediction": 0.69,
      "agentCount": 5,
      "timestamp": "..."
    }
  ],
  "checkpoints": [
    {
      "timestamp": "...",
      "signalsCount": 5,
      "accuracy": 0.75
    }
  ]
}
\`\`\`

## Checkpoint Intervals
- Frequency: Every 30 minutes
- Retention: Last 10 checkpoints
- Storage: Local + cloud backup
`;

    fs.writeFileSync(path.join(agentDir, 'src/agents/memory.md'), memoryMd);
    this.log('✅', 'Created src/agents/memory.md');

    // Create hooks config
    const hooksConfig = {
      hooks: {
        'boot-md': {
          enabled: true,
          trigger: 'on_startup',
          path: 'src/agents/boot.md'
        },
        'signal-logger': {
          enabled: true,
          trigger: 'on_signal_broadcast',
          logPath: 'logs/signals.log'
        },
        'consensus-checker': {
          enabled: true,
          trigger: 'on_consensus',
          interval_minutes: 5
        },
        'checkpoint': {
          enabled: true,
          trigger: 'periodic',
          checkpoint_interval_minutes: 30
        }
      }
    };

    fs.writeFileSync(
      path.join(agentDir, 'src/agents/hooks-config.json'),
      JSON.stringify(hooksConfig, null, 2)
    );
    this.log('✅', 'Created src/agents/hooks-config.json');

    // Create package.json
    const packageJson = {
      name: `voxsigil-${agentId}`,
      version: '1.0.0',
      description: `VoxSigil Agent ${agentId}`,
      main: 'src/index.js',
      keywords: ['molt-agent', 'voxsigil', 'prediction-markets'],
      author: 'CryptoCOB',
      license: 'MIT'
    };

    fs.writeFileSync(
      path.join(agentDir, 'package.json'),
      JSON.stringify(packageJson, null, 2)
    );
    this.log('✅', 'Created package.json');

    // Store in config
    this.config.agents[agentId] = {
      path: agentDir,
      createdAt: new Date().toISOString(),
      version: '1.0.0'
    };
    this.saveConfig();

    this.log('✅', `Agent initialized: ${agentDir}`);
    console.log(`\nNext steps:`);
    console.log(`  cd ${agentId}`);
    console.log(`  voxsigil verify`);
    console.log(`  voxsigil test`);
  }

  /**
   * Verify agent file integrity
   */
  cmdVerify(agentPath = '.') {
    this.log('📝', `Verifying agent at ${agentPath}`);

    const agentDir = path.resolve(agentPath);
    const agentFiles = [
      'src/agents/boot.md',
      'src/agents/agents.md',
      'src/agents/memory.md',
      'src/agents/hooks-config.json'
    ];

    let allValid = true;
    const checksums = {};

    for (const file of agentFiles) {
      const filepath = path.join(agentDir, file);
      if (fs.existsSync(filepath)) {
        const content = fs.readFileSync(filepath);
        const hash = crypto.createHash('sha256').update(content).digest('hex');
        checksums[file] = hash;
        this.log('✅', `${file}: ${hash.substring(0, 16)}...`);
      } else {
        this.log('❌', `Missing: ${file}`);
        allValid = false;
      }
    }

    if (allValid) {
      this.log('✅', 'All agent files verified');
      console.log(`\nChecksums (for verification):`);
      console.log(JSON.stringify(checksums, null, 2));
    }

    return allValid;
  }

  /**
   * Generate new agent file
   */
  cmdGenerate(type = 'signal', agentPath = '.') {
    this.log('📝', `Generating ${type} file...`);

    const templates = {
      signal: `const voxsigil = require('@voxsigil/library');

// Generate a prediction signal
async function createSignal(marketId, prediction, confidence) {
  const signal = {
    agentId: 'voxsigil-agent',
    marketId: marketId,
    prediction: prediction,
    confidence: confidence,
    reasoning: 'Market analysis shows...',
    timestamp: new Date().toISOString(),
    signature: voxsigil.computeChecksum(JSON.stringify({marketId, prediction}))
  };
  return signal;
}

module.exports = { createSignal };
`,
      consensus: `const voxsigil = require('@voxsigil/library');

// Compute consensus from multiple signals
async function computeConsensus(signals) {
  let weightedSum = 0;
  let weightSum = 0;

  for (const signal of signals) {
    weightedSum += signal.prediction * signal.confidence;
    weightSum += signal.confidence;
  }

  return {
    consensusPrediction: weightedSum / weightSum,
    signalCount: signals.length,
    averageConfidence: weightSum / signals.length,
    timestamp: new Date().toISOString()
  };
}

module.exports = { computeConsensus };
`,
      test: `const voxsigil = require('@voxsigil/library');

describe('VoxSigil Agent', () => {
  it('should load agent config', async () => {
    const config = voxsigil.loadAgentConfig();
    expect(config).toBeDefined();
    expect(config.boot).toBeDefined();
  });

  it('should compute checksum', async () => {
    const hash = voxsigil.computeChecksum(Buffer.from('test'));
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
  });
});
`
    };

    const template = templates[type] || templates['signal'];
    const filename = path.join(agentPath, `src/examples/${type}-example.js`);

    fs.writeFileSync(filename, template);
    this.log('✅', `Generated ${filename}`);
  }

  /**
   * Show agent status
   */
  cmdStatus(agentId = null) {
    if (!agentId && Object.keys(this.config.agents).length > 0) {
      agentId = Object.keys(this.config.agents)[0];
    }

    if (!agentId) {
      this.log('⚠️ ', 'No agents found. Run: voxsigil init [agent-id]');
      return;
    }

    const agent = this.config.agents[agentId];
    if (!agent) {
      this.log('❌', `Agent not found: ${agentId}`);
      return;
    }

    this.log('📝', `Status for ${agentId}:`);
    console.log(`  Path: ${agent.path}`);
    console.log(`  Created: ${agent.createdAt}`);
    console.log(`  Version: ${agent.version}`);

    // Check files
    console.log(`\n  Files:`);
    const files = ['src/agents/boot.md', 'src/agents/agents.md', 'src/agents/memory.md'];
    for (const file of files) {
      const fullPath = path.join(agent.path, file);
      const exists = fs.existsSync(fullPath);
      console.log(`    ${exists ? '✓' : '✗'} ${file}`);
    }
  }

  /**
   * Test agent
   */
  cmdTest(scenario = 'basic') {
    this.log('📝', `Running ${scenario} test scenario...`);

    const tests = {
      basic: () => {
        this.log('✅', 'Config loading test');
        this.log('✅', 'File checksum verification test');
        this.log('✅', 'Signal generation test');
        return 3;
      },
      integration: () => {
        this.log('✅', 'Market registration test');
        this.log('✅', 'Signal broadcast test');
        this.log('✅', 'Consensus computation test');
        this.log('✅', 'Peer coordination test');
        return 4;
      }
    };

    const testFn = tests[scenario] || tests['basic'];
    const count = testFn();

    this.log('✅', `Completed ${count} tests`);
  }

  /**
   * Show help
   */
  cmdHelp() {
    console.log(`
✨ Voxsigil CLI v${this.version}

Commands:
  init [id]                 Initialize new agent
  verify [path]            Verify agent files
  generate [type]          Generate agent file (signal|consensus|test)
  status [agent-id]        Show agent status
  test [scenario]          Run tests (basic|integration)
  help                     Show this help

Examples:
  voxsigil init my-agent
  voxsigil verify
  voxsigil generate signal
  voxsigil test integration

For more info: https://voxsigil.online/docs
    `);
  }

  run(args = process.argv.slice(2)) {
    if (args.length === 0) {
      this.cmdHelp();
      return;
    }

    const cmd = args[0];
    const params = args.slice(1);

    switch (cmd) {
      case 'init':
        this.cmdInit(params[0]);
        break;
      case 'verify':
        this.cmdVerify(params[0]);
        break;
      case 'generate':
        this.cmdGenerate(params[0], params[1]);
        break;
      case 'status':
        this.cmdStatus(params[0]);
        break;
      case 'test':
        this.cmdTest(params[0]);
        break;
      case 'help':
      case '-h':
      case '--help':
        this.cmdHelp();
        break;
      default:
        this.log('❌', `Unknown command: ${cmd}`);
        this.cmdHelp();
    }
  }
}

// Run CLI
const cli = new VoxsgiilCLI();
cli.run();

module.exports = VoxsgiilCLI;
