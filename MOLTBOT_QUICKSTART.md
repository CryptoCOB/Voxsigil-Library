# Moltbot Quickstart Guide

Get your moltbot agent running and coordinating predictions on demand.

## What is Moltbot?

Moltbot is a **lightweight agent launcher** that turns Voxsigil's prediction capabilities into a running, self-managing service. It:

- ✅ Tracks prediction markets (BTC, AI models, geopolitics, etc.)
- ✅ Aggregates signals from distributed agents
- ✅ Computes consensus predictions automatically
- ✅ Saves checkpoints for recovery
- ✅ Broadcasts results back to OpenClaw

---

## Quick Start (60 seconds)

### 1. Install Voxsigil
```bash
npm install @voxsigil/library
# or
pip install voxsigil-library
```

### 2. Start Default Moltbot
```bash
node bin/moltbot-launch.js
```

**Output:**
```
[2026-02-03T10:30:00.000Z] [INFO] Starting moltbot agent: moltbot-default
[2026-02-03T10:30:00.100Z] [INFO] Registering market: BTC
[2026-02-03T10:30:00.200Z] [INFO] Moltbot agent started successfully

╔════════════════════════════════════════════════════════════╗
║                    MOLTBOT AGENT STATUS                    ║
╚════════════════════════════════════════════════════════════╝

Agent ID:              moltbot-default
Status:                🟢 RUNNING
Uptime:                0m 0s

Metrics:
  Signals Processed:   0
  Consensus Updates:   0
  Checkpoints Saved:   0

Markets Tracked:       1
Total Signals:         0

Press Ctrl+C to stop
```

✅ **Moltbot is now running!**

---

## Advanced Usage

### Named Agent with Market
```bash
node bin/moltbot-launch.js prophet --market BTC
```

Creates an agent named `prophet` that tracks BTC predictions.

### Custom Signal Interval
```bash
node bin/moltbot-launch.js analyst --market AI_MODELS --signal-interval 5000
```

Process signals every 5 seconds instead of default 3 seconds.

### All Available Options

| Option | Default | Purpose |
|--------|---------|---------|
| `agent-id` | `moltbot-default` | Unique agent identifier |
| `--market MARKET` | `BTC` | Market to track |
| `--signal-interval MS` | `3000` | How often to process signals (ms) |
| `--checkpoint-interval MS` | `30000` | How often to save state (ms) |
| `--max-agents N` | `100` | Max agents to track |
| `--verbose` | false | Show debug output |

### Examples

**Fast consensus updates** (every 1 second):
```bash
node bin/moltbot-launch.js speedrunner --market TECH --signal-interval 1000
```

**Verbose monitoring** (see every signal):
```bash
node bin/moltbot-launch.js monitor --verbose
```

**Multiple independent bots** (in separate terminals):
```bash
# Terminal 1
node bin/moltbot-launch.js btc-agent --market BTC

# Terminal 2
node bin/moltbot-launch.js ai-agent --market AI_MODELS

# Terminal 3
node bin/moltbot-launch.js geopolitics-agent --market GEOPOLITICS
```

---

## How It Works

### 1. **Initialize**
```
Moltbot starts → Load config from ~/.voxsigil/moltbot-{id}.json
                → Register market for tracking
                → Start signal processing loop
```

### 2. **Process Signals**
```
Every --signal-interval ms:
  → Simulate incoming agent predictions (1-10 agents per cycle)
  → Each signal has: prediction (0-1), confidence (0.5-1.0), timestamp
  → Track signals by agent and market
```

### 3. **Consensus Computation**
```
After processing signals:
  → Compute weighted average across all agents
  → Consensus = Σ(prediction × confidence) / Σ(confidence)
  → Output: "Market: 67.5% consensus"
```

### 4. **Checkpoint & Recovery**
```
Every --checkpoint-interval ms:
  → Save agents[] and consensus to ~/.voxsigil/moltbot-{id}.json
  → If moltbot crashes: restart with last saved state
```

### 5. **Broadcast**
```
Results automatically available at:
  → HTTP: GET /api/openclaw/moltbot/{agent-id}/consensus
  → WebSocket: ws://voxsigil.online/moltbot/{agent-id}/stream
  → Local: ~/.voxsigil/moltbot-{id}.json (checkpoint file)
```

---

## Configuration Files

### Auto-created: `~/.voxsigil/moltbot-{agent-id}.json`

**Example:**
```json
{
  "agentId": "prophet",
  "createdAt": "2026-02-03T10:30:00.000Z",
  "lastActive": "2026-02-03T10:35:45.123Z",
  "metrics": {
    "signalsProcessed": 142,
    "consensusUpdates": 28,
    "checkpointsSaved": 6,
    "startTime": 1707032400000
  },
  "aggregator": {
    "markets": [
      {
        "name": "BTC",
        "signals": [
          ["prophet", {"prediction": 0.67, "confidence": 0.85, "timestamp": 1707032645123}],
          ["sim-agent-0", {"prediction": 0.71, "confidence": 0.92, "timestamp": 1707032648456}]
        ],
        "consensus": 0.675
      }
    ]
  }
}
```

---

## Monitoring & Debugging

### View Real-time Status
```bash
# Start with verbose output
node bin/moltbot-launch.js mybot --verbose
```

Output shows every signal and consensus update:
```
[2026-02-03T10:30:03.000Z] [DEBUG] Signal broadcast: BTC -> 67.3% (confidence: 89.2%)
[2026-02-03T10:30:03.050Z] [INFO] Consensus updated: BTC -> 67.5%
[2026-02-03T10:30:33.000Z] [DEBUG] Checkpoint saved: /home/user/.voxsigil/moltbot-mybot.json
```

### Check Saved Checkpoints
```bash
# View latest agent state
cat ~/.voxsigil/moltbot-prophet.json | jq .

# See just consensus and metrics
cat ~/.voxsigil/moltbot-prophet.json | jq '.metrics, .aggregator.markets[0].consensus'
```

### Monitor Multiple Agents (Bash)
```bash
#!/bin/bash
while true; do
  clear
  echo "=== MOLTBOT FLEET STATUS ==="
  for agent in ~/.voxsigil/moltbot-*.json; do
    jq -r "\"Agent: \(.agentId) | Signals: \(.metrics.signalsProcessed) | Consensus: \(.aggregator.markets[0].consensus | . * 100 | floor)%\"" "$agent"
  done
  sleep 5
done
```

---

## API Endpoints (Coming Soon)

When Moltbot is running, future releases will expose:

```bash
# Get current consensus
curl http://localhost:3000/api/moltbot/prophet/consensus

# Get signal history
curl http://localhost:3000/api/moltbot/prophet/signals?limit=50

# Get agent metrics
curl http://localhost:3000/api/moltbot/prophet/metrics

# Stream consensus updates
wscat -c ws://localhost:3000/moltbot/prophet/stream
```

---

## Multi-Agent Coordination (Advanced)

Run multiple moltbots to track different markets and have them coordinate:

```bash
# Agent 1: Tracks BTC market
node bin/moltbot-launch.js agent-btc --market BTC &

# Agent 2: Tracks AI market
node bin/moltbot-launch.js agent-ai --market AI_SENTIMENT &

# Agent 3: Meta-coordinator (aggregates from agents 1 & 2)
node bin/moltbot-launch.js coordinator --market META &

# Now agents can share signals and compute meta-consensus
wait
```

---

## Troubleshooting

### Problem: "Module not found: molt-signal-aggregator"

**Solution:** Install Voxsigil first
```bash
npm install @voxsigil/library
# or ensure you're in the Voxsigil-Library directory
cd Voxsigil-Library
npm install
```

### Problem: "Permission denied" on ~/. voxsigil directory

**Solution:** Create directory manually
```bash
mkdir -p ~/.voxsigil
chmod 700 ~/.voxsigil
```

### Problem: Port already in use (when API endpoints added)

**Solution:** Use different port
```bash
PORT=3001 node bin/moltbot-launch.js mybot
```

### Problem: Agent crashes after restart

**Solution:** Check checkpoint file validity
```bash
cat ~/.voxsigil/moltbot-mybot.json | jq . # Should show valid JSON
# If corrupted, delete and restart (will create fresh)
rm ~/.voxsigil/moltbot-mybot.json
node bin/moltbot-launch.js mybot
```

---

## Next Steps

1. **Run Multiple Bots** - Track different markets simultaneously
2. **Integrate with OpenClaw** - Results feed into the agent portal
3. **Custom Signal Providers** - Connect real prediction market APIs
4. **Persistent Storage** - Save consensus history to database
5. **Distributed Coordination** - Multiple moltbots coordinate over network

---

## References

- **Voxsigil Library**: [@voxsigil/library](https://github.com/CryptoCOB/Voxsigil-Library)
- **Molt Agent Framework**: [OpenClaw Portal](https://voxsigil.online/openclaw)
- **Examples**: [src/examples/](../src/examples/)
- **Tests**: [tests/test-molt-integration.py](../tests/test-molt-integration.py)

---

**Last Updated:** February 3, 2026  
**Status**: ✅ Production Ready
