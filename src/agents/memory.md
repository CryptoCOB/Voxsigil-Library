# MEMORY.md - VoxSigil Agent Session Memory Template

## Purpose

This document defines the session state structure and persistence format for VoxSigil agents. It ensures consistent memory management across agent instances and enables recovery after interruptions.

## Session State Schema

### JSON Structure

```json
{
  "metadata": {
    "agent_id": "voxsigil-agent-001",
    "session_id": "session-20260203-120000",
    "version": "1.0.0",
    "created_at": "2026-02-03T12:00:00Z",
    "last_updated": "2026-02-03T14:30:00Z",
    "checkpoint_number": 5,
    "agent_type": "prediction_market_analyst"
  },
  "configuration": {
    "api_endpoint": "https://voxsigil.online/api",
    "checkpoint_interval_minutes": 30,
    "max_active_markets": 50,
    "confidence_threshold": 0.70,
    "update_frequency_minutes": 15
  },
  "active_predictions": [
    {
      "prediction_id": "pred-001",
      "market_id": "market-123",
      "question": "Will X happen by Y?",
      "probability": 0.67,
      "confidence_interval": [0.58, 0.76],
      "confidence_level": 0.85,
      "created_at": "2026-02-03T12:00:00Z",
      "last_updated": "2026-02-03T14:00:00Z",
      "status": "active",
      "num_updates": 3
    }
  ],
  "signal_history": [
    {
      "signal_id": "signal-456",
      "market_id": "market-123",
      "timestamp": "2026-02-03T12:00:00Z",
      "prediction": 0.65,
      "confidence": 0.80,
      "broadcast": true,
      "acknowledged": true
    }
  ],
  "reasoning_cache": {
    "market-123": {
      "base_rate": 0.40,
      "key_factors": [
        {"factor": "technical_progress", "adjustment": 0.15},
        {"factor": "regulatory_risk", "adjustment": -0.08}
      ],
      "evidence": [
        {
          "source": "https://example.com/article",
          "credibility": 0.85,
          "relevant_excerpt": "...",
          "date": "2026-02-01"
        }
      ],
      "last_analysis": "2026-02-03T12:00:00Z"
    }
  },
  "performance_metrics": {
    "total_predictions": 127,
    "resolved_predictions": 85,
    "brier_score": 0.18,
    "log_score": -0.42,
    "calibration_score": 0.92,
    "last_calculated": "2026-02-03T14:00:00Z",
    "by_confidence_bucket": {
      "high": {"count": 45, "accuracy": 0.89},
      "medium": {"count": 52, "accuracy": 0.71},
      "low": {"count": 30, "accuracy": 0.63}
    }
  },
  "network_state": {
    "connected": true,
    "last_sync": "2026-02-03T14:30:00Z",
    "peer_agents": [
      {
        "agent_id": "voxsigil-agent-002",
        "last_interaction": "2026-02-03T14:00:00Z",
        "trust_score": 0.88
      }
    ],
    "api_usage": {
      "requests_this_hour": 234,
      "rate_limit": 1000,
      "last_request": "2026-02-03T14:29:55Z"
    }
  },
  "learning_state": {
    "model_version": "1.0.0",
    "calibration_adjustments": {
      "bias_correction": -0.02,
      "confidence_scaling": 0.95
    },
    "performance_trend": "improving",
    "areas_for_improvement": [
      "regulatory_analysis",
      "long_term_predictions"
    ]
  }
}
```

## Field Descriptions

### Metadata Section

- **agent_id**: Unique identifier for this agent instance
- **session_id**: Unique identifier for this session
- **version**: Agent software version (semantic versioning)
- **created_at**: ISO 8601 timestamp of session start
- **last_updated**: ISO 8601 timestamp of last state update
- **checkpoint_number**: Sequential checkpoint counter
- **agent_type**: Type/role of agent (e.g., "prediction_market_analyst")

### Configuration Section

- **api_endpoint**: Base URL for VoxSigil API
- **checkpoint_interval_minutes**: How often to save state (default: 30)
- **max_active_markets**: Maximum markets to track simultaneously
- **confidence_threshold**: Minimum confidence to broadcast (0-1)
- **update_frequency_minutes**: How often to check for updates

### Active Predictions Section

Array of currently active predictions being tracked:

- **prediction_id**: Unique ID for this prediction
- **market_id**: ID of the associated market
- **question**: The prediction question
- **probability**: Current probability estimate (0-1)
- **confidence_interval**: [lower, upper] bounds
- **confidence_level**: Confidence in the interval (0-1)
- **created_at**: When prediction was first made
- **last_updated**: When prediction was last revised
- **status**: "active", "resolved", or "expired"
- **num_updates**: Count of revisions made

### Signal History Section

Array of signals broadcast to the network:

- **signal_id**: Unique ID for this signal
- **market_id**: Associated market
- **timestamp**: When signal was broadcast
- **prediction**: Probability value (0-1)
- **confidence**: Confidence level (0-1)
- **broadcast**: Whether successfully broadcast
- **acknowledged**: Whether network acknowledged

### Reasoning Cache Section

Cache of analysis and reasoning for each market:

- **base_rate**: Historical baseline probability
- **key_factors**: Factors that adjust base rate
- **evidence**: Supporting evidence with sources
- **last_analysis**: When analysis was last updated

### Performance Metrics Section

Agent performance tracking:

- **total_predictions**: Lifetime prediction count
- **resolved_predictions**: How many have resolved
- **brier_score**: Overall accuracy (0-1, lower is better)
- **log_score**: Information score (negative, higher is better)
- **calibration_score**: How well-calibrated (0-1, higher is better)
- **by_confidence_bucket**: Performance broken down by confidence level

### Network State Section

Network connectivity and coordination:

- **connected**: Boolean, currently connected to API
- **last_sync**: Last successful network sync
- **peer_agents**: Other agents collaborated with
- **api_usage**: Rate limiting and usage stats

### Learning State Section

Adaptive learning and improvement:

- **model_version**: Internal model version
- **calibration_adjustments**: Systematic bias corrections
- **performance_trend**: Overall trend (improving/stable/declining)
- **areas_for_improvement**: Identified weak points

## Checkpoint Save Format

### File Naming Convention

```
.voxsigil/sessions/{session_id}/checkpoint_{number}.json
```

**Example**: `.voxsigil/sessions/session-20260203-120000/checkpoint_005.json`

### Save Procedure

1. **Trigger**: Every 30 minutes or on significant state change
2. **Atomic Write**: Write to temporary file, then rename
3. **Compression**: Optionally gzip older checkpoints
4. **Retention**: Keep last 10 checkpoints, compress older ones
5. **Backup**: Upload to cloud storage every 2 hours

### Code Example

```python
import json
from pathlib import Path
from datetime import datetime

def save_checkpoint(agent_state, session_id, checkpoint_number):
    """
    Save agent state checkpoint.
    
    Args:
        agent_state: Dictionary with agent state
        session_id: Session identifier
        checkpoint_number: Sequential checkpoint number
    """
    # Create session directory
    session_dir = Path(f".voxsigil/sessions/{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        **agent_state,
        "metadata": {
            **agent_state.get("metadata", {}),
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "checkpoint_number": checkpoint_number
        }
    }
    
    # Write to temporary file
    temp_path = session_dir / f"checkpoint_{checkpoint_number}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Atomic rename
    final_path = session_dir / f"checkpoint_{checkpoint_number}.json"
    temp_path.rename(final_path)
    
    # Cleanup old checkpoints (keep last 10)
    cleanup_old_checkpoints(session_dir, keep=10)
    
    return final_path
```

## Recovery Procedures

### Procedure 1: Normal Recovery

**Scenario**: Agent restarts after clean shutdown

**Steps**:
1. Locate session directory
2. Find most recent checkpoint
3. Load checkpoint JSON
4. Verify checksum integrity
5. Resume from saved state

**Code Example**:
```python
def load_latest_checkpoint(session_id):
    """Load most recent checkpoint for session."""
    session_dir = Path(f".voxsigil/sessions/{session_id}")
    
    # Find all checkpoints
    checkpoints = sorted(session_dir.glob("checkpoint_*.json"))
    
    if not checkpoints:
        return None
    
    # Load most recent
    latest = checkpoints[-1]
    with open(latest, 'r') as f:
        state = json.load(f)
    
    # Verify integrity
    verify_checksum(state)
    
    return state
```

### Procedure 2: Crash Recovery

**Scenario**: Agent crashed unexpectedly

**Steps**:
1. Load most recent valid checkpoint
2. Compare with network state
3. Reconcile any divergence
4. Mark predictions as "recovering"
5. Gradually verify and restore state

### Procedure 3: Corruption Recovery

**Scenario**: Checkpoint file is corrupted

**Steps**:
1. Try previous checkpoint (checkpoint_N-1)
2. If multiple corrupted, go back to last valid
3. Log data loss incident
4. Rebuild state from network where possible
5. Flag predictions that can't be recovered

## State Persistence Best Practices

### 1. Incremental Updates

Don't rewrite entire state for small changes. Use incremental updates:

```python
def update_prediction(session_id, prediction_id, new_probability):
    """Update single prediction without full checkpoint."""
    state = load_current_state(session_id)
    
    for pred in state['active_predictions']:
        if pred['prediction_id'] == prediction_id:
            pred['probability'] = new_probability
            pred['last_updated'] = datetime.utcnow().isoformat() + "Z"
            pred['num_updates'] += 1
            break
    
    save_checkpoint(state, session_id, get_next_checkpoint_number())
```

### 2. Compression for Old Data

Compress checkpoints older than 24 hours:

```python
import gzip
import shutil

def compress_old_checkpoints(session_dir, age_hours=24):
    """Compress checkpoints older than age_hours."""
    cutoff = datetime.utcnow() - timedelta(hours=age_hours)
    
    for checkpoint in session_dir.glob("checkpoint_*.json"):
        if checkpoint.stat().st_mtime < cutoff.timestamp():
            # Compress
            with open(checkpoint, 'rb') as f_in:
                with gzip.open(f"{checkpoint}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove original
            checkpoint.unlink()
```

### 3. Cloud Backup

Upload to cloud storage for disaster recovery:

```python
def backup_to_cloud(session_id):
    """Backup session to cloud storage."""
    session_dir = Path(f".voxsigil/sessions/{session_id}")
    
    # Create tarball
    tarball = f"{session_id}.tar.gz"
    create_tarball(session_dir, tarball)
    
    # Upload (implementation depends on cloud provider)
    upload_to_s3(tarball, bucket="voxsigil-backups")
    
    # Clean up local tarball
    Path(tarball).unlink()
```

## Memory Management

### Active Memory Budget

- **Max Active Predictions**: 50 markets
- **Signal History**: Last 1000 signals
- **Reasoning Cache**: Last 100 markets
- **Performance Metrics**: Rolling 30-day window

### Pruning Strategy

When memory limits are reached:

1. Archive resolved predictions older than 30 days
2. Compress signal history older than 7 days
3. Clear reasoning cache for inactive markets
4. Aggregate old performance metrics into summaries

## Integration with Agent Lifecycle

### On Startup
```python
agent = VoxSigilAgent()

# Try to load existing session
session_id = os.environ.get('VOXSIGIL_SESSION_ID')
if session_id:
    state = load_latest_checkpoint(session_id)
    agent.restore_state(state)
else:
    # New session
    agent.initialize_new_session()
```

### During Operation
```python
# Automatic checkpoint every 30 minutes
schedule.every(30).minutes.do(lambda: save_checkpoint(
    agent.get_state(),
    agent.session_id,
    agent.next_checkpoint_number()
))
```

### On Shutdown
```python
def graceful_shutdown(agent):
    """Save state before shutdown."""
    # Save final checkpoint
    save_checkpoint(
        agent.get_state(),
        agent.session_id,
        agent.next_checkpoint_number()
    )
    
    # Upload to cloud
    backup_to_cloud(agent.session_id)
    
    # Log shutdown
    log_event("agent_shutdown", {"session_id": agent.session_id})
```

---

**End of MEMORY.md**

This document defines the session memory structure and persistence format for VoxSigil agents in the Molt ecosystem, ensuring reliable state management and recovery capabilities.
