# System Overview

This document provides a high level summary of the VoxSigil / Vanta ecosystem.

## Components

- **UnifiedVantaCore** – central orchestration engine that registers and manages all agents. It exposes an async bus and event bus so subsystems can communicate.
- **Agents** – 30+ modular agents defined in `AGENTS.md`. Each agent implements `initialize_subsystem()` and `on_gui_call()` for GUI triggers.
- **ART Module** – adaptive resonance theory engine used for pattern recognition. Integrated via the `Dreamer` guardian agent.
- **BLT‑RAG** – retrieval augmented generation pipeline connected through the `EntropyBard` guardian.
- **GridFormer Connector** – links trained GridFormer models for ARC tasks. Managed by the `PulseSmith` agent.
- **Meta Learner** – meta-learning interface mapped to the `MirrorWarden` and `CodeWeaver` agents.
- **Speech System** – async TTS and STT engines available via the `SpeechIntegrationHandler`. Agents `Carla` and `Wendy` expose speech control through the GUI.
- **GUI** – `dynamic_gridformer_gui.py` launches a Tkinter interface and automatically binds buttons for each agent.

## Data Flow

1. `launch_gui.py` initializes `UnifiedVantaCore`, loads the speech and VMB integrations, and registers all agents.
2. The GUI discovers registered agents and creates buttons for each using `gui_utils.bind_agent_buttons`.
3. When a user activates an agent button, the agent publishes a message on the async bus. Subsystems (RAG, GridFormer, ART, speech, etc.) listen and respond.
4. Results are routed back through the event bus and displayed in the GUI.

## Logs and Validation

Run `python agent_validation.py` to generate:
- `agents.json` – current agent definitions
- `agent_status.log` – registration and import status
- `agent_graph.json` – basic connectivity graph

Log files such as `vantacore_grid_former_integration.log` and component-specific logs provide additional diagnostics during runtime.

