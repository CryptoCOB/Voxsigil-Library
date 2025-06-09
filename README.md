# Voxsigil-Library

This repository hosts the VoxSigil experimental agent framework. It bundles the
``UnifiedVantaCore`` orchestration engine, a collection of 30+ agents and a
Tkinter based GUI for visual model interaction. Phase 6 and Phase 7 of the
integration plan are now complete: agents live under the ``agents`` package and
validation tools generate status reports.

## Usage

1. **Generate status reports**

   ```bash
   python agent_validation.py
   ```

   This creates `agents.json`, `agent_graph.json` and `agent_status.log` showing
   which agents are available and registered.

2. **Launch the GUI**

   ```bash
   python scripts/launch_gui.py
   ```

   Each registered agent will appear as a button at the bottom of the interface
   allowing quick invocation. Speech controls (TTS/STT) are available through
   the ``Carla`` and ``Wendy`` agents when the speech integration handler is
   loaded.

## Contents

* `AGENTS.md` – manifest describing every VoxSigil agent
* `docs/PROGRESS_PLAN.md` – high level integration roadmap
* `docs/SYSTEM_OVERVIEW.md` – summary of components and data flow
* `agent_validation.py` – simple integrity checker
* `Vanta/` – Vanta core packages and integration modules
* `handlers/` – integration handler implementations
* `gui/` – GUI components and tab interfaces
* `scripts/` – helper scripts and launchers

### Folder Structure

```
Vanta/           # Core orchestration modules and integration utilities
handlers/        # Pluggable integration handlers
gui/             # Tkinter GUI components and tab interfaces
agents/          # Individual agent class implementations
scripts/         # Entry points and demo launchers
```
