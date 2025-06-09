# Voxsigil-Library

This repository hosts the VoxSigil experimental agent framework. It bundles the
``UnifiedVantaCore`` orchestration engine, a collection of 30+ agents and a
Tkinter based GUI for visual model interaction.

## Usage

1. **Generate status reports**

   ```bash
   python agent_validation.py
   ```

   This creates `agents.json`, `agent_graph.json` and `agent_status.log` showing
   which agents are available and registered.

2. **Launch the GUI**

   ```bash
   python launch_gui.py
   ```

   Each registered agent will appear as a button at the bottom of the interface
   allowing quick invocation.

## Contents

* `AGENTS.md` – manifest describing every VoxSigil agent
* `docs/PROGRESS_PLAN.md` – high level integration roadmap
* `agent_validation.py` – simple integrity checker
