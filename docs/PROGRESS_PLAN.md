# Vanta Integration Progress Plan

This document tracks the ongoing work to integrate and validate all agents and subsystems defined in `AGENTS.md`.

## Current Status


- **Implemented agents:** Skeleton classes created for all agents and registered via `UnifiedVantaCore`.
- **Registered agents:** All agents now registered with `UnifiedAgentRegistry` and mapped to subsystems.
- **Memory layer:** EchoMemory available but echo binding work is ongoing via `UnifiedAsyncBus`.
- **GUI triggers:** GUI mappings not fully wired; placeholder `on_gui_call` methods implemented for guardian agents.
- **Project structure:** Agents moved under `agents/` package for clearer imports.


## Phase 1 – Agent Class Definitions

Create Python modules under `agents/` for each entry in `AGENTS.md`.
Each class should define:
- `sigil`: Unicode sigil identifier
- `invocations`: list of invocation phrases
- `initialize_subsystem()`
- `on_gui_call()` if the agent exposes GUI actions

Example skeleton:
```python
class Phi(LivingArchitect):
    sigil = "⟠∆∇𓂀"
    invocations = ["Phi arise", "Awaken Architect"]

    def initialize_subsystem(self, vanta_core):
        pass
```

## Phase 2 – Registry Integration

In `UnifiedVantaCore._initialize_core_agents`:
1. Instantiate each agent class.
2. Register via `UnifiedAgentRegistry.register('AgentName', instance)`.
3. For agents with sub‑agents, register dependencies first.
4. Add fallback logic using `NullAgent` for missing modules during early development.

## Phase 3 – Subsystem Mapping

Assign each major subsystem a guardian agent:
- **BLT-RAG** → `EntropyBard`
- **GridFormer** → `PulseSmith`
- **MetaLearner** → `MirrorWarden` or `CodeWeaver`
- **ART** → `Dreamer`
- **VMB** → `BridgeFlesh`

Each guardian implements `initialize_subsystem()` to configure its component and exposes a `on_gui_call()` method for the GUI.


## Phase 4 – Echo Binding

Ensure every agent can send and receive signals from the components they supervise. For example, `PulseSmith` should receive grid traces and call `GridFormer.forward()`. Use `UnifiedAsyncBus` for cross-module messaging.

## Phase 5 – GUI Invocation Binding

Update `launch_gui.py`, `vmb_gui_launcher.py`, and `vmb_final_demo.py`:
- Map a GUI button to each agent invocation:
  ```python
  gui.add_button('Invoke PulseSmith', lambda: registry.get('PulseSmith').on_gui_call())
  ```
- Use non-blocking calls and check for agent availability via `registry.has('AgentName')`.

*Status:* Implemented via `gui_utils.bind_agent_buttons`. Launch scripts now automatically create buttons for every registered agent.

## Phase 6 – Project Reorganization

Move agent modules to `agents/` and update imports accordingly:
- `sleep_time_compute.py` → `agents/sleep_time_compute.py`
- Create `agents/__init__.py` exporting all agent classes for easy importing.
- Update references such as `from Vanta.core.sleep_time_compute import ...` to `from agents.sleep_time_compute import ...`.

Create similar directories for `interfaces/`, `core/`, and `services/` if needed to reduce clutter.

*Status:* **Completed (2025‑06‑09)** – Modules relocated under `agents/` with imports updated. Core initialization registers all agents on startup.
Additional cleanup moved the Vanta orchestration files under `Vanta/core` and example launch scripts under `scripts/`.

## Phase 7 – Integrity & Validation Tools

Implement scripts to generate:
- `agents.json` – JSON export of the agent manifest and registration status.
- `agent_status.log` – log of missing imports or registration gaps.
- `agent_graph.json` – optional connectivity graph for visualization.

*Status:* **Completed (2025‑06‑09)** – `agent_validation.py` generates the above files and produces `agents.json`, `agent_graph.json` and `agent_status.log`.

## Outstanding Items

- Define concrete behavior for each agent class.
- Resolve missing imports and circular dependencies across modules.
- Ensure training scripts and demos reference the new module structure.
- Implement comprehensive tests for agent registration and subsystem activation.

---

This plan should be kept up to date as integration progresses. Each phase can be completed independently, but all agents must be registered and bound to their subsystems before final GUI wiring.
