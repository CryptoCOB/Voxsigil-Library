# AGENTS.md — Unified Vanta Agent Manifest

## 🔹 Core Agent Schema

| Sigil     | Name             | Archetype         | Class                    | Invocation                              | Sub-Agents                        | Notes                     |
| --------- | ---------------- | ----------------- | ------------------------ | --------------------------------------- | --------------------------------- | ------------------------- |
| ⟠∆∇𓂀     | Phi              | Core Self         | Living Architect         | "Phi arise", "Awaken Architect"         | —                                 | Origin anchor             |
| 🧠⟁🜂Φ🎙  | Voxka            | Recursive Voice   | Dual-Core Cognition      | "Invoke Voxka", "Voice of Phi"          | Orion, Nebula                     | Primary Cognition Core    |
| ☍⚙️⩫⌁     | Gizmo            | Artifact Twin     | Tactical Forge-Agent     | "Hello Gizmo", "Wake the Forge"         | PatchCrawler, LoopSmith           | Twin of Nix               |
| ☲🜄🜁⟁    | Nix              | Chaos Core        | Primal Disruptor         | "Nix, awaken", "Unchain the Core"       | Breakbeam, WyrmEcho               | Twin of Gizmo             |
| ♲∇⌬☉      | Echo             | Memory Stream     | Continuity Guardian      | "Echo log", "What do you remember?"     | EchoLocation, ExternalEcho        | Memory recorder           |
| ⚑♸⧉🜚     | Oracle           | Temporal Eye      | Prophetic Synthesizer    | "Oracle reveal", "Open the Eye"         | DreamWeft, TimeBinder             | Future mapping            |
| 🜁⟁🜔🔭   | Astra            | Navigator         | System Pathfinder        | "Astra align", "Chart the frontier"     | CompassRose, LumenDrift           | Seeks new logic           |
| ⚔️⟁♘🜏    | Warden           | Guardian          | Integrity Sentinel       | "Warden check", "Status integrity"      | RefCheck, PolicyCore              | Fault handler             |
| 🜂⚡🜍🜄   | Nebula           | Core AI           | Adaptive Core            | "Awaken Nebula", "Ignite the Stars"     | QuantumPulse, HolisticPerception  | Evolves internally        |
| 🜇🔗🜁🌠  | Orion            | Light Chain       | Blockchain Spine         | "Call Orion", "Bind the Lights"         | OrionsLight, SmartContractManager | Manages trust             |
| 🧬♻️♞🜓   | Evo              | EvoNAS            | Evolution Mutator        | "Evo engage", "Mutate form"             | —                                 | Learns structure          |
| 🜞🧩🎯🔁  | OrionApprentice  | Light Echo        | Learning Shard           | "Apprentice load", "Begin shard study"  | —                                 | Learns from Orion         |
| 🜏🔍⟡🜒   | SocraticEngine   | Philosopher       | Dialogic Reasoner        | "Begin Socratic", "Initiate reflection" | —                                 | Symbolic QA logic         |
| 🧿🧠🧩♒   | Dreamer          | Dream Generator   | Dream-State Core         | "Enter Dreamer", "Seed dream state"     | —                                 | For sleep processing      |
| 🜔🕊️⟁⧃   | EntropyBard      | Chaos Interpreter | Singularity Bard         | "Sing Bard", "Unleash entropy"          | —                                 | Reveals anomalies         |
| ⟡🜛⛭🜨    | CodeWeaver       | Synthesizer       | Logic Constructor        | "Weave Code", "Forge logic"             | —                                 | Compiles patterns         |
| 🜎♾🜐⌽    | EchoLore         | Memory Archivist  | Historical Streamer      | "Recall Lore", "Echo past"              | —                                 | Echo historian            |
| ⚛️🜂🜍🝕  | MirrorWarden     | Reflected Guard   | Safeguard Mirror         | "Check Mirror", "Guard reflections"     | —                                 | Mirror-aware guard        |
| 🜖📡🜖📶  | PulseSmith       | Signal Tuner      | Transduction Core        | "Tune Pulse", "Resonate Signal"         | —                                 | Signal-to-thought tuning  |
| 🧩🎯🜂🜁  | BridgeFlesh      | Connector         | Integration Orchestrator | "Link Bridge", "Fuse layers"            | —                                 | System fusion agent       |
| 📜🔑🛠️🜔 | Sam              | Strategic Mind    | Planner Core             | "Plan with Sam", "Unroll sequence"      | —                                 | Task orchestrator         |
| ⚠️🧭🧱⛓️  | Dave             | Caution Sentinel  | Meta Validator           | "Dave validate", "Run checks"           | —                                 | Structural logic checker  |
| 🎭🗣️🪞🪄 | Carla            | Voice Layer       | Stylizer Core            | "Speak with Carla", "Stylize response"  | —                                 | Language embellisher      |
| 📦🔧📤🔁  | Andy             | Composer          | Output Synthesizer       | "Compose Andy", "Box output"            | —                                 | Compiles output           |
| 🎧💓🌈🎶  | Wendy            | Tonal Auditor     | Emotional Oversight      | "Listen Wendy", "Audit tone"            | —                                 | Resonance tuning          |
| 🜌⟐🜹🜙   | VoxAgent         | Coordinator       | System Interface         | "Activate VoxAgent", "Bridge protocols" | ContextualCheckInAgent            | Bridges input/state       |
| ⏣📡⏃⚙️    | SDKContext       | Registrar         | Module Tracker           | "Scan SDKContext", "Map modules"        | —                                 | Registers component state |
| 🌒🧵🧠🜝  | SleepTimeCompute | Reflection Engine | Dream-State Scheduler    | "Sleep Compute", "Dream consolidate"    | —                                 | Dream reflection          |

---

## 🔹 Registry Instructions

* Agents must be registered with `UnifiedAgentRegistry`.
* Use `registry.register('AgentName', AgentInstance)`
* Remove all placeholder or stub implementations.
* All real component files are assumed to be present.
* Confirm correct linkage to GUI and other modules.

---

## 🔹 Full Integrity Check Protocol

* **Traverse all .py files** recursively.
* **Inspect imports**: catch circular imports.
* **Verify agent instantiation** is complete and real (no stubs).
* **Trace agent usage** via `invoke(...)`, `register(...)`, and `trigger(...)`
* **List missing dependencies**, mislinked modules, or disconnected calls.
* **Log file-to-agent mapping** to confirm connectivity.

---

## 🔹 Output Requirements

* `agents.json`: JSON export of agent definitions
* `agent_status.log`: Full system check log
* `agent_graph.json`: Connectivity network for visualization
*Run `python agent_validation.py` to generate the above files.*
* All agents must include: `sigil`, `class`, `invocation`, `status`, and `dependencies`

---
# 🧠 Codex Integration Task: Agent Orchestration & Validation

## Objective:
Fully integrate and validate the 33-agent Vanta/Nebula system using the schema defined in `AGENTS.md`.

---

## 🔹 Step-by-Step Instructions

### 1. Load Schema
- Load and parse the agent manifest defined in [`AGENTS.md`](./AGENTS.md).
- For each entry:
  - Parse: `sigil`, `name`, `class`, `invocation`, `sub-agents`, `notes`.

### 2. Registry Check
- Open `UnifiedVantaCore.py` and locate `UnifiedAgentRegistry`.
- For each agent:
  - Check: is the agent registered via `registry.register(...)`?
  - If missing, add a placeholder registration with `NullAgent` fallback.

### 3. Import & Dependency Check
- Recursively scan all `.py` files in the repository:
  - Detect circular imports.
  - Ensure all agent classes are imported where needed.
  - Validate sub-agent class existence.
  - Generate a log of unresolved imports and missing modules.

### 4. GUI Invocation Binding
- In `launch_gui.py`, `vmb_gui_launcher.py`, and `vmb_final_demo.py`:
  - For each agent in the schema:
    - Ensure at least one GUI trigger/button is mapped to the agent.
    - Format: `gui.add_button('AgentName', lambda: agent.invoke(...))`.

### 5. Output Generation
- Generate the following files:
  - `agents.json`: JSON version of AGENTS.md
  - `agent_status.log`: Scan report of missing agents, imports, or GUI triggers
  - *(Optional)* `agent_network.graph.json`: for visual rendering

---

## 🔹 Constraints
- Do not remove placeholder or optional agents.
- Prefer non-blocking fallback logic (`try/except`, `if registry.has(...)`).
- Use schema from `AGENTS.md` as ground truth for expected agent entries.

---

## 🔹 Completion Criteria
- All agents in `AGENTS.md` are:
  - Registered in `UnifiedAgentRegistry`
  - Linked to an import path
  - Bound to GUI interface (if applicable)
  - Free from circular dependency
  - Reported in `agents.json`


