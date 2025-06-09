# Vanta Bug Sweep Findings

The following issues were identified while inspecting the Vanta code base.
Each entry includes a suggested fix. Patches applied in this commit are marked accordingly.

1. **Invalid agent import path** – `UnifiedVantaCore` imported agents from a
   non-existent module `Vanta.core.agents`.
   *Fixed by importing from the top-level `agents` package.*
2. **GUI invocation not streamed** – `BaseAgent.on_gui_call` did not emit
   events for GUI panels.
   *Added event emissions so GUI panels can react to agent activity.*
3. **Duplicate SleepTimeCompute alias** – `agents/__init__.py` exported both
   `SleepTimeCompute` and `SleepTimeComputeAgent` which can cause duplicate
   registration.
   *Added a warning comment to highlight the issue.*
4. **Agents fail to register** – many agents remain unregistered according to
   `agent_status.log`; missing dependencies cause import failures.
5. **Missing launch script** – README references `scripts/launch_gui.py` but the
   file is absent.
6. **Missing `blt_encoder.py`** – referenced in docs yet not present in the
   repository.
7. **Mesh module mismatch** – expected `mesh.py` or `voxsigil_mesh.py`; only
   `holo_mesh.py` exists.
8. **Messages not compressed** – no calls to `BLTEncoder.compress()` before
   sending data on the bus.
9. **GUI panels disconnected** – no events for `EchoLogPanel`, `AgentStatusPanel`
   or `MeshMapPanel` were observed.
10. **Legacy launcher path issues** – `legacy_gui/vmb_gui_launcher.py` modifies
    `sys.path` with relative entries that may break when executed elsewhere.
11. **Agent stubs** – most agent classes contain placeholder methods with no
    operational logic.

Patches applied in this commit address items 1–3 and partially item 2.
