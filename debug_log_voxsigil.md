# VoxSigil Debug Log


Detected critical bugs and structural flaws across the system.
Each entry lists the file location, error type, and a suggested fix.

1. **Wrong import path** – `Vanta/core/UnifiedVantaCore.py` lines 1324-1327 import `speech_integration_handler` from `Vanta.integration` but the module resides in `handlers/`. **Fixed** by updating the import path.
2. **Wrong import path** – `Vanta/core/UnifiedVantaCore.py` lines 1357-1358 import `vmb_integration_handler` from `Vanta.integration` though it's located in `handlers/`. **Fixed**.
3. **Hardcoded Windows path** – `Vanta/core/VantaCognitiveEngine.py` lines 22-27 set `_PROJECT_ROOT` to a Windows directory. **Fixed** using a relative path based on `__file__`.
4. **Missing utility module** – `Vanta/core/VantaCognitiveEngine.py` line 36 attempts to import from `tools.utilities.utils`, which is absent. **Fixed** with a local helper function.
5. **Registry not thread-safe** – `Vanta/core/UnifiedAgentRegistry.py` uses a plain dict without locks, risking race conditions in multi-threaded scenarios. **Fixed** using `threading.RLock`.
6. **Mutable class defaults** – `agents/base.py` lines 13-15 define `invocations` and `sub_agents` as mutable class-level lists. **Fixed** by initializing per instance.
7. **Async publish not awaited** – `agents/base.py` line 91 calls `async_bus.publish()` without awaiting. **Fixed** with `asyncio.create_task`.
8. **Subsystem placeholder** – Many agents (e.g., `agents/wendy.py` lines 9-19) contained `pass` in `initialize_subsystem`. **Fixed** by calling superclass methods.
9. **Unbounded event history** – `EventBus` in `UnifiedVantaCore` stores up to 1000 events but never prunes. Could cause memory buildup. Add pruning logic or limit.
10. **Duplicate checks** – `discover_agents_by_capabilities` (lines 1076-1082) repeats registry checks. Simplify conditions to avoid redundancy.
11. **Async bus not started** – `UnifiedVantaCore` never calls `async_bus.start()`, so async messages queue indefinitely. Start the bus on initialization.
12. **Undefined handler paths** – `_initialize_speech_integration` and `_initialize_vmb_integration` don't verify handler success; failures go unnoticed. Add status checks.
13. **Hard dependency** – `_initialize_cognitive_layer` assumes `RealSupervisorConnector` is available. Wrap in try/except to handle missing dependency.
14. **GUI import failure** – `scripts/launch_gui.py` lines 181-189 expect `dynamic_gridformer_gui` module which may not exist, leading to ImportError. Provide module or adjust fallback.
15. **Global core None risk** – `scripts/launch_gui.py` functions reference global `core` without checking after initialization. Guard access with `if core` checks.
16. **Event loop misuse** – `handlers/vmb_integration_handler.py` runs async init using `get_event_loop()` which can fail if no loop. **Fixed** by creating a new event loop and closing it.
17. **Loop not closed** – same handler did not close the loop after `run_until_complete`. **Fixed**.
18. **STT handler duration assumption** – `handlers/speech_integration_handler.py` lines around 250 assume message has `get` method. Validate type before use.
19. **No agent existence check** – `UnifiedVantaCore.delegate_task_to_agent` returns error only after retrieving agent, but not when `agent_registry` missing. Add registry check.
20. **Placeholder cross-system link** – `UnifiedVantaCore` lacks Nebula integration; `bind_cross_system_link()` is left as a stub for future connection.
21. **RAG integration unused** – `handlers/rag_integration_handler.py` exists but is never initialized from core, leaving RAG features disconnected.
22. **Inconsistent tag data** – multiple agents defined `'None'` as a tag. **Fixed** by removing invalid tags.
23. **Shared invocations** – because of class-level lists, adding an invocation to one agent affects all others. **Fixed** with instance-specific lists.
24. **Duplicate path inserts** – `scripts/launch_gui.py` adds several paths to `sys.path` without checking for duplicates, causing path bloat. Use a set check.
25. **Missing cleanup** – `UnifiedVantaCore.shutdown` emits events but does not stop the async bus or other components. Ensure graceful shutdown.
26. **No error propagation** – many try/except blocks in core simply log errors without re-raising or handling, hiding failures. Review error handling strategy.
27. **Unused convenience alias** – `Vanta/core/UnifiedVantaCore.py` defines `VantaCore = UnifiedVantaCore`, risking confusion during imports. Consider removing alias.
28. **Registry overwrite warning** – `UnifiedAgentRegistry.register_agent` warns then overwrites existing agents silently; may hide duplicates. Consider raising exception.
29. **GUI tooltip creation** – `gui/gui_utils.py` does not destroy tooltip windows on root close, leading to ghost windows. Bind destroy event.
30. **Asynchronous tasks** – `handlers/vmb_integration_handler.execute_task` awaits `production_executor` but errors are swallowed; return results to caller for debugging.
