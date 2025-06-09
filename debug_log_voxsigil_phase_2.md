# VoxSigil Debug Log Phase 2

New issues discovered during deep logic sweep. ✅ indicates fix applied.

1. interfaces/llm_interface.py line 16 – Non ASCII variable name `VOXSİGİL_LLM_HANDLER_AVAILABLE` caused import errors. Renamed to ASCII `VOXSIGIL_LLM_HANDLER_AVAILABLE`. ✅
2. interfaces/llm_interface.py lines 35,49,141 – Updated references to new variable name. ✅
3. path_helper.py line 83 – Wrong project path key `Voxsigil_Library` used underscore. Changed to `Voxsigil-Library`. ✅
4. Vanta/core/UnifiedAsyncBus.py stop() – Did not cancel processing task causing hang. Added cancel logic. ✅
5. Vanta/core/UnifiedVantaCore.py shutdown() – Async bus not stopped. Added stop call with error logging. ✅
6. Vanta/core/UnifiedVantaCore.py init – async bus start failure silently ignored. Now logs error. ✅
7. Vanta/core/UnifiedVantaCore.py EventBus.emit – Did not log when no subscribers. Added debug log. ✅
8. legacy_gui/gui_utils.py _ToolTip – Tooltip windows persisted after root close. Bound destroy event. ✅
9. async_tts_engine.py _start_background_processing – Used get_event_loop without running loop. Now obtains running loop or creates background loop. ✅
10. agent_validation.py – agent_status.log lacked trailing newline. Added. ✅
11. agent_status.log – file recreated with newline. ✅
12. event "vmb.swarm.initialized" has no subscribers; log at emit covers this. (not fixed)
13. Multiple legacy Tkinter modules remain under legacy_gui/. Documented in migrated_gui_status.md. (not fixed)
14. path_helper.py – create_sigil_supervisor_instance fallback logging lacked module-level logger. (not fixed)
15. UnifiedAsyncBus.stop – race condition when stop called while processing queue. Cancel added (see 4). ✅
16. AsyncTTSEngine background tasks may run without shutdown due to thread loops. Partially mitigated with new loop management. ✅
17. agent_graph.json generation unaffected. (info)
18. Unused backup file production_config.py.bak may confuse config selection. (not fixed)
19. Some event types emitted without receivers (vmb.task.*, etc.) – logged as debug by event bus. (not fixed)
20. Inconsistent path checks around VoxSigilRag import options. (not fixed)
21. Global integration manager in voxsigil_integration.py uses global var; potential race. (not fixed)
22. UnifiedVantaCore alias `VantaCore` might cause confusion but kept for compatibility. (not fixed)
23. AsyncProcessingEngine uses get_event_loop directly in callbacks; may break in non-main thread. (not fixed)
24. Many .bak/.old files remain; may cause stale imports. (not fixed)
25. Some placeholder tests refer to removed modules. (not fixed)
26. event_bus._max_history set to 1000; not configurable. (not fixed)
27. Logging configuration repeated in several modules; could lead to duplicates. (not fixed)
28. path_helper.create_sigil_supervisor_instance uses nested imports; error handling improved. (not fixed)
29. VMBIntegrationHandler emits events before verifying subscribers. Covered by new EventBus logging. ✅
30. Several async functions in speech_integration_handler call get_event_loop; still unpatched. (not fixed)
