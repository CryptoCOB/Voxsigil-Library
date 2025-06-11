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
13. Legacy Tkinter modules moved to `archive/`. See migrated_gui_status.md.
10. agent_validation.py – agent_status.log lacked trailing newline. Added. ✅
11. agent_status.log – file recreated with newline. ✅
12. event "vmb.swarm.initialized" has no subscribers; log at emit covers this. (not fixed)
13. Legacy Tkinter modules archived. (resolved)
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
31. services/memory_service_connector.py lines 69-75 – `namespace` and `metadata` parameters lacked Optional typing. Added Optional hints. ✅
32. services/memory_service_connector.py line 91 – Metadata passed as None caused errors in backends. Now defaults to empty dict. ✅
33. services/memory_service_connector.py line 97 – `retrieve` used `str` type for namespace; changed to `Optional[str]`. ✅
34. services/memory_service_connector.py line 114 – `retrieve_similar` namespace parameter updated to Optional. ✅
35. path_helper.py top – No module-level logger. Added logger and imported logging. ✅
36. path_helper.py line 198 – Exception logging now uses module logger instead of inline import. ✅
37. real_supervisor_connector.py lines 50-57 – Hardcoded Windows paths replaced with cross-platform detection using `Path.home()`. ✅
38. real_supervisor_connector.py line 64 – Default fallback path now uses user home directory. ✅
39. UnifiedVantaCore.shutdown line 1330 – Used `asyncio.run` even if loop running; now checks loop state and schedules stop accordingly. ✅
40. memory_service_connector.store returns empty string when not initialized – still not ideal. (not fixed)
41. vmb_integration_handler.initialize_vmb_system creates new event loop without restoring original; potential side effect. (not fixed)
42. arc_utils.cache_response removes only one entry when cache full; may still exceed max size. (not fixed)
43. real_supervisor_connector.store_sigil_content lacks atomic write, may corrupt files on crash. (not fixed)
44. unified_async_bus.register_component does not check for reserved ids. (not fixed)
45. scripts contain `sys.exit(asyncio.run(...))`; fails if called from running loop. (not fixed)
46. real_supervisor_connector._find_voxsigil_library does not validate env path exists. (not fixed)
47. memory_service_connector methods not thread-safe. (not fixed)
48. debug summary not updated when agents missing – informational. (not fixed)
49. event_bus history not pruned beyond max size. (not fixed)
50. Many test files contain placeholders with `pass`. (not fixed)
51. asynchronous STT initialization uses get_event_loop which may fail. (not fixed)
52. STT handler loops may not cancel on shutdown. (not fixed)
53. production_config.py.bak leftover may confuse deployment. (not fixed)
54. multiple modules repeat logging.basicConfig causing duplicate handlers. (not fixed)
55. VMBIntegrationHandler._register_event_handlers may subscribe duplicates on repeated calls. (not fixed)
56. RealSupervisorConnector._save_sigil_to_filesystem does not sanitize all invalid characters. (not fixed)
57. path_helper.verify_module_paths does not catch ImportError messages. (not fixed)
58. Some event bus emitters pass dicts but handlers expect AsyncMessage objects. (not fixed)
59. create_sigil_supervisor_instance uses fallback rag/llm but not typed. (not fixed)
60. AsyncProcessingEngine creates tasks without storing handles, causing leaks. (not fixed)
