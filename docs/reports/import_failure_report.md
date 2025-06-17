# Import Failure Analysis Report

**Scan Date:** D:\Vox\Voxsigil-Library
**Files Scanned:** 416
**Total Imports:** 4803
**Failed Imports:** 942

## Files with Import Issues

### `core\novel_reasoning\__init__.py`
**Failed imports (28):**
- `logical_neural_units.LogicalNeuralUnit`
- `logical_neural_units.LogicalReasoningEngine`
- `logical_neural_units.LogicalState`
- `logical_neural_units.LogicalRule`
- `logical_neural_units.LogicOperation`
- `logical_neural_units.DifferentiableLogicGate`
- `logical_neural_units.VariableBinding`
- `logical_neural_units.create_logical_state`
- `logical_neural_units.create_reasoning_engine`
- `kuramoto_oscillatory.AKOrNBindingNetwork`
- `kuramoto_oscillatory.KuramotoOscillator`
- `kuramoto_oscillatory.SpatialCouplingNetwork`
- `kuramoto_oscillatory.ObjectSegmentationHead`
- `kuramoto_oscillatory.OscillatorState`
- `kuramoto_oscillatory.BindingResult`
- `kuramoto_oscillatory.compute_phase_synchrony`
- `kuramoto_oscillatory.extract_synchrony_clusters`
- `kuramoto_oscillatory.create_akorn_network`
- `spiking_neural_networks.SPLRSpikingNetwork`
- `spiking_neural_networks.LIFNeuron`
- `spiking_neural_networks.SparsePlasticityRule`
- `spiking_neural_networks.SynapticLayer`
- `spiking_neural_networks.GridToSpikeEncoder`
- `spiking_neural_networks.SpikeEvent`
- `spiking_neural_networks.SpikeTrain`
- `spiking_neural_networks.LIFNeuronState`
- `spiking_neural_networks.spike_train_statistics`
- `spiking_neural_networks.create_splr_network`

### `demo_novel_paradigms.py`
**Failed imports (21):**
- `core.novel_efficiency.DeltaNetAttention`
- `core.novel_efficiency.AdaptiveMemoryManager`
- `core.novel_efficiency.DatasetManager`
- `core.novel_efficiency.create_dataset_manager`
- `core.novel_reasoning.LogicalReasoningEngine`
- `core.novel_reasoning.AKOrNBindingNetwork`
- `core.novel_reasoning.SPLRSpikingNetwork`
- `core.novel_reasoning.create_logical_state`
- `core.novel_reasoning.create_reasoning_engine`
- `core.novel_reasoning.create_akorn_network`
- `core.novel_reasoning.create_splr_network`
- `core.meta_control.EffortController`
- `core.meta_control.ComplexityMonitor`
- `core.meta_control.ComplexityLevel`
- `core.meta_control.create_effort_controller`
- `core.meta_control.create_complexity_monitor`
- `core.ensemble_integration.ARCEnsembleOrchestrator`
- `core.ensemble_integration.SPLREncoderAgent`
- `core.ensemble_integration.AKOrNBinderAgent`
- `core.ensemble_integration.LNUReasonerAgent`
- `core.ensemble_integration.create_arc_ensemble`

### `core\novel_efficiency\__init__.py`
**Failed imports (20):**
- `minicache.MiniCacheWrapper`
- `minicache.KVCacheCompressor`
- `minicache.OutlierTokenDetector`
- `deltanet_attention.DeltaNetAttention`
- `deltanet_attention.LinearAttentionConfig`
- `deltanet_attention.DeltaRuleOperator`
- `adaptive_memory.AdaptiveMemoryManager`
- `adaptive_memory.MemoryPool`
- `adaptive_memory.ResourceOptimizer`
- `dataset_manager.DatasetManager`
- `dataset_manager.DatasetRegistry`
- `dataset_manager.DatasetMetadata`
- `dataset_manager.DatasetLicense`
- `dataset_manager.LogicalGroundTruth`
- `dataset_manager.LicenseType`
- `dataset_manager.DatasetType`
- `dataset_manager.ComplianceLevel`
- `dataset_manager.create_dataset_manager`
- `minicache_blt.BLTMiniCacheWrapper`
- `minicache_blt.SemanticHashCache`

### `config\production_config.py`
**Failed imports (19):**
- `Vanta.interfaces.real_supervisor_connector.RealSupervisorConnector`
- `Vanta.core.cat_engine.CATEngine`
- `Vanta.core.cat_engine.CATEngineConfig`
- `Vanta.core.proactive_intelligence.ProactiveIntelligence`
- `Vanta.core.proactive_intelligence.ProactiveIntelligenceConfig`
- `Vanta.core.hybrid_cognition_engine.HybridCognitionConfig`
- `Vanta.core.hybrid_cognition_engine.HybridCognitionEngine`
- `Vanta.core.tot_engine.ToTEngine`
- `Vanta.core.tot_engine.ToTEngineConfig`
- `Vanta.async_stt_engine.AsyncSTTEngine`
- `Vanta.async_tts_engine.AsyncTTSEngine`
- `Vanta.async_processing_engine.AsyncProcessingEngine`
- `Vanta.core.echo_memory.EchoMemory`
- `Vanta.core.memory_braid.MemoryBraid`
- `Vanta.core.sleep_time_compute.SleepTimeCompute`
- `Vanta.interfaces.art_interface.StubARTInterface`
- `Vanta.interfaces.art_interface.create_art_interface`
- `Vanta.interfaces.art_interface.StubARTInterface`
- `Vanta.interfaces.art_interface.StubARTInterface`

### `core\meta_control\__init__.py`
**Failed imports (19):**
- `effort_controller.EffortController`
- `effort_controller.ComplexityEstimator`
- `effort_controller.EffortBudgetOptimizer`
- `effort_controller.EffortBudget`
- `effort_controller.EffortMetrics`
- `effort_controller.ComplexityLevel`
- `effort_controller.EffortAllocationStrategy`
- `effort_controller.create_effort_controller`
- `effort_controller.effort_budget_from_complexity`
- `complexity_monitor.ComplexityMonitor`
- `complexity_monitor.VisualComplexityAnalyzer`
- `complexity_monitor.LogicalComplexityAnalyzer`
- `complexity_monitor.ResourceRequirementPredictor`
- `complexity_monitor.ComplexityMeasurement`
- `complexity_monitor.ComplexityProfile`
- `complexity_monitor.ComplexityDimension`
- `complexity_monitor.ComplexityTrend`
- `complexity_monitor.create_complexity_monitor`
- `complexity_monitor.analyze_complexity_dimensions`

### `core\dialogue_manager.py`
**Failed imports (17):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.generate_internal_dialogue_message`
- `Vanta.core.UnifiedVantaCore.get_component`
- `Vanta.core.UnifiedVantaCore.publish_event`
- `Vanta.core.UnifiedVantaCore.register_component`
- `Vanta.core.UnifiedVantaCore.safe_component_call`
- `Vanta.core.UnifiedVantaCore.subscribe_to_event`
- `Vanta.core.UnifiedVantaCore.trace_event`
- `UnifiedVantaCore.generate_internal_dialogue_message`
- `UnifiedVantaCore.get_component`
- `UnifiedVantaCore.publish_event`
- `UnifiedVantaCore.register_component`
- `UnifiedVantaCore.safe_component_call`
- `UnifiedVantaCore.subscribe_to_event`
- `UnifiedVantaCore.trace_event`

### `Vanta\interfaces\__init__.py`
**Failed imports (17):**
- `base_interfaces.BaseRagInterface`
- `base_interfaces.BaseLlmInterface`
- `base_interfaces.BaseMemoryInterface`
- `base_interfaces.BaseAgentInterface`
- `base_interfaces.BaseModelInterface`
- `specialized_interfaces.MetaLearnerInterface`
- `specialized_interfaces.ModelManagerInterface`
- `specialized_interfaces.BLTInterface`
- `specialized_interfaces.ARCInterface`
- `specialized_interfaces.ARTInterface`
- `specialized_interfaces.MiddlewareInterface`
- `blt_encoder_interface.BaseBLTEncoder`
- `hybrid_middleware_interface.BaseHybridMiddleware`
- `supervisor_connector_interface.BaseSupervisorConnector`
- `protocol_interfaces.VantaProtocol`
- `protocol_interfaces.ModuleAdapterProtocol`
- `protocol_interfaces.IntegrationProtocol`

### `core\ensemble_integration\arc_ensemble_orchestrator.py`
**Failed imports (14):**
- `novel_efficiency.MiniCacheWrapper`
- `novel_efficiency.DeltaNetAttention`
- `novel_efficiency.AdaptiveMemoryManager`
- `novel_efficiency.DatasetManager`
- `novel_reasoning.LogicalReasoningEngine`
- `novel_reasoning.AKOrNBindingNetwork`
- `novel_reasoning.SPLRSpikingNetwork`
- `meta_control.EffortController`
- `meta_control.ComplexityMonitor`
- `meta_control.EffortBudget`
- `meta_control.ComplexityMeasurement`
- `meta_control.create_effort_controller`
- `meta_control.create_complexity_monitor`
- `novel_reasoning.logical_neural_units.create_logical_state`

### `ART\adapter.py`
**Failed imports (13):**
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`
- `Vanta.interfaces.base_interfaces.BaseMemoryInterface`
- `Vanta.interfaces.base_interfaces.BaseRagInterface`
- `art_manager.ARTManager`
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`
- `Voxsigil_Library.Scaffolds.scaffold_router.ScaffoldRouter`
- `Voxsigil_Library.Scaffolds.scaffold_router.ScaffoldRouter`
- `Vanta.interfaces.memory_interface.BaseMemoryInterface`
- `Vanta.interfaces.rag_interface.BaseRagInterface`
- `art_manager.ARTManager`
- `art_logger.get_art_logger`
- `core.vanta_core.VantaCore`
- `core.vanta_core.VantaCore`

### `engines\hybrid_cognition_engine.py`
**Failed imports (13):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`
- `cat_engine.CATEngine`
- `tot_engine.ToTEngine`
- `cat_engine.CATEngineConfig`
- `tot_engine.BranchEvaluator`
- `tot_engine.BranchValidator`
- `tot_engine.ContextProvider`
- `tot_engine.MetaLearningAgent`
- `tot_engine.ThoughtSeeder`
- `tot_engine.ToTEngineConfig`

### `VoxSigilRag\__init__.py`
**Failed imports (13):**
- `voxsigil_rag.VoxSigilRAG`
- `voxsigil_evaluator.VoxSigilResponseEvaluator`
- `voxsigil_evaluator.VoxSigilConfig`
- `voxsigil_evaluator.VoxSigilError`
- `voxsigil_rag_compression.RAGCompressionEngine`
- `voxsigil_rag_compression.RAGCompressionError`
- `voxsigil_blt.ByteLatentTransformerEncoder`
- `voxsigil_blt.BLTEnhancedMiddleware`
- `sigil_patch_encoder.SigilPatchEncoder`
- `hybrid_blt.HybridMiddleware`
- `voxsigil_semantic_cache.SemanticCacheManager`
- `hybrid_blt.HybridMiddleware`
- `sigil_patch_encoder.SigilPatchEncoder`

### `engines\cat_engine.py`
**Failed imports (12):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`
- `Vanta.interfaces.specialized_interfaces.MetaLearnerInterface`
- `Vanta.interfaces.specialized_interfaces.ModelManagerInterface`
- `Vanta.interfaces.protocol_interfaces.MemoryBraidInterface`
- `Vanta.core.fallback_implementations.FallbackLlmInterface`
- `Vanta.core.fallback_implementations.FallbackMemoryInterface`
- `Vanta.core.fallback_implementations.FallbackRagInterface`

### `integration\voxsigil_integration.py`
**Failed imports (12):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`
- `Vanta.interfaces.memory_interface.BaseMemoryInterface`
- `Vanta.interfaces.rag_interface.BaseRagInterface`
- `Vanta.async_training_engine.AsyncTrainingEngine`
- `Vanta.async_training_engine.TrainingConfig`
- `Vanta.async_training_engine.TrainingJob`
- `Vanta.interfaces.checkin_manager_vosk.VantaInteractionManager`
- `Vanta.interfaces.learning_manager.LearningManager`
- `Vanta.interfaces.memory_interface.JsonFileMemoryInterface`
- `Vanta.interfaces.model_manager.ModelManager`
- `Vanta.interfaces.rag_interface.SupervisorRagInterface`

### `ARC\arc_comparison.py`
**Failed imports (11):**
- `arc_config.ARC_DATA_DIR`
- `arc_config.LMSTUDIO_API_BASE_URL`
- `arc_config.OLLAMA_API_BASE_URL`
- `arc_config.RESULTS_OUTPUT_DIR`
- `arc_config.USE_LLM_CACHE`
- `arc_config.VoxSigilComponent`
- `arc_config.analyze_task_for_categorization_needs`
- `arc_config.initialize_and_validate_models_config`
- `arc_config.load_llm_response_cache`
- `arc_config.load_voxsigil_entries`
- `arc_config.save_llm_response_cache`

### `core\ensemble_integration\__init__.py`
**Failed imports (11):**
- `arc_ensemble_orchestrator.ARCEnsembleOrchestrator`
- `arc_ensemble_orchestrator.AgentContract`
- `arc_ensemble_orchestrator.SPLREncoderAgent`
- `arc_ensemble_orchestrator.AKOrNBinderAgent`
- `arc_ensemble_orchestrator.LNUReasonerAgent`
- `arc_ensemble_orchestrator.ConsensusBuilder`
- `arc_ensemble_orchestrator.ProcessingStage`
- `arc_ensemble_orchestrator.EnsembleStrategy`
- `arc_ensemble_orchestrator.ProcessingResult`
- `arc_ensemble_orchestrator.EnsembleState`
- `arc_ensemble_orchestrator.create_arc_ensemble`

### `ART\__init__.py`
**Failed imports (10):**
- `art_controller.ARTController`
- `art_manager.ARTManager`
- `art_trainer.ArtTrainer`
- `art_adapter.ArtAdapter`
- `art_adapter.create_art_adapter`
- `generative_art.GenerativeArt`
- `art_entropy_bridge.ArtEntropyBridge`
- `art_hybrid_blt_bridge.ARTHybridBLTBridge`
- `art_rag_bridge.ARTRAGBridge`
- `art_logger.ARTLogger`

### `training\vanta_registration.py`
**Failed imports (10):**
- `Vanta.integration.module_adapters.module_registry`
- `training.rag_interface.SupervisorRagInterface`
- `training.rag_interface.SimpleRagInterface`
- `training.rag_interface.VOXSIGIL_RAG_AVAILABLE`
- `Vanta.core.fallback_implementations.FallbackRagInterface`
- `training.rag_interface.SupervisorRagInterface`
- `training.rag_interface.SimpleRagInterface`
- `training.rag_interface.VOXSIGIL_RAG_AVAILABLE`
- `Vanta.core.fallback_implementations.FallbackRagInterface`
- `Vanta.integration.module_adapters.ClassBasedAdapter`

### `engines\async_training_engine.py`
**Failed imports (9):**
- `datasets`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`
- `datasets.Dataset`
- `datasets.dataset_dict.DatasetDict`
- `datasets.iterable_dataset.IterableDataset`

### `Vanta\core\UnifiedVantaCore.py`
**Failed imports (9):**
- `Vanta.interfaces.blt_encoder_interface.BaseBLTEncoder`
- `Vanta.interfaces.hybrid_middleware_interface.BaseHybridMiddleware`
- `Vanta.interfaces.supervisor_connector_interface.BaseSupervisorConnector`
- `UnifiedAgentRegistry.UnifiedAgentRegistry`
- `UnifiedAsyncBus.UnifiedAsyncBus`
- `voxsigil_mesh.VoxSigilMesh`
- `vanta_mesh_graph.VantaMeshGraph`
- `handlers.speech_integration_handler.initialize_speech_system`
- `handlers.vmb_integration_handler.initialize_vmb_system`

### `gui\components\vmb_gui_launcher.py`
**Failed imports (9):**
- `vmb_activation.CopilotSwarm`
- `vmb_config_status.VMBCompletionReport`
- `vmb_config_status.VMBStatus`
- `vmb_config_status.VMBSwarmConfig`
- `vmb_config_status.VMBSystemStatus`
- `vmb_production_executor.ProductionTaskExecutor`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `vmb_status.generate_vmb_status_report`
- `gui_utils.bind_agent_buttons`

### `ART\art_manager.py`
**Failed imports (8):**
- `art_controller.ARTController`
- `art_logger.get_art_logger`
- `core.base_agent.VantaAgentCapability`
- `art_trainer.ArtTrainer`
- `generative_art.GenerativeArt`
- `art_entropy_bridge.ArtEntropyBridge`
- `art_rag_bridge.ARTRAGBridge`
- `blt.art_blt_bridge.ARTBLTBridge`

### `core\grid_distillation.py`
**Failed imports (8):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.base_core.BaseCore`
- `Vanta.core.cognitive_mesh.CognitiveMesh`
- `Vanta.core.cognitive_mesh.vanta_core_module`
- `grid_former.GRID_Former`
- `training.grid_model_trainer.GridFormerTrainer`
- `arc_llm_handler.ARCAwareLLMInterface`
- `arc_llm_handler.ARCAwareLLMInterface`

### `core\grid_former_evaluator.py`
**Failed imports (8):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `inference.GridFormerInference`
- `inference.InferenceStrategy`
- `model_utils.ModelLoader`
- `model_utils.discover_models`
- `model_utils.get_latest_models`

### `core\proactive_intelligence.py`
**Failed imports (8):**
- `Vanta.interfaces.specialized_interfaces.ModelManagerInterface`
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `core.learning_manager.DefaultModelManager`

### `handlers\speech_integration_handler.py`
**Failed imports (8):**
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`
- `Vanta.async_tts_engine.AsyncTTSEngine`
- `Vanta.async_tts_engine.TTSConfig`
- `Vanta.async_tts_engine.create_async_tts_engine`
- `Vanta.async_stt_engine.AsyncSTTEngine`
- `Vanta.async_stt_engine.STTConfig`

### `monitoring\exporter.py`
**Failed imports (8):**
- `prometheus_client`
- `GPUtil`
- `prometheus_client.Gauge`
- `prometheus_client.Counter`
- `prometheus_client.Histogram`
- `prometheus_client.Info`
- `prometheus_client.CollectorRegistry`
- `prometheus_client.core.REGISTRY`

### `Vanta\core\__init__.py`
**Failed imports (8):**
- `UnifiedVantaCore.UnifiedVantaCore`
- `UnifiedVantaCore.get_vanta_core`
- `UnifiedAgentRegistry.UnifiedAgentRegistry`
- `UnifiedAsyncBus.UnifiedAsyncBus`
- `UnifiedMemoryInterface.UnifiedMemoryInterface`
- `VantaBLTMiddleware.VantaBLTMiddleware`
- `VantaCognitiveEngine.VantaCognitiveEngine`
- `VantaOrchestrationEngine.VantaOrchestrationEngine`

### `Vanta\integration\vanta_supervisor.py`
**Failed imports (8):**
- `Vanta.interfaces.memory_interface.BaseMemoryInterface`
- `Vanta.interfaces.rag_interface.BaseRagInterface`
- `Voxsigil_Library.Scaffolds.scaffold_router.ScaffoldRouter`
- `Vanta.core.sleep_time_compute.CognitiveState`
- `Vanta.core.sleep_time_compute.SleepTimeCompute`
- `Vanta.interfaces.learning_manager.LearningManager`
- `Vanta.core.fallback_implementations.FallbackLlmInterface`
- `Vanta.core.fallback_implementations.FallbackMemoryInterface`

### `core\hyperparameter_search.py`
**Failed imports (7):**
- `Vanta.core.VantaCore.BaseCore`
- `Vanta.core.VantaCore.VantaCore`
- `Vanta.core.holo15_core.vanta_core_module`
- `Vanta.core.holo15_core.CognitiveRole`
- `Vanta.core.recursive_symbolic_mesh.SymbolicNode`
- `Vanta.core.recursive_symbolic_mesh.RecursiveProcessor`
- `Vanta.core.recursive_symbolic_mesh.CognitiveMesh`

### `core\hyperparameter_search_enhanced.py`
**Failed imports (7):**
- `Vanta.core.VantaCore.BaseCore`
- `Vanta.core.VantaCore.VantaCore`
- `Vanta.core.holo15_core.vanta_core_module`
- `Vanta.core.holo15_core.CognitiveRole`
- `Vanta.core.recursive_symbolic_mesh.SymbolicNode`
- `Vanta.core.recursive_symbolic_mesh.RecursiveProcessor`
- `Vanta.core.recursive_symbolic_mesh.CognitiveMesh`

### `handlers\rag_integration_handler.py`
**Failed imports (7):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`
- `Vanta.interfaces.base_interfaces.BaseRagInterface`
- `Vanta.core.fallback_implementations.FallbackRagInterface`
- `Vanta.interfaces.rag_interface.VOXSIGIL_RAG_AVAILABLE`
- `Vanta.interfaces.rag_interface.SimpleRagInterface`
- `Vanta.interfaces.rag_interface.SupervisorRagInterface`

### `training\arc_grid_trainer.py`
**Failed imports (7):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Voxsigil_Library.ARC.core.arc_data_processor.ARCGridDataProcessor`
- `Voxsigil_Library.ARC.core.arc_data_processor.create_arc_dataloaders`
- `Voxsigil_Library.Gridformer.core.grid_former.GRID_Former`
- `Voxsigil_Library.Gridformer.core.vantacore_grid_connector.GridFormerConnector`
- `Voxsigil_Library.Gridformer.training.grid_model_trainer.GridFormerTrainer`
- `Voxsigil_Library.Gridformer.training.grid_sigil_handler.GridSigilHandler`

### `test_complete_registration.py`
**Failed imports (6):**
- `Vanta.registration.get_registration_status`
- `Vanta.integration.module_adapters.module_registry`
- `Vanta.core.orchestrator.vanta_orchestrator`
- `Vanta.integration.module_adapters.BaseModuleAdapter`
- `Vanta.integration.module_adapters.ClassBasedAdapter`
- `Vanta.integration.module_adapters.ModuleRegistry`

### `validate_production_readiness.py`
**Failed imports (6):**
- `core.safety.canary_validator.CanaryGridValidator`
- `core.deployment.shadow_mode.initialize_shadow_mode`
- `core.deployment.shadow_mode.ShadowModeConfig`
- `core.explainability.reasoning_traces.initialize_trace_capture`
- `vanta_cli.VantaCLI`
- `core.explainability.reasoning_traces.ReasoningStepType`

### `agents\sleep_time_compute_agent.py`
**Failed imports (6):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `core.vanta_core.VantaCore`

### `agents\voxagent.py`
**Failed imports (6):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`
- `blt_compression_middleware.compress_outbound`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`

### `ARC\arc_reasoner.py`
**Failed imports (6):**
- `arc_config.ARC_SOLVER_SIGIL_NAME`
- `arc_config.CATENGINE_SIGIL_NAME`
- `arc_config.LLM_SYNTHESIZER_TEMPERATURE`
- `arc_config.DEFAULT_STRATEGY_PRIORITY`
- `arc_config.SYNTHESIS_FAILURE_FALLBACK_STRATEGY`
- `arc_config.DETAILED_PROMPT_METADATA`

### `ART\art_adapter.py`
**Failed imports (6):**
- `art_controller.ARTController`
- `art_trainer.ArtTrainer`
- `art_logger.get_art_logger`
- `adapter.VANTAFactory`
- `art_entropy_bridge.ArtEntropyBridge`
- `art_rag_bridge.ARTRAGBridge`

### `core\AdvancedMetaLearner.py`
**Failed imports (6):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.interfaces.specialized_interfaces.MetaLearnerInterface`
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.VantaCore`

### `core\evolutionary_optimizer.py`
**Failed imports (6):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `deap.creator`
- `deap.base`
- `deap.tools`

### `handlers\arc_llm_handler.py`
**Failed imports (6):**
- `arc_utils.LLM_RESPONSE_CACHE`
- `arc_voxsigil_loader.load_voxsigil_system_prompt`
- `arc_utils.get_cached_response`
- `arc_utils.LLM_RESPONSE_CACHE`
- `arc_utils.cache_response`
- `arc_voxsigil_loader.load_voxsigil_system_prompt`

### `handlers\vanta_registration.py`
**Failed imports (6):**
- `Vanta.get_vanta_core_instance`
- `Vanta.get_vanta_core_instance`
- `handlers.arc_llm_handler.ARCLLMHandler`
- `handlers.rag_integration_handler.RagIntegrationHandler`
- `handlers.speech_integration_handler.SpeechIntegrationHandler`
- `handlers.vmb_integration_handler.VMBIntegrationHandler`

### `interfaces\__init__.py`
**Failed imports (6):**
- `Vanta.interfaces.BaseRagInterface`
- `Vanta.interfaces.BaseLlmInterface`
- `Vanta.interfaces.BaseMemoryInterface`
- `rag_interface.BaseRagInterface`
- `llm_interface.BaseLlmInterface`
- `memory_interface.BaseMemoryInterface`

### `training\phi2_finetune.py`
**Failed imports (6):**
- `peft.prepare_model_for_kbit_training`
- `peft.LoraConfig`
- `peft.get_peft_model`
- `peft.LoraConfig`
- `peft.get_peft_model`
- `peft.prepare_model_for_kbit_training`

### `training\tinyllama_voxsigil_finetune.py`
**Failed imports (6):**
- `peft.prepare_model_for_kbit_training`
- `peft.LoraConfig`
- `peft.get_peft_model`
- `peft.prepare_model_for_kbit_training`
- `peft.LoraConfig`
- `peft.get_peft_model`

### `Vanta\integration\vanta_orchestrator.py`
**Failed imports (6):**
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`
- `Vanta.interfaces.base_interfaces.BaseMemoryInterface`
- `Vanta.interfaces.base_interfaces.BaseRagInterface`
- `Vanta.core.sleep_time_compute.CognitiveState`
- `Vanta.core.sleep_time_compute.SleepTimeCompute`

### `Vanta\integration\vanta_orchestrator_clean.py`
**Failed imports (6):**
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`
- `Vanta.interfaces.base_interfaces.BaseMemoryInterface`
- `Vanta.interfaces.base_interfaces.BaseRagInterface`
- `Vanta.core.sleep_time_compute.CognitiveState`
- `Vanta.core.sleep_time_compute.SleepTimeCompute`

### `Vanta\registration\master_registration.py`
**Failed imports (6):**
- `Vanta.tabletop`
- `Vanta.integration.module_adapters.module_registry`
- `Vanta.core.orchestrator.vanta_orchestrator`
- `engines.vanta_registration.register_engines`
- `core.vanta_registration.register_core_modules`
- `memory.vanta_registration.register_memory_modules`

### `tests\regression\test_arc_batch.py`
**Failed imports (6):**
- `core.ensemble_integration.ARCEnsembleOrchestrator`
- `core.ensemble_integration.create_arc_ensemble`
- `core.novel_reasoning.create_reasoning_engine`
- `core.meta_control.create_effort_controller`
- `core.meta_control.ComplexityLevel`
- `demo_novel_paradigms.ARCTaskGenerator`

### `agents\ensemble\music\music_composer_agent.py`
**Failed imports (6):**
- `librosa`
- `soundfile`
- `core.vanta_core.VantaCore`
- `core.vanta_core.BaseCore`
- `training.music.blt_reindex.BLTMusicReindexer`
- `training.music.blt_reindex.BLTFineTuneConfig`

### `core\enhanced_grid_connector.py`
**Failed imports (5):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `grid_former.GRID_Former`

### `core\learning_manager.py`
**Failed imports (5):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `core.UnifiedVantaCore.UnifiedVantaCore`

### `core\meta_cognitive.py`
**Failed imports (5):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `MetaConsciousness.core.context.SDKContext`
- `MetaConsciousness.core.context.SDKContext`

### `core\model_manager.py`
**Failed imports (5):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.trace_event`

### `core\vantacore_grid_former_integration.py`
**Failed imports (5):**
- `Vanta.integration.vantacore_grid_former_integration.GridFormerVantaIntegration`
- `Vanta.integration.vantacore_grid_former_integration.HybridARCSolver`
- `Vanta.integration.vantacore_grid_former_integration.integrate_with_vantacore`
- `Vanta.integration.vantacore_grid_former_integration.main`
- `Vanta.integration.vantacore_grid_former_integration.parse_arguments`

### `engines\async_stt_engine.py`
**Failed imports (5):**
- `aiofiles`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`

### `engines\async_tts_engine.py`
**Failed imports (5):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`

### `engines\tot_engine.py`
**Failed imports (5):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`
- `Vanta.interfaces.protocol_interfaces.MemoryBraidInterface`

### `llm\register_llm_module.py`
**Failed imports (5):**
- `arc_llm_bridge.ARCLLMBridge`
- `arc_utils.ARCUtils`
- `arc_voxsigil_loader.ARCVoxSigilLoader`
- `llm_api_compat.LLMAPICompat`
- `main.MainLLM`

### `llm\vanta_registration.py`
**Failed imports (5):**
- `arc_llm_bridge.ARCLLMBridge`
- `arc_utils.ARCUtils`
- `arc_voxsigil_loader.ARCVoxSigilLoader`
- `llm_api_compat.LLMAPICompat`
- `main.MainLLMHandler`

### `memory\vanta_registration.py`
**Failed imports (5):**
- `Vanta.get_vanta_core_instance`
- `Vanta.get_vanta_core_instance`
- `memory.echo_memory.EchoMemory`
- `memory.external_echo_layer.ExternalEchoLayer`
- `memory.memory_braid.MemoryBraid`

### `middleware\hybrid_middleware.py`
**Failed imports (5):**
- `Vanta.interfaces.blt_encoder_interface.BaseBLTEncoder`
- `Vanta.interfaces.hybrid_middleware_interface.BaseHybridMiddleware`
- `Vanta.interfaces.supervisor_connector_interface.BaseSupervisorConnector`
- `hybrid_blt.HybridMiddleware`
- `hybrid_blt.HybridMiddlewareConfig`

### `middleware\vanta_registration.py`
**Failed imports (5):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`
- `Vanta.get_vanta_core_instance`
- `Vanta.get_vanta_core_instance`

### `voxsigil_supervisor\vanta_registration.py`
**Failed imports (5):**
- `Vanta.get_vanta_core_instance`
- `supervisor_engine.SupervisorEngine`
- `supervisor_engine_compat.SupervisorEngineCompat`
- `supervisor_wrapper.SupervisorWrapper`
- `blt_supervisor_integration.BLTSupervisorIntegration`

### `Vanta\core\UnifiedMemoryInterface.py`
**Failed imports (5):**
- `echo_memory.EchoMemory`
- `memory_braid.MemoryBraid`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `Vanta.interfaces.memory_interface.JsonFileMemoryInterface`

### `Vanta\interfaces\specialized_interfaces.py`
**Failed imports (5):**
- `base_interfaces.BaseRagInterface`
- `base_interfaces.BaseLlmInterface`
- `base_interfaces.BaseMemoryInterface`
- `base_interfaces.BaseAgentInterface`
- `base_interfaces.BaseModelInterface`

### `training\music\blt_reindex.py`
**Failed imports (5):**
- `core.vanta_core.VantaCore`
- `core.vanta_core.BaseCore`
- `core.vanta_core.vanta_core_module`
- `core.vanta_core.CognitiveMeshRole`
- `core.vanta_core.VantaCore`

### `gui\components\pyqt_main.py`
**Failed imports (5):**
- `echo_log_panel.EchoLogPanel`
- `mesh_map_panel.MeshMapPanel`
- `agent_status_panel.AgentStatusPanel`
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`

### `BLT\blt_supervisor_integration.py`
**Failed imports (4):**
- `Vanta.interfaces.base_interfaces.BaseRagInterface`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`
- `Scaffolds.scaffold_router.ScaffoldRouter`

### `core\checkin_manager_vosk.py`
**Failed imports (4):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`

### `core\communication_orchestrator.py`
**Failed imports (4):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.core.UnifiedVantaCore.VantaCore`

### `core\vanta_registration.py`
**Failed imports (4):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`
- `Vanta.get_vanta_core_instance`

### `core\__init__.py`
**Failed imports (4):**
- `Vanta.integration.vantacore_grid_connector.GridFormerConnector`
- `Vanta.integration.vantacore_grid_former_integration.GridFormerVantaIntegration`
- `enhanced_grid_connector.EnhancedGridFormerConnector`
- `grid_former.GRID_Former`

### `engines\async_processing_engine.py`
**Failed imports (4):**
- `Vanta.core.UnifiedAsyncBus.MessageType`
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`

### `handlers\vmb_integration_handler.py`
**Failed imports (4):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`
- `Vanta.vmb.vmb_activation.CopilotSwarm`
- `Vanta.vmb.vmb_production_final.ProductionTaskExecutor`

### `handlers\__init__.py`
**Failed imports (4):**
- `arc_llm_handler.ARCLLMHandler`
- `rag_integration_handler.RagIntegrationHandler`
- `speech_integration_handler.SpeechIntegrationHandler`
- `vmb_integration_handler.VMBIntegrationHandler`

### `strategies\__init__.py`
**Failed imports (4):**
- `scaffold_router.ScaffoldRouter`
- `evaluation_heuristics.ResponseEvaluator`
- `retry_policy.RetryPolicy`
- `execution_strategy.BaseExecutionStrategy`

### `vmb\vanta_registration.py`
**Failed imports (4):**
- `vmb_operations.VMBOperations`
- `vmb_activation.VMBActivation`
- `vmb_production_executor.VMBProductionExecutor`
- `vmb_advanced_demo.VMBAdvancedDemo`

### `vmb\vmb_import_test.py`
**Failed imports (4):**
- `vmb_completion_report`
- `vmb_activation.CopilotSwarm`
- `vmb_production_executor.ProductionTaskExecutor`
- `vmb_activation.CopilotSwarm`

### `voxsigil_supervisor\strategies\__init__.py`
**Failed imports (4):**
- `scaffold_router.ScaffoldRouter`
- `evaluation_heuristics.ResponseEvaluator`
- `retry_policy.RetryPolicy`
- `execution_strategy.BaseExecutionStrategy`

### `Vanta\core\VantaCognitiveEngine.py`
**Failed imports (4):**
- `Vanta.interfaces.supervisor_connector_interface.BaseSupervisorConnector`
- `Vanta.interfaces.blt_encoder_interface.BaseBLTEncoder`
- `Vanta.interfaces.hybrid_middleware_interface.BaseHybridMiddleware`
- `Vanta.interfaces.real_supervisor_connector.RealSupervisorConnector`

### `Vanta\integration\art_integration_example.py`
**Failed imports (4):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.interfaces.rag_interface.BaseRagInterface`
- `Vanta.interfaces.memory_interface.BaseMemoryInterface`
- `Vanta.interfaces.model_manager.ModelManager`

### `Vanta\integration\vanta_integration.py`
**Failed imports (4):**
- `Vanta.async_training_engine.AsyncTrainingEngine`
- `Vanta.async_training_engine.TrainingConfig`
- `Vanta.async_training_engine.TrainingJob`
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`

### `Vanta\integration\vanta_runner.py`
**Failed imports (4):**
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`
- `Vanta.integration.vanta_supervisor.VANTA_SYSTEM_PROMPT`
- `Vanta.core.fallback_implementations.FallbackRagInterface`
- `Vanta.core.fallback_implementations.FallbackMemoryInterface`

### `Vanta\registration\__init__.py`
**Failed imports (4):**
- `master_registration.register_all_modules`
- `master_registration.get_registration_status`
- `master_registration.registration_orchestrator`
- `master_registration.RegistrationOrchestrator`

### `agents\ensemble\music\voice_modulator_agent.py`
**Failed imports (4):**
- `librosa`
- `soundfile`
- `core.vanta_core.VantaCore`
- `core.vanta_core.BaseCore`

### `tinyllama_assistant.py`
**Failed imports (3):**
- `peft.LoraConfig`
- `peft.get_peft_model`
- `peft.prepare_model_for_kbit_training`

### `vanta_cli.py`
**Failed imports (3):**
- `demo_novel_paradigms.main`
- `core.safety.canary_validator.main`
- `core.deployment.shadow_mode.main`

### `__init__.py`
**Failed imports (3):**
- `enhanced_testing_interface.EnhancedVoxSigilTestingInterface`
- `voxsigil_integration.get_voxsigil_integration`
- `voxsigil_integration.initialize_voxsigil_integration`

### `agents\andy.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\astra.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\bridgeflesh.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\carla.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\codeweaver.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\dave.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\dreamer.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\echo.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\echolore.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\entropybard.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\evo.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\game_master_agent.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\gizmo.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\mirrorwarden.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\nebula.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\nix.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\oracle.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\orion.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\orionapprentice.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\phi.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\pulsesmith.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\rules_ref_agent.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\sam.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\sdkcontext.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\socraticengine.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\vanta_registration.py`
**Failed imports (3):**
- `Vanta.get_vanta_core_instance`
- `Vanta.get_vanta_core_instance`
- `base.BaseAgent`

### `agents\voice_table_agent.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\voxka.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\warden.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `agents\wendy.py`
**Failed imports (3):**
- `base.BaseAgent`
- `base.vanta_agent`
- `base.CognitiveMeshRole`

### `ART\art_controller.py`
**Failed imports (3):**
- `art_logger.get_art_logger`
- `core.base_agent.VantaAgentCapability`
- `art_manager.ARTManager`

### `ART\test_art_controller.py`
**Failed imports (3):**
- `art_controller.ARTController`
- `art_logger.get_art_logger`
- `art_manager.ARTManager`

### `BLT\blt_enhanced_extension.py`
**Failed imports (3):**
- `voxsigil_rag.VoxSigilRAG`
- `blt_encoder.ByteLatentTransformerEncoder`
- `blt_encoder.SigilPatchEncoder`

### `BLT\vanta_registration.py`
**Failed imports (3):**
- `Vanta.integration.module_adapters.LegacyModuleAdapter`
- `Vanta.integration.module_adapters.module_registry`
- `Vanta.integration.module_adapters.ClassBasedAdapter`

### `core\download_arc_data.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\end_to_end_arc_validation.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\evo_nas.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\iterative_gridformer.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\iterative_reasoning_gridformer.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\model_architecture.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.CognitiveMeshRole`
- `base.vanta_core_module`

### `core\model_architecture_fixer.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `core\neuro_symbolic_network.py`
**Failed imports (3):**
- `base.BaseCore`
- `base.vanta_core_module`
- `base.CognitiveMeshRole`

### `engines\rag_compression_engine.py`
**Failed imports (3):**
- `base.BaseEngine`
- `base.vanta_engine`
- `base.CognitiveMeshRole`

### `gui\launcher.py`
**Failed imports (3):**
- `Vanta.async_training_engine.AsyncTrainingEngine`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`
- `Vanta.integration.vmb_integration_handler.VMBIntegrationHandler`

### `llm\arc_llm_bridge.py`
**Failed imports (3):**
- `Voxsigil_Library.Scaffolds.scaffold_router.ScaffoldRouter`
- `Vanta.core.echo_memory.EchoMemory`
- `Vanta.core.memory_braid.MemoryBraid`

### `memory\__init__.py`
**Failed imports (3):**
- `echo_memory.EchoMemory`
- `external_echo_layer.ExternalEchoLayer`
- `memory_braid.MemoryBraid`

### `monitoring\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `rules\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `scaffolds\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `schema\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `sigils\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `strategies\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `tags\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `training\mistral_finetune.py`
**Failed imports (3):**
- `peft.prepare_model_for_kbit_training`
- `peft.LoraConfig`
- `peft.get_peft_model`

### `utils\path_helper.py`
**Failed imports (3):**
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`
- `Vanta.interfaces.real_supervisor_connector.RealSupervisorConnector`
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`

### `utils\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `Vanta\vanta_registration.py`
**Failed imports (3):**
- `core.base.BaseCore`
- `core.base.CognitiveMeshRole`
- `core.base.vanta_core_module`

### `Vanta\__init__.py`
**Failed imports (3):**
- `core.orchestrator.vanta_orchestrator`
- `core.fallback_implementations.initialize_fallbacks`
- `core.orchestrator.VantaClient`

### `vmb\register_vmb_module.py`
**Failed imports (3):**
- `vmb_activation.VMBActivation`
- `vmb_operations.VMBOperations`
- `vmb_status.VMBStatus`

### `voxsigil_supervisor\utils\__init__.py`
**Failed imports (3):**
- `validation_utils.validate_payload_structure`
- `validation_utils.check_llm_message_format`
- `sigil_formatting.format_sigil_detail`

### `Vanta\integration\art_blt_bridge.py`
**Failed imports (3):**
- `art_logger.get_art_logger`
- `art_controller.ARTManager`
- `Voxsigil_Library.VoxSigilRag.sigil_patch_encoder.SigilPatchEncoder`

### `core\safety\canary_validator.py`
**Failed imports (3):**
- `core.ensemble_integration.ARCEnsembleOrchestrator`
- `core.ensemble_integration.create_arc_ensemble`
- `core.meta_control.ComplexityMonitor`

### `ARC\core\arc_data_processor.py`
**Failed imports (3):**
- `arc_data_processor.ARCGridDataProcessor`
- `arc_data_processor.create_arc_dataloaders`
- `arc_data_processor.visualize_grid`

### `validate.py`
**Failed imports (2):**
- `VoxSigilDatasetTools.generator`
- `VoxSigilDatasetTools.field_generators.utils.export_utils`

### `agents\base.py`
**Failed imports (2):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`

### `agents\__init__.py`
**Failed imports (2):**
- `base.BaseAgent`
- `base.NullAgent`

### `ARC\arc_task_example.py`
**Failed imports (2):**
- `Vanta.integration.vanta_integration.create_vanta_supervisor`
- `Vanta.integration.vanta_supervisor.VantaSigilSupervisor`

### `ART\art_blt_bridge.py`
**Failed imports (2):**
- `art_manager.ARTManager`
- `art_logger.get_art_logger`

### `ART\art_hybrid_blt_bridge.py`
**Failed imports (2):**
- `art_logger.get_art_logger`
- `art_manager.ARTManager`

### `ART\art_rag_bridge.py`
**Failed imports (2):**
- `art_logger.get_art_logger`
- `Voxsigil_Library.VoxSigilRag.voxsigil_blt_rag.BLTEnhancedRAG`

### `ART\art_trainer.py`
**Failed imports (2):**
- `art_controller.ARTController`
- `art_logger.get_art_logger`

### `integration\vanta_registration.py`
**Failed imports (2):**
- `real_supervisor_connector.RealSupervisorConnector`
- `voxsigil_integration.VoxSigilIntegration`

### `llm\llm_api_compat.py`
**Failed imports (2):**
- `arc_llm_handler._llm_call_api_internal`
- `arc_llm_handler._llm_call_api_internal`

### `llm\__init__.py`
**Failed imports (2):**
- `register_llm_module.register_llm`
- `register_llm_module.LLMModuleAdapter`

### `memory\echo_memory.py`
**Failed imports (2):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`

### `memory\external_echo_layer.py`
**Failed imports (2):**
- `UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.interfaces.protocol_interfaces.MemoryBraidInterface`

### `memory\memory_braid.py`
**Failed imports (2):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`

### `services\memory_service_connector.py`
**Failed imports (2):**
- `Vanta.core.UnifiedMemoryInterface.UnifiedMemoryInterface`
- `Vanta.core.UnifiedVantaCore.get_vanta_core`

### `services\vanta_registration.py`
**Failed imports (2):**
- `Vanta.get_vanta_core_instance`
- `Vanta.get_vanta_core_instance`

### `training\rag_interface.py`
**Failed imports (2):**
- `Vanta.interfaces.BaseRagInterface`
- `Vanta.interfaces.base_interfaces.BaseRagInterface`

### `training\__init__.py`
**Failed imports (2):**
- `arc_grid_trainer.ARCGridTrainer`
- `arc_grid_trainer.VantaGridFormerBridge`

### `utils\sleep_time_compute.py`
**Failed imports (2):**
- `Vanta.core.UnifiedAsyncBus.AsyncMessage`
- `Vanta.core.UnifiedAsyncBus.MessageType`

### `Vanta\core\enhanced_gridformer_manager.py`
**Failed imports (2):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`
- `Vanta.core.UnifiedVantaCore.get_unified_core`

### `tools\utilities\model_utils.py`
**Failed imports (2):**
- `tensorflow`
- `onnxruntime`

### `tools\utilities\model_utils_fixed.py`
**Failed imports (2):**
- `tensorflow`
- `onnxruntime`

### `core\deployment\shadow_mode.py`
**Failed imports (2):**
- `core.ensemble_integration.ARCEnsembleOrchestrator`
- `core.ensemble_integration.create_arc_ensemble`

### `agents\ensemble\music\music_sense_agent.py`
**Failed imports (2):**
- `demucs.pretrained`
- `crepe`

### `quick_setup.py`
**Failed imports (1):**
- `install_with_uv`

### `agents\enhanced_vanta_registration.py`
**Failed imports (1):**
- `Vanta.get_vanta_core_instance`

### `agents\holo_mesh.py`
**Failed imports (1):**
- `peft.PeftModel`

### `ARC\arc_grid_former_pipeline.py`
**Failed imports (1):**
- `core.arc_data_processor.create_arc_dataloaders`

### `ARC\arc_grid_former_runner.py`
**Failed imports (1):**
- `core.arc_data_processor.create_arc_dataloaders`

### `ARC\arc_integration.py`
**Failed imports (1):**
- `arc_reasoner.ARCReasoner`

### `ARC\vanta_registration.py`
**Failed imports (1):**
- `Vanta.get_vanta_core_instance`

### `ART\art_entropy_bridge.py`
**Failed imports (1):**
- `art_logger.get_art_logger`

### `ART\art_logger.py`
**Failed imports (1):**
- `core.vanta_core.VantaCore`

### `ART\train_art_with_local_llm.py`
**Failed imports (1):**
- `blt.art_blt_bridge.ARTBLTBridge`

### `BLT\__init__.py`
**Failed imports (1):**
- `Vanta.interfaces.blt_encoder_interface.BaseBLTEncoder`

### `core\vantacore_grid_connector.py`
**Failed imports (1):**
- `Vanta.integration.vantacore_grid_connector.GridFormerConnector`

### `engines\vanta_registration.py`
**Failed imports (1):**
- `Vanta.get_vanta_core_instance`

### `integration\real_supervisor_connector.py`
**Failed imports (1):**
- `Vanta.interfaces.supervisor_connector_interface.BaseSupervisorConnector`

### `interfaces\arc_llm_interface.py`
**Failed imports (1):**
- `llm_interface.BaseLlmInterface`

### `interfaces\llm_interface.py`
**Failed imports (1):**
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`

### `interfaces\memory_interface.py`
**Failed imports (1):**
- `Vanta.interfaces.base_interfaces.BaseMemoryInterface`

### `interfaces\rag_interface.py`
**Failed imports (1):**
- `Vanta.interfaces.base_interfaces.BaseRagInterface`

### `middleware\blt_middleware_loader.py`
**Failed imports (1):**
- `voxsigil_blt_rag.BLTEnhancedRAG`

### `middleware\voxsigil_middleware.py`
**Failed imports (1):**
- `hybrid_blt.HybridMiddleware`

### `scripts\run_vantacore_grid_connector.py`
**Failed imports (1):**
- `Voxsigil_Library.Gridformer.core.vantacore_grid_connector.test_grid_former_connector`

### `services\dice_roller_service.py`
**Failed imports (1):**
- `Vanta.core.UnifiedVantaCore.get_vanta_core`

### `services\game_state_store.py`
**Failed imports (1):**
- `Vanta.core.UnifiedVantaCore.get_vanta_core`

### `services\inventory_manager.py`
**Failed imports (1):**
- `Vanta.core.UnifiedVantaCore.get_vanta_core`

### `test\agent_validation.py`
**Failed imports (1):**
- `Vanta.core.UnifiedVantaCore.UnifiedVantaCore`

### `test\quick_step4_test.py`
**Failed imports (1):**
- `GUI.components.voxsigil_integration.VoxSigilIntegrationManager`

### `test\test_integration.py`
**Failed imports (1):**
- `voxsigil_integration.VoxSigilIntegrationManager`

### `test\test_mesh_echo_chain.py`
**Failed imports (1):**
- `voxsigil_mesh.VoxSigilMesh`

### `test\test_mesh_echo_chain_legacy.py`
**Failed imports (1):**
- `voxsigil_mesh.VoxSigilMesh`

### `test\test_step4.py`
**Failed imports (1):**
- `voxsigil_integration.VoxSigilIntegrationManager`

### `vmb\vmb_production_executor.py`
**Failed imports (1):**
- `vmb_activation.CopilotSwarm`

### `VoxSigilRag\hybrid_blt.py`
**Failed imports (1):**
- `sigil_patch_encoder.SigilPatchEncoder`

### `VoxSigilRag\vanta_registration.py`
**Failed imports (1):**
- `Vanta.get_vanta_core_instance`

### `VoxSigilRag\voxsigil_blt.py`
**Failed imports (1):**
- `sigil_patch_encoder.SigilPatchEncoder`

### `VoxSigilRag\voxsigil_blt_rag.py`
**Failed imports (1):**
- `sigil_patch_encoder.SigilPatchEncoder`

### `voxsigil_supervisor\blt_supervisor_integration.py`
**Failed imports (1):**
- `Vanta.interfaces.base_interfaces.BaseRagInterface`

### `Vanta\core\orchestrator.py`
**Failed imports (1):**
- `fallback_implementations.fallback_registry`

### `Vanta\core\UnifiedAsyncBus.py`
**Failed imports (1):**
- `Vanta.interfaces.blt_encoder_interface.BaseBLTEncoder`

### `Vanta\integration\module_adapters.py`
**Failed imports (1):**
- `core.orchestrator.vanta_orchestrator`

### `Vanta\interfaces\blt_encoder_interface.py`
**Failed imports (1):**
- `specialized_interfaces.BLTInterface`

### `Vanta\interfaces\hybrid_middleware_interface.py`
**Failed imports (1):**
- `specialized_interfaces.MiddlewareInterface`

### `Vanta\tabletop\__init__.py`
**Failed imports (1):**
- `Vanta.core.UnifiedVantaCore.get_vanta_core`

### `gui\components\dynamic_gridformer_gui.py`
**Failed imports (1):**
- `GUI.components.vanta_integration.VantaGUIIntegration`

### `gui\components\music_tab.py`
**Failed imports (1):**
- `core.vanta_core.VantaCore`

### `core\novel_efficiency\dataset_manager.py`
**Failed imports (1):**
- `aiofiles`

### `core\novel_efficiency\minicache.py`
**Failed imports (1):**
- `Vanta.core.base.BaseCore`

### `ARC\llm\llm_interface.py`
**Failed imports (1):**
- `Vanta.interfaces.base_interfaces.BaseLlmInterface`

