# engines/base.py
"""
HOLO-1.5 Recursive Symbolic Cognition Mesh - Base Engine Infrastructure
Provides encapsulated registration pattern for all VoxSigil Library processing engines
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Type, Union, Protocol
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger("VoxSigil.Engines.Base")

# HOLO-1.5 Cognitive Mesh Role System
class CognitiveMeshRole(Enum):
    """HOLO-1.5 Recursive Symbolic Cognition Mesh roles for processing engines"""
    ORCHESTRATOR = "orchestrator"  # High-level coordination and resource management
    PROCESSOR = "processor"        # Core processing and transformation engines  
    EVALUATOR = "evaluator"        # Analysis, validation, and assessment engines
    SYNTHESIZER = "synthesizer"    # Integration, fusion, and creation engines

# Base Engine Class with HOLO-1.5 Integration
class BaseEngine:
    """
    Base class for all VoxSigil processing engines with HOLO-1.5 integration.
    Provides standard interface and cognitive mesh collaboration capabilities.
    """
    
    COMPONENT_NAME = "base_engine"
    
    def __init__(self, vanta_core, config=None):
        self.vanta_core = vanta_core
        self.config = config or {}
        self.running = False
        self.last_error = None
        self.mesh_role = getattr(self, '_mesh_role', CognitiveMeshRole.PROCESSOR)
        
        # HOLO-1.5 Cognitive Mesh Attributes
        self.mesh_connections = {}
        self.symbolic_state = {}
        self.recursive_depth = 0
        self.collaboration_history = []
        
        logger.info(f"BaseEngine {self.COMPONENT_NAME} initialized with mesh role: {self.mesh_role.value}")
    
    async def initialize_subsystem(self, vanta_core):
        """Initialize engine subsystem with VantaCore integration"""
        self.vanta_core = vanta_core
        await self.setup_mesh_connections()
        logger.info(f"Engine {self.COMPONENT_NAME} subsystem initialized")
    
    async def setup_mesh_connections(self):
        """Setup HOLO-1.5 cognitive mesh connections with other engines"""
        try:
            # Connect to complementary mesh roles
            mesh_registry = getattr(self.vanta_core, 'mesh_registry', {})
            
            for role in CognitiveMeshRole:
                if role != self.mesh_role:
                    role_engines = mesh_registry.get(role.value, [])
                    self.mesh_connections[role.value] = role_engines
            
            logger.info(f"Mesh connections established for {self.COMPONENT_NAME}")
        except Exception as e:
            logger.error(f"Failed to setup mesh connections: {e}")
    
    async def collaborate_with_mesh(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other engines in the cognitive mesh"""
        try:
            collaboration_id = f"collab_{id(task_data)}"
            self.collaboration_history.append(collaboration_id)
            
            # Determine optimal collaboration strategy based on mesh role
            if self.mesh_role == CognitiveMeshRole.ORCHESTRATOR:
                return await self._orchestrate_mesh_task(task_data)
            elif self.mesh_role == CognitiveMeshRole.PROCESSOR:
                return await self._process_with_mesh_support(task_data)
            elif self.mesh_role == CognitiveMeshRole.EVALUATOR:
                return await self._evaluate_with_mesh_input(task_data)
            elif self.mesh_role == CognitiveMeshRole.SYNTHESIZER:
                return await self._synthesize_mesh_results(task_data)
            
            return task_data
        except Exception as e:
            logger.error(f"Mesh collaboration failed: {e}")
            return {"error": str(e)}
    
    async def _orchestrate_mesh_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate task across multiple mesh engines"""
        results = {"orchestration_id": f"orch_{id(task_data)}", "engine": self.COMPONENT_NAME}
        
        # Distribute to processors
        processors = self.mesh_connections.get('processor', [])
        if processors:
            for proc in processors[:2]:  # Limit to 2 for efficiency
                try:
                    if hasattr(proc, 'process_request'):
                        proc_result = await proc.process_request(task_data)
                        results[f"processor_{proc.COMPONENT_NAME}"] = proc_result
                except Exception as e:
                    logger.warning(f"Processor {proc.COMPONENT_NAME} failed: {e}")
        
        return results
    
    async def _process_with_mesh_support(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with mesh evaluation support"""
        # Core processing logic (to be overridden)
        result = await self.process_core(task_data)
        
        # Get evaluation from mesh evaluators
        evaluators = self.mesh_connections.get('evaluator', [])
        if evaluators and result:
            for evaluator in evaluators[:1]:  # Single evaluator for efficiency
                try:
                    if hasattr(evaluator, 'evaluate'):
                        eval_result = await evaluator.evaluate(result)
                        result['mesh_evaluation'] = eval_result
                except Exception as e:
                    logger.warning(f"Evaluator {evaluator.COMPONENT_NAME} failed: {e}")
        
        return result
    
    async def _evaluate_with_mesh_input(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate with input from mesh processors"""
        evaluation = {
            "evaluation_id": f"eval_{id(task_data)}",
            "engine": self.COMPONENT_NAME,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Core evaluation logic (to be overridden)
        evaluation.update(await self.evaluate_core(task_data))
        
        return evaluation
    
    async def _synthesize_mesh_results(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple mesh engines"""
        synthesis = {
            "synthesis_id": f"synth_{id(task_data)}",
            "engine": self.COMPONENT_NAME,
            "mesh_inputs": []
        }
        
        # Gather inputs from all mesh roles
        for role, engines in self.mesh_connections.items():
            for engine in engines[:1]:  # One per role for efficiency
                try:
                    if hasattr(engine, 'get_state'):
                        state = await engine.get_state()
                        synthesis["mesh_inputs"].append({
                            "role": role,
                            "engine": engine.COMPONENT_NAME,
                            "state": state
                        })
                except Exception as e:
                    logger.warning(f"Failed to get state from {engine.COMPONENT_NAME}: {e}")
        
        # Core synthesis logic (to be overridden)
        synthesis.update(await self.synthesize_core(task_data))
        
        return synthesis
    
    async def process_core(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core processing logic - to be overridden by specific engines"""
        return {"processed": True, "engine": self.COMPONENT_NAME}
    
    async def evaluate_core(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core evaluation logic - to be overridden by evaluator engines"""
        return {"evaluated": True, "score": 0.8}
    
    async def synthesize_core(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core synthesis logic - to be overridden by synthesizer engines"""
        return {"synthesized": True, "components": len(self.mesh_connections)}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current engine state for mesh collaboration"""
        return {
            "engine": self.COMPONENT_NAME,
            "running": self.running,
            "mesh_role": self.mesh_role.value,
            "connections": len(self.mesh_connections),
            "symbolic_state": self.symbolic_state,
            "recursive_depth": self.recursive_depth
        }
    
    def start(self):
        """Start the engine"""
        self.running = True
        logger.info(f"Engine {self.COMPONENT_NAME} started")
    
    def stop(self):
        """Stop the engine"""
        self.running = False
        logger.info(f"Engine {self.COMPONENT_NAME} stopped")

# HOLO-1.5 Engine Registration Decorator
def vanta_engine(name: str, subsystem: str, mesh_role: CognitiveMeshRole, 
                description: Optional[str] = None, capabilities: Optional[List[str]] = None):
    """
    HOLO-1.5 Recursive Symbolic Cognition Mesh engine registration decorator.
    
    Args:
        name: Engine identifier for VantaCore registration
        subsystem: Target VoxSigil subsystem for integration
        mesh_role: HOLO-1.5 cognitive mesh role
        description: Human-readable description
        capabilities: List of engine capabilities
    """
    def decorator(cls):
        # Store registration metadata
        cls._vanta_name = name
        cls._vanta_subsystem = subsystem
        cls._mesh_role = mesh_role
        cls._vanta_description = description or cls.__doc__ or f"{name} processing engine"
        cls._vanta_capabilities = capabilities or []
        
        # Enhanced mesh role capabilities
        role_capabilities = {
            CognitiveMeshRole.ORCHESTRATOR: ['task_coordination', 'resource_management', 'workflow_control'],
            CognitiveMeshRole.PROCESSOR: ['data_processing', 'transformation', 'computation'],
            CognitiveMeshRole.EVALUATOR: ['analysis', 'validation', 'assessment', 'quality_control'],
            CognitiveMeshRole.SYNTHESIZER: ['integration', 'fusion', 'creation', 'composition']
        }
        
        cls._vanta_capabilities.extend(role_capabilities.get(mesh_role, []))
        cls._vanta_capabilities.append(f'mesh_role_{mesh_role.value}')
        cls._vanta_capabilities.append('holo_1_5_compatible')
        
        # Add auto-registration method
        @classmethod
        async def register_with_vanta(cls, vanta_core):
            """Auto-register this engine with VantaCore"""
            try:
                # Create engine adapter with HOLO-1.5 capabilities
                from .vanta_registration import EngineModuleAdapter
                
                adapter = EngineModuleAdapter(
                    module_id=f'engine_{name}',
                    engine_class=cls,
                    description=cls._vanta_description
                )
                
                # Enhanced metadata with HOLO-1.5 mesh info
                metadata = adapter.get_metadata()
                metadata.update({
                    'mesh_role': mesh_role.value,
                    'subsystem': subsystem,
                    'holo_version': '1.5',
                    'cognitive_mesh_enabled': True,
                    'symbolic_processing': True,
                    'recursive_cognition': True
                })
                
                # Register with VantaCore
                await vanta_core.register_module(f'engine_{name}', adapter, metadata)
                
                # Register in mesh registry
                if not hasattr(vanta_core, 'mesh_registry'):
                    vanta_core.mesh_registry = {}
                if mesh_role.value not in vanta_core.mesh_registry:
                    vanta_core.mesh_registry[mesh_role.value] = []
                
                # Initialize adapter to get engine instance
                await adapter.initialize(vanta_core)
                if adapter.engine_instance:
                    vanta_core.mesh_registry[mesh_role.value].append(adapter.engine_instance)
                
                logger.info(f"✅ {cls.__name__} registered with HOLO-1.5 mesh role: {mesh_role.value}")
                return adapter
                
            except Exception as e:
                logger.error(f"❌ Failed to register {cls.__name__}: {e}")
                raise
        
        cls.register_with_vanta = register_with_vanta
        return cls
    
    return decorator

# Enhanced Engine Module Adapter with HOLO-1.5 support
class HOLO15EngineAdapter:
    """Enhanced engine adapter with HOLO-1.5 cognitive mesh capabilities"""
    
    def __init__(self, engine_class: Type, mesh_role: CognitiveMeshRole):
        self.engine_class = engine_class
        self.mesh_role = mesh_role
        self.engine_instance = None
        self.mesh_connections = {}
    
    async def initialize_with_mesh(self, vanta_core):
        """Initialize engine with mesh connections"""
        # Create engine instance
        if hasattr(self.engine_class, '__init__'):
            self.engine_instance = self.engine_class(vanta_core=vanta_core)
        else:
            self.engine_instance = self.engine_class()
        
        # Setup mesh connections
        if hasattr(self.engine_instance, 'setup_mesh_connections'):
            await self.engine_instance.setup_mesh_connections()
        
        return self.engine_instance
    
    async def process_with_mesh(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using mesh collaboration"""
        if not self.engine_instance:
            return {"error": "Engine not initialized"}
        
        if hasattr(self.engine_instance, 'collaborate_with_mesh'):
            return await self.engine_instance.collaborate_with_mesh(request)
        elif hasattr(self.engine_instance, 'process_request'):
            return await self.engine_instance.process_request(request)
        else:
            return {"message": f"Engine {self.engine_class.__name__} processed request"}

# Subsystem Mapping for VoxSigil Integration
VOXSIGIL_SUBSYSTEMS = {
    "async_processing_engine": "async_processing_core",
    "async_stt_engine": "speech_processing_layer", 
    "async_tts_engine": "speech_synthesis_layer",
    "async_training_engine": "meta_learning_core",
    "cat_engine": "cognitive_architecture_frame",
    "hybrid_cognition_engine": "dual_cognition_core",
    "rag_compression_engine": "rag_optimization_subsystem",
    "tot_engine": "reasoning_architecture_frame"
}

logger.info("HOLO-1.5 Engine Base Infrastructure initialized")
