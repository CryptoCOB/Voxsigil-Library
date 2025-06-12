#!/usr/bin/env python
"""
Scaffolds Module Vanta Registration - HOLO-1.5 Enhanced

This module provides registration capabilities for the scaffolds module
with the Vanta orchestrator system using HOLO-1.5 cognitive mesh integration.

Components registered:
- ReasoningScaffolds: Structured reasoning framework components
- ProblemSolvingScaffolds: Problem decomposition and solution scaffolds
- CognitiveScaffolds: Cognitive support structures and frameworks
- MetaScaffolds: Meta-level reasoning and reflection scaffolds

HOLO-1.5 Integration: Full cognitive mesh integration with scaffolding capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# HOLO-1.5 imports
import sys
import os
# Add the parent directory to the path to ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module

# Configure logging
logger = logging.getLogger(__name__)


@vanta_core_module(
    name="scaffolds_module_processor",
    subsystem="cognitive_frameworks",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="Scaffolding framework processor for reasoning structures and cognitive support",
    capabilities=[
        "reasoning_scaffolds",
        "problem_decomposition",
        "cognitive_frameworks",
        "meta_reasoning",
        "structured_thinking",
        "solution_scaffolds",
        "reflection_frameworks",
        "learning_scaffolds"
    ],
    cognitive_load=3.0,
    symbolic_depth=4,
    collaboration_patterns=[
        "scaffolded_reasoning",
        "framework_synthesis",
        "cognitive_structuring",
        "meta_scaffolding"
    ],
)
class ScaffoldsModuleAdapter(BaseCore):
    """
    Scaffolds Module Adapter with HOLO-1.5 integration.
    Provides comprehensive scaffolding frameworks for reasoning,
    problem-solving, and cognitive support structures.
    """
    
    def __init__(self, vanta_core: Any, config: Dict[str, Any]):
        super().__init__(vanta_core, config)
        
        self.module_id = "scaffolds"
        self.display_name = "Scaffolds Module"
        self.version = "1.0.0"
        self.description = "Comprehensive scaffolding frameworks for reasoning and cognitive support"
        
        # Scaffold component instances
        self.reasoning_scaffolds = None
        self.problem_solving_scaffolds = None
        self.cognitive_scaffolds = None
        self.meta_scaffolds = None
        self.initialized = False
        
        # Available scaffold components
        self.available_components = {
            'reasoning_scaffolds': 'reasoning_scaffolds.ReasoningScaffolds',
            'problem_solving_scaffolds': 'problem_solving_scaffolds.ProblemSolvingScaffolds',
            'cognitive_scaffolds': 'cognitive_scaffolds.CognitiveScaffolds',
            'meta_scaffolds': 'meta_scaffolds.MetaScaffolds',
            'learning_scaffolds': 'learning_scaffolds.LearningScaffolds',
            'reflection_scaffolds': 'reflection_scaffolds.ReflectionScaffolds'
        }
        
        self.logger = config.get("logger", logging.getLogger(__name__))
        self.logger.setLevel(config.get("log_level", logging.INFO))
    
    async def initialize(self) -> bool:
        """Initialize the Scaffolds module with vanta core."""
        try:
            self.logger.info(f"Initializing Scaffolds module with Vanta core...")
            
            # Initialize scaffold components
            await self._initialize_reasoning_scaffolds()
            await self._initialize_problem_solving_scaffolds()
            await self._initialize_cognitive_scaffolds()
            await self._initialize_meta_scaffolds()
            
            # Register with HOLO-1.5 cognitive mesh
            if hasattr(self.vanta_core, "register_component"):
                self.vanta_core.register_component(
                    "scaffolds_module_processor",
                    self,
                    {"type": "scaffolding_service", "cognitive_role": "SYNTHESIZER"}
                )
            
            self.initialized = True
            self.logger.info(f"Scaffolds module initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Scaffolds module: {e}")
            return False
    
    async def _initialize_reasoning_scaffolds(self):
        """Initialize reasoning scaffolds."""
        try:
            reasoning_module = importlib.import_module('scaffolds.reasoning_scaffolds')
            ReasoningScaffoldsClass = getattr(reasoning_module, 'ReasoningScaffolds', None)
            
            if ReasoningScaffoldsClass:
                try:
                    self.reasoning_scaffolds = ReasoningScaffoldsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.reasoning_scaffolds = ReasoningScaffoldsClass()
                    if hasattr(self.reasoning_scaffolds, 'set_vanta_core'):
                        self.reasoning_scaffolds.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.reasoning_scaffolds, 'async_init'):
                    await self.reasoning_scaffolds.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Reasoning scaffolds not available: {e}")
            self.reasoning_scaffolds = None
    
    async def _initialize_problem_solving_scaffolds(self):
        """Initialize problem solving scaffolds."""
        try:
            problem_solving_module = importlib.import_module('scaffolds.problem_solving_scaffolds')
            ProblemSolvingScaffoldsClass = getattr(problem_solving_module, 'ProblemSolvingScaffolds', None)
            
            if ProblemSolvingScaffoldsClass:
                try:
                    self.problem_solving_scaffolds = ProblemSolvingScaffoldsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.problem_solving_scaffolds = ProblemSolvingScaffoldsClass()
                    if hasattr(self.problem_solving_scaffolds, 'set_vanta_core'):
                        self.problem_solving_scaffolds.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.problem_solving_scaffolds, 'async_init'):
                    await self.problem_solving_scaffolds.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Problem solving scaffolds not available: {e}")
            self.problem_solving_scaffolds = None
    
    async def _initialize_cognitive_scaffolds(self):
        """Initialize cognitive scaffolds."""
        try:
            cognitive_module = importlib.import_module('scaffolds.cognitive_scaffolds')
            CognitiveScaffoldsClass = getattr(cognitive_module, 'CognitiveScaffolds', None)
            
            if CognitiveScaffoldsClass:
                try:
                    self.cognitive_scaffolds = CognitiveScaffoldsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.cognitive_scaffolds = CognitiveScaffoldsClass()
                    if hasattr(self.cognitive_scaffolds, 'set_vanta_core'):
                        self.cognitive_scaffolds.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.cognitive_scaffolds, 'async_init'):
                    await self.cognitive_scaffolds.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Cognitive scaffolds not available: {e}")
            self.cognitive_scaffolds = None
    
    async def _initialize_meta_scaffolds(self):
        """Initialize meta scaffolds."""
        try:
            meta_module = importlib.import_module('scaffolds.meta_scaffolds')
            MetaScaffoldsClass = getattr(meta_module, 'MetaScaffolds', None)
            
            if MetaScaffoldsClass:
                try:
                    self.meta_scaffolds = MetaScaffoldsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.meta_scaffolds = MetaScaffoldsClass()
                    if hasattr(self.meta_scaffolds, 'set_vanta_core'):
                        self.meta_scaffolds.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.meta_scaffolds, 'async_init'):
                    await self.meta_scaffolds.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Meta scaffolds not available: {e}")
            self.meta_scaffolds = None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for scaffolding operations."""
        if not self.initialized:
            return {"error": "Scaffolds module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'reasoning_scaffold':
                return await self._handle_reasoning_scaffold(request)
            elif request_type == 'problem_solving_scaffold':
                return await self._handle_problem_solving_scaffold(request)
            elif request_type == 'cognitive_scaffold':
                return await self._handle_cognitive_scaffold(request)
            elif request_type == 'meta_scaffold':
                return await self._handle_meta_scaffold(request)
            elif request_type == 'learning_scaffold':
                return await self._handle_learning_scaffold(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            self.logger.error(f"Error processing scaffolds request: {e}")
            return {"error": str(e)}
    
    async def _handle_reasoning_scaffold(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reasoning scaffold requests."""
        if not self.reasoning_scaffolds:
            return {"error": "Reasoning scaffolds not available"}
        
        scaffold_type = request.get('scaffold_type')
        problem_data = request.get('problem_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.reasoning_scaffolds, scaffold_type):
                handler = getattr(self.reasoning_scaffolds, scaffold_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(problem_data, **params)
                else:
                    result = handler(problem_data, **params)
            else:
                result = {"warning": f"Scaffold type {scaffold_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Reasoning scaffold failed: {e}"}
    
    async def _handle_problem_solving_scaffold(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle problem solving scaffold requests."""
        if not self.problem_solving_scaffolds:
            return {"error": "Problem solving scaffolds not available"}
        
        scaffold_type = request.get('scaffold_type')
        problem_data = request.get('problem_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.problem_solving_scaffolds, scaffold_type):
                handler = getattr(self.problem_solving_scaffolds, scaffold_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(problem_data, **params)
                else:
                    result = handler(problem_data, **params)
            else:
                result = {"warning": f"Scaffold type {scaffold_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Problem solving scaffold failed: {e}"}
    
    async def _handle_cognitive_scaffold(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cognitive scaffold requests."""
        if not self.cognitive_scaffolds:
            return {"error": "Cognitive scaffolds not available"}
        
        scaffold_type = request.get('scaffold_type')
        cognitive_data = request.get('cognitive_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.cognitive_scaffolds, scaffold_type):
                handler = getattr(self.cognitive_scaffolds, scaffold_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(cognitive_data, **params)
                else:
                    result = handler(cognitive_data, **params)
            else:
                result = {"warning": f"Scaffold type {scaffold_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Cognitive scaffold failed: {e}"}
    
    async def _handle_meta_scaffold(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle meta scaffold requests."""
        if not self.meta_scaffolds:
            return {"error": "Meta scaffolds not available"}
        
        scaffold_type = request.get('scaffold_type')
        meta_data = request.get('meta_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.meta_scaffolds, scaffold_type):
                handler = getattr(self.meta_scaffolds, scaffold_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(meta_data, **params)
                else:
                    result = handler(meta_data, **params)
            else:
                result = {"warning": f"Scaffold type {scaffold_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Meta scaffold failed: {e}"}
    
    async def _handle_learning_scaffold(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning scaffold requests."""
        learning_type = request.get('learning_type')
        learning_data = request.get('learning_data')
        params = request.get('params', {})
        
        try:
            # Import learning scaffolds
            learning_module = importlib.import_module('scaffolds.learning_scaffolds')
            LearningScaffoldsClass = getattr(learning_module, 'LearningScaffolds')
            
            learning_scaffold = LearningScaffoldsClass()
            if hasattr(learning_scaffold, learning_type):
                handler = getattr(learning_scaffold, learning_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(learning_data, **params)
                else:
                    result = handler(learning_data, **params)
            else:
                result = {"warning": f"Learning type {learning_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Learning scaffold failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Scaffolds module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "reasoning_scaffolds",
                "problem_decomposition",
                "cognitive_frameworks",
                "meta_reasoning",
                "structured_thinking",
                "solution_scaffolds",
                "reflection_frameworks",
                "learning_scaffolds"
            ],
            "supported_operations": [
                "reasoning_scaffold",
                "problem_solving_scaffold",
                "cognitive_scaffold",
                "meta_scaffold",
                "learning_scaffold"
            ],
            "components": list(self.available_components.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "SYNTHESIZER",
            "symbolic_depth": 4
        }
    
    async def shutdown(self):
        """Shutdown the Scaffolds module gracefully."""
        try:
            self.logger.info("Shutting down Scaffolds module...")
            
            # Shutdown all scaffold components that support it
            for scaffold_name in ['reasoning_scaffolds', 'problem_solving_scaffolds', 
                                 'cognitive_scaffolds', 'meta_scaffolds']:
                scaffold_instance = getattr(self, scaffold_name, None)
                if scaffold_instance and hasattr(scaffold_instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(scaffold_instance.shutdown):
                        await scaffold_instance.shutdown()
                    else:
                        scaffold_instance.shutdown()
            
            self.initialized = False
            self.logger.info("Scaffolds module shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Scaffolds module shutdown: {e}")


# Registration function for the master orchestrator
async def register_scaffolds_module(vanta_core) -> ScaffoldsModuleAdapter:
    """Register the Scaffolds module with Vanta orchestrator."""
    logger.info("Registering Scaffolds module with Vanta orchestrator...")
    
    adapter = ScaffoldsModuleAdapter(vanta_core, {})
    success = await adapter.initialize()
    
    if success:
        logger.info("Scaffolds module registration successful")
    else:
        logger.error("Scaffolds module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['ScaffoldsModuleAdapter', 'register_scaffolds_module']
