#!/usr/bin/env python
"""
Sigils Module Vanta Registration - HOLO-1.5 Enhanced

This module provides registration capabilities for the sigils module
with the Vanta orchestrator system using HOLO-1.5 cognitive mesh integration.

Components registered:
- SigilDefinitions: Core sigil definitions and metadata
- SigilProcessor: Sigil processing and transformation engine
- SigilGenerator: Dynamic sigil generation and creation
- SigilInterpreter: Sigil interpretation and meaning extraction

HOLO-1.5 Integration: Full cognitive mesh integration with symbolic processing capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict

# HOLO-1.5 imports
import sys
import os
# Add the parent directory to the path to ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module

# Configure logging
logger = logging.getLogger(__name__)


@vanta_core_module(
    name="sigils_module_processor",
    subsystem="symbolic_processing",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="Sigil processing and symbolic representation engine for VoxSigil ecosystem",
    capabilities=[
        "sigil_definitions",
        "symbolic_processing",
        "sigil_generation",
        "sigil_interpretation",
        "symbolic_transformation",
        "meaning_extraction",
        "symbolic_synthesis",
        "pattern_recognition"
    ],
    cognitive_load=3.5,
    symbolic_depth=5,
    collaboration_patterns=[
        "symbolic_synthesis",
        "pattern_emergence",
        "meaning_construction",
        "symbolic_reasoning"
    ],
)
class SigilsModuleAdapter(BaseCore):
    """
    Sigils Module Adapter with HOLO-1.5 integration.
    Provides comprehensive sigil processing, symbolic representation,
    and meaning extraction for the VoxSigil symbolic reasoning system.
    """
    
    def __init__(self, vanta_core: Any, config: Dict[str, Any]):
        super().__init__(vanta_core, config)
        
        self.module_id = "sigils"
        self.display_name = "Sigils Module"
        self.version = "1.0.0"
        self.description = "Comprehensive sigil processing and symbolic representation system"
        
        # Sigil component instances
        self.sigil_definitions = None
        self.sigil_processor = None
        self.sigil_generator = None
        self.sigil_interpreter = None
        self.initialized = False
        
        # Available sigil components
        self.available_components = {
            'sigil_definitions': 'sigil_definitions.SigilDefinitions',
            'sigil_processor': 'sigil_processor.SigilProcessor',
            'sigil_generator': 'sigil_generator.SigilGenerator',
            'sigil_interpreter': 'sigil_interpreter.SigilInterpreter',
            'symbolic_transformer': 'symbolic_transformer.SymbolicTransformer',
            'pattern_recognizer': 'pattern_recognizer.PatternRecognizer'
        }
        
        self.logger = config.get("logger", logging.getLogger(__name__))
        self.logger.setLevel(config.get("log_level", logging.INFO))
    
    async def initialize(self) -> bool:
        """Initialize the Sigils module with vanta core."""
        try:
            self.logger.info(f"Initializing Sigils module with Vanta core...")
            
            # Initialize sigil components
            await self._initialize_sigil_definitions()
            await self._initialize_sigil_processor()
            await self._initialize_sigil_generator()
            await self._initialize_sigil_interpreter()
            
            # Register with HOLO-1.5 cognitive mesh
            if hasattr(self.vanta_core, "register_component"):
                self.vanta_core.register_component(
                    "sigils_module_processor",
                    self,
                    {"type": "symbolic_service", "cognitive_role": "SYNTHESIZER"}
                )
            
            self.initialized = True
            self.logger.info(f"Sigils module initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sigils module: {e}")
            return False
    
    async def _initialize_sigil_definitions(self):
        """Initialize sigil definitions."""
        try:
            definitions_module = importlib.import_module('sigils.sigil_definitions')
            SigilDefinitionsClass = getattr(definitions_module, 'SigilDefinitions', None)
            
            if SigilDefinitionsClass:
                try:
                    self.sigil_definitions = SigilDefinitionsClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.sigil_definitions = SigilDefinitionsClass()
                    if hasattr(self.sigil_definitions, 'set_vanta_core'):
                        self.sigil_definitions.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.sigil_definitions, 'async_init'):
                    await self.sigil_definitions.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Sigil definitions not available: {e}")
            self.sigil_definitions = None
    
    async def _initialize_sigil_processor(self):
        """Initialize sigil processor."""
        try:
            processor_module = importlib.import_module('sigils.sigil_processor')
            SigilProcessorClass = getattr(processor_module, 'SigilProcessor', None)
            
            if SigilProcessorClass:
                try:
                    self.sigil_processor = SigilProcessorClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.sigil_processor = SigilProcessorClass()
                    if hasattr(self.sigil_processor, 'set_vanta_core'):
                        self.sigil_processor.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.sigil_processor, 'async_init'):
                    await self.sigil_processor.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Sigil processor not available: {e}")
            self.sigil_processor = None
    
    async def _initialize_sigil_generator(self):
        """Initialize sigil generator."""
        try:
            generator_module = importlib.import_module('sigils.sigil_generator')
            SigilGeneratorClass = getattr(generator_module, 'SigilGenerator', None)
            
            if SigilGeneratorClass:
                try:
                    self.sigil_generator = SigilGeneratorClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.sigil_generator = SigilGeneratorClass()
                    if hasattr(self.sigil_generator, 'set_vanta_core'):
                        self.sigil_generator.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.sigil_generator, 'async_init'):
                    await self.sigil_generator.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Sigil generator not available: {e}")
            self.sigil_generator = None
    
    async def _initialize_sigil_interpreter(self):
        """Initialize sigil interpreter."""
        try:
            interpreter_module = importlib.import_module('sigils.sigil_interpreter')
            SigilInterpreterClass = getattr(interpreter_module, 'SigilInterpreter', None)
            
            if SigilInterpreterClass:
                try:
                    self.sigil_interpreter = SigilInterpreterClass(vanta_core=self.vanta_core)
                except TypeError:
                    self.sigil_interpreter = SigilInterpreterClass()
                    if hasattr(self.sigil_interpreter, 'set_vanta_core'):
                        self.sigil_interpreter.set_vanta_core(self.vanta_core)
                        
                if hasattr(self.sigil_interpreter, 'async_init'):
                    await self.sigil_interpreter.async_init()
                    
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Sigil interpreter not available: {e}")
            self.sigil_interpreter = None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for sigil operations."""
        if not self.initialized:
            return {"error": "Sigils module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'sigil_processing':
                return await self._handle_sigil_processing(request)
            elif request_type == 'sigil_generation':
                return await self._handle_sigil_generation(request)
            elif request_type == 'sigil_interpretation':
                return await self._handle_sigil_interpretation(request)
            elif request_type == 'symbolic_transformation':
                return await self._handle_symbolic_transformation(request)
            elif request_type == 'pattern_recognition':
                return await self._handle_pattern_recognition(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            self.logger.error(f"Error processing sigils request: {e}")
            return {"error": str(e)}
    
    async def _handle_sigil_processing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sigil processing requests."""
        if not self.sigil_processor:
            return {"error": "Sigil processor not available"}
        
        operation = request.get('operation')
        sigil_data = request.get('sigil_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.sigil_processor, operation):
                handler = getattr(self.sigil_processor, operation)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(sigil_data, **params)
                else:
                    result = handler(sigil_data, **params)
            else:
                result = {"warning": f"Operation {operation} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Sigil processing failed: {e}"}
    
    async def _handle_sigil_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sigil generation requests."""
        if not self.sigil_generator:
            return {"error": "Sigil generator not available"}
        
        generation_type = request.get('generation_type')
        template_data = request.get('template_data')
        params = request.get('params', {})
        
        try:
            if hasattr(self.sigil_generator, generation_type):
                handler = getattr(self.sigil_generator, generation_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(template_data, **params)
                else:
                    result = handler(template_data, **params)
            else:
                result = {"warning": f"Generation type {generation_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Sigil generation failed: {e}"}
    
    async def _handle_sigil_interpretation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sigil interpretation requests."""
        if not self.sigil_interpreter:
            return {"error": "Sigil interpreter not available"}
        
        interpretation_type = request.get('interpretation_type')
        sigil_data = request.get('sigil_data')
        context = request.get('context', {})
        
        try:
            if hasattr(self.sigil_interpreter, interpretation_type):
                handler = getattr(self.sigil_interpreter, interpretation_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(sigil_data, context)
                else:
                    result = handler(sigil_data, context)
            else:
                result = {"warning": f"Interpretation type {interpretation_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Sigil interpretation failed: {e}"}
    
    async def _handle_symbolic_transformation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle symbolic transformation requests."""
        transformation_type = request.get('transformation_type')
        input_symbols = request.get('input_symbols')
        params = request.get('params', {})
        
        try:
            # Import symbolic transformer
            transformer_module = importlib.import_module('sigils.symbolic_transformer')
            SymbolicTransformerClass = getattr(transformer_module, 'SymbolicTransformer')
            
            transformer = SymbolicTransformerClass()
            if hasattr(transformer, transformation_type):
                handler = getattr(transformer, transformation_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(input_symbols, **params)
                else:
                    result = handler(input_symbols, **params)
            else:
                result = {"warning": f"Transformation type {transformation_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Symbolic transformation failed: {e}"}
    
    async def _handle_pattern_recognition(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pattern recognition requests."""
        pattern_type = request.get('pattern_type')
        pattern_data = request.get('pattern_data')
        params = request.get('params', {})
        
        try:
            # Import pattern recognizer
            recognizer_module = importlib.import_module('sigils.pattern_recognizer')
            PatternRecognizerClass = getattr(recognizer_module, 'PatternRecognizer')
            
            recognizer = PatternRecognizerClass()
            if hasattr(recognizer, pattern_type):
                handler = getattr(recognizer, pattern_type)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(pattern_data, **params)
                else:
                    result = handler(pattern_data, **params)
            else:
                result = {"warning": f"Pattern type {pattern_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Pattern recognition failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Sigils module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "sigil_definitions",
                "symbolic_processing",
                "sigil_generation",
                "sigil_interpretation",
                "symbolic_transformation",
                "meaning_extraction",
                "symbolic_synthesis",
                "pattern_recognition"
            ],
            "supported_operations": [
                "sigil_processing",
                "sigil_generation",
                "sigil_interpretation",
                "symbolic_transformation",
                "pattern_recognition"
            ],
            "components": list(self.available_components.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "SYNTHESIZER",
            "symbolic_depth": 5
        }
    
    async def shutdown(self):
        """Shutdown the Sigils module gracefully."""
        try:
            self.logger.info("Shutting down Sigils module...")
            
            # Shutdown all sigil components that support it
            for sigil_name in ['sigil_definitions', 'sigil_processor', 
                              'sigil_generator', 'sigil_interpreter']:
                sigil_instance = getattr(self, sigil_name, None)
                if sigil_instance and hasattr(sigil_instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(sigil_instance.shutdown):
                        await sigil_instance.shutdown()
                    else:
                        sigil_instance.shutdown()
            
            self.initialized = False
            self.logger.info("Sigils module shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Sigils module shutdown: {e}")


# Registration function for the master orchestrator
async def register_sigils_module(vanta_core) -> SigilsModuleAdapter:
    """Register the Sigils module with Vanta orchestrator."""
    logger.info("Registering Sigils module with Vanta orchestrator...")
    
    adapter = SigilsModuleAdapter(vanta_core, {})
    success = await adapter.initialize()
    
    if success:
        logger.info("Sigils module registration successful")
    else:
        logger.error("Sigils module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['SigilsModuleAdapter', 'register_sigils_module']
