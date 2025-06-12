"""
ART Module Vanta Registration

This module provides registration capabilities for the Adaptive Resonance Theory (ART)
module with the Vanta orchestrator system.

Components registered:
- ARTController: Main ART neural network controller
- ARTUnifiedAdapter: Comprehensive ART system adapter
- ARTManager: ART module management and coordination
- Various ART bridge components (BLT, RAG, Entropy)

HOLO-1.5 Integration: Full cognitive mesh integration with adaptive resonance capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# Configure logging
logger = logging.getLogger(__name__)

class ARTModuleAdapter:
    """Adapter for integrating ART module with Vanta orchestrator."""
    
    def __init__(self):
        self.module_id = "art"
        self.display_name = "Adaptive Resonance Theory Module"
        self.version = "1.0.0"
        self.description = "Adaptive Resonance Theory neural network for pattern recognition and category learning"
        
        # ART component instances
        self.art_controller = None
        self.art_adapter = None
        self.art_manager = None
        self.initialized = False
        
        # Available ART components
        self.available_components = {
            'art_controller': 'art_controller.ARTController',
            'art_adapter': 'art_adapter.ARTUnifiedAdapter', 
            'art_manager': 'art_manager.ARTManager',
            'art_trainer': 'art_trainer.ARTTrainer',
            'generative_art': 'generative_art.GenerativeARTNetwork'
        }
        
    async def initialize(self, vanta_core):
        """Initialize the ART module with vanta core."""
        try:
            logger.info(f"Initializing ART module with Vanta core...")
            
            # Import and initialize ART Controller
            art_controller_module = importlib.import_module('ART.art_controller')
            ARTControllerClass = getattr(art_controller_module, 'ARTController')
            
            # Initialize with vanta_core if constructor supports it
            try:
                self.art_controller = ARTControllerClass(vanta_core=vanta_core)
            except TypeError:
                # Fallback: initialize without vanta_core, then set it
                self.art_controller = ARTControllerClass()
                if hasattr(self.art_controller, 'set_vanta_core'):
                    self.art_controller.set_vanta_core(vanta_core)
                elif hasattr(self.art_controller, 'vanta_core'):
                    self.art_controller.vanta_core = vanta_core
            
            # Import and initialize ART Adapter
            art_adapter_module = importlib.import_module('ART.art_adapter')
            ARTAdapterClass = getattr(art_adapter_module, 'ARTUnifiedAdapter')
            
            try:
                self.art_adapter = ARTAdapterClass(vanta_core=vanta_core)
            except TypeError:
                self.art_adapter = ARTAdapterClass()
                if hasattr(self.art_adapter, 'set_vanta_core'):
                    self.art_adapter.set_vanta_core(vanta_core)
                elif hasattr(self.art_adapter, 'vanta_core'):
                    self.art_adapter.vanta_core = vanta_core
            
            # Import and initialize ART Manager if available
            try:
                art_manager_module = importlib.import_module('ART.art_manager')
                ARTManagerClass = getattr(art_manager_module, 'ARTManager')
                
                try:
                    self.art_manager = ARTManagerClass(vanta_core=vanta_core)
                except TypeError:
                    self.art_manager = ARTManagerClass()
                    if hasattr(self.art_manager, 'set_vanta_core'):
                        self.art_manager.set_vanta_core(vanta_core)
                    elif hasattr(self.art_manager, 'vanta_core'):
                        self.art_manager.vanta_core = vanta_core
                        
            except (ImportError, AttributeError) as e:
                logger.warning(f"ART Manager not available: {e}")
                self.art_manager = None
            
            # Initialize async components if they have async_init methods
            if hasattr(self.art_controller, 'async_init'):
                await self.art_controller.async_init()
                
            if hasattr(self.art_adapter, 'async_init'):
                await self.art_adapter.async_init()
                
            if self.art_manager and hasattr(self.art_manager, 'async_init'):
                await self.art_manager.async_init()
            
            self.initialized = True
            logger.info(f"ART module initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ART module: {e}")
            return False
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for ART module operations."""
        if not self.initialized:
            return {"error": "ART module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'pattern_recognition':
                return await self._handle_pattern_recognition(request)
            elif request_type == 'category_learning':
                return await self._handle_category_learning(request)
            elif request_type == 'adaptive_resonance':
                return await self._handle_adaptive_resonance(request)
            elif request_type == 'art_training':
                return await self._handle_art_training(request)
            elif request_type == 'bridge_operation':
                return await self._handle_bridge_operation(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing ART request: {e}")
            return {"error": str(e)}
    
    async def _handle_pattern_recognition(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pattern recognition requests."""
        if not self.art_controller:
            return {"error": "ART Controller not available"}
        
        pattern_data = request.get('pattern_data')
        if pattern_data is None:
            return {"error": "Missing pattern_data"}
        
        try:
            # Use ART controller for pattern recognition
            if hasattr(self.art_controller, 'recognize_pattern'):
                result = await self.art_controller.recognize_pattern(pattern_data)
            elif hasattr(self.art_controller, 'process_pattern'):
                result = self.art_controller.process_pattern(pattern_data)
            else:
                result = {"warning": "Pattern recognition method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Pattern recognition failed: {e}"}
    
    async def _handle_category_learning(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle category learning requests."""
        if not self.art_controller:
            return {"error": "ART Controller not available"}
        
        training_data = request.get('training_data')
        if training_data is None:
            return {"error": "Missing training_data"}
        
        try:
            # Use ART controller for category learning
            if hasattr(self.art_controller, 'learn_categories'):
                result = await self.art_controller.learn_categories(training_data)
            elif hasattr(self.art_controller, 'train'):
                result = self.art_controller.train(training_data)
            else:
                result = {"warning": "Category learning method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Category learning failed: {e}"}
    
    async def _handle_adaptive_resonance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adaptive resonance requests."""
        if not self.art_controller:
            return {"error": "ART Controller not available"}
        
        input_pattern = request.get('input_pattern')
        if input_pattern is None:
            return {"error": "Missing input_pattern"}
        
        try:
            # Use ART controller for adaptive resonance
            if hasattr(self.art_controller, 'adaptive_resonance'):
                result = await self.art_controller.adaptive_resonance(input_pattern)
            elif hasattr(self.art_controller, 'resonate'):
                result = self.art_controller.resonate(input_pattern)
            else:
                result = {"warning": "Adaptive resonance method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Adaptive resonance failed: {e}"}
    
    async def _handle_art_training(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ART training requests."""
        training_params = request.get('training_params', {})
        
        try:
            # Import and use ART trainer
            art_trainer_module = importlib.import_module('ART.art_trainer')
            ARTTrainerClass = getattr(art_trainer_module, 'ARTTrainer')
            
            trainer = ARTTrainerClass()
            if hasattr(trainer, 'train_async'):
                result = await trainer.train_async(**training_params)
            else:
                result = trainer.train(**training_params)
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"ART training failed: {e}"}
    
    async def _handle_bridge_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bridge operations (BLT, RAG, Entropy)."""
        if not self.art_adapter:
            return {"error": "ART Adapter not available"}
        
        bridge_type = request.get('bridge_type')
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.art_adapter, f'handle_{bridge_type}_bridge'):
                handler = getattr(self.art_adapter, f'handle_{bridge_type}_bridge')
                result = await handler(operation, **params)
            else:
                result = {"warning": f"Bridge type {bridge_type} not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Bridge operation failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the ART module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "pattern_recognition",
                "category_learning", 
                "adaptive_resonance",
                "unsupervised_learning",
                "novelty_detection",
                "art_training",
                "bridge_operations"
            ],
            "supported_operations": [
                "pattern_recognition",
                "category_learning",
                "adaptive_resonance", 
                "art_training",
                "bridge_operation"
            ],
            "components": list(self.available_components.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "PROCESSOR",
            "symbolic_depth": 3
        }
    
    async def shutdown(self):
        """Shutdown the ART module gracefully."""
        try:
            logger.info("Shutting down ART module...")
            
            # Shutdown components that support it
            if self.art_controller and hasattr(self.art_controller, 'shutdown'):
                if asyncio.iscoroutinefunction(self.art_controller.shutdown):
                    await self.art_controller.shutdown()
                else:
                    self.art_controller.shutdown()
            
            if self.art_adapter and hasattr(self.art_adapter, 'shutdown'):
                if asyncio.iscoroutinefunction(self.art_adapter.shutdown):
                    await self.art_adapter.shutdown()
                else:
                    self.art_adapter.shutdown()
            
            if self.art_manager and hasattr(self.art_manager, 'shutdown'):
                if asyncio.iscoroutinefunction(self.art_manager.shutdown):
                    await self.art_manager.shutdown()
                else:
                    self.art_manager.shutdown()
            
            self.initialized = False
            logger.info("ART module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ART module shutdown: {e}")


# Registration function for the master orchestrator
async def register_art_module(vanta_core) -> ARTModuleAdapter:
    """Register the ART module with Vanta orchestrator."""
    logger.info("Registering ART module with Vanta orchestrator...")
    
    adapter = ARTModuleAdapter()
    success = await adapter.initialize(vanta_core)
    
    if success:
        logger.info("ART module registration successful")
    else:
        logger.error("ART module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['ARTModuleAdapter', 'register_art_module']
