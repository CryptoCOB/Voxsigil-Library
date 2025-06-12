"""
Interfaces Module Vanta Registration

This module provides registration capabilities for the Interfaces module
with the Vanta orchestrator system.

Components registered:
- LLMInterface: Large Language Model interaction interface
- NeuralInterface: Neural network inference interface  
- MemoryInterface: Memory system interface
- RAGInterface: Retrieval-Augmented Generation interface
- BLTEncoderInterface: BLT encoder interface
- ARCLLMInterface: ARC-specific LLM interface
- TrainingInterface: Training workflow interface
- ModelDiscoveryInterface: Model discovery and loading interface

HOLO-1.5 Integration: Interface adaptation and cross-system communication capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# Configure logging
logger = logging.getLogger(__name__)

class InterfacesModuleAdapter:
    """Adapter for integrating Interfaces module with Vanta orchestrator."""
    
    def __init__(self):
        self.module_id = "interfaces"
        self.display_name = "System Interfaces Module"
        self.version = "1.0.0"
        self.description = "Standardized interfaces for cross-system communication and integration"
        
        # Interface instances
        self.llm_interface = None
        self.neural_interface = None
        self.memory_interface = None
        self.rag_interface = None
        self.blt_interface = None
        self.arc_interface = None
        self.training_interface = None
        self.model_discovery_interface = None
        self.initialized = False
        
        # Available interface components
        self.available_interfaces = {
            'llm_interface': 'llm_interface.LLMInterface',
            'neural_interface': 'neural_interface.NeuralInterface',
            'memory_interface': 'memory_interface.MemoryInterface',
            'rag_interface': 'rag_interface.RAGInterface', 
            'blt_encoder_interface': 'blt_encoder_interface.BLTEncoderInterface',
            'arc_llm_interface': 'arc_llm_interface.ARCLLMInterface',
            'training_interface': 'training_interface.TrainingInterface',
            'model_discovery_interface': 'model_discovery_interface.ModelDiscoveryInterface'
        }
        
    async def initialize(self, vanta_core):
        """Initialize the Interfaces module with vanta core."""
        try:
            logger.info(f"Initializing Interfaces module with Vanta core...")
            
            # Initialize LLM Interface
            try:
                llm_interface_module = importlib.import_module('interfaces.llm_interface')
                if hasattr(llm_interface_module, 'LLMInterface'):
                    LLMInterfaceClass = getattr(llm_interface_module, 'LLMInterface')
                    self.llm_interface = self._initialize_component(LLMInterfaceClass, vanta_core)
                else:
                    logger.info("LLM Interface class not found, using module functions")
                    self.llm_interface = llm_interface_module
            except ImportError as e:
                logger.warning(f"LLM Interface not available: {e}")
            
            # Initialize Neural Interface
            try:
                neural_interface_module = importlib.import_module('interfaces.neural_interface')
                NeuralInterfaceClass = getattr(neural_interface_module, 'NeuralInterface')
                self.neural_interface = self._initialize_component(NeuralInterfaceClass, vanta_core)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Neural Interface not available: {e}")
            
            # Initialize Memory Interface
            try:
                memory_interface_module = importlib.import_module('interfaces.memory_interface')
                if hasattr(memory_interface_module, 'MemoryInterface'):
                    MemoryInterfaceClass = getattr(memory_interface_module, 'MemoryInterface')
                    self.memory_interface = self._initialize_component(MemoryInterfaceClass, vanta_core)
                else:
                    self.memory_interface = memory_interface_module
            except ImportError as e:
                logger.warning(f"Memory Interface not available: {e}")
            
            # Initialize RAG Interface
            try:
                rag_interface_module = importlib.import_module('interfaces.rag_interface')
                if hasattr(rag_interface_module, 'RAGInterface'):
                    RAGInterfaceClass = getattr(rag_interface_module, 'RAGInterface')
                    self.rag_interface = self._initialize_component(RAGInterfaceClass, vanta_core)
                else:
                    self.rag_interface = rag_interface_module
            except ImportError as e:
                logger.warning(f"RAG Interface not available: {e}")
            
            # Initialize BLT Encoder Interface
            try:
                blt_interface_module = importlib.import_module('interfaces.blt_encoder_interface')
                if hasattr(blt_interface_module, 'BLTEncoderInterface'):
                    BLTInterfaceClass = getattr(blt_interface_module, 'BLTEncoderInterface')
                    self.blt_interface = self._initialize_component(BLTInterfaceClass, vanta_core)
                else:
                    self.blt_interface = blt_interface_module
            except ImportError as e:
                logger.warning(f"BLT Encoder Interface not available: {e}")
            
            # Initialize ARC LLM Interface
            try:
                arc_interface_module = importlib.import_module('interfaces.arc_llm_interface')
                if hasattr(arc_interface_module, 'ARCLLMInterface'):
                    ARCInterfaceClass = getattr(arc_interface_module, 'ARCLLMInterface')
                    self.arc_interface = self._initialize_component(ARCInterfaceClass, vanta_core)
                else:
                    self.arc_interface = arc_interface_module
            except ImportError as e:
                logger.warning(f"ARC LLM Interface not available: {e}")
            
            # Initialize Training Interface
            try:
                training_interface_module = importlib.import_module('interfaces.training_interface')
                if hasattr(training_interface_module, 'TrainingInterface'):
                    TrainingInterfaceClass = getattr(training_interface_module, 'TrainingInterface')
                    self.training_interface = self._initialize_component(TrainingInterfaceClass, vanta_core)
                else:
                    self.training_interface = training_interface_module
            except ImportError as e:
                logger.warning(f"Training Interface not available: {e}")
            
            # Initialize Model Discovery Interface
            try:
                model_discovery_module = importlib.import_module('interfaces.model_discovery_interface')
                if hasattr(model_discovery_module, 'ModelDiscoveryInterface'):
                    ModelDiscoveryClass = getattr(model_discovery_module, 'ModelDiscoveryInterface')
                    self.model_discovery_interface = self._initialize_component(ModelDiscoveryClass, vanta_core)
                else:
                    self.model_discovery_interface = model_discovery_module
            except ImportError as e:
                logger.warning(f"Model Discovery Interface not available: {e}")
            
            # Initialize async components
            for interface in [self.llm_interface, self.neural_interface, self.memory_interface, 
                            self.rag_interface, self.blt_interface, self.arc_interface,
                            self.training_interface, self.model_discovery_interface]:
                if interface and hasattr(interface, 'async_init'):
                    await interface.async_init()
            
            self.initialized = True
            logger.info(f"Interfaces module initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Interfaces module: {e}")
            return False
    
    def _initialize_component(self, component_class, vanta_core):
        """Helper method to initialize a component with vanta_core."""
        try:
            # Try to initialize with vanta_core
            return component_class(vanta_core=vanta_core)
        except TypeError:
            try:
                # Try without vanta_core
                instance = component_class()
                # Try to set vanta_core afterwards
                if hasattr(instance, 'set_vanta_core'):
                    instance.set_vanta_core(vanta_core)
                elif hasattr(instance, 'vanta_core'):
                    instance.vanta_core = vanta_core
                return instance
            except Exception as e:
                logger.warning(f"Failed to initialize component {component_class.__name__}: {e}")
                return None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for Interface module operations."""
        if not self.initialized:
            return {"error": "Interfaces module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            interface_name = request.get('interface', 'unknown')
            
            if request_type == 'llm_operation':
                return await self._handle_llm_operation(request)
            elif request_type == 'neural_inference':
                return await self._handle_neural_inference(request)
            elif request_type == 'memory_operation':
                return await self._handle_memory_operation(request)
            elif request_type == 'rag_operation':
                return await self._handle_rag_operation(request)
            elif request_type == 'blt_encoding':
                return await self._handle_blt_encoding(request)
            elif request_type == 'arc_operation':
                return await self._handle_arc_operation(request)
            elif request_type == 'training_operation':
                return await self._handle_training_operation(request)
            elif request_type == 'model_discovery':
                return await self._handle_model_discovery(request)
            elif request_type == 'interface_call':
                return await self._handle_interface_call(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing Interfaces request: {e}")
            return {"error": str(e)}
    
    async def _handle_llm_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM interface operations."""
        if not self.llm_interface:
            return {"error": "LLM Interface not available"}
        
        operation = request.get('operation', 'chat')
        params = request.get('params', {})
        
        try:
            if hasattr(self.llm_interface, operation):
                method = getattr(self.llm_interface, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            elif hasattr(self.llm_interface, 'llm_chat_completion') and operation == 'chat':
                result = self.llm_interface.llm_chat_completion(**params)
            else:
                return {"error": f"LLM operation {operation} not supported"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"LLM operation failed: {e}"}
    
    async def _handle_neural_inference(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle neural inference operations."""
        if not self.neural_interface:
            return {"error": "Neural Interface not available"}
        
        input_data = request.get('input_data')
        model_config = request.get('model_config', {})
        
        try:
            if hasattr(self.neural_interface, 'inference'):
                result = await self.neural_interface.inference(input_data, **model_config)
            elif hasattr(self.neural_interface, 'predict'):
                result = self.neural_interface.predict(input_data, **model_config)
            else:
                return {"error": "Neural inference method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Neural inference failed: {e}"}
    
    async def _handle_memory_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory interface operations."""
        if not self.memory_interface:
            return {"error": "Memory Interface not available"}
        
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.memory_interface, operation):
                method = getattr(self.memory_interface, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            else:
                return {"error": f"Memory operation {operation} not supported"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Memory operation failed: {e}"}
    
    async def _handle_rag_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG interface operations.""" 
        if not self.rag_interface:
            return {"error": "RAG Interface not available"}
        
        query = request.get('query')
        context = request.get('context', {})
        
        try:
            if hasattr(self.rag_interface, 'retrieve_and_generate'):
                result = await self.rag_interface.retrieve_and_generate(query, context)
            elif hasattr(self.rag_interface, 'query'):
                result = self.rag_interface.query(query, **context)
            else:
                return {"error": "RAG operation method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"RAG operation failed: {e}"}
    
    async def _handle_blt_encoding(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle BLT encoder operations."""
        if not self.blt_interface:
            return {"error": "BLT Interface not available"}
        
        input_data = request.get('input_data')
        encoding_params = request.get('encoding_params', {})
        
        try:
            if hasattr(self.blt_interface, 'encode'):
                result = await self.blt_interface.encode(input_data, **encoding_params)
            elif hasattr(self.blt_interface, 'transform'):
                result = self.blt_interface.transform(input_data, **encoding_params)
            else:
                return {"error": "BLT encoding method not found"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"BLT encoding failed: {e}"}
    
    async def _handle_arc_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ARC interface operations."""
        if not self.arc_interface:
            return {"error": "ARC Interface not available"}
        
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.arc_interface, operation):
                method = getattr(self.arc_interface, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            else:
                return {"error": f"ARC operation {operation} not supported"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"ARC operation failed: {e}"}
    
    async def _handle_training_operation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training interface operations."""
        if not self.training_interface:
            return {"error": "Training Interface not available"}
        
        operation = request.get('operation')
        params = request.get('params', {})
        
        try:
            if hasattr(self.training_interface, operation):
                method = getattr(self.training_interface, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            else:
                return {"error": f"Training operation {operation} not supported"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Training operation failed: {e}"}
    
    async def _handle_model_discovery(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model discovery operations."""
        if not self.model_discovery_interface:
            return {"error": "Model Discovery Interface not available"}
        
        operation = request.get('operation', 'discover')
        params = request.get('params', {})
        
        try:
            if hasattr(self.model_discovery_interface, operation):
                method = getattr(self.model_discovery_interface, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            else:
                return {"error": f"Model discovery operation {operation} not supported"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Model discovery failed: {e}"}
    
    async def _handle_interface_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic interface calls."""
        interface_name = request.get('interface')
        method_name = request.get('method')
        params = request.get('params', {})
        
        # Map interface names to instances
        interface_map = {
            'llm': self.llm_interface,
            'neural': self.neural_interface,
            'memory': self.memory_interface,
            'rag': self.rag_interface,
            'blt': self.blt_interface,
            'arc': self.arc_interface,
            'training': self.training_interface,
            'model_discovery': self.model_discovery_interface
        }
        
        interface = interface_map.get(interface_name)
        if not interface:
            return {"error": f"Interface {interface_name} not available"}
        
        try:
            if hasattr(interface, method_name):
                method = getattr(interface, method_name)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**params)
                else:
                    result = method(**params)
            else:
                return {"error": f"Method {method_name} not found in {interface_name} interface"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Interface call failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Interfaces module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "llm_interface",
                "neural_inference",
                "memory_interface",
                "rag_interface",
                "blt_encoding",
                "arc_interface",
                "training_interface",
                "model_discovery",
                "cross_system_communication"
            ],
            "supported_operations": [
                "llm_operation",
                "neural_inference",
                "memory_operation",
                "rag_operation",
                "blt_encoding",
                "arc_operation", 
                "training_operation",
                "model_discovery",
                "interface_call"
            ],
            "interfaces": list(self.available_interfaces.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "PROCESSOR",
            "symbolic_depth": 2
        }
    
    async def shutdown(self):
        """Shutdown the Interfaces module gracefully."""
        try:
            logger.info("Shutting down Interfaces module...")
            
            # Shutdown all interfaces that support it
            interfaces = [self.llm_interface, self.neural_interface, self.memory_interface,
                         self.rag_interface, self.blt_interface, self.arc_interface,
                         self.training_interface, self.model_discovery_interface]
            
            for interface in interfaces:
                if interface and hasattr(interface, 'shutdown'):
                    if asyncio.iscoroutinefunction(interface.shutdown):
                        await interface.shutdown()
                    else:
                        interface.shutdown()
            
            self.initialized = False
            logger.info("Interfaces module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Interfaces module shutdown: {e}")


# Registration function for the master orchestrator
async def register_interfaces_module(vanta_core) -> InterfacesModuleAdapter:
    """Register the Interfaces module with Vanta orchestrator."""
    logger.info("Registering Interfaces module with Vanta orchestrator...")
    
    adapter = InterfacesModuleAdapter()
    success = await adapter.initialize(vanta_core)
    
    if success:
        logger.info("Interfaces module registration successful")
    else:
        logger.error("Interfaces module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['InterfacesModuleAdapter', 'register_interfaces_module']
