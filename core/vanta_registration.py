# core/vanta_registration.py
"""
Core Utilities Registration with Vanta
Registers core utility modules and managers

HOLO-1.5 Enhanced Core Module Registration Manager:
- Cognitive mesh registration and coordination for core utilities
- VantaCore-integrated module lifecycle management
- Neural-symbolic coordination of core module capabilities
- Recursive symbolic cognition mesh for enhanced system coordination
"""

import logging
from typing import Dict, Any, List, Optional, Type

# HOLO-1.5 Core Integration
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

logger = logging.getLogger("Vanta.CoreRegistration")


@vanta_core_module(
    name="core_module_adapter",
    subsystem="system_management",
    mesh_role=CognitiveMeshRole.MANAGER,
    description="HOLO-1.5 enhanced core module registration and lifecycle management adapter",
    capabilities=[
        "module_registration",
        "lifecycle_management",
        "adaptive_initialization",
        "capability_extraction",
        "cognitive_coordination"
    ],
    cognitive_load=3.0,
    symbolic_depth=2,
    collaboration_patterns=[
        "module_orchestration",
        "registration_synthesis",
        "adaptive_management"
    ]
)
class CoreModuleAdapter(BaseCore):
    """
    HOLO-1.5 Enhanced Adapter for registering core utility modules as Vanta modules.
    
    Provides cognitive mesh registration and coordination capabilities for:
    - Intelligent module lifecycle management with adaptive initialization
    - Neural-symbolic capability extraction and registration
    - Cognitive mesh coordination for enhanced system integration
    """
    
    def __init__(self, vanta_core=None, config: Dict[str, Any] = None, 
                 module_id: str = None, module_class: Type = None, description: str = None):
        super().__init__(vanta_core, config)
        self.module_id = module_id
        self.module_class = module_class
        self.description = description
        self.module_instance = None
        self.cognitive_metrics = {
            "registration_attempts": 0,
            "successful_initializations": 0,
            "capability_extractions": 0,
            "mesh_coordination_events": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the HOLO-1.5 enhanced core module adapter."""
        try:
            if self.holo15_adapter:
                await self.holo15_adapter.initialize()
            
            logger.info(f"🧠 HOLO-1.5 CoreModuleAdapter initialized for: {self.module_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HOLO-1.5 CoreModuleAdapter: {e}")
            return False        
    async def initialize_module(self, vanta_core):
        """Initialize the core module instance with cognitive enhancement."""
        try:
            self.cognitive_metrics["registration_attempts"] += 1
            
            # Try to initialize with vanta_core parameter first
            if hasattr(self.module_class, '__init__'):
                import inspect
                sig = inspect.signature(self.module_class.__init__)
                params = list(sig.parameters.keys())
                
                if 'vanta_core' in params:
                    self.module_instance = self.module_class(vanta_core=vanta_core)
                elif 'core' in params:
                    self.module_instance = self.module_class(core=vanta_core)
                elif 'config' in params:
                    # For modules that take config, try to create a config
                    try:
                        config = {}  # Basic config
                        self.module_instance = self.module_class(config=config)
                    except:
                        self.module_instance = self.module_class()
                else:
                    self.module_instance = self.module_class()
            else:
                self.module_instance = self.module_class()
                
            # If the module has an initialize method, call it
            if hasattr(self.module_instance, 'initialize'):
                await self.module_instance.initialize()
            elif hasattr(self.module_instance, 'setup'):
                self.module_instance.setup()
                
            self.cognitive_metrics["successful_initializations"] += 1
            
            # Update cognitive mesh with registration event
            if self.holo15_adapter:
                await self.holo15_adapter.process_symbolic_request(
                    "module_registered",
                    {"module_id": self.module_id, "type": "core_utility"}                )
                self.cognitive_metrics["mesh_coordination_events"] += 1
                
            logger.info(f"✅ Core module {self.module_id} initialized successfully with cognitive enhancement")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize core module {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the core module with cognitive enhancement."""
        if not self.module_instance:
            return {"error": f"Core module {self.module_id} not initialized"}
            
        try:
            # Route request to appropriate module method
            if hasattr(self.module_instance, 'process'):
                result = await self.module_instance.process(request)
            elif hasattr(self.module_instance, 'handle_request'):
                result = await self.module_instance.handle_request(request)
            elif hasattr(self.module_instance, 'execute'):
                result = await self.module_instance.execute(request)
            elif hasattr(self.module_instance, 'run'):
                result = self.module_instance.run(request)
            else:
                result = {"message": f"Core module {self.module_id} processed request"}
                
            # Update cognitive mesh with processing event
            if self.holo15_adapter:
                await self.holo15_adapter.process_symbolic_request(
                    "request_processed",
                    {"module_id": self.module_id, "request_type": request.get("type", "unknown")}
                )
                self.cognitive_metrics["mesh_coordination_events"] += 1
                
            return {"module": self.module_id, "result": result, "cognitive_enhanced": True}
        except Exception as e:
            logger.error(f"Error processing request in core module {self.module_id}: {e}")
            return {"error": str(e)}            
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get enhanced core module metadata for Vanta registration."""
        self.cognitive_metrics["capability_extractions"] += 1
        
        metadata = {
            "type": "core_utility",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "module_class": self.module_class.__name__,
            "holo15_enhanced": True,
            "cognitive_metrics": self.cognitive_metrics.copy(),
            "mesh_role": "MANAGER",
            "subsystem": "system_management"
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on module type with cognitive enhancement."""        
        
        module_name = self.module_id.lower()
        capabilities = ["utility", "cognitive_enhanced"]
        
        if 'learning' in module_name:
            capabilities.extend(['learning', 'training'])
        elif 'manager' in module_name:
            capabilities.extend(['management', 'coordination'])
        elif 'meta' in module_name:
            capabilities.extend(['meta_learning', 'reflection'])
        elif 'dialogue' in module_name:
            capabilities.extend(['conversation', 'dialogue_management'])
        elif 'grid' in module_name:
            capabilities.extend(['grid_processing', 'arc_tasks'])
        elif 'model' in module_name:
            capabilities.extend(['model_management', 'architecture'])
        elif 'neuro' in module_name:
            capabilities.extend(['neural_networks', 'symbolic_reasoning'])
        elif 'cognitive' in module_name:
            capabilities.extend(['cognitive_processing', 'reasoning'])
        elif 'hyperparameter' in module_name:
            capabilities.extend(['optimization', 'hyperparameter_tuning'])
        elif 'validation' in module_name:
            capabilities.extend(['validation', 'testing'])
        elif 'distillation' in module_name:
            capabilities.extend(['knowledge_distillation', 'compression'])
        elif 'intelligence' in module_name:
            capabilities.extend(['ai_reasoning', 'intelligence'])
        elif 'download' in module_name:
            capabilities.extend(['data_download', 'dataset_management'])
        elif 'checkin' in module_name:
            capabilities.extend(['monitoring', 'status_checking'])
            
        return capabilities


def import_core_class(module_name: str):
    """Dynamically import a core module class."""
    try:
        module = __import__(f'core.{module_name}', fromlist=[module_name])
        
        # Class mapping for known modules
        class_mapping = {
            'AdvancedMetaLearner': 'AdvancedMetaLearner',
            'checkin_manager_vosk': 'CheckinManagerVosk',
            'default_learning_manager': 'DefaultLearningManager',
            'dialogue_manager': 'DialogueManager',
            'download_arc_data': 'ARCDataDownloader',
            'end_to_end_arc_validation': 'ARCValidator',
            'enhanced_grid_connector': 'EnhancedGridConnector',
            'grid_distillation': 'GridDistillation',
            'grid_former_evaluator': 'GridFormerEvaluator',
            'hyperparameter_search': 'HyperparameterSearch',
            'iterative_gridformer': 'IterativeGridFormer',
            'iterative_reasoning_gridformer': 'IterativeReasoningGridFormer',
            'learning_manager': 'LearningManager',
            'meta_cognitive': 'MetaCognitive',
            'model_architecture_fixer': 'ModelArchitectureFixer',
            'model_manager': 'ModelManager',
            'neuro_symbolic_network': 'NeuroSymbolicNetwork',
            'proactive_intelligence': 'ProactiveIntelligence',
        }
        
        class_name = class_mapping.get(module_name)
        if not class_name:
            # Fallback: try to find the main class in the module
            classes = [name for name in dir(module) if name[0].isupper() and not name.startswith('_')]
            if classes:
                class_name = classes[0]  # Take the first class found
            else:
                logger.error(f"No suitable class found in core module {module_name}")
                return None
                
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import core module {module_name}: {e}")
        return None


async def register_core_modules():
    """Register core utility modules."""
    from Vanta import get_vanta_core_instance
    
    vanta = get_vanta_core_instance()
    
    core_modules = [
        ('AdvancedMetaLearner', 'Advanced meta-learning capabilities'),
        ('checkin_manager_vosk', 'Vosk-based check-in management'),
        ('default_learning_manager', 'Default learning coordination'),
        ('dialogue_manager', 'Conversation flow management'),
        ('download_arc_data', 'ARC dataset downloader'),
        ('end_to_end_arc_validation', 'Complete ARC validation'),
        ('enhanced_grid_connector', 'Enhanced grid processing'),
        ('grid_distillation', 'Grid knowledge distillation'),
        ('grid_former_evaluator', 'GridFormer evaluation'),
        ('hyperparameter_search', 'Hyperparameter optimization'),
        ('iterative_gridformer', 'Iterative grid processing'),
        ('iterative_reasoning_gridformer', 'Reasoning-enhanced GridFormer'),
        ('learning_manager', 'Learning process coordination'),
        ('meta_cognitive', 'Meta-cognitive processing'),
        ('model_architecture_fixer', 'Model architecture repair'),
        ('model_manager', 'Model lifecycle management'),
        ('neuro_symbolic_network', 'Neuro-symbolic integration'),
        ('proactive_intelligence', 'Proactive AI capabilities'),
    ]
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🧠 Starting registration of {len(core_modules)} core modules...")
    
    for module_name, description in core_modules:
        try:
            # Import the core module class
            module_class = import_core_class(module_name)
            if module_class is None:
                logger.warning(f"Skipping core module {module_name} - failed to import")
                failed_count += 1
                continue
                
            # Create adapter
            adapter = CoreModuleAdapter(
                module_id=f'core_{module_name}',
                module_class=module_class,
                description=description
            )
            
            # Register with Vanta
            await vanta.register_module(f'core_{module_name}', adapter)
            registered_count += 1
            logger.info(f"✅ Registered core module: {module_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register core module {module_name}: {e}")
            failed_count += 1
    
    logger.info(f"🎉 Core module registration complete: {registered_count} successful, {failed_count} failed")
    return registered_count, failed_count


if __name__ == "__main__":
    import asyncio
    asyncio.run(register_core_modules())
