# core/vanta_registration.py
"""
Core Utilities Registration with Vanta
Registers core utility modules and managers
"""

import logging
from typing import Dict, Any, List, Optional, Type

logger = logging.getLogger("Vanta.CoreRegistration")


class CoreModuleAdapter:
    """Adapter for registering core utility modules as Vanta modules."""
    
    def __init__(self, module_id: str, module_class: Type, description: str):
        self.module_id = module_id
        self.module_class = module_class
        self.description = description
        self.module_instance = None
        
    async def initialize(self, vanta_core):
        """Initialize the core module instance with vanta core."""
        try:
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
                
            logger.info(f"Core module {self.module_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize core module {self.module_id}: {e}")
            return False
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the core module."""
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
                
            return {"module": self.module_id, "result": result}
        except Exception as e:
            logger.error(f"Error processing request in core module {self.module_id}: {e}")
            return {"error": str(e)}
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get core module metadata for Vanta registration."""
        metadata = {
            "type": "core_utility",
            "description": self.description,
            "capabilities": self._extract_capabilities(),
            "module_class": self.module_class.__name__,
        }
        return metadata
        
    def _extract_capabilities(self) -> List[str]:
        """Extract capabilities based on module type."""
        module_name = self.module_id.lower()
        capabilities = ["utility"]
        
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
