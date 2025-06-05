#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportUnknownMemberType=false, reportIncompatibleVariableOverride=false
"""
VoxSigil Supervisor Integration Module

This module bridges the GUI interface components with the VoxSigil supervisor interfaces,
ensuring that GUI tabs can access the core VoxSigil functionality including:
- Memory management
- RAG (Retrieval Augmented Generation)
- Learning management
- Model management
- Check-in management

Created to fix the disconnection between GUI tabs and VoxSigil supervisor interfaces.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

# Try to import torch for GPU monitoring
try:
    import torch
except ImportError:
    torch = None

# Use standard path helper for imports
try:
    from utils.path_helper import add_project_root_to_path
    add_project_root_to_path()
except ImportError:
    # Fallback if path_helper isn't available
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(PROJECT_ROOT))

# Setup logging
logger = logging.getLogger("VoxSigil.GUI.Integration")

# Import UnifiedVantaCore for centralized orchestration
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore, get_vanta_core
    
    # Import base interfaces for fallback compatibility
    from Vanta.interfaces.memory_interface import (
        BaseMemoryInterface as VantaBaseMemoryInterface,
    )
    from Vanta.interfaces.rag_interface import BaseRagInterface as VantaBaseRagInterface

    # Import training engine components for comprehensive VantaCore integration
    from Vanta.async_training_engine import (
        AsyncTrainingEngine,
        TrainingConfig,
        TrainingJob,
    )

    UNIFIED_VANTA_AVAILABLE = True
    TRAINING_ENGINE_AVAILABLE = True
    logger.info("Successfully imported UnifiedVantaCore and training components for GUI integration")
except ImportError as e:
    logger.warning(f"UnifiedVantaCore or training components not available: {e}")
    UNIFIED_VANTA_AVAILABLE = False
    TRAINING_ENGINE_AVAILABLE = False
    UnifiedVantaCore = None
    get_vanta_core = None
    VantaBaseMemoryInterface = None
    VantaBaseRagInterface = None
    AsyncTrainingEngine = None
    TrainingConfig = None
    TrainingJob = None

# Legacy import fallback for compatibility
try:
    from Vanta.interfaces.checkin_manager_vosk import (
        VantaInteractionManager as VantaInteractionManagerClass,
    )
    from Vanta.interfaces.learning_manager import (
        LearningManager as VantaLearningManagerClass,
    )
    from Vanta.interfaces.memory_interface import (
        JsonFileMemoryInterface as JSONFileMemoryManagerClass,
    )
    from Vanta.interfaces.model_manager import ModelManager as VantaModelManagerClass
    from Vanta.interfaces.rag_interface import (
        SupervisorRagInterface as VoxSigilRAGManagerClass,
    )

    LEGACY_INTERFACES_AVAILABLE = True
    logger.info("Legacy VoxSigil interfaces available as fallback")
except ImportError as e:
    logger.warning(f"Legacy interfaces not available: {e}")
    LEGACY_INTERFACES_AVAILABLE = False
    VantaInteractionManagerClass = None
    VantaLearningManagerClass = None
    JSONFileMemoryManagerClass = None
    VantaModelManagerClass = None
    VoxSigilRAGManagerClass = None


class SimpleTrainingEngine:
    """Simple fallback training engine for when AsyncTrainingEngine is not available"""
    
    def __init__(self, vanta_core, config):
        self.vanta_core = vanta_core
        self.config = config
        self.current_job = None
        
    def start_training(self, model_path: str, dataset_path: str, config_updates: Optional[Dict] = None):
        """Start a training job"""
        logger.info(f"Starting simple training: {model_path}")
        return {"status": "started", "job_id": "simple_job"}
        
    def stop_training(self):
        """Stop current training"""
        logger.info("Stopping simple training")
        return {"status": "stopped"}
        
    def get_status(self):
        """Get training status"""
        return {"status": "idle", "progress": 0}


class VoxSigilIntegrationManager:
    """
    Comprehensive VoxSigil integration manager that orchestrates all VoxSigil GUI operations
    through UnifiedVantaCore for complete system connectivity.
    
    This class serves as the bridge between GUI components and the VoxSigil supervisor
    interfaces, ensuring proper connectivity and functionality access.
    """

    def __init__(self, parent_gui=None):
        """
        Initialize VoxSigil integration with comprehensive VantaCore connectivity.
        
        Args:
            parent_gui: Reference to the parent GUI component for callbacks
        """
        self.parent_gui = parent_gui
        self.logger = logging.getLogger("VoxSigil.Integration.Manager")
        # Core integration state
        self.unified_core = None
        self.use_unified_core = False
        # Training engine integration
        self.training_engine = None
        self.current_training_job = None
        self.training_status_callbacks = []
        
        # GPU monitoring
        self.gpu_monitor_thread = None
        self.gpu_monitoring_active = False
        self.gpu_status = {
            "available": False,
            "device_count": 0,
            "memory_allocated": 0,
            "memory_reserved": 0,
            "memory_total": 0,
            "device_name": "Unknown",
            "cuda_version": "Unknown"
        }
        
        # Component managers (legacy fallback)
        self.memory_manager = None
        self.rag_manager = None
        self.learning_manager = None
        self.model_manager = None
        self.checkin_manager = None
        
        # Component status tracking
        self.component_status = {
            "memory": False,
            "rag": False,
            "learning": False,
            "model": False,
            "checkin": False,
            "training": False,
            "gpu_monitoring": False
        }
        
        # Initialize integration
        self.initialize_integration()

    def initialize_integration(self):
        """Initialize the comprehensive VoxSigil integration system"""
        self.logger.info("Initializing VoxSigil integration with UnifiedVantaCore")
        
        try:
            if UNIFIED_VANTA_AVAILABLE:
                self.setup_unified_vanta_integration()
            else:
                self.logger.warning("UnifiedVantaCore not available, falling back to legacy interfaces")
                self.setup_legacy_integration()
                
            # Initialize training engine regardless of core type
            self.initialize_training_engine()
            
            # Start GPU monitoring
            self.start_gpu_monitoring()
            
            self.logger.info("VoxSigil integration initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VoxSigil integration: {e}")
            self.setup_emergency_fallback()

    def setup_unified_vanta_integration(self):
        """Setup integration with UnifiedVantaCore"""
        try:
            # Get or create UnifiedVantaCore instance
            if get_vanta_core:
                self.unified_core = get_vanta_core()
            else:
                self.unified_core = UnifiedVantaCore()
            
            self.use_unified_core = True
            
            # Setup event subscriptions if events system is available
            if hasattr(self.unified_core, 'events'):
                self.setup_event_subscriptions()
            
            # Register GUI integration with the core
            if hasattr(self.unified_core, 'registry'):
                self.unified_core.registry.register(
                    "voxsigil_gui_integration",
                    self,
                    {"type": "VoxSigilIntegrationManager", "version": "1.0"}
                )
            
            self.component_status["memory"] = True
            self.component_status["rag"] = True
            self.component_status["learning"] = True
            self.component_status["model"] = True
            self.component_status["checkin"] = True
            
            self.logger.info("UnifiedVantaCore integration established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup UnifiedVantaCore integration: {e}")
            self.setup_legacy_integration()

    def setup_event_subscriptions(self):
        """Setup event subscriptions for training and system events"""
        if not self.unified_core or not hasattr(self.unified_core, 'events'):
            self.logger.warning("Event system not available for subscriptions")
            return
            
        try:
            # Subscribe to training events
            self.unified_core.events.subscribe("training.job_started", self.on_training_started)
            self.unified_core.events.subscribe("training.job_progress", self.on_training_progress)
            self.unified_core.events.subscribe("training.job_completed", self.on_training_completed)
            self.unified_core.events.subscribe("training.job_failed", self.on_training_failed)
            self.unified_core.events.subscribe("training.job_paused", self.on_training_paused)
            
            # Subscribe to GPU events
            self.unified_core.events.subscribe("gpu.status_update", self.on_gpu_status_update)
            
            self.logger.info("Event subscriptions established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup event subscriptions: {e}")

    def setup_legacy_integration(self):
        """Setup legacy interface integration as fallback"""
        if not LEGACY_INTERFACES_AVAILABLE:
            self.logger.warning("Legacy interfaces not available, using emergency fallback")
            return
            
        try:
            # Initialize legacy managers with simple fallback approach
            if JSONFileMemoryManagerClass:
                try:
                    self.memory_manager = JSONFileMemoryManagerClass()
                    self.component_status["memory"] = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize memory manager: {e}")
                
            if VoxSigilRAGManagerClass:
                try:
                    self.rag_manager = VoxSigilRAGManagerClass()
                    self.component_status["rag"] = True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize RAG manager: {e}")
                
            # Skip complex managers that require specific parameters for now
            self.logger.info("Legacy interface integration established with basic components")
            
        except Exception as e:
            self.logger.error(f"Failed to setup legacy integration: {e}")

    def setup_emergency_fallback(self):
        """Setup emergency fallback with minimal functionality"""
        self.logger.warning("Setting up emergency fallback integration")
        # All component_status remains False, indicating unavailable services

    def initialize_training_engine(self):
        """Initialize the training engine for VoxSigil"""
        try:
            if TRAINING_ENGINE_AVAILABLE and self.unified_core:
                # Create training configuration
                training_config = {
                    "max_epochs": 10,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "save_steps": 100,
                    "eval_steps": 50,
                    "warmup_steps": 50,
                    "device": "auto",
                    "output_dir": str(Path.cwd() / "voxsigil_training_outputs"),
                    "checkpoint_dir": str(Path.cwd() / "voxsigil_checkpoints"),
                    "tensorboard_log_dir": str(Path.cwd() / "logs" / "voxsigil_tensorboard"),
                }
                
                if TrainingConfig and AsyncTrainingEngine:
                    config_obj = TrainingConfig(**training_config)
                    self.training_engine = AsyncTrainingEngine(self.unified_core, config_obj)
                else:
                    self.training_engine = SimpleTrainingEngine(self.unified_core, training_config)
                    
                # Register with core if available
                if hasattr(self.unified_core, 'registry'):
                    self.unified_core.registry.register(
                        "voxsigil_training_engine",
                        self.training_engine,
                        {"type": "VoxSigilTrainingEngine", "version": "1.0"}
                    )
                    
                self.component_status["training"] = True
                self.logger.info("VoxSigil training engine initialized successfully")
                
            else:
                # Create simple fallback training engine
                self.training_engine = SimpleTrainingEngine(self.unified_core, {})
                self.component_status["training"] = True
                self.logger.info("Simple fallback training engine created")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize training engine: {e}")
            self.training_engine = None

    def start_gpu_monitoring(self):
        """Start GPU monitoring for VoxSigil operations"""
        if self.gpu_monitoring_active:
            return
            
        self.gpu_monitoring_active = True
        self.gpu_monitor_thread = threading.Thread(
            target=self._gpu_monitor_loop,
            daemon=True,
            name="VoxSigil-GPU-Monitor"
        )
        self.gpu_monitor_thread.start()
        self.component_status["gpu_monitoring"] = True
        self.logger.info("GPU monitoring started for VoxSigil")

    def _gpu_monitor_loop(self):
        """GPU monitoring loop for VoxSigil"""
        while self.gpu_monitoring_active:
            try:
                if torch and torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        # Get memory info for primary GPU
                        memory_allocated = torch.cuda.memory_allocated(0)
                        memory_reserved = torch.cuda.memory_reserved(0)
                        
                        # Get total memory
                        try:
                            total_memory = torch.cuda.get_device_properties(0).total_memory
                        except Exception:
                            total_memory = memory_reserved + (1024**3)  # Fallback estimate
                            
                        self.gpu_status = {
                            "available": True,
                            "device_count": device_count,
                            "memory_allocated": memory_allocated,
                            "memory_reserved": memory_reserved,
                            "memory_total": total_memory,
                            "device_name": torch.cuda.get_device_name(0),
                            "cuda_version": torch.__version__,
                        }
                    else:
                        self.gpu_status["available"] = False
                else:
                    self.gpu_status["available"] = False
                    
                # Publish GPU status update if event system available
                if (self.unified_core and 
                    hasattr(self.unified_core, 'events') and 
                    hasattr(self.unified_core.events, 'publish')):
                    self.unified_core.events.publish("gpu.status_update", self.gpu_status)
                    
            except Exception as e:
                self.logger.error(f"GPU monitoring error: {e}")
                
            time.sleep(5)  # Update every 5 seconds

    # Training Event Handlers
    def on_training_started(self, event_data):
        """Handle training started event"""
        self.logger.info(f"Training started: {event_data}")
        for callback in self.training_status_callbacks:
            try:
                callback("started", event_data)
            except Exception as e:
                self.logger.error(f"Training callback error: {e}")

    def on_training_progress(self, event_data):
        """Handle training progress event"""
        for callback in self.training_status_callbacks:
            try:
                callback("progress", event_data)
            except Exception as e:
                self.logger.error(f"Training callback error: {e}")

    def on_training_completed(self, event_data):
        """Handle training completed event"""
        self.logger.info(f"Training completed: {event_data}")
        self.current_training_job = None
        for callback in self.training_status_callbacks:
            try:
                callback("completed", event_data)
            except Exception as e:
                self.logger.error(f"Training callback error: {e}")

    def on_training_failed(self, event_data):
        """Handle training failed event"""
        self.logger.error(f"Training failed: {event_data}")
        self.current_training_job = None
        for callback in self.training_status_callbacks:
            try:
                callback("failed", event_data)
            except Exception as e:
                self.logger.error(f"Training callback error: {e}")

    def on_training_paused(self, event_data):
        """Handle training paused event"""
        self.logger.info(f"Training paused: {event_data}")
        for callback in self.training_status_callbacks:
            try:
                callback("paused", event_data)
            except Exception as e:
                self.logger.error(f"Training callback error: {e}")

    def on_gpu_status_update(self, event_data):
        """Handle GPU status update event"""
        self.gpu_status.update(event_data)

    # Training Management Methods
    def start_training(self, model_path: str, dataset_path: str, config_updates: Optional[Dict] = None):
        """Start a training job through VoxSigil integration"""
        if not self.training_engine:
            return {"error": "Training engine not available"}
            
        try:
            result = self.training_engine.start_training(model_path, dataset_path, config_updates)
            if hasattr(result, 'job_id'):
                self.current_training_job = result
            return {"status": "started", "result": result}
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            return {"error": str(e)}

    def stop_training(self):
        """Stop current training job"""
        if not self.training_engine:
            return {"error": "Training engine not available"}
            
        try:
            result = self.training_engine.stop_training()
            self.current_training_job = None
            return {"status": "stopped", "result": result}
        except Exception as e:
            self.logger.error(f"Failed to stop training: {e}")
            return {"error": str(e)}

    def get_training_status(self):
        """Get current training status"""
        if not self.training_engine:
            return {"status": "unavailable"}
            
        try:
            return self.training_engine.get_status()
        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}")
            return {"error": str(e)}

    def register_training_callback(self, callback):
        """Register a callback for training status updates"""
        self.training_status_callbacks.append(callback)

    # Core VoxSigil Interface Methods
    def get_memory_interface(self):
        """Get the memory management interface"""
        if self.use_unified_core and self.unified_core:
            return getattr(self.unified_core, 'memory', self.memory_manager)
        return self.memory_manager

    def get_rag_interface(self):
        """Get the RAG interface"""
        if self.use_unified_core and self.unified_core:
            return getattr(self.unified_core, 'rag', self.rag_manager)
        return self.rag_manager

    def get_learning_interface(self):
        """Get the learning management interface"""
        if self.use_unified_core and self.unified_core:
            return getattr(self.unified_core, 'learning', self.learning_manager)
        return self.learning_manager

    def get_model_interface(self):
        """Get the model management interface"""
        if self.use_unified_core and self.unified_core:
            return getattr(self.unified_core, 'model', self.model_manager)
        return self.model_manager

    def get_checkin_interface(self):
        """Get the check-in management interface"""
        if self.use_unified_core and self.unified_core:
            return getattr(self.unified_core, 'checkin', self.checkin_manager)
        return self.checkin_manager

    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            "integration_type": "unified" if self.use_unified_core else "legacy",
            "components": self.component_status.copy(),
            "gpu_status": self.gpu_status.copy(),
            "training_status": self.get_training_status(),
            "unified_core_available": self.unified_core is not None,
            "legacy_fallback_available": LEGACY_INTERFACES_AVAILABLE,
        }
        
        # Calculate overall health
        total_components = len(status["components"])
        active_components = sum(1 for active in status["components"].values() if active)
        status["health_percentage"] = (active_components / total_components) * 100 if total_components > 0 else 0
        
        return status

    def shutdown(self):
        """Shutdown the integration system gracefully"""
        self.logger.info("Shutting down VoxSigil integration")
        
        # Stop GPU monitoring
        self.gpu_monitoring_active = False
        if self.gpu_monitor_thread and self.gpu_monitor_thread.is_alive():
            self.gpu_monitor_thread.join(timeout=2)
            
        # Stop training if active
        if self.current_training_job:
            self.stop_training()
            
        # Clear callbacks
        self.training_status_callbacks.clear()
        
        # Cleanup references
        self.unified_core = None
        self.training_engine = None
        
        self.logger.info("VoxSigil integration shutdown complete")


# Convenience function for GUI access
def get_voxsigil_integration(parent_gui=None):
    """Get or create VoxSigil integration manager instance"""
    return VoxSigilIntegrationManager(parent_gui)


def test_voxsigil_integration():
    """Test VoxSigil integration functionality"""
    logger.info("Testing VoxSigil integration...")
    
    integration = get_voxsigil_integration()
    status = integration.get_system_status()
    
    logger.info(f"Integration Status: {json.dumps(status, indent=2)}")
    
    # Test interfaces
    memory_interface = integration.get_memory_interface()
    rag_interface = integration.get_rag_interface()
    learning_interface = integration.get_learning_interface()
    model_interface = integration.get_model_interface()
    checkin_interface = integration.get_checkin_interface()
    
    logger.info(f"Memory Interface: {memory_interface is not None}")
    logger.info(f"RAG Interface: {rag_interface is not None}")
    logger.info(f"Learning Interface: {learning_interface is not None}")
    logger.info(f"Model Interface: {model_interface is not None}")
    logger.info(f"Check-in Interface: {checkin_interface is not None}")    # Test training functionality
    training_status = integration.get_training_status()
    logger.info(f"Training Status: {training_status}")
    
    integration.shutdown()
    logger.info("VoxSigil integration test complete")


def initialize_voxsigil_integration(gui_instance=None) -> VoxSigilIntegrationManager:
    """
    Initialize VoxSigil integration for GUI components.

    Args:
        gui_instance: Reference to the main GUI instance

    Returns:
        VoxSigilIntegrationManager: Configured integration manager
    """
    return VoxSigilIntegrationManager(gui_instance)


# Global integration manager instance
_integration_manager = None


def get_integration_manager() -> VoxSigilIntegrationManager:
    """Get the global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = initialize_voxsigil_integration()
    return _integration_manager


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_voxsigil_integration()
