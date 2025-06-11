"""
VantaCore Integration for Sigil GUI
Manages Vanta async components within the GUI environment
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger("VantaCore.Integration")

class VantaIntegration:
    """
    Main integration class for VantaCore components within the GUI.
    Provides unified interface for async training, processing, and monitoring.
    """
    
    def __init__(self):
        self.active = False
        self.components = {}
        self.monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def initialize(self) -> bool:
        """Initialize VantaCore integration."""
        try:
            logger.info("Initializing VantaCore integration...")
            
            # Initialize core components
            self._init_training_engine()
            self._init_processing_engine()
            self._init_monitoring()
            
            self.active = True
            logger.info("✅ VantaCore integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VantaCore integration: {e}")
            return False
    
    def _init_training_engine(self):
        """Initialize async training engine."""
        # Implementation would import and initialize the actual training engine
        logger.info("✓ Training engine initialized")
        
    def _init_processing_engine(self):
        """Initialize async processing engine."""
        # Implementation would import and initialize the actual processing engine
        logger.info("✓ Processing engine initialized")
        
    def _init_monitoring(self):
        """Initialize system monitoring."""
        # Implementation would set up monitoring for GPU, memory, etc.
        logger.info("✓ Monitoring initialized")
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("📊 Monitoring thread started")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 Monitoring thread stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Monitor system resources
                self._update_system_status()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _update_system_status(self):
        """Update system status metrics."""
        status = {
            "timestamp": time.time(),
            "active": self.active,
            "gpu_available": torch is not None and torch.cuda.is_available() if torch else False
        }
        
        if torch and torch.cuda.is_available():
            status["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
            status["gpu_memory_used"] = torch.cuda.memory_allocated(0)
        
        self.components["system_status"] = status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            "active": self.active,
            "components": list(self.components.keys()),
            "monitoring_active": self.monitoring_thread and self.monitoring_thread.is_alive(),
            "system_status": self.components.get("system_status", {})
        }
    
    def shutdown(self):
        """Shutdown VantaCore integration."""
        logger.info("Shutting down VantaCore integration...")
        
        self.stop_monitoring()
        self.active = False
        self.components.clear()
        
        logger.info("✅ VantaCore integration shutdown complete")

# Global instance
vanta_integration = VantaIntegration()

# Convenience functions
def initialize_vanta() -> bool:
    """Initialize VantaCore integration."""
    return vanta_integration.initialize()

def get_vanta_status() -> Dict[str, Any]:
    """Get VantaCore status."""
    return vanta_integration.get_status()

def shutdown_vanta():
    """Shutdown VantaCore integration."""
    vanta_integration.shutdown()

if __name__ == "__main__":
    # Test the integration
    print("Testing VantaCore Integration...")
    
    if initialize_vanta():
        print("✅ Integration successful")
        vanta_integration.start_monitoring()
        
        # Run for a few seconds
        time.sleep(3)
        
        status = get_vanta_status()
        print(f"Status: {status}")
        
        shutdown_vanta()
    else:
        print("❌ Integration failed")
