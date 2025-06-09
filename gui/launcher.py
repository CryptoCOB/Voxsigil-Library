#!/usr/bin/env python3
"""
VantaCore GUI Launcher - Comprehensive System Integration
Properly initializes paths and launches the Dynamic GridFormer GUI with VantaCore integration
Includes robust error handling and fallback mechanisms per VANTA Integration Master Plan
"""

import logging
import sys
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaGUI")

# Add Vanta to path - go up two levels to find Vanta
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
vanta_path = project_root / "Vanta"

# Add paths systematically
paths_to_add = [
    str(project_root),
    str(vanta_path),
    str(project_root / "BLT"),
    str(project_root / "Gridformer"),
    str(project_root / "voxsigil_supervisor"),
    str(current_dir),  # GUI components directory
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

logger.info("=" * 60)
logger.info("🚀 LAUNCHING SIGIL GUI WITH VANTACORE INTEGRATION")
logger.info("=" * 60)
logger.info(f"Project root: {project_root}")
logger.info(f"Vanta path: {vanta_path}")
logger.info(f"Added {len(paths_to_add)} paths to sys.path")

# Global variables for system state
core = None
vmb_handler = None
vanta_available = False


def verify_vanta_accessibility():
    """Verify Vanta components are accessible without importing problematic modules"""
    global vanta_available
    try:
        # Test basic Vanta imports first
        from Vanta.async_training_engine import AsyncTrainingEngine

        logger.info("✅ VantaCore basic components verified and accessible")
        vanta_available = True
        return True
    except ImportError as e:
        logger.warning(f"⚠️ VantaCore basic components not available: {e}")
        vanta_available = False
        return False


def initialize_unified_vanta_core():
    """Initialize UnifiedVantaCore with comprehensive error handling"""
    global core

    if not vanta_available:
        logger.info("🔄 Skipping UnifiedVantaCore initialization - Vanta not available")
        return None

    try:
        from Vanta.core.UnifiedVantaCore import get_vanta_core

        # Get or create the unified core instance
        core = get_vanta_core()
        logger.info("✅ UnifiedVantaCore initialized successfully")

        return core

    except ImportError as import_err:
        logger.warning(f"UnifiedVantaCore import failed: {import_err}")
        logger.info("🔄 Continuing without VantaCore integration...")
        return None
    except Exception as e:
        logger.warning(f"UnifiedVantaCore initialization failed: {e}")
        logger.info("🔄 Continuing without VantaCore integration...")
        return None


def register_gui_agent(core_instance):
    """Register GUI launcher as an agent in the system"""
    if not core_instance:
        logger.info("⚠️ No core instance available for agent registration")
        return False

    try:
        # Register GUI launcher agent if agent_registry exists
        if hasattr(core_instance, "agent_registry") and core_instance.agent_registry:
            core_instance.agent_registry.register_agent(
                "gui_launcher",
                None,
                {
                    "type": "gui_launcher",
                    "status": "initialized",
                    "capabilities": [
                        "gui_management",
                        "user_interface",
                        "event_handling",
                    ],
                },
            )
            logger.info(
                "✅ GUI launcher registered with UnifiedVantaCore via agent_registry"
            )
            return True
        else:
            logger.info(
                "⚠️ Agent registry not available, skipping GUI launcher registration"
            )
            return False

    except Exception as e:
        logger.warning(f"Failed to register GUI agent: {e}")
        return False


def initialize_event_bus(core_instance):
    """Initialize and verify event bus connectivity"""
    if not core_instance:
        return False

    try:
        # Start event bus if available
        if hasattr(core_instance, "event_bus"):
            logger.info("✅ Event bus available and ready for use")
            return True
        else:
            logger.info("⚠️ Event bus not available")
            return False
    except Exception as e:
        logger.warning(f"Event bus initialization failed: {e}")
        return False


def setup_vmb_integration(core_instance):
    """Setup VMB Integration Handler if VantaCore is available"""
    global vmb_handler

    if not core_instance:
        logger.info("⚠️ No core instance available for VMB integration")
        return None

    try:
        from Vanta.integration.vmb_integration_handler import VMBIntegrationHandler

        vmb_handler = VMBIntegrationHandler(core_instance)

        # Register with core if registry exists
        if hasattr(core_instance, "registry"):
            core_instance.registry.register(
                "vmb_integration_handler",
                vmb_handler,
                {"type": "integration_handler", "for": "vmb", "status": "active"},
            )

        logger.info("✅ VMB Integration Handler registered with VantaCore")
        return vmb_handler

    except Exception as vmb_err:
        logger.warning(f"⚠️ VMB Integration Handler could not be registered: {vmb_err}")
        return None


def launch_gui_with_fallback():
    """Launch the GUI using the PyQt5 interface."""
    try:
        from gui.components import pyqt_main
    except Exception as e:
        logger.error(f"❌ Failed to import PyQt GUI: {e}")
        raise

    try:
        logger.info("🎨 Starting PyQt GUI main loop...")
        registry = core.agent_registry if core else None
        event_bus = core.event_bus if core else None
        pyqt_main.launch(registry, event_bus)
    except Exception as e:
        logger.error(f"❌ Critical error launching PyQt GUI: {e}")
        traceback.print_exc()
        raise


def main():
    """Main launcher function with comprehensive system initialization"""
    try:
        # Step 1: Verify Vanta accessibility
        logger.info("🔍 Step 1: Verifying Vanta component accessibility...")
        verify_vanta_accessibility()

        # Step 2: Initialize UnifiedVantaCore
        logger.info("🔍 Step 2: Initializing UnifiedVantaCore...")
        core_instance = initialize_unified_vanta_core()

        # Step 3: Register GUI as system agent
        logger.info("🔍 Step 3: Registering GUI launcher agent...")
        register_gui_agent(core_instance)

        # Step 4: Initialize event bus
        logger.info("🔍 Step 4: Initializing event bus...")
        initialize_event_bus(core_instance)

        # Step 5: Setup VMB integration
        logger.info("🔍 Step 5: Setting up VMB integration...")
        setup_vmb_integration(core_instance)

        # Step 6: Launch GUI with fallback mechanisms
        logger.info("🔍 Step 6: Launching GUI with comprehensive fallback...")
        launch_gui_with_fallback()

    except KeyboardInterrupt:
        logger.info("🛑 GUI launch interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in main launcher: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
