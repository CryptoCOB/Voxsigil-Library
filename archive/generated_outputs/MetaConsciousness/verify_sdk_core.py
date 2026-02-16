#!/usr/bin/env python
"""
SDK Core Verification Script

This script verifies basic functionality of the MetaConsciousness SDK.
It initializes the SDK, checks for core components, and tests basic functionality.
"""

import os
import sys
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sdk_verification.log")
    ]
)
logger = logging.getLogger("sdk_verification")

def verify_imports():
    """Verify that core SDK components can be imported."""
    logger.info("Testing SDK imports...")
    try:
        import MetaConsciousness
        logger.info(f"✓ MetaConsciousness SDK version {MetaConsciousness.__version__}")
        
        from MetaConsciousness import SDKContext
        logger.info("✓ SDKContext imported")
        
        from MetaConsciousness.core.meta_core import MetaConsciousness
        logger.info("✓ MetaConsciousness core imported")
        
        from MetaConsciousness.agent.metaconscious_agent import MetaconsciousAgent
        logger.info("✓ MetaconsciousAgent imported")
        
        from MetaConsciousness.memory.memory_cluster import MemoryCluster
        logger.info("✓ MemoryCluster imported")
        
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def initialize_sdk():
    """Initialize the SDK and bootstrap components."""
    logger.info("Initializing SDK components...")
    try:
        # Import from core.system_init instead of bootstrap
        from MetaConsciousness.core.system_init import initialize_system
        
        # Initialize the system - this should handle all component registration
        logger.info("Calling initialize_system...")
        initialize_result = initialize_system()
        logger.info(f"System initialization result: {initialize_result}")
        
        # If initialize_system returns False, consider initialization failed
        if not initialize_result:
            logger.error("❌ initialize_system reported failure")
            return False
        
        # Try bootstrap initialization as fallback if needed
        try:
            from MetaConsciousness import bootstrap
            if hasattr(bootstrap, "initialize"):
                logger.info("Calling bootstrap.initialize as fallback...")
                bootstrap_result = bootstrap.initialize()
                logger.info(f"Bootstrap initialization result: {bootstrap_result}")
        except Exception as e:
            logger.warning(f"Bootstrap fallback initialization failed: {e}")
        
        # Apply Omega3 registration fix
        try:
            logger.info("Applying Omega3 registration fix...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("fix_omega_registration", 
                                                          "fix_omega_registration.py")
            fix_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fix_module)
            fix_result = fix_module.fix_omega_registration()
            logger.info(f"Omega3 registration fix result: {fix_result}")
        except Exception as e:
            logger.warning(f"Omega3 registration fix failed: {e}")
        
        from MetaConsciousness import SDKContext
        components = SDKContext.list_components()
        logger.info(f"Registered components: {components}")
        
        return True
    except Exception as e:
        logger.error(f"SDK initialization failed: {e}", exc_info=True)
        return False

def verify_core_components():
    """Verify that core components are properly initialized and accessible."""
    logger.info("Verifying core components...")
    components_ok = True  # Flag to track component verification success
    
    try:
        from MetaConsciousness import SDKContext
        
        # Check MetaCore
        meta_core = SDKContext.get("meta_core")
        if meta_core:
            logger.info(f"✓ MetaCore found: {type(meta_core).__name__}")
            # Try to call a method
            state = meta_core.get_state() if hasattr(meta_core, "get_state") else None
            logger.info(f"  - MetaCore state: {state}")
        else:
            logger.error("✗ MetaCore not found in SDKContext")
            components_ok = False
            # No longer try to create components locally - strict verification only
        
        # Check MetaconsciousAgent
        agent = SDKContext.get("metaconscious_agent")
        if agent:
            logger.info(f"✓ MetaconsciousAgent found: {type(agent).__name__}")
            # Try to call a method
            if hasattr(agent, "get_status"):
                status = agent.get_status()
                logger.info(f"  - Agent status: {status}")
        else:
            logger.error("✗ MetaconsciousAgent not found in SDKContext")
            components_ok = False
            # No longer try to create components locally - strict verification only
        
        # Check MemoryCluster - check for both common registry keys
        memory = SDKContext.get("memory")
        memory_cluster = SDKContext.get("memory_cluster")
        if memory or memory_cluster:
            memory = memory or memory_cluster  # Use whichever one is found
            logger.info(f"✓ Memory component found: {type(memory).__name__}")
            # Try to call a method
            if hasattr(memory, "store_event"):
                try:
                    memory.store_event({"type": "verification", "message": "SDK verification test"})
                    logger.info("  - Successfully stored event in memory")
                except Exception as e:
                    logger.error(f"  - Failed to store event in memory: {e}")
                    components_ok = False
        else:
            logger.error("✗ Memory component not found in SDKContext (checked 'memory' and 'memory_cluster')")
            components_ok = False
            # No longer try to create components locally - strict verification only
        
        # Check OmegaCluster
        omega = SDKContext.get("omega_cluster")
        if omega:
            logger.info(f"✓ Omega3 cluster found: {type(omega).__name__}")
        else:
            logger.error("✗ Omega3 cluster not found in SDKContext")
            components_ok = False
            # No longer try to create components locally - strict verification only
        
        # Check ARTController - since this is a critical component
        art_controller = SDKContext.get("art_controller")
        if art_controller:
            logger.info(f"✓ ARTController found: {type(art_controller).__name__}")
            # Verify recent_resonance attribute exists
            if hasattr(art_controller, "recent_resonance"):
                logger.info("  - ARTController.recent_resonance attribute confirmed")
            else:
                logger.error("✗ ARTController missing recent_resonance attribute")
                components_ok = False
        else:
            logger.error("✗ ARTController not found in SDKContext")
            components_ok = False
            # No longer try to create components locally - strict verification only
        
        # Return overall component verification status
        return components_ok
    except Exception as e:
        logger.error(f"Error verifying core components: {e}", exc_info=True)
        return False

def test_basic_operations():
    """Test basic operations with SDK components."""
    logger.info("Testing basic operations...")
    operations_ok = True
    
    try:
        from MetaConsciousness import SDKContext
        
        # Test with MetaCore
        meta_core = SDKContext.get("meta_core")
        if meta_core and hasattr(meta_core, "process_input"):
            try:
                response = meta_core.process_input("Hello, MetaConsciousness!")
                logger.info(f"✓ MetaCore process_input response: {response}")
            except Exception as e:
                logger.error(f"✗ MetaCore process_input failed: {e}")
                operations_ok = False
        else:
            logger.warning("MetaCore unavailable for process_input testing - skipping test")
        
        # Test with Agent - only if available
        agent = SDKContext.get("metaconscious_agent")
        if agent and hasattr(agent, "process_message"):
            try:
                response = agent.process_message("Test message for verification")
                logger.info(f"✓ Agent process_message response: {response}")
            except Exception as e:
                logger.error(f"✗ Agent process_message failed: {e}")
                operations_ok = False
        else:
            logger.warning("MetaconsciousAgent unavailable for process_message testing - skipping test")
        
        # Test with Memory - only if available
        memory = SDKContext.get("memory")
        if memory:
            try:
                if hasattr(memory, "store_event"):
                    memory.store_event({"verification_test": "completed", "timestamp": time.time()})
                    logger.info("✓ Memory store_event called successfully")
                
                if hasattr(memory, "recall"):
                    results = memory.recall("verification", limit=5)
                    logger.info(f"✓ Memory recall results: {results}")
            except Exception as e:
                logger.error(f"✗ Memory operations failed: {e}")
                operations_ok = False
        else:
            logger.warning("Memory unavailable for testing - skipping test")
        
        # If no components were available for testing, we can't really say operations succeeded
        available_components = [c for c in [meta_core, agent, memory] if c is not None]
        if not available_components:
            logger.warning("No components available for operational testing")
            operations_ok = False
        
        return operations_ok
    except Exception as e:
        logger.error(f"Error testing basic operations: {e}", exc_info=True)
        return False

def main():
    """Main verification function."""
    logger.info("=" * 50)
    logger.info("Starting MetaConsciousness SDK Verification")
    logger.info("=" * 50)
    
    step_results = {}
    
    # Step 1: Verify imports
    step_results["imports"] = verify_imports()
    
    # Step 2: Initialize SDK
    step_results["initialization"] = initialize_sdk()
    
    # Step 3: Verify core components
    step_results["components"] = verify_core_components()
    
    # Step 4: Test basic operations
    step_results["operations"] = test_basic_operations()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Verification Results Summary:")
    for step, result in step_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status} - {step}")
    logger.info("=" * 50)
    
    # Overall result
    if all(step_results.values()):
        logger.info("✅ VERIFICATION SUCCESSFUL: All tests passed")
        return 0
    else:
        logger.error("❌ VERIFICATION FAILED: Some tests did not pass")
        return 1

if __name__ == "__main__":
    sys.exit(main())
