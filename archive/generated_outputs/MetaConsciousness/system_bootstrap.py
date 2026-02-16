"""
MetaConsciousness System Bootstrap

This script serves as the central initialization point for the complete MetaConsciousness system.
It connects all core modules, cluster systems, and meta agents to form a cohesive runtime environment.
"""
import os
import sys
import logging
import time
import json
import argparse
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("metaconsciousness.bootstrap")

# Add project root to path if needed
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    logger.info(f"Added project root to sys.path: {project_root}")
except Exception as e:
    logger.warning(f"Failed to add project root to path: {e}")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load system configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "features": {
            "enable_memory_cluster": True,
            "enable_agent_cluster": True,
            "enable_rag_cluster": True,
            "enable_omega_cluster": True,
            "enable_art_cluster": True,
            "enable_tool_cluster": True,
            "enable_meta_learning_cluster": True,
            "enable_compression_cluster": True,
            "enable_narrative_cluster": True
        },
        "meta_core": {
            "initial_awareness": 0.5,
            "initial_regulation": 0.3,
            "neuro_symbolic_enabled": True
        },
        "clusters": {
            "memory": {"enable_auto_snapshots": True},
            "agent": {"max_agents": 5},
            "rag": {"chunk_size": 512},
            "omega": {"strategy_count": 5}
        },
        "vox_agent": {
            "enable": True,
            "agent_id": "Vox",
            "check_in_callback_key": "user_output",
            "enable_think_integration": True,
            "enable_self_reflex": True,
            "prioritize_goal_idle_tasks": True,
            "rate_limit_on_sentiment": True,
            "model_manager_config": {
                "lm_studio_url": "http://localhost:1234",
                "ollama_url": "http://localhost:11434",
                "max_concurrent_models": 3,
                "default_models": {
                    "chat": "mistral-7b-instruct",
                    "embed": "text-embedding-nomic",
                    "rag": "phi-3"
                },
                "training_guard": True,
                "monitor_interval_s": 60
            }
        }
    }
    
    if not config_path:
        logger.info("No config path provided, using default configuration")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.warning(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration instead")
        return default_config

def bootstrap_system(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Bootstrap the complete MetaConsciousness system.
    
    Args:
        config: System configuration
        
    Returns:
        Dict with bootstrap results
    """
    # Start timer
    start_time = time.time()
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # --- EXPLICITLY RUN BOOTSTRAP FIRST ---
    logger.info("Initiating MetaConsciousness bootstrap...")
    import bootstrap
    if not bootstrap.run_bootstrap():
        logger.critical("Core bootstrap failed. Cannot continue with system initialization.")
        return {
            "status": "failed",
            "error": "Core bootstrap failed",
            "components_registered": 0,
            "time": time.time() - start_time
        }
    logger.info("Core bootstrap successful, continuing with system initialization...")
    
    # Initialize components
    components_registered = 0
    results = {"registered": [], "failed": []}
    
    try:
        # --- INITIALIZE CONTEXT ---
        try:
            from MetaConsciousness.core.context import SDKContext, initialize_context
            # Initialize core context 
            context = initialize_context()
            logger.info("SDK Context initialized")
            results["registered"].append("sdk_context")
            components_registered += 1
        except ImportError as e:
            logger.error(f"Failed to import SDKContext: {e}")
            logger.error("Cannot continue without core context")
            return {
                "status": "failed",
                "error": f"Context initialization failed: {str(e)}",
                "components_registered": 0,
                "time": time.time() - start_time
            }
        
        # --- CORE & ENHANCED META CORE ---
        try:
            from MetaConsciousness.core.meta_core import MetaConsciousness
            meta_core = MetaConsciousness(config=config.get("meta_core", {}))
            SDKContext.register("meta_core", meta_core)
            logger.info("Registered meta_core")
            results["registered"].append("meta_core")
            components_registered += 1
        except ImportError as e:
            logger.error(f"Failed to import MetaConsciousness: {e}")
            results["failed"].append("meta_core")
        
        try:
            from MetaConsciousness.core.enhanced_meta_core import EnhancedMetaConsciousness
            enhanced_meta = EnhancedMetaConsciousness(config=config)
            SDKContext.register("enhanced_meta_core", enhanced_meta)
            logger.info("Registered enhanced_meta_core")
            results["registered"].append("enhanced_meta_core")
            components_registered += 1
        except ImportError as e:
            logger.error(f"Failed to import EnhancedMetaConsciousness: {e}")
            results["failed"].append("enhanced_meta_core")
        
        # --- INITIALIZE CORE CLUSTERS ---
        from MetaConsciousness.core.cluster_adapters import initialize_core_clusters
        try:
            cluster_results = initialize_core_clusters(config)
            # Log successful initializations
            for cluster, status in cluster_results.items():
                if status.get("initialized", False):
                    logger.info(f"Initialized {cluster}")
                    results["registered"].append(cluster)
                    components_registered += 1
                else:
                    logger.warning(f"Failed to initialize {cluster}")
                    results["failed"].append(cluster)
            
            # Explicitly verify memory_cluster and memory_cluster_status
            if SDKContext.get("memory_cluster") is None:
                logger.error("memory_cluster not registered in SDKContext")
                results["failed"].append("memory_cluster")
            else:
                logger.info("Verified memory_cluster registration")
                results["registered"].append("memory_cluster")
                components_registered += 1
            
            if SDKContext.get("memory_cluster_status") is None:
                logger.error("memory_cluster_status not registered in SDKContext")
                results["failed"].append("memory_cluster_status")
            else:
                logger.info("Verified memory_cluster_status registration")
                results["registered"].append("memory_cluster_status")
                components_registered += 1
        except Exception as e:
            logger.error(f"Error initializing core clusters: {e}")
        
        # --- INITIALIZE VOX AGENT ---
        if config.get("vox_agent", {}).get("enable", True):
            try:
                from MetaConsciousness.agent.vox_agent import VoxAgent
                vox_agent = VoxAgent(config=config.get("vox_agent", {}))
                # Note: VoxAgent registers itself with SDKContext during initialization
                logger.info("Initialized VoxAgent with runtime model management")
                results["registered"].append("vox_agent")
                results["registered"].append("model_manager")
                components_registered += 2
            except ImportError as e:
                logger.error(f"Failed to import VoxAgent: {e}")
                results["failed"].append("vox_agent")
        else:
            logger.info("VoxAgent disabled in configuration")
        
        # --- META SYSTEMS (optional) ---
        try:
            from MetaConsciousness.core.meta_cognitive import MetaCognitiveEngine
            meta_cognitive = MetaCognitiveEngine()
            SDKContext.register("meta_cognitive", meta_cognitive)
            logger.info("Registered meta_cognitive")
            results["registered"].append("meta_cognitive")
            components_registered += 1
        except ImportError as e:
            logger.warning(f"Failed to import MetaCognitiveEngine (optional): {e}")
            results["failed"].append("meta_cognitive")
        
        try:
            from MetaConsciousness.AdvancedMetaLearner import AdvancedMetaLearner
            meta_learner = AdvancedMetaLearner()
            SDKContext.register("advanced_meta_learner", meta_learner)
            logger.info("Registered advanced_meta_learner")
            results["registered"].append("advanced_meta_learner")
            components_registered += 1
        except ImportError as e:
            logger.warning(f"Failed to import AdvancedMetaLearner (optional): {e}")
            results["failed"].append("advanced_meta_learner")
        
        # --- CONNECT COMPONENTS TO META CORE ---
        try:
            from MetaConsciousness.utils.component_linker import auto_link_components
            linking_results = auto_link_components()
            logger.info(f"Auto-linked {linking_results.get('components_connected', 0)} components")
            results["auto_linked"] = linking_results
        except ImportError as e:
            logger.warning(f"Failed to import component_linker (optional): {e}")
        
        # --- REGISTER SYSTEM STATUS ---
        SDKContext.register("system_status", {
            "initialized": True,
            "initialization_time": time.time() - start_time,
            "components_registered": components_registered,
            "timestamp": time.time()
        })
        
        logger.info(f"System bootstrap complete in {time.time() - start_time:.2f} seconds")
        logger.info(f"Registered {components_registered} components")
        
        return {
            "status": "success",
            "components_registered": components_registered,
            "registered": results["registered"],
            "failed": results["failed"],
            "time": time.time() - start_time,
            "launch_ready": True
        }
    except Exception as e:
        logger.error(f"Error during system bootstrap: {e}")
        return {
            "status": "error",
            "error": str(e),
            "components_registered": components_registered,
            "time": time.time() - start_time
        }

def launch_gui() -> None:
    """Launch the GUI interface."""
    try:
        from MetaConsciousness.interface.launch_gui import main
        main()
    except ImportError as e:
        logger.error(f"Failed to import GUI launcher: {e}")
        
def launch_runtime() -> None:
    """Launch the runtime environment."""
    try:
        # Get enhanced_meta_core from context
        from MetaConsciousness.core.context import SDKContext
        enhanced_meta = SDKContext.get("enhanced_meta_core")
        if enhanced_meta:
            # Run the main event loop if available
            if hasattr(enhanced_meta, "run_main_loop"):
                enhanced_meta.run_main_loop()
            else:
                logger.error("Enhanced MetaCore does not have a run_main_loop method")
        else:
            logger.error("Enhanced MetaCore not found in context")
    except Exception as e:
        logger.error(f"Failed to launch runtime environment: {e}")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MetaConsciousness System Bootstrap")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--gui", action="store_true", help="Launch GUI after bootstrap")
    parser.add_argument("--runtime", action="store_true", help="Launch runtime after bootstrap")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Bootstrap the system
    result = bootstrap_system(config)
    
    # Print result summary
    if result["status"] == "success":
        print("\n=== System Bootstrap Successful ===")
        print(f"Registered components: {result['components_registered']}")
        print(f"Initialization time: {result['time']:.2f} seconds")
        print("\nRegistered components:")
        for component in result["registered"]:
            print(f"  - {component}")
        if result["failed"] != []:
            print("\nFailed components:")
            for component in result["failed"]:
                print(f"  - {component}")
    else:
        print("\n=== System Bootstrap Failed ===")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Registered components: {result['components_registered']}")
        print(f"Initialization time: {result['time']:.2f} seconds")
    
    # Launch GUI or runtime if requested
    if args.gui and result["status"] == "success":
        print("\nLaunching GUI...")
        launch_gui()
    elif args.runtime and result["status"] == "success":
        print("\nLaunching runtime environment...")
        launch_runtime()

if __name__ == "__main__":
    main()
