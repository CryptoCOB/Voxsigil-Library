# System initialization main file that should handle all component instantiation and registration

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def initialize_system(config=None) -> None:
    """
    Main system initialization function.
    Creates and registers all core components.
    """
    from MetaConsciousness import SDKContext
    from MetaConsciousness.core.enhanced_meta_core import EnhancedMetaConsciousness
    from MetaConsciousness.memory.memory_cluster import MemoryCluster
    from MetaConsciousness.agent.metaconscious_agent import MetaconsciousAgent
    # Import initialize_omega_cluster instead of OmegaCluster directly
    from MetaConsciousness.omega3 import initialize_omega_cluster
    
    config = config or {}
    results = {}
    
    # Initialize MetaCore
    logger.info("Initializing MetaCore...")
    meta_core = EnhancedMetaConsciousness()
    SDKContext.register("meta_core", meta_core)
    results["meta_core"] = True
    
    # Initialize Memory Cluster
    logger.info("Initializing Memory Cluster...")
    memory = MemoryCluster.initialize(config.get("memory", {}))
    SDKContext.register("memory", memory)
    results["memory"] = True
    
    # Initialize Omega Cluster using the new helper function
    logger.info("Initializing Omega Cluster...")
    omega = initialize_omega_cluster(config=config.get("omega", {}))
    # Registration already handled by initialize_omega_cluster
    results["omega_cluster"] = True
    
    # Initialize Agent (after Memory and Omega are available)
    logger.info("Initializing Metaconscious Agent...")
    agent = MetaconsciousAgent(config=config.get("agent", {}))
    SDKContext.register("metaconscious_agent", agent)
    results["agent"] = True
    
    logger.info(f"System initialization complete. Components registered: {SDKContext.list_components()}")
    return results
