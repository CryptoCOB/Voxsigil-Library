#!/usr/bin/env python
"""
Core Verification Runner

This script runs the core verification process to check that
all core components are correctly initialized.
"""
import os
import sys
import logging
import traceback
import argparse
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main() -> None:
    """Run core verification."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MetaConsciousness core verification")
    parser.add_argument('--force', action='store_true', help='Force re-initialization of all clusters')
    parser.add_argument('--reset', action='store_true', help='Reset all clusters before initialization')
    parser.add_argument('--cluster', type=str, help='Initialize only a specific cluster')
    args = parser.parse_args()

    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        # Import verification modules
        from MetaConsciousness.core.verify_core import run_verification
        from MetaConsciousness.core.utils.context_guard import ClusterInitGuard

        # Reset clusters if requested
        if args.reset:
            print("🔄 Resetting clusters before initialization...")
            if args.cluster:
                ClusterInitGuard.reset_cluster(args.cluster)
            else:
                # Reset all common clusters
                for cluster in ["memory", "agent", "tool", "art", "omega", "rag",
                               "compression", "meta_learning", "narrative"]:
                    ClusterInitGuard.reset_cluster(cluster)

        # Create config with all features enabled
        config = {
            "enable_advanced_clusters": True,
            "features": {
                "enable_memory_cluster": True,
                "enable_agent_cluster": True,
                "enable_tool_cluster": True,
                "enable_art_cluster": True,
                "enable_omega_cluster": True,
                "enable_rag_cluster": True,
                "enable_compression_cluster": True,
                "enable_meta_learning_cluster": True,
                "enable_narrative_cluster": True
            }
        }

        # Run verification with the specified options
        run_verification(config=config, force=args.force, specific_cluster=args.cluster)
        return 0
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nDETAILED ERROR:")
        traceback.print_exc()
        print("\nThis error usually indicates a missing module or incorrect import path.")
        return 1
    except Exception as e:
        print(f"Error running verification: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
