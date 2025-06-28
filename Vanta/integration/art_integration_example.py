#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating the integration of ARTManager into the VANTA Supervisor.

This script showcases how to use the refactored voxsigil.art.ARTManager with VANTA,
demonstrating pattern recognition, category detection, and logging integration.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Setup path for imports
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import ART components
try:
    from ART.art_manager import ARTManager
    from ART.art_logger import get_art_logger
    from ART.adapter import VANTAFactory

    HAS_ART = True
except ImportError:
    print("Error importing ART components. Please ensure they're properly installed.")
    print("This example requires the ART module.")
    HAS_ART = False
    sys.exit(1)

# Import VANTA components
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore
    from Vanta.interfaces.rag_interface import BaseRagInterface
    from Vanta.interfaces.memory_interface import BaseMemoryInterface
    from Vanta.interfaces.model_manager import ModelManager
    from BLT.hybrid_blt import BLTEnhancedRAG as VoxSigilRAG
    from ARC.llm.llm_interface import BaseLlmInterface
    from ARC.llm.arc_llm_handler import initialize_llm_handler, _llm_call_api_internal
except ImportError as e:
    print(f"Error importing VANTA components: {e}")
    print("Please ensure the Vanta, BLT, and ARC modules are properly installed.")
    sys.exit(1)


# Define the logging setup function
def setup_supervisor_logging(level=logging.INFO, log_file_path=None):
    """
    Set up logging for the VANTA Supervisor demonstration.

    Args:
        level: The logging level
        log_file_path: Path to the log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("VoxSigilSupervisor")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Create file handler if log_file_path is provided
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            logger.error(f"Failed to create log file at {log_file_path}: {e}")

    return logger


# Configure logging
log_file_path = project_root / "logs" / "vanta_runtime.log"
os.makedirs(log_file_path.parent, exist_ok=True)

logger = setup_supervisor_logging(level=logging.INFO, log_file_path=str(log_file_path))

# Create real implementations of the interfaces


class RealRAGInterface(BaseRagInterface):
    """
    Real implementation of the RAG Interface using VoxSigilRAG.
    """

    def __init__(self, voxsigil_library_path=None):
        """Initialize the RAG Interface with the VoxSigil library path."""
        self.voxsigil_library_path = (
            voxsigil_library_path or project_root / "Voxsigil_Library"
        )
        logger.info(
            f"Initializing RAG Interface with library path: {self.voxsigil_library_path}"
        )
        self.rag_engine = VoxSigilRAG(
            voxsigil_library_path=Path(str(self.voxsigil_library_path)),
            embedding_model="all-MiniLM-L6-v2",  # Use a small model for example
            cache_enabled=True,
        )

    def retrieve_sigils(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant sigils using VoxSigilRAG."""
        logger.info(f"Retrieving sigils for: {query[:50]}...")
        _, results = self.rag_engine.create_rag_context(
            query=query, num_sigils=top_k, min_score_threshold=0.6
        )
        return results

    def retrieve_context(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Retrieves and formats context for a query as a formatted string."""
        logger.info(f"Retrieving context for: {query[:50]}...")
        context, _ = self.rag_engine.create_rag_context(
            query=query,
            num_sigils=params.get("num_sigils", 5) if params else 5,
            min_score_threshold=params.get("min_score_threshold", 0.6)
            if params
            else 0.6,
        )
        return context

    def retrieve_scaffolds(
        self, query: str, filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves reasoning scaffolds relevant to the query."""
        logger.info(f"Retrieving scaffolds for: {query[:50]}...")
        # Not implemented in this example
        return []

    def get_scaffold_definition(
        self, scaffold_name_or_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the full definition of a specific reasoning scaffold."""
        logger.info(f"Getting scaffold definition for: {scaffold_name_or_id}")
        # Not implemented in this example
        return None

    def get_sigil_by_id(self, sigil_id_glyph: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific sigil by its unique ID or glyph."""
        logger.info(f"Getting sigil by ID: {sigil_id_glyph}")
        # Not implemented in this example
        return None


class RealLLMInterface(BaseLlmInterface):
    """
    Real implementation of the LLM Interface using arc_llm_handler.
    """

    def __init__(self):
        """Initialize the LLM Interface."""
        logger.info("Initializing LLM Interface")
        initialize_llm_handler(force_discover_models=True)

    def generate_response(
        self,
        messages: str,
        task_requirements: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = 0.2,
        system_prompt_override: Optional[str] = None,
        use_global_system_prompt: bool = True,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Generate a response from the LLM."""
        logger.info(f"LLM Request: {str(messages)[:50]}...")

        # Handle both string and message list formats
        if isinstance(messages, str):
            messages_payload = [{"role": "user", "content": messages}]
        else:
            messages_payload = messages

        # Set default task requirements if none provided
        if task_requirements is None:
            task_requirements = {
                "min_strength_tier": 1,
                "required_capabilities": ["general"],
            }

        # Use the arc_llm_handler to generate a response
        from ARC.llm.arc_llm_handler import llm_chat_completion

        # Ensure temperature is not None
        safe_temperature = 0.2 if temperature is None else temperature

        response_text, model_info, response_metadata = llm_chat_completion(
            user_prompt=messages
            if isinstance(messages, str)
            else messages[-1]["content"],
            task_requirements=task_requirements,
            system_prompt_override=system_prompt_override,
            use_global_voxsigil_system_prompt=use_global_system_prompt,
            temperature=safe_temperature,
        )

        return response_text, model_info, response_metadata

    def select_model(
        self, task_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Select an appropriate model based on task requirements."""
        logger.info("Selecting appropriate model based on task requirements")

        # In a real implementation, this would use complex logic to select the best model
        # For this example, we'll just return a simple model info
        if task_requirements is None:
            task_requirements = {
                "min_strength_tier": 1,
                "required_capabilities": ["general"],
            }

        # Default model info
        model_info = {
            "model_id": "default_model",
            "provider": "example_provider",
            "strength_tier": task_requirements.get("min_strength_tier", 1),
            "capabilities": task_requirements.get("required_capabilities", ["general"]),
        }

        return model_info


class RealMemoryInterface(BaseMemoryInterface):
    """
    Real implementation of the Memory Interface.
    """

    def __init__(self):
        """Initialize the Memory Interface."""
        logger.info("Initializing Memory Interface")
        self.memory_storage = {}
        self.memory_counter = 0

    def store_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Store a complete interaction."""
        interaction_id = f"interaction_{self.memory_counter}"
        self.memory_counter += 1
        self.memory_storage[interaction_id] = {
            **interaction_data,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(f"Stored interaction with ID: {interaction_id}")
        return True

    def store(
        self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a query-response pair with optional metadata."""
        interaction_id = f"interaction_{self.memory_counter}"
        self.store_interaction(
            {"query": query, "response": response, "metadata": metadata or {}}
        )
        return interaction_id

    def retrieve_similar_interactions(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieves interactions with similar queries."""
        logger.info(f"Retrieving similar interactions for: {query[:50]}...")

        # In a real implementation, this would use semantic search
        # For this example, we'll just return the most recent memories
        recent_memories = sorted(
            self.memory_storage.values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:limit]

        return recent_memories

    def retrieve_interaction_by_id(
        self, interaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a specific interaction by ID."""
        logger.info(f"Retrieving interaction by ID: {interaction_id}")
        return self.memory_storage.get(interaction_id)

    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interaction history."""
        logger.info(f"Retrieving interaction history, limit: {limit}")

        # Return the most recent interactions up to the limit
        history = sorted(
            self.memory_storage.values(),
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:limit]

        return history

    def update_interaction(
        self, interaction_id: str, update_data: Dict[str, Any]
    ) -> bool:
        """Update an existing interaction with new data."""
        logger.info(f"Updating interaction {interaction_id}")

        if interaction_id not in self.memory_storage:
            logger.warning(f"Interaction ID {interaction_id} not found, cannot update")
            return False

        # Update the interaction with new data
        self.memory_storage[interaction_id].update(update_data)

        # Add an updated timestamp if not explicitly provided
        if "updated_at" not in update_data:
            self.memory_storage[interaction_id]["updated_at"] = (
                datetime.now().isoformat()
            )

        return True

    def get_all_interactions(
        self,
        limit: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all stored interactions, optionally filtered and limited."""
        interactions = list(self.memory_storage.values())

        # Apply filters if provided
        if filter_conditions:
            for key, value in filter_conditions.items():
                interactions = [i for i in interactions if i.get(key) == value]

        # Sort by timestamp descending (newest first)
        interactions = sorted(
            interactions, key=lambda x: x.get("timestamp", ""), reverse=True
        )

        # Apply limit if provided
        if limit is not None:
            interactions = interactions[:limit]

        return interactions

    def clear_memory(self) -> bool:
        """Clear all stored memories."""
        logger.info("Clearing all memory storage")
        self.memory_storage = {}
        self.memory_counter = 0
        return True


class ARTVantaIntegrationDemo:
    """
    Demonstration of ARTManager integration with VANTA Supervisor.
    """

    def __init__(self, voxsigil_library_path: Optional[Path] = None):
        """
        Initialize the demonstration.

        Args:
            voxsigil_library_path: Path to the VoxSigil library
        """
        self.voxsigil_library_path = voxsigil_library_path or project_root

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize ARTManager and VANTA Supervisor."""
        logger.info("Initializing ARTManager and VANTA components...")

        # Create ARTManager with proper logger configuration
        art_logger = get_art_logger("VoxSigilSupervisor.vanta.art")
        # Create a config that includes the logger
        art_config = {"logger": art_logger}
        self.art_manager = ARTManager(config=art_config)
        logger.info("ARTManager initialized with logger: VoxSigilSupervisor.vanta.art")

        # Create real components for VANTA
        rag_interface = RealRAGInterface(self.voxsigil_library_path)
        llm_interface = RealLLMInterface()
        memory_interface = RealMemoryInterface()

        # Create VANTA Supervisor with ARTManager
        self.vanta = VANTAFactory.create_new(
            rag_interface=rag_interface,
            llm_interface=llm_interface,
            memory_interface=memory_interface,
            art_manager_instance=self.art_manager,
        )
        logger.info("VANTA Supervisor initialized with ARTManager")

    def demonstrate_integration(self):
        """
        Run demonstrations of ART + VANTA integration.
        """
        logger.info("\n=== Starting ART + VANTA Integration Demonstration ===\n")

        # Demonstrate pattern recognition with a sequence of related inputs
        self._demonstrate_pattern_recognition()

        # Demonstrate category formation and resonance
        self._demonstrate_category_resonance()

        # Demonstrate adaptive learning through batch training
        self._demonstrate_adaptive_learning()

        logger.info("\n=== ART + VANTA Integration Demonstration Complete ===\n")

    def _demonstrate_pattern_recognition(self):
        """Demonstrate ART pattern recognition with VANTA."""
        logger.info("\n=== Demonstrating Pattern Recognition ===\n")

        # Process a sequence of related queries
        queries = [
            "How do neural networks learn patterns from data?",
            "What mechanisms allow deep learning to recognize features?",
            "How does backpropagation help neural networks adapt to new information?",
            "Can you explain gradient descent in machine learning?",
        ]

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i + 1}/{len(queries)}: '{query}'")

            # Process through VANTA's orchestrate_thought_cycle
            result = self.vanta.orchestrate_thought_cycle(query)

            # Print the ART analysis if available
            art_analysis = result.get("art_analysis")
            if art_analysis and "category" in art_analysis:
                category_id = art_analysis["category"].get("id", "unknown")
                is_novel = art_analysis.get("is_novel_category", False)
                logger.info(f"ART Analysis: Category={category_id}, Novel={is_novel}")
            else:
                logger.info("No ART analysis available in result")

            # Brief pause between queries
            print("\n")

    def _demonstrate_category_resonance(self):
        """Demonstrate how ART categories resonate with similar inputs."""
        logger.info("\n=== Demonstrating Category Resonance ===\n")

        # Train ART on a distinct category
        self.art_manager.analyze_input(
            "The recursive structure of symbolic languages enables hierarchical representations"
        )

        # Now test with a similar query that should resonate with the same category
        query = (
            "How do hierarchical symbolic structures enable recursive representations?"
        )
        logger.info(f"Testing resonance with query: '{query}'")

        # Process through VANTA
        result = self.vanta.orchestrate_thought_cycle(query)

        # Analyze the response
        art_analysis = result.get("art_analysis")
        if art_analysis and "category" in art_analysis:
            category_id = art_analysis["category"].get("id", "unknown")
            is_novel = art_analysis.get("is_novel_category", False)
            resonance = art_analysis.get("resonance_score", 0)

            logger.info(
                f"ART Resonance: Category={category_id}, Score={resonance:.4f}, Novel={is_novel}"
            )
            logger.info(
                "This demonstrates how VANTA can use ART's pattern recognition to identify similar queries"
            )
        else:
            logger.info("No ART analysis available in result")

    def _demonstrate_adaptive_learning(self):
        """Demonstrate ART's adaptive learning through batch training."""
        logger.info("\n=== Demonstrating Adaptive Learning ===\n")

        # Create a batch of query-response pairs for training
        training_data = [
            (
                "How can artificial intelligence understand context?",
                "AI systems understand context through various mechanisms like attention, embeddings, and memory networks.",
            ),
            (
                "What makes language models context-aware?",
                "Language models become context-aware through transformer architectures, self-attention, and contextual embeddings.",
            ),
            (
                "How do transformer models process context?",
                "Transformers process context using self-attention mechanisms that model relationships between all elements in a sequence.",
            ),
        ]

        # Train ART on these pairs
        logger.info(f"Training ARTManager on {len(training_data)} query-response pairs")
        result = self.art_manager.train_on_batch(training_data)

        # Log the training results
        if result and isinstance(result, dict):
            logger.info(f"Training completed: {result}")
        else:
            logger.info("Training completed but no detailed results available")

        # Now test what ART has learned
        test_query = "How do attention mechanisms help with context understanding?"
        logger.info(f"Testing what ART learned with query: '{test_query}'")

        # Process through VANTA
        result = self.vanta.orchestrate_thought_cycle(test_query)

        # Check if ART recognized the pattern
        art_analysis = result.get("art_analysis")
        if art_analysis and "category" in art_analysis:
            category_id = art_analysis["category"].get("id", "unknown")
            is_novel = art_analysis.get("is_novel_category", False)

            logger.info(
                f"ART Analysis after learning: Category={category_id}, Novel={is_novel}"
            )
            logger.info(
                "This demonstrates how ART adaptively learns from training data"
            )
        else:
            logger.info("No ART analysis available in result")  # Report ART statistics
        try:
            # Try to access information from ARTManager
            logger.info("ART information from the demonstration:")

            # Access basic information from the art_manager
            if (
                hasattr(self.art_manager, "art_controller")
                and self.art_manager.art_controller
            ):
                controller = self.art_manager.art_controller

                # Print some general information about the controller
                logger.info(f"  - ART Controller: {controller.__class__.__name__}")

                # Report general pattern information
                num_inputs = getattr(controller, "_input_count", 0)
                logger.info(f"  - Total processed inputs: {num_inputs}")

                # Try to gather basic statistics from introspection
                stats = {}
                for attr_name in dir(controller):
                    if attr_name.startswith("_") or callable(
                        getattr(controller, attr_name)
                    ):
                        continue
                    try:
                        attr_value = getattr(controller, attr_name)
                        if isinstance(attr_value, (int, float, str, bool)):
                            stats[attr_name] = attr_value
                    except:
                        pass

                # Log any stats we found
                if stats:
                    logger.info("  ART Controller Statistics:")
                    for name, value in stats.items():
                        logger.info(f"    - {name}: {value}")

        except Exception as e:
            logger.info(f"Could not retrieve ART information: {e}")


def main():
    """Main entry point for the demonstration."""
    parser = argparse.ArgumentParser(
        description="Demonstrate ARTManager integration with VANTA Supervisor"
    )
    parser.add_argument(
        "--library-path",
        type=str,
        default=None,
        help="Path to the VoxSigil library (optional)",
    )

    args = parser.parse_args()

    voxsigil_library_path = Path(args.library_path) if args.library_path else None

    # Run the demonstration
    try:
        demo = ARTVantaIntegrationDemo(voxsigil_library_path)
        demo.demonstrate_integration()
    except Exception as e:
        logger.error(f"Error running demonstration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
