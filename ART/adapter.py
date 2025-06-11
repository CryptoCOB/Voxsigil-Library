"""
Adapter module to integrate VANTA Supervisor with existing VoxSigil components.
This handles the conversion between different interface styles and provides factory
methods for creating VANTA supervisor instances.
"""

import importlib
import importlib.util
import logging
import sys
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .art_manager import ARTManager
else:
    # Fallback if ARTManager is not found, to prevent import errors during type hinting
    class ARTManager:
        pass


# Import VANTA Supervisor
# Use try-except for external imports that might not be available
try:
    from Vanta.integration.vanta_supervisor import VantaSigilSupervisor
except ImportError:
    # Define a placeholder class to prevent type errors
    class VANTASupervisor:
        """Placeholder for VANTASupervisor when the module is not available."""

        def __init__(self, **kwargs):
            self.rag_interface = kwargs.get("rag_interface")
            self.llm_interface = kwargs.get("llm_interface")
            self.memory_interface = kwargs.get("memory_interface")
            self.scaffold_router = kwargs.get("scaffold_router")
            self.evaluation_heuristics = kwargs.get("evaluation_heuristics")
            self.retry_policy = kwargs.get("retry_policy")
            self.art_manager = kwargs.get("art_manager_instance")
            self.enable_sleep_time_compute = kwargs.get(
                "enable_sleep_time_compute", False
            )
            self.enable_art_training = kwargs.get("enable_art_training", False)

        def orchestrate_thought_cycle(self, query, context=None):
            """Placeholder for orchestrate_thought_cycle method."""
            return {
                "response": "Placeholder response",
                "sigils_used": [],
                "scaffold": "PLACEHOLDER",
                "execution_time": 0.0,
            }


# Try to import from VoxSigil package
VOXSIGIL_AVAILABLE = False
HAS_ADAPTIVE = False
HAS_ART = False

# Import unified interfaces from Vanta
from Vanta.interfaces.base_interfaces import BaseLlmInterface, BaseMemoryInterface, BaseRagInterface

# Define placeholder classes to satisfy type checking
if TYPE_CHECKING:
    # Only used for type checking, never imported at runtime
    from Voxsigil_Library.Scaffolds.scaffold_router import ScaffoldRouter
    from voxsigil_supervisor.evaluation_heuristics import ResponseEvaluator
    from voxsigil_supervisor.retry_policy import RetryPolicy
else:    # Runtime placeholders for components not yet unified
    class ScaffoldRouter:
        """Placeholder for ScaffoldRouter."""

        pass

    class ResponseEvaluator:
        """Placeholder for ResponseEvaluator."""

        pass

    class RetryPolicy:
        """Placeholder for RetryPolicy."""

        pass


# Try to import real components
try:
    if importlib.util.find_spec("voxsigil_supervisor.interfaces.rag_interface"):
        # These imports will override the placeholders at runtime, but keep type checking happy
        from ARC.llm.llm_interface import BaseLlmInterface
        from Voxsigil_Library.Scaffolds.scaffold_router import ScaffoldRouter
        from voxsigil_supervisor.evaluation_heuristics import ResponseEvaluator
        from Vanta.interfaces.memory_interface import BaseMemoryInterface
        from Vanta.interfaces.rag_interface import BaseRagInterface
        from voxsigil_supervisor.retry_policy import RetryPolicy

        VOXSIGIL_AVAILABLE = True
except ImportError:
    # Failed to import, use placeholders defined above
    pass

# Check for adaptive components
if importlib.util.find_spec("voxsigil_supervisor.adaptive.task_analyzer"):
    HAS_ADAPTIVE = True

# Check for ART components
if importlib.util.find_spec("voxsigil_supervisor.art.art_controller"):
    try:
        from .art_manager import ARTManager as ImportedARTManager
        from .art_logger import get_art_logger

        # Use the imported class but keep type checking happy
        ARTManager = ImportedARTManager
        HAS_ART = True
    except ImportError:
        pass

# Set up logging - use ART logger if available, otherwise standard logging
if HAS_ART:
    logger = get_art_logger("vanta.adapter")
else:
    logger = logging.getLogger("vanta.adapter")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VANTAFactory:
    """Factory class for creating VANTA Supervisor instances."""

    @staticmethod
    def create_from_voxsigil(
        voxsigil_supervisor_instance,
        resonance_threshold: float = 0.7,
        enable_echo_harmonization: bool = True,
        art_manager_instance=None,
    ):
        """
        Create a VANTA Supervisor from an existing VoxSigil Supervisor instance.

        Args:
            voxsigil_supervisor_instance: Existing VoxSigil Supervisor instance.
            resonance_threshold: Threshold for sigil resonance.
            enable_echo_harmonization: Whether to enable echo memory harmonization.
            art_manager_instance: Optional ARTManager for pattern recognition and categorization.

        Returns:
            VANTA Supervisor instance or None if conversion fails.
        """
        if not VOXSIGIL_AVAILABLE:
            logger.error("VoxSigil components not available")
            return None

        try:
            # Extract components from existing supervisor
            rag_interface = voxsigil_supervisor_instance.rag_interface
            llm_interface = voxsigil_supervisor_instance.llm_interface
            memory_interface = voxsigil_supervisor_instance.memory_interface
            scaffold_router = voxsigil_supervisor_instance.scaffold_router
            evaluation_heuristics = voxsigil_supervisor_instance.evaluation_heuristics
            retry_policy = voxsigil_supervisor_instance.retry_policy

            # Check if the existing supervisor has an ART manager
            existing_art_manager = getattr(
                voxsigil_supervisor_instance, "art_manager", None
            )
            art_manager = art_manager_instance or existing_art_manager

            # Create VANTA instance
            vanta = VANTASupervisor(
                rag_interface=rag_interface,
                llm_interface=llm_interface,
                memory_interface=memory_interface,
                scaffold_router=scaffold_router,
                evaluation_heuristics=evaluation_heuristics,
                retry_policy=retry_policy,
                resonance_threshold=resonance_threshold,
                enable_echo_harmonization=enable_echo_harmonization,
                enable_adaptive=hasattr(
                    voxsigil_supervisor_instance, "enable_self_adaptation"
                )
                and voxsigil_supervisor_instance.enable_self_adaptation,
                art_manager_instance=art_manager,
            )

            logger.info(
                "Successfully created VANTA Supervisor from VoxSigil Supervisor"
            )
            return vanta

        except Exception as e:
            logger.error(f"Failed to create VANTA from VoxSigil Supervisor: {e}")
            return None

    @staticmethod
    def create_new(
        rag_interface,
        llm_interface,
        memory_interface=None,
        scaffold_router=None,
        evaluation_heuristics=None,
        retry_policy=None,
        system_prompt=None,
        resonance_threshold=0.7,
        enable_adaptive=True,
        enable_echo_harmonization=True,
        art_manager_instance=None,
        enable_sleep_time_compute=True,
    ):
        """
        Create a new VANTA Supervisor with the specified components.

        Args:
            rag_interface: RAG interface for retrieving symbolic contexts.
            llm_interface: LLM interface for generating responses.
            memory_interface: Optional memory interface.
            scaffold_router: Optional scaffold router.
            evaluation_heuristics: Optional response evaluator.
            retry_policy: Optional retry policy.
            system_prompt: Optional system prompt.
            resonance_threshold: Threshold for sigil resonance.
            enable_adaptive: Whether to enable adaptive learning.
            enable_echo_harmonization: Whether to enable echo memory harmonization.
            art_manager_instance: Optional ARTManager for pattern recognition and categorization.
            enable_sleep_time_compute: Whether to enable memory consolidation via SleepTimeCompute.

        Returns:
            New VANTA Supervisor instance.
        """
        # Import sleep_time_compute module if needed
        sleep_time_compute_instance = None
        if enable_sleep_time_compute:
            sleep_time_compute_spec = importlib.util.find_spec(
                "voxsigil_supervisor.strategies.sleep_time_compute"
            )
            if sleep_time_compute_spec is not None:
                try:
                    sleep_time_compute_module = importlib.import_module(
                        "voxsigil_supervisor.strategies.sleep_time_compute"
                    )
                    if hasattr(sleep_time_compute_module, "SleepTimeCompute"):
                        # Get the SleepTimeCompute class and instantiate it
                        SleepTimeComputeClass = getattr(
                            sleep_time_compute_module, "SleepTimeCompute"
                        )
                        sleep_time_compute_instance = SleepTimeComputeClass(
                            external_memory_interface=memory_interface
                            if memory_interface
                            else None
                        )
                        logger.info(
                            "SleepTimeCompute module loaded and instantiated successfully"
                        )
                    else:
                        logger.warning("SleepTimeCompute class not found in module")
                        enable_sleep_time_compute = False
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Failed to load SleepTimeCompute: {e}")
                    enable_sleep_time_compute = False
            else:
                logger.warning(
                    "SleepTimeCompute module not found, memory consolidation disabled"
                )
                enable_sleep_time_compute = False

        vanta = VANTASupervisor(
            rag_interface=rag_interface,
            llm_interface=llm_interface,
            memory_interface=memory_interface,
            scaffold_router=scaffold_router,
            evaluation_heuristics=evaluation_heuristics,
            retry_policy=retry_policy,
            default_system_prompt=system_prompt,
            resonance_threshold=resonance_threshold,
            enable_adaptive=enable_adaptive and HAS_ADAPTIVE,
            enable_echo_harmonization=enable_echo_harmonization,
            art_manager_instance=art_manager_instance,
            sleep_time_compute_instance=sleep_time_compute_instance,
        )

        logger.info("Created new VANTA Supervisor")
        return vanta


class SimpleScaffoldRouter(ScaffoldRouter):
    """
    A simple implementation of the ScaffoldRouter interface.
    Maps tags to scaffold types and selects appropriate scaffolds.
    """

    def __init__(self, scaffolds_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the router with optional custom mapping.

        Args:
            scaffolds_mapping: Optional mapping from tags to scaffold names.
        """
        self.scaffolds_mapping = scaffolds_mapping or {
            "reflection": "HEGELIAN_KERNEL",
            "reasoning": "PERCEPTION_ACTION_CYCLE",
            "memory": "AMN_Scaffold",
            "ethical": "AREM_Scaffold",
            "communication": "ICNP_Scaffold",
            "worldview": "WCCE_Scaffold",
            "creativity": "NGCSE_Scaffold",
            "tree_of_thought": "TREETHOUGHT",
            "novelty": "TREE_NAVIGATOR",
            "arc": "ARC_SOLVER",
        }

        # Add ART category mappings for enhanced routing
        self.art_category_scaffolds = {
            "novel_pattern": "NOVELTY_GENERATOR",
            "anomaly": "ANOMALY_SCAFFOLD",
            "recurring_pattern": "PATTERN_CONSOLIDATOR",
            # Add more mappings as needed based on ART categories
        }

    def select_scaffold(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select an appropriate scaffold based on query and context.

        Args:
            query: User query.
            context: Optional context information, including resonant sigils and ART analysis.

        Returns:
            Selected scaffold name.
        """
        # Default scaffold
        default_scaffold = "PERCEPTION_ACTION_CYCLE"

        # Check for ART analysis first - highest priority routing
        if context and "art_analysis" in context:
            art_analysis = context["art_analysis"]
            # Handle different possible structures of art_analysis
            if isinstance(art_analysis, dict):
                # Get category_id from different possible locations
                category_id = None
                if "category" in art_analysis and isinstance(
                    art_analysis["category"], dict
                ):
                    category_id = art_analysis["category"].get("id")
                elif "category_id" in art_analysis:
                    category_id = art_analysis["category_id"]

                if category_id and category_id in self.art_category_scaffolds:
                    logger.info(
                        f"Selected scaffold '{self.art_category_scaffolds[category_id]}' based on ART category '{category_id}'"
                    )
                    return self.art_category_scaffolds[category_id]

                # Check for novel category flag
                is_novel = art_analysis.get("is_novel_category", False)
                if is_novel:
                    logger.info(
                        "Selected 'NOVELTY_GENERATOR' scaffold for novel ART pattern"
                    )
                    return "NOVELTY_GENERATOR"

        # Check for explicit query keyword matches - second priority
        query_lower = query.lower()
        query_scaffolds = {
            "reflect": "HEGELIAN_KERNEL",
            "reason": "PERCEPTION_ACTION_CYCLE",
            "remember": "AMN_Scaffold",
            "ethical": "AREM_Scaffold",
            "communicate": "ICNP_Scaffold",
            "world": "WCCE_Scaffold",
            "create": "NGCSE_Scaffold",
            "think": "TREETHOUGHT",
            "novel": "TREE_NAVIGATOR",
            "arc": "ARC_SOLVER",
        }

        for keyword, scaffold in query_scaffolds.items():
            if keyword in query_lower:
                logger.info(
                    f"Selected scaffold '{scaffold}' based on query keyword '{keyword}'"
                )
                return scaffold

        # Check sigils from context - third priority
        if context and "sigils" in context and context["sigils"]:
            sigils = context["sigils"]
            tag_counts = {}

            # Process sigils based on format
            if isinstance(sigils, list):
                if sigils and isinstance(sigils[0], dict):
                    # Count tag occurrences from dict-based sigils
                    for sigil in sigils:
                        for tag in sigil.get("tags", []):
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                elif sigils and isinstance(sigils[0], str):
                    # Count direct tag occurrences
                    for tag in sigils:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # If tags found, find most common mapped tag
            if tag_counts:
                best_scaffold = None
                best_count = 0

                for tag, count in sorted(
                    tag_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    if tag in self.scaffolds_mapping and count > best_count:
                        best_scaffold = self.scaffolds_mapping[tag]
                        best_count = count

                if best_scaffold:
                    logger.info(
                        f"Selected scaffold '{best_scaffold}' based on sigil tags"
                    )
                    return best_scaffold

        logger.info(f"Using default scaffold '{default_scaffold}'")
        return default_scaffold


# Define a standalone function (not a method of a class)
def orchestrate_query(
    vanta_supervisor,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to orchestrate a query through VANTA.

    Args:
        vanta_supervisor: VANTA Supervisor instance.
        query: User query.
        context: Optional context information.

    Returns:
        Response and metadata.
    """
    # Initialize context if None
    if context is None:
        context = {}

    # Process query through ART if available
    if hasattr(vanta_supervisor, "art_manager") and vanta_supervisor.art_manager:
        try:
            # Analyze the query using ART for pattern recognition
            art_result = vanta_supervisor.art_manager.analyze_inputs(query)

            # Add ART analysis to context for scaffold selection
            context["art_analysis"] = art_result

            # Log the ART analysis
            logger.info(
                f"ART analysis completed: category={art_result.get('category_id', '')}, "
                f"resonance={art_result.get('resonance', 0):.2f}, "
                f"is_novel={art_result.get('is_novel_category', False)}"
            )
        except Exception as e:
            logger.warning(f"ART analysis failed: {e}")

    # Check if we should run memory consolidation via SleepTimeCompute
    if (
        hasattr(vanta_supervisor, "enable_sleep_time_compute")
        and vanta_supervisor.enable_sleep_time_compute
    ):
        try:
            # Use importlib to check for module availability
            sleep_time_compute_module = None

            # Try different import paths with importlib for maximum compatibility
            for module_path in [
                "voxsigil_supervisor.strategies.sleep_time_compute",
                "..strategies.sleep_time_compute",
                "Voxsigil_Library.voxsigil_supervisor.strategies.sleep_time_compute",
            ]:
                if importlib.util.find_spec(module_path) is not None:
                    sleep_time_compute_module = importlib.import_module(module_path)
                    logger.debug(f"Found SleepTimeCompute module at {module_path}")
                    break

            if sleep_time_compute_module is None:
                logger.warning(
                    "SleepTimeCompute module not found in any expected location"
                )
            else:
                SleepTimeCompute = getattr(
                    sleep_time_compute_module, "SleepTimeCompute"
                )

                # Check if it's time to consolidate memories
                if (
                    hasattr(vanta_supervisor, "memory_interface")
                    and vanta_supervisor.memory_interface
                ):
                    sleep_compute = SleepTimeCompute(
                        memory_interface=vanta_supervisor.memory_interface
                    )

                    # Check if consolidation should run
                    if sleep_compute.should_consolidate():
                        logger.info("Running memory consolidation via SleepTimeCompute")
                        sleep_compute.consolidate_memories()
                        logger.info("Memory consolidation complete")
        except ImportError:
            logger.warning(
                "SleepTimeCompute module not available for memory consolidation"
            )
        except AttributeError as e:
            logger.warning(f"Error accessing SleepTimeCompute class: {e}")
        except Exception as e:
            logger.warning(f"Error during memory consolidation: {e}")

    # Orchestrate the thought cycle
    result = vanta_supervisor.orchestrate_thought_cycle(query, context)

    # Create a simplified response format
    simplified_result = {
        "response": result["response"],
        "sigil_count": len(result["sigils_used"])
        if isinstance(result["sigils_used"], list)
        else 1,
        "scaffold": result["scaffold"],
        "execution_time": result["execution_time"],
    }

    # Add ART analysis to result if available
    if "art_analysis" in context:
        simplified_result["art_analysis"] = {
            "category": context["art_analysis"].get("category_id", "unknown"),
            "resonance": context["art_analysis"].get("resonance", 0),
            "is_novel": context["art_analysis"].get("is_novel_category", False),
        }

    # Train ART on this interaction if configured
    if (
        hasattr(vanta_supervisor, "art_manager")
        and vanta_supervisor.art_manager
        and hasattr(vanta_supervisor, "enable_art_training")
        and vanta_supervisor.enable_art_training
    ):
        try:
            # Create training data from query and response
            training_data = (query, result["response"])
            vanta_supervisor.art_manager.train_on_batch([training_data])
            logger.debug("ART training completed for this interaction")
        except Exception as e:
            logger.warning(f"ART training failed: {e}")

    return simplified_result
