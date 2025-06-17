"""
Adapter module to integrate VANTA Supervisor with existing VoxSigil components.
This handles the conversion between different interface styles and provides factory
methods for creating VANTA supervisor instances.

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh for adaptive integration
and autonomous optimization of cross-system communications.
"""

import asyncio
import importlib
import importlib.util
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

# Define placeholder classes to satisfy type checking
if TYPE_CHECKING:
    # Only used for type checking, never imported at runtime
    from voxsigil_supervisor.evaluation_heuristics import ResponseEvaluator
    from voxsigil_supervisor.retry_policy import RetryPolicy
    from voxsigil_supervisor.strategies.scaffold_router import ScaffoldRouter
else:  # Runtime placeholders for components not yet unified

    class ScaffoldRouter:
        """Placeholder for ScaffoldRouter."""

        pass

    class ResponseEvaluator:
        """Placeholder for ResponseEvaluator."""

        pass

    class RetryPolicy:
        """Placeholder for RetryPolicy."""

        pass


# Initialize global variables
VOXSIGIL_AVAILABLE = False
HAS_ADAPTIVE = False
HAS_ART = False

# Try to import real components
try:
    if importlib.util.find_spec("voxsigil_supervisor.interfaces.rag_interface"):
        # These imports will override the placeholders at runtime, but keep type checking happy
        from voxsigil_supervisor.evaluation_heuristics import (
            ResponseEvaluator as _ResponseEvaluator,
        )
        from voxsigil_supervisor.retry_policy import RetryPolicy as _RetryPolicy
        from voxsigil_supervisor.strategies.scaffold_router import ScaffoldRouter as _ScaffoldRouter

        # Override the placeholder classes
        ResponseEvaluator = _ResponseEvaluator
        RetryPolicy = _RetryPolicy
        ScaffoldRouter = _ScaffoldRouter

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
        from .art_logger import get_art_logger
        from .art_manager import ARTManager as ImportedARTManager

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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# HOLO-1.5 Core Imports
try:
    from ..agents.base import BaseAgent, CognitiveMeshRole, vanta_agent
except (ImportError, ValueError):
    # Fallback for non-HOLO environments
    def vanta_agent(role=None, name=None, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    class CognitiveMeshRole:
        PROCESSOR = "processor"

    class BaseAgent:
        pass


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
    from Vanta.integration.vanta_supervisor import VantaSigilSupervisor as _VantaSigilSupervisor

    VantaSigilSupervisor = _VantaSigilSupervisor
except ImportError:
    # Define a placeholder class to prevent type errors
    class VantaSigilSupervisor:
        """Placeholder for VantaSigilSupervisor when the module is not available."""

        def __init__(self, **kwargs):
            pass
            self.rag_interface = kwargs.get("rag_interface")
            self.llm_interface = kwargs.get("llm_interface")
            self.memory_interface = kwargs.get("memory_interface")
            self.scaffold_router = kwargs.get("scaffold_router")
            self.evaluation_heuristics = kwargs.get("evaluation_heuristics")
            self.retry_policy = kwargs.get("retry_policy")
            self.art_manager = kwargs.get("art_manager_instance")
            self.enable_sleep_time_compute = kwargs.get("enable_sleep_time_compute", False)
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


@vanta_agent(role=CognitiveMeshRole.PROCESSOR)
class VANTAFactory(BaseAgent):
    """
    Factory class for creating VANTA Supervisor instances.
    Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh for autonomous optimization.
    """

    def __init__(self):
        super().__init__()
        self.cognitive_metrics = {
            "integration_events": 0,
            "supervisor_creations": 0,
            "conversion_successes": 0,
            "optimization_cycles": 0,
            "cognitive_load": 0.0,
            "symbolic_depth": 0.0,
        }
        self.vanta_core = None
        self._background_tasks = []

    async def async_init(self):
        """Initialize async components and cognitive monitoring"""
        try:
            from ..core.vanta_core import VantaCore

            self.vanta_core = VantaCore.get_instance()
            await self.register_cognitive_capabilities()
            await self.start_cognitive_monitoring()
            logger.info("🧠 VANTAFactory HOLO-1.5 initialization complete")
        except ImportError:
            logger.warning("VantaCore not available - running in standalone mode")

    async def register_cognitive_capabilities(self):
        """Register factory capabilities with VantaCore mesh"""
        if self.vanta_core:
            capabilities = {
                "supervisor_creation": "Advanced VANTA supervisor instantiation",
                "cross_system_integration": "Seamless component bridge creation",
                "adaptive_routing": "Intelligent scaffold selection",
                "memory_consolidation": "SleepTimeCompute integration",
                "art_pattern_recognition": "ART-driven supervisor enhancement",
            }

            for capability, description in capabilities.items():
                await self.vanta_core.register_capability(
                    f"vanta_factory.{capability}", description, self
                )

    async def start_cognitive_monitoring(self):
        """Start background cognitive monitoring and optimization"""

        async def monitor_loop():
            while True:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                self._update_cognitive_metrics()

                # Generate cognitive trace for mesh learning
                if self.vanta_core:
                    trace = self._generate_cognitive_trace()
                    await self.vanta_core.emit_cognitive_trace(trace)

        task = asyncio.create_task(monitor_loop())
        self._background_tasks.append(task)

    def _update_cognitive_metrics(self):
        """Update real-time cognitive metrics"""
        self.cognitive_metrics["cognitive_load"] = self._calculate_cognitive_load()
        self.cognitive_metrics["symbolic_depth"] = self._calculate_symbolic_depth()

    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on factory operations"""
        base_load = 0.1
        integration_load = min(self.cognitive_metrics["integration_events"] * 0.05, 0.5)
        creation_load = min(self.cognitive_metrics["supervisor_creations"] * 0.1, 0.3)
        return min(base_load + integration_load + creation_load, 1.0)

    def _calculate_symbolic_depth(self) -> float:
        """Calculate symbolic processing depth"""
        base_depth = 0.2
        success_depth = min(self.cognitive_metrics["conversion_successes"] * 0.1, 0.6)
        optimization_depth = min(self.cognitive_metrics["optimization_cycles"] * 0.05, 0.2)
        return min(base_depth + success_depth + optimization_depth, 1.0)

    def _generate_cognitive_trace(self) -> Dict[str, Any]:
        """Generate cognitive trace for mesh learning"""
        return {
            "component": "VANTAFactory",
            "role": "PROCESSOR",
            "timestamp": time.time(),
            "metrics": self.cognitive_metrics.copy(),
            "cognitive_state": {
                "integration_efficiency": self.cognitive_metrics["conversion_successes"]
                / max(1, self.cognitive_metrics["integration_events"]),
                "factory_utilization": min(
                    self.cognitive_metrics["supervisor_creations"] / 10.0, 1.0
                ),
                "optimization_ratio": self.cognitive_metrics["optimization_cycles"]
                / max(1, self.cognitive_metrics["integration_events"]),
            },
        }

    @staticmethod
    async def create_from_voxsigil_async(
        voxsigil_supervisor_instance,
        resonance_threshold: float = 0.7,
        enable_echo_harmonization: bool = True,
        art_manager_instance=None,
    ):
        """
        Create a VANTA Supervisor from an existing VoxSigil Supervisor instance.
        Enhanced with async cognitive monitoring and optimization.
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
            existing_art_manager = getattr(voxsigil_supervisor_instance, "art_manager", None)
            art_manager = art_manager_instance or existing_art_manager

            # Create VANTA instance
            vanta = VantaSigilSupervisor(
                rag_interface=rag_interface,
                llm_interface=llm_interface,
                memory_interface=memory_interface,
                scaffold_router=scaffold_router,
                evaluation_heuristics=evaluation_heuristics,
                retry_policy=retry_policy,
                resonance_threshold=resonance_threshold,
                enable_echo_harmonization=enable_echo_harmonization,
                enable_adaptive=hasattr(voxsigil_supervisor_instance, "enable_self_adaptation")
                and voxsigil_supervisor_instance.enable_self_adaptation,
                art_manager_instance=art_manager,
            )

            logger.info("Successfully created VANTA Supervisor from VoxSigil Supervisor")
            return vanta

        except Exception as e:
            logger.error(f"Failed to create VANTA from VoxSigil Supervisor: {e}")
            return None

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
            art_manager_instance: Optional ARTManager for pattern recognition and categorization.        Returns:
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
            existing_art_manager = getattr(voxsigil_supervisor_instance, "art_manager", None)
            art_manager = art_manager_instance or existing_art_manager

            # Create VANTA instance
            vanta = VantaSigilSupervisor(
                rag_interface=rag_interface,
                llm_interface=llm_interface,
                memory_interface=memory_interface,
                scaffold_router=scaffold_router,
                evaluation_heuristics=evaluation_heuristics,
                retry_policy=retry_policy,
                resonance_threshold=resonance_threshold,
                enable_echo_harmonization=enable_echo_harmonization,
                enable_adaptive=hasattr(voxsigil_supervisor_instance, "enable_self_adaptation")
                and voxsigil_supervisor_instance.enable_self_adaptation,
                art_manager_instance=art_manager,
            )

            logger.info("Successfully created VANTA Supervisor from VoxSigil Supervisor")
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
                            external_memory_interface=memory_interface if memory_interface else None
                        )
                        logger.info("SleepTimeCompute module loaded and instantiated successfully")
                    else:
                        logger.warning("SleepTimeCompute class not found in module")
                        enable_sleep_time_compute = False
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Failed to load SleepTimeCompute: {e}")
                    enable_sleep_time_compute = False
            else:
                logger.warning("SleepTimeCompute module not found, memory consolidation disabled")
                enable_sleep_time_compute = False

        vanta = VantaSigilSupervisor(
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


@vanta_agent(role=CognitiveMeshRole.PROCESSOR)
class SimpleScaffoldRouter(ScaffoldRouter, BaseAgent):
    """
    A simple implementation of the ScaffoldRouter interface.
    Maps tags to scaffold types and selects appropriate scaffolds.
    Enhanced with HOLO-1.5 cognitive routing and adaptive optimization.
    """

    def __init__(self, scaffolds_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the router with optional custom mapping.
        Enhanced with cognitive mesh capabilities.
        """
        BaseAgent.__init__(self)
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

        # HOLO-1.5 Cognitive Metrics
        self.cognitive_metrics = {
            "routing_decisions": 0,
            "art_driven_routes": 0,
            "pattern_matches": 0,
            "adaptation_cycles": 0,
            "cognitive_load": 0.0,
            "symbolic_depth": 0.0,
        }
        self.vanta_core = None

    async def async_init(self):
        """Initialize async components and cognitive monitoring"""
        try:
            from ..core.vanta_core import VantaCore

            self.vanta_core = VantaCore.get_instance()
            await self.register_cognitive_capabilities()
            logger.info("🧠 SimpleScaffoldRouter HOLO-1.5 initialization complete")
        except ImportError:
            logger.warning("VantaCore not available - running in standalone mode")

    async def register_cognitive_capabilities(self):
        """Register routing capabilities with VantaCore mesh"""
        if self.vanta_core:
            capabilities = {
                "adaptive_routing": "Intelligent scaffold selection with ART integration",
                "pattern_mapping": "Dynamic category-to-scaffold mapping",
                "cognitive_routing": "Context-aware routing decisions",
                "route_optimization": "Continuous routing efficiency improvement",
            }

            for capability, description in capabilities.items():
                await self.vanta_core.register_capability(
                    f"scaffold_router.{capability}", description, self
                )

    async def select_scaffold_async(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select an appropriate scaffold based on query and context.
        Enhanced with cognitive routing and adaptive optimization.
        """
        self.cognitive_metrics["routing_decisions"] += 1

        # Perform cognitive route selection
        scaffold = self.select_scaffold(query, context)

        # Update cognitive metrics based on selection
        if context and "art_analysis" in context:
            self.cognitive_metrics["art_driven_routes"] += 1

        # Generate cognitive trace for mesh learning
        if self.vanta_core:
            trace = {
                "component": "SimpleScaffoldRouter",
                "role": "PROCESSOR",
                "timestamp": time.time(),
                "operation": "scaffold_selection",
                "input_query": query[:100],  # Truncated for privacy
                "selected_scaffold": scaffold,
                "context_available": bool(context),
                "art_driven": context and "art_analysis" in context,
                "metrics": self.cognitive_metrics.copy(),
            }
            await self.vanta_core.emit_cognitive_trace(trace)

        return scaffold

    def select_scaffold(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
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
                if "category" in art_analysis and isinstance(art_analysis["category"], dict):
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
                    logger.info("Selected 'NOVELTY_GENERATOR' scaffold for novel ART pattern")
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
                logger.info(f"Selected scaffold '{scaffold}' based on query keyword '{keyword}'")
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

                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                    if tag in self.scaffolds_mapping and count > best_count:
                        best_scaffold = self.scaffolds_mapping[tag]
                        best_count = count

                if best_scaffold:
                    logger.info(f"Selected scaffold '{best_scaffold}' based on sigil tags")
                    return best_scaffold

        logger.info(f"Using default scaffold '{default_scaffold}'")
        return default_scaffold


# Enhanced orchestrate_query function with HOLO-1.5 cognitive integration
async def orchestrate_query_async(
    vanta_supervisor,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to orchestrate a query through VANTA.
    Enhanced with HOLO-1.5 cognitive monitoring and adaptive optimization.

    Args:
        vanta_supervisor: VANTA Supervisor instance.
        query: User query.
        context: Optional context information.

    Returns:
        Response and metadata with cognitive metrics.
    """
    start_time = time.time()

    # Initialize context if None
    if context is None:
        context = {}

    # Cognitive metrics for this orchestration
    orchestration_metrics = {
        "query_length": len(query),
        "context_complexity": len(context),
        "art_analysis_available": False,
        "memory_consolidation_ran": False,
        "training_completed": False,
    }

    # Process query through ART if available
    if hasattr(vanta_supervisor, "art_manager") and vanta_supervisor.art_manager:
        try:
            # Analyze the query using ART for pattern recognition
            art_result = vanta_supervisor.art_manager.analyze_inputs(query)

            # Add ART analysis to context for scaffold selection
            context["art_analysis"] = art_result
            orchestration_metrics["art_analysis_available"] = True

            # Log the ART analysis
            logger.info(
                f"ART analysis completed: category={art_result.get('category_id', '')}, "
                f"resonance={art_result.get('resonance', 0):.2f}, "
                f"is_novel={art_result.get('is_novel_category', False)}"
            )
        except Exception as e:
            logger.warning(
                f"ART analysis failed: {e}"
            )  # Enhanced memory consolidation with cognitive monitoring
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
            ]:
                if importlib.util.find_spec(module_path) is not None:
                    sleep_time_compute_module = importlib.import_module(module_path)
                    logger.debug(f"Found SleepTimeCompute module at {module_path}")
                    break

            if sleep_time_compute_module is None:
                logger.warning("SleepTimeCompute module not found in any expected location")
            else:
                SleepTimeCompute = getattr(sleep_time_compute_module, "SleepTimeCompute")

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
                        orchestration_metrics["memory_consolidation_ran"] = True
                        logger.info("Memory consolidation complete")
        except ImportError:
            logger.warning("SleepTimeCompute module not available for memory consolidation")
        except AttributeError as e:
            logger.warning(f"Error accessing SleepTimeCompute class: {e}")
        except Exception as e:
            logger.warning(f"Error during memory consolidation: {e}")

    # Orchestrate the thought cycle
    result = vanta_supervisor.orchestrate_thought_cycle(query, context)

    execution_time = time.time() - start_time

    # Create a simplified response format with cognitive metrics
    simplified_result = {
        "response": result["response"],
        "sigil_count": len(result["sigils_used"]) if isinstance(result["sigils_used"], list) else 1,
        "scaffold": result["scaffold"],
        "execution_time": execution_time,
        "cognitive_metrics": orchestration_metrics,
    }

    # Add ART analysis to result if available
    if "art_analysis" in context:
        simplified_result["art_analysis"] = {
            "category": context["art_analysis"].get("category_id", "unknown"),
            "resonance": context["art_analysis"].get("resonance", 0),
            "is_novel": context["art_analysis"].get("is_novel_category", False),
        }

    # Enhanced ART training with cognitive monitoring
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
            orchestration_metrics["training_completed"] = True
            logger.debug("ART training completed for this interaction")
        except Exception as e:
            logger.warning(f"ART training failed: {e}")

    return simplified_result


# Keep original sync function for backwards compatibility
def orchestrate_query(
    vanta_supervisor,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to orchestrate a query through VANTA.
    Synchronous wrapper around async version for backwards compatibility.
    """
    try:
        # Try to run async version if event loop is available
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we can't use asyncio.run
            # Return synchronous result
            return _orchestrate_query_sync(vanta_supervisor, query, context)
        else:
            return asyncio.run(orchestrate_query_async(vanta_supervisor, query, context))
    except RuntimeError:
        # No event loop, use synchronous version
        return _orchestrate_query_sync(vanta_supervisor, query, context)


def _orchestrate_query_sync(
    vanta_supervisor,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronous implementation for backwards compatibility"""
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
            logger.warning(
                f"ART analysis failed: {e}"
            )  # Check if we should run memory consolidation via SleepTimeCompute
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
            ]:
                if importlib.util.find_spec(module_path) is not None:
                    sleep_time_compute_module = importlib.import_module(module_path)
                    logger.debug(f"Found SleepTimeCompute module at {module_path}")
                    break

            if sleep_time_compute_module is None:
                logger.warning("SleepTimeCompute module not found in any expected location")
            else:
                SleepTimeCompute = getattr(sleep_time_compute_module, "SleepTimeCompute")

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
            logger.warning("SleepTimeCompute module not available for memory consolidation")
        except AttributeError as e:
            logger.warning(f"Error accessing SleepTimeCompute class: {e}")
        except Exception as e:
            logger.warning(f"Error during memory consolidation: {e}")

    # Orchestrate the thought cycle
    result = vanta_supervisor.orchestrate_thought_cycle(query, context)

    # Create a simplified response format
    simplified_result = {
        "response": result["response"],
        "sigil_count": len(result["sigils_used"]) if isinstance(result["sigils_used"], list) else 1,
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
