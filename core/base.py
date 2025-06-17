# core/base.py
"""
HOLO-1.5 Recursive Symbolic Cognition Mesh Base Infrastructure for Core Modules
Enhanced with advanced cognitive collaboration and symbolic processing capabilities
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Vanta.CoreBase")


class CognitiveMeshRole(Enum):  # Copied from agents/base.py
    """HOLO-1.5 Recursive Symbolic Cognition Mesh Role definitions."""

    PLANNER = "planner"  # Strategic planning and task decomposition
    GENERATOR = "generator"  # Content generation and solution synthesis
    CRITIC = "critic"  # Analysis and evaluation of solutions
    EVALUATOR = "evaluator"  # Final assessment and quality control
    PROCESSOR = "processor"  # Data processing and transformation
    SYNTHESIZER = "synthesizer"  # Synthesis and integration of results
    MANAGER = "manager"  # Management and coordination
    MONITOR = "monitor"  # Monitoring and observation
    PUBLISHER = "publisher"  # Publishing and broadcasting
    ORCHESTRATOR = "orchestrator"  # Added from original core/base.py definition
    CONNECTOR = "connector"  # Added from original core/base.py definition


@dataclass
class HOLO15CoreMetadata:
    """Metadata for HOLO-1.5 core module registration."""

    name: str
    subsystem: str
    mesh_role: CognitiveMeshRole
    description: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    cognitive_load: float = 1.0
    symbolic_depth: int = 1
    collaboration_patterns: List[str] = field(default_factory=list)


# Subsystem mappings for core modules
VOXSIGIL_CORE_SUBSYSTEMS = {
    "learning_management": [
        "learning_manager",
        "default_learning_manager",
        "AdvancedMetaLearner",
    ],
    "cognitive_processing": [
        "meta_cognitive",
        "dialogue_manager",
        "proactive_intelligence",
    ],
    "model_management": ["model_manager", "model_architecture_fixer"],
    "grid_processing": [
        "enhanced_grid_connector",
        "grid_distillation",
        "grid_former_evaluator",
        "iterative_gridformer",
        "iterative_reasoning_gridformer",
    ],
    "data_management": ["download_arc_data", "end_to_end_arc_validation"],
    "system_management": ["checkin_manager_vosk"],
    "neural_symbolic": ["neuro_symbolic_network"],
    "optimization": ["hyperparameter_search"],
}


class HOLO15CoreAdapter:
    """Enhanced cognitive mesh adapter for core modules."""

    def __init__(self, core_instance, metadata: HOLO15CoreMetadata):
        self.core_instance = core_instance
        self.metadata = metadata
        self.mesh_connections: Dict[str, Any] = {}
        self.symbolic_registry: Dict[str, Any] = {}
        self.collaboration_history: List[Dict] = []

    async def process_symbolic_request(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process symbolic cognitive requests through the mesh."""
        try:
            result = {
                "symbol": symbol,
                "processed_by": self.metadata.name,
                "role": self.metadata.mesh_role.value,
                "context_keys": list(context.keys()),
                "symbolic_depth": self.metadata.symbolic_depth,
                "timestamp": "current",
                "status": "processed",
            }

            # Route based on cognitive role
            if self.metadata.mesh_role == CognitiveMeshRole.ORCHESTRATOR:
                result["orchestration"] = await self._orchestrate_symbolic_flow(
                    symbol, context
                )
            elif self.metadata.mesh_role == CognitiveMeshRole.PROCESSOR:
                result["processing"] = await self._process_symbolic_data(
                    symbol, context
                )
            elif self.metadata.mesh_role == CognitiveMeshRole.EVALUATOR:
                result["evaluation"] = await self._evaluate_symbolic_content(
                    symbol, context
                )
            elif self.metadata.mesh_role == CognitiveMeshRole.SYNTHESIZER:
                result["synthesis"] = await self._synthesize_symbolic_knowledge(
                    symbol, context
                )
            elif self.metadata.mesh_role == CognitiveMeshRole.MANAGER:
                result["management"] = await self._manage_symbolic_resources(
                    symbol, context
                )
            elif self.metadata.mesh_role == CognitiveMeshRole.CONNECTOR:
                result["connection"] = await self._connect_symbolic_elements(
                    symbol, context
                )

            self.collaboration_history.append(result)
            return result
        except AttributeError as e:
            logger.error(f"HOLO-1.5 missing attribute in {self.metadata.name}: {e}")
            return {
                "error": "missing_attribute",
                "details": str(e),
                "symbol": symbol,
                "processed_by": self.metadata.name,
            }
        except TypeError as e:
            logger.error(f"HOLO-1.5 type error in {self.metadata.name}: {e}")
            return {
                "error": "type_error",
                "details": str(e),
                "symbol": symbol,
                "processed_by": self.metadata.name,
            }
        except ValueError as e:
            logger.error(f"HOLO-1.5 value error in {self.metadata.name}: {e}")
            return {
                "error": "value_error",
                "details": str(e),
                "symbol": symbol,
                "processed_by": self.metadata.name,
            }
        except Exception as e:
            logger.error(
                f"HOLO-1.5 unexpected symbolic processing error in {self.metadata.name}: {e}",
                exc_info=True,
            )
            return {
                "error": "unexpected_error",
                "details": str(e),
                "symbol": symbol,
                "processed_by": self.metadata.name,
            }

    async def _orchestrate_symbolic_flow(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate high-level symbolic cognitive flows."""
        return {
            "flow_type": "orchestration",
            "symbol": symbol,
            "coordination_strategy": "hierarchical",
            "cognitive_load_distribution": self.metadata.cognitive_load,
        }

    async def _process_symbolic_data(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process core symbolic data and logic."""
        return {
            "processing_type": "symbolic_computation",
            "symbol": symbol,
            "capabilities_applied": self.metadata.capabilities,
            "depth_level": self.metadata.symbolic_depth,
        }

    async def _evaluate_symbolic_content(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate and assess symbolic content."""
        return {
            "evaluation_type": "symbolic_assessment",
            "symbol": symbol,
            "validation_criteria": self.metadata.capabilities,
            "quality_metrics": {"coherence": 0.8, "relevance": 0.9},
        }

    async def _synthesize_symbolic_knowledge(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize and integrate symbolic knowledge."""
        return {
            "synthesis_type": "knowledge_integration",
            "symbol": symbol,
            "fusion_patterns": self.metadata.collaboration_patterns,
            "emergent_properties": ["enhanced_understanding", "cross_domain_insights"],
        }

    async def _manage_symbolic_resources(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Manage symbolic cognitive resources."""
        return {
            "management_type": "resource_optimization",
            "symbol": symbol,
            "resource_allocation": self.metadata.cognitive_load,
            "lifecycle_stage": "active_processing",
        }

    async def _connect_symbolic_elements(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Connect and integrate symbolic elements."""
        return {
            "connection_type": "symbolic_bridging",
            "symbol": symbol,
            "bridge_patterns": self.metadata.collaboration_patterns,
            "integration_depth": self.metadata.symbolic_depth,
        }


class BaseCore(ABC):
    """Abstract base class for HOLO-1.5 enhanced core modules."""

    def __init__(self, vanta_core, config: Optional[Dict[str, Any]] = None):
        self.vanta_core = vanta_core
        self.config = config or {}
        self.holo15_adapter: Optional[HOLO15CoreAdapter] = None
        self.is_initialized = False

        # Initialize HOLO-1.5 metadata from decorator
        if hasattr(self.__class__, "_vanta_metadata"):
            metadata = self.__class__._vanta_metadata
            self.holo15_adapter = HOLO15CoreAdapter(self, metadata)

        self._register_with_vanta()

    def _register_with_vanta(self):
        """Register the core module with VantaCore automatically."""
        try:
            if hasattr(self.vanta_core, "register_component"):
                component_info = {
                    "type": "holo15_core_module",
                    "mesh_role": self.holo15_adapter.metadata.mesh_role.value
                    if self.holo15_adapter
                    else "unknown",
                    "subsystem": self.holo15_adapter.metadata.subsystem
                    if self.holo15_adapter
                    else "core",
                    "capabilities": self.holo15_adapter.metadata.capabilities
                    if self.holo15_adapter
                    else [],
                    "holo15_enabled": True,
                }
                self.vanta_core.register_component(
                    self.__class__.__name__.lower(), self, component_info
                )
                logger.info(
                    f"🧠 Registered HOLO-1.5 core module: {self.__class__.__name__}"
                )
        except ImportError as e:
            logger.warning(
                f"Failed to auto-register core module {self.__class__.__name__} - missing dependency: {e}"
            )
        except AttributeError as e:
            logger.warning(
                f"Failed to auto-register core module {self.__class__.__name__} - missing method: {e}"
            )
        except TypeError as e:
            logger.warning(
                f"Failed to auto-register core module {self.__class__.__name__} - invalid arguments: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error auto-registering core module {self.__class__.__name__}: {e}",
                exc_info=True,
            )

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the core module. Must be implemented by subclasses."""
        pass

    async def process_mesh_request(
        self, symbol: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process HOLO-1.5 mesh cognitive requests."""
        if self.holo15_adapter:
            return await self.holo15_adapter.process_symbolic_request(symbol, context)
        return {"error": "HOLO-1.5 adapter not available"}

    def get_mesh_status(self) -> Dict[str, Any]:
        """Get current HOLO-1.5 mesh status."""
        if self.holo15_adapter:
            return {
                "name": self.holo15_adapter.metadata.name,
                "role": self.holo15_adapter.metadata.mesh_role.value,
                "subsystem": self.holo15_adapter.metadata.subsystem,
                "capabilities": self.holo15_adapter.metadata.capabilities,
                "collaboration_history_length": len(
                    self.holo15_adapter.collaboration_history
                ),
                "symbolic_depth": self.holo15_adapter.metadata.symbolic_depth,
                "cognitive_load": self.holo15_adapter.metadata.cognitive_load,
                "mesh_connections": len(self.holo15_adapter.mesh_connections),
            }
        return {"status": "HOLO-1.5 not enabled"}


def vanta_core_module(
    name: str,
    subsystem: str,
    mesh_role: CognitiveMeshRole,
    description: str,
    capabilities: List[str],
    dependencies: Optional[List[str]] = None,
    cognitive_load: float = 1.0,
    symbolic_depth: int = 1,
    collaboration_patterns: Optional[List[str]] = None,
):
    """
    Decorator for HOLO-1.5 encapsulated registration of core modules.

    Args:
        name: Module name for registration
        subsystem: Functional subsystem (from VOXSIGIL_CORE_SUBSYSTEMS)
        mesh_role: Cognitive mesh role specialization
        description: Module description
        capabilities: List of module capabilities
        dependencies: Optional dependencies list
        cognitive_load: Cognitive processing load (0.1-10.0)
        symbolic_depth: Symbolic processing depth (1-5)
        collaboration_patterns: Collaboration patterns for mesh integration
    """

    def decorator(cls):
        # Store metadata as class attribute
        cls._vanta_metadata = HOLO15CoreMetadata(
            name=name,
            subsystem=subsystem,
            mesh_role=mesh_role,
            description=description,
            capabilities=capabilities,
            dependencies=dependencies or [],
            cognitive_load=cognitive_load,
            symbolic_depth=symbolic_depth,
            collaboration_patterns=collaboration_patterns or [],
        )

        # Store original init
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Call original init first
            original_init(self, *args, **kwargs)

            # If not already a BaseCore subclass, add HOLO-1.5 capabilities
            if not isinstance(self, BaseCore):
                self.holo15_adapter = HOLO15CoreAdapter(self, cls._vanta_metadata)

        cls.__init__ = new_init

        # Add HOLO-1.5 methods if not present
        if not hasattr(cls, "process_mesh_request"):

            def process_mesh_request(self, symbol: str, context: Dict[str, Any]):
                if hasattr(self, "holo15_adapter") and self.holo15_adapter:
                    return self.holo15_adapter.process_symbolic_request(symbol, context)
                return {"error": "HOLO-1.5 adapter not available"}

            cls.process_mesh_request = process_mesh_request

        if not hasattr(cls, "get_mesh_status"):

            def get_mesh_status(self):
                if hasattr(self, "holo15_adapter") and self.holo15_adapter:
                    return {
                        "name": self.holo15_adapter.metadata.name,
                        "role": self.holo15_adapter.metadata.mesh_role.value
                        if hasattr(self.holo15_adapter.metadata.mesh_role, "value")
                        else self.holo15_adapter.metadata.mesh_role,
                        "subsystem": self.holo15_adapter.metadata.subsystem,
                        "capabilities": self.holo15_adapter.metadata.capabilities,
                        "holo15_enabled": True,
                    }
                return {"holo15_enabled": False}

            cls.get_mesh_status = get_mesh_status

        logger.info(
            f"🧠 HOLO-1.5 core module decorator applied: {name} ({mesh_role.value if hasattr(mesh_role, 'value') else mesh_role})"
        )
        return cls

    return decorator
