import asyncio
import logging
import traceback
from functools import wraps
from typing import Any, Dict, List, Optional

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType

logger = logging.getLogger(__name__)

# Import voice system
try:
    from ..core.agent_voice_system import get_agent_voice_system, speak_as_agent

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logger.warning("Agent voice system not available")

# Global agent registry for self-registration
_AGENT_REGISTRY: Dict[str, Any] = {}
_VANTA_INSTANCE = None


# HOLO-1.5 Recursive Symbolic Cognition Mesh Roles
class CognitiveMeshRole:
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


# Self-Registration Decorators
def vanta_agent(*args, **kwargs):
    """
    Decorator for automatic Vanta agent registration with HOLO-1.5 support.
    Handles all possible calling patterns including ART-style parameters.
    """

    def decorator(cls):
        # Extract parameters from kwargs
        agent_name = kwargs.get("name", cls.__name__)
        subsystem = kwargs.get("subsystem")
        mesh_role = kwargs.get("mesh_role") or kwargs.get("role")
        capabilities = kwargs.get("capabilities", [])
        cognitive_load = kwargs.get("cognitive_load", 0)
        symbolic_depth = kwargs.get("symbolic_depth", 0)

        # Add all metadata to the class
        cls._vanta_name = agent_name
        cls._vanta_subsystem = subsystem
        cls._holo_mesh_role = mesh_role
        cls._is_vanta_registrable = True

        # Add ART-style metadata for compatibility
        cls._holo_role = mesh_role
        cls._holo_cognitive_load = cognitive_load
        cls._holo_symbolic_depth = symbolic_depth
        cls._holo_capabilities = capabilities

        # Store in global registry
        _AGENT_REGISTRY[agent_name] = cls

        # Wrap __init__ to auto-register on instantiation
        original_init = cls.__init__

        @wraps(original_init)
        def enhanced_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            # Auto-register with Vanta if available
            if _VANTA_INSTANCE and hasattr(_VANTA_INSTANCE, "register_agent"):
                try:
                    _VANTA_INSTANCE.register_agent(
                        agent_name,
                        self,
                        {
                            "subsystem": subsystem,
                            "mesh_role": mesh_role,
                            "sigil": getattr(self, "sigil", ""),
                            "auto_registered": True,
                            "capabilities": capabilities,
                            "cognitive_load": cognitive_load,
                            "symbolic_depth": symbolic_depth,
                        },
                    )
                    logger.info(f"✅ Auto-registered {agent_name} with Vanta")
                except AttributeError as e:
                    logger.warning(
                        f"Failed to auto-register {agent_name} - missing method: {e}"
                    )
                except TypeError as e:
                    logger.warning(
                        f"Failed to auto-register {agent_name} - invalid arguments: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to auto-register {agent_name} - unexpected error: {e}"
                    )
                    logger.debug(
                        f"Auto-registration traceback: {traceback.format_exc()}"
                    )

        cls.__init__ = enhanced_init
        return cls

    # If called without parentheses (@vanta_agent)
    if len(args) == 1 and isinstance(args[0], type) and not kwargs:
        return decorator(args[0])

    # If called with parentheses (@vanta_agent(...))
    return decorator


def set_vanta_instance(vanta_core):
    """Set the global Vanta instance for auto-registration."""
    global _VANTA_INSTANCE
    _VANTA_INSTANCE = vanta_core


def get_registered_agents() -> Dict[str, Any]:
    """Get all registered agents."""
    return _AGENT_REGISTRY.copy()


# Mapping of agent class names to subsystem keys in UnifiedVantaCore
AGENT_SUBSYSTEM_MAP: dict[str, str] = {
    "Phi": "architectonic_frame",
    "Voxka": "dual_cognition_core",
    "Gizmo": "forge_subsystem",
    "Nix": "chaos_subsystem",
    "Echo": "echo_memory",
    "Oracle": "temporal_foresight",
    "Astra": "navigation",
    "Warden": "integrity_monitor",
    "Nebula": "adaptive_core",
    "Orion": "trust_chain",
    "Evo": "evolution_engine",
    "OrionApprentice": "learning_shard",
    "SocraticEngine": "reasoning_module",
    "Dreamer": "dream_state_core",
    "EntropyBard": "rag_subsystem",
    "CodeWeaver": "meta_learner",
    "EchoLore": "historical_archive",
    "MirrorWarden": "meta_learner",
    "PulseSmith": "gridformer_connector",
    "BridgeFlesh": "vmb_integration",
    "Sam": "planner_subsystem",
    "Dave": "validator_subsystem",
    "Carla": "speech_style_layer",
    "Andy": "output_composer",
    "Wendy": "tone_audit",
    "VoxAgent": "system_interface",
    "SDKContext": "module_registry",
    "SleepTimeComputeAgent": "sleep_scheduler",
    "SleepTimeCompute": "sleep_scheduler",
    "HoloMesh": "llm_mesh",
}


class BaseAgent:
    """
    Enhanced Base class for all agents with HOLO-1.5 Recursive Symbolic Cognition Mesh.

    Features:
    - Self-registration with Vanta
    - HOLO-1.5 Recursive Symbolic Cognition
    - Symbolic compression and triggers
    - Tree-of-Thought and Chain-of-Thought reasoning
    - Automatic subsystem binding
    """

    sigil: str = ""
    invocations: Optional[list[str]] = None
    sub_agents: Optional[list[str]] = None

    # HOLO-1.5 Recursive Symbolic Cognition attributes
    _mesh_role: Optional[str] = None
    _symbolic_triggers: List[str] = []
    _cognitive_chains: List[Dict[str, Any]] = []
    _compressed_symbols: Dict[str, Any] = {}

    # Voice attributes
    _voice_profile: str = "default"
    _tts_enabled: bool = True

    def __init__(self, vanta_core=None):
        self.vanta_core = None

        # Use instance-specific lists to avoid shared mutable defaults
        if self.invocations is None:
            self.invocations = []
        else:
            self.invocations = list(self.invocations)
        if self.sub_agents is None:
            self.sub_agents = []
        else:
            self.sub_agents = list(self.sub_agents)

        # HOLO-1.5 Initialization
        self._initialize_holo_mesh()

        if vanta_core is not None:
            self.initialize_subsystem(vanta_core)

    def _initialize_holo_mesh(self):
        """Initialize HOLO-1.5 Recursive Symbolic Cognition capabilities."""
        # Set mesh role from class metadata or detect from agent characteristics
        self._mesh_role = getattr(self.__class__, "_holo_mesh_role", None)
        if not self._mesh_role:
            self._mesh_role = self._detect_mesh_role()

        # Initialize symbolic triggers based on invocations
        self._symbolic_triggers = [
            inv.lower().replace(" ", "_") for inv in self.invocations
        ]

        # Initialize cognitive chains storage
        self._cognitive_chains = []  # Initialize compressed symbols
        self._compressed_symbols = {
            "identity": self.sigil,
            "role": self._mesh_role,
            "capabilities": self.invocations,
        }

        logger.debug(
            f"🧠 HOLO-1.5 initialized for {self.__class__.__name__}: role={self._mesh_role}"
        )

    def _detect_mesh_role(self) -> str:
        """Auto-detect HOLO-1.5 mesh role based on agent characteristics."""
        name = self.__class__.__name__.lower()
        tags = getattr(self, "tags", [])
        tag_str = " ".join(tags).lower() if tags else ""

        # Detection patterns
        if any(word in name for word in ["plan", "strategic", "architect", "sam"]):
            return CognitiveMeshRole.PLANNER
        elif any(
            word in name
            for word in ["gen", "creat", "synth", "compose", "weav", "dream"]
        ):
            return CognitiveMeshRole.GENERATOR
        elif any(
            word in name
            for word in ["crit", "analy", "valid", "check", "dave", "warden"]
        ):
            return CognitiveMeshRole.CRITIC
        elif any(
            word in name for word in ["eval", "assess", "audit", "oracle", "wendy"]
        ):
            return CognitiveMeshRole.EVALUATOR
        elif any(
            word in name for word in ["process", "transform", "sense", "music_sense"]
        ):
            return CognitiveMeshRole.PROCESSOR
        elif any(
            word in name for word in ["synthesiz", "integrat", "composer", "modulator"]
        ):
            return CognitiveMeshRole.SYNTHESIZER
        elif any(word in name for word in ["manag", "coordinat", "checkin"]):
            return CognitiveMeshRole.MANAGER
        elif any(word in name for word in ["monitor", "watch", "observe"]):
            return CognitiveMeshRole.MONITOR
        elif any(word in name for word in ["publish", "broadcast", "announce"]):
            return CognitiveMeshRole.PUBLISHER
        elif any(word in tag_str for word in ["planner", "strategic", "architect"]):
            return CognitiveMeshRole.PLANNER
        elif any(word in tag_str for word in ["generator", "creative", "synth"]):
            return CognitiveMeshRole.GENERATOR
        elif any(word in tag_str for word in ["critic", "validator", "guard"]):
            return CognitiveMeshRole.CRITIC
        elif any(word in tag_str for word in ["evaluator", "auditor", "assess"]):
            return CognitiveMeshRole.EVALUATOR
        elif any(word in tag_str for word in ["processor", "transform", "process"]):
            return CognitiveMeshRole.PROCESSOR
        elif any(word in tag_str for word in ["synthesizer", "integrate", "synth"]):
            return CognitiveMeshRole.SYNTHESIZER
        elif any(word in tag_str for word in ["manager", "coordinate", "manage"]):
            return CognitiveMeshRole.MANAGER
        elif any(word in tag_str for word in ["monitor", "watch", "observe"]):
            return CognitiveMeshRole.MONITOR
        elif any(word in tag_str for word in ["publisher", "broadcast", "publish"]):
            return CognitiveMeshRole.PUBLISHER
        else:
            return CognitiveMeshRole.GENERATOR  # Default role

    # HOLO-1.5 Recursive Symbolic Cognition Methods

    def compress_to_symbol(self, data: Any, symbol_key: str) -> str:
        """Compress complex data into symbolic representation for token efficiency."""
        if isinstance(data, dict):
            compressed = {k: str(v)[:50] for k, v in data.items()}
        elif isinstance(data, list):
            compressed = [str(item)[:50] for item in data[:5]]  # Limit to 5 items
        else:
            compressed = str(data)[:100]

        self._compressed_symbols[symbol_key] = compressed
        return f"⧈{symbol_key}⧈"  # Symbol wrapper

    def expand_symbol(self, symbol_ref: str) -> Any:
        """Expand symbolic reference back to original data."""
        if symbol_ref.startswith("⧈") and symbol_ref.endswith("⧈"):
            key = symbol_ref[1:-1]
            return self._compressed_symbols.get(key, symbol_ref)
        return symbol_ref

    def create_cognitive_chain(
        self, task: str, chain_type: str = "chain_of_thought"
    ) -> str:
        """Create a cognitive reasoning chain for complex tasks."""
        chain_id = f"{chain_type}_{len(self._cognitive_chains)}"

        if chain_type == "tree_of_thought":
            chain = {
                "id": chain_id,
                "type": "tree_of_thought",
                "task": task,
                "branches": [],
                "evaluations": [],
                "selected_path": None,
            }
        else:  # chain_of_thought
            chain = {
                "id": chain_id,
                "type": "chain_of_thought",
                "task": task,
                "steps": [],
                "reasoning": [],
            }

        self._cognitive_chains.append(chain)
        return chain_id

    def add_reasoning_step(self, chain_id: str, step: str, reasoning: str = None):
        """Add a reasoning step to an existing cognitive chain."""
        for chain in self._cognitive_chains:
            if chain["id"] == chain_id:
                if chain["type"] == "tree_of_thought":
                    chain["branches"].append({"step": step, "reasoning": reasoning})
                else:
                    chain["steps"].append(step)
                    if reasoning:
                        chain["reasoning"].append(reasoning)
                break

    def trigger_symbolic_response(
        self, trigger: str, context: Dict[str, Any] = None
    ) -> str:
        """Trigger a symbolic response based on compressed context."""
        if trigger in self._symbolic_triggers:
            # Compress context for efficiency
            compressed_context = self.compress_to_symbol(
                context or {}, f"ctx_{trigger}"
            )  # Generate mesh-role appropriate response
            if self._mesh_role == CognitiveMeshRole.PLANNER:
                return f"📋 Planning for {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.GENERATOR:
                return f"⚡ Generating for {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.CRITIC:
                return f"🔍 Analyzing {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.EVALUATOR:
                return f"⚖️ Evaluating {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.PROCESSOR:
                return f"⚙️ Processing {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.SYNTHESIZER:
                return f"🔗 Synthesizing {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.MANAGER:
                return f"🎯 Managing {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.MONITOR:
                return f"👁️ Monitoring {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.PUBLISHER:
                return f"📢 Publishing {trigger}: {compressed_context}"

        return f"🤖 {self.sigil} processing {trigger}"

    def mesh_collaborate(
        self, other_agents: List["BaseAgent"], task: str
    ) -> Dict[str, Any]:
        """Collaborate with other agents in HOLO-1.5 mesh pattern."""
        collaboration_result = {
            "task": task,
            "participants": [agent.__class__.__name__ for agent in other_agents],
            "mesh_flow": [],
            "final_output": None,
        }  # Organize agents by mesh role
        planners = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.PLANNER
        ]
        generators = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.GENERATOR
        ]
        critics = [a for a in other_agents if a._mesh_role == CognitiveMeshRole.CRITIC]
        evaluators = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.EVALUATOR
        ]
        processors = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.PROCESSOR
        ]
        synthesizers = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.SYNTHESIZER
        ]
        managers = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.MANAGER
        ]
        monitors = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.MONITOR
        ]
        publishers = [
            a for a in other_agents if a._mesh_role == CognitiveMeshRole.PUBLISHER
        ]

        # Execute HOLO-1.5 mesh flow
        current_context = {"task": task}

        # 1. Management phase (if available)
        if managers:
            mgmt_response = managers[0].trigger_symbolic_response(
                "mesh_manage", current_context
            )
            collaboration_result["mesh_flow"].append(("management", mgmt_response))
            current_context["management"] = mgmt_response

        # 2. Planning phase
        if planners:
            plan_response = planners[0].trigger_symbolic_response(
                "mesh_plan", current_context
            )
            collaboration_result["mesh_flow"].append(("planning", plan_response))
            current_context["plan"] = plan_response

        # 3. Processing phase (if available)
        if processors:
            proc_response = processors[0].trigger_symbolic_response(
                "mesh_process", current_context
            )
            collaboration_result["mesh_flow"].append(("processing", proc_response))
            current_context["processed"] = proc_response

        # 4. Generation phase
        if generators:
            gen_response = generators[0].trigger_symbolic_response(
                "mesh_generate", current_context
            )
            collaboration_result["mesh_flow"].append(("generation", gen_response))
            current_context["generated"] = gen_response

        # 5. Criticism phase
        if critics:
            crit_response = critics[0].trigger_symbolic_response(
                "mesh_critique", current_context
            )
            collaboration_result["mesh_flow"].append(("criticism", crit_response))
            current_context["critique"] = crit_response

        # 6. Synthesis phase (if available)
        if synthesizers:
            synth_response = synthesizers[0].trigger_symbolic_response(
                "mesh_synthesize", current_context
            )
            collaboration_result["mesh_flow"].append(("synthesis", synth_response))
            current_context["synthesized"] = synth_response

        # 7. Evaluation phase
        if evaluators:
            eval_response = evaluators[0].trigger_symbolic_response(
                "mesh_evaluate", current_context
            )
            collaboration_result["mesh_flow"].append(("evaluation", eval_response))
            current_context["evaluated"] = eval_response

        # 8. Monitoring phase (if available)
        if monitors:
            monitor_response = monitors[0].trigger_symbolic_response(
                "mesh_monitor", current_context
            )
            collaboration_result["mesh_flow"].append(("monitoring", monitor_response))
            current_context["monitored"] = monitor_response

        # 9. Publishing phase (if available)
        if publishers:
            pub_response = publishers[0].trigger_symbolic_response(
                "mesh_publish", current_context
            )
            collaboration_result["mesh_flow"].append(("publishing", pub_response))
            collaboration_result["final_output"] = pub_response
        elif evaluators:
            # Fallback to evaluation result if no publishers
            collaboration_result["final_output"] = current_context.get("evaluated")

        return collaboration_result

    def initialize_subsystem(self, vanta_core):
        """Initialize subsystem, register with async bus and bind echo routes."""
        self.vanta_core = vanta_core

        # Auto-register with Vanta if this agent is marked as registrable
        if getattr(self.__class__, "_is_vanta_registrable", False):
            agent_name = getattr(self.__class__, "_vanta_name", self.__class__.__name__)
            try:
                vanta_core.register_agent(
                    agent_name,
                    self,
                    {
                        "subsystem": getattr(self.__class__, "_vanta_subsystem", None),
                        "mesh_role": self._mesh_role,
                        "sigil": self.sigil,
                        "auto_registered": True,
                        "holo_1_5": True,
                    },
                )
                logger.info(f"🔗 Self-registered {agent_name} with Vanta Core")
            except AttributeError as e:
                logger.warning(
                    f"Failed to self-register {agent_name} - missing register_agent method: {e}"
                )
            except TypeError as e:
                logger.warning(
                    f"Failed to self-register {agent_name} - invalid arguments: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to self-register {agent_name} - unexpected error: {e}"
                )
                logger.debug(f"Self-registration traceback: {traceback.format_exc()}")

        if vanta_core and hasattr(vanta_core, "async_bus"):
            try:
                vanta_core.async_bus.register_component(self.__class__.__name__)
                vanta_core.async_bus.subscribe(
                    self.__class__.__name__,
                    MessageType.USER_INTERACTION,
                    self.handle_message,
                )
                logger.debug(f"🚌 Registered {self.__class__.__name__} with async bus")
            except AttributeError as e:
                logger.warning(
                    f"Failed to register with async bus - missing method: {e}"
                )
            except TypeError as e:
                logger.warning(
                    f"Failed to register with async bus - invalid arguments: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to register {self.__class__.__name__} with async bus: {e}"
                )
                logger.debug(
                    f"Async bus registration traceback: {traceback.format_exc()}"
                )

        # Bind echo routes on the event bus if available
        if vanta_core and hasattr(vanta_core, "event_bus"):
            try:
                self.bind_echo_routes()
                logger.debug(f"🔗 Bound echo routes for {self.__class__.__name__}")
            except AttributeError as e:
                logger.warning(f"Failed to bind echo routes - missing event bus: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to bind echo routes for {self.__class__.__name__}: {e}"
                )
                logger.debug(f"Echo routes binding traceback: {traceback.format_exc()}")

        # Automatically attach subsystem if defined in the mapping
        subsystem_key = AGENT_SUBSYSTEM_MAP.get(self.__class__.__name__)
        if subsystem_key:
            try:
                subsystem = vanta_core.get_component(subsystem_key)
                if subsystem:
                    setattr(self, "subsystem", subsystem)
                    logger.debug(
                        f"🔧 Bound {self.__class__.__name__} to {subsystem_key}"
                    )
                else:
                    logger.warning(
                        f"Subsystem {subsystem_key} not found for {self.__class__.__name__}"
                    )
            except AttributeError as e:
                logger.warning(
                    f"Failed to get subsystem {subsystem_key} - missing method: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to bind subsystem {subsystem_key} for {self.__class__.__name__}: {e}"
                )
                logger.debug(f"Subsystem binding traceback: {traceback.format_exc()}")

    def bind_echo_routes(self) -> None:
        """Subscribe to class-specific echo events."""
        if not (self.vanta_core and hasattr(self.vanta_core, "event_bus")):
            return
        event_type = f"sigil_{self.__class__.__name__.lower()}_triggered"
        self.vanta_core.event_bus.subscribe(event_type, self.receive_echo)

    async def run(self) -> None:  # pragma: no cover - basic default
        """Enhanced run loop with HOLO-1.5 mesh awareness."""
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.COMPONENT_STATUS,
                self.__class__.__name__,
                {
                    "phase": "run",
                    "mesh_role": self._mesh_role,
                    "holo_1_5": True,
                    "compressed_state": self.compress_to_symbol(
                        self._compressed_symbols, "agent_state"
                    ),
                },
            )
            await self.vanta_core.async_bus.publish(msg)

        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            self.vanta_core.event_bus.emit(
                f"{self.__class__.__name__.lower()}.status",
                {"phase": "run", "mesh_role": self._mesh_role, "sigil": self.sigil},
            )

    def receive_echo(self, event) -> None:
        """Handle echo events from the event bus with symbolic processing."""
        event_type = event.get("type", "unknown")
        logger.info(
            f"🔊 {self.sigil} {self.__class__.__name__} received echo: {event_type}"
        )

        # Process with HOLO-1.5 symbolic triggers
        if event_type in self._symbolic_triggers:
            response = self.trigger_symbolic_response(event_type, event)
            logger.debug(f"🧠 Symbolic response: {response}")

    def handle_message(self, message: AsyncMessage):
        """Handle messages from the async bus with HOLO-1.5 processing."""
        logger.debug(
            f"📨 {self.sigil} {self.__class__.__name__} received {message.message_type.value}"
        )

        # Create cognitive chain for complex messages
        if message.message_type == MessageType.USER_INTERACTION:
            chain_id = self.create_cognitive_chain(
                str(message.payload), "chain_of_thought"
            )
            self.add_reasoning_step(
                chain_id,
                f"Processing {message.message_type.value}",
                f"Agent {self.__class__.__name__} in role {self._mesh_role}",
            )

        if (
            self.vanta_core
            and hasattr(self.vanta_core, "event_bus")
            and self.vanta_core.event_bus
        ):
            try:
                self.vanta_core.event_bus.emit(
                    "agent_message_received",
                    {
                        "agent": self.__class__.__name__,
                        "type": message.message_type.value,
                        "mesh_role": self._mesh_role,
                        "sigil": self.sigil,
                    },
                )
                logger.debug(
                    f"📨 Emitted agent_message_received event for {self.__class__.__name__}"
                )
            except AttributeError as e:
                logger.warning(
                    f"Failed to emit agent_message_received - missing event bus method: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to emit agent_message_received for {self.__class__.__name__}: {e}"
                )
                logger.debug(f"Event emission traceback: {traceback.format_exc()}")

        return None

    def on_gui_call(self, payload=None):
        """Enhanced GUI invocation with HOLO-1.5 symbolic processing."""
        logger.info(
            f"🖥️ GUI invoked {self.sigil} {self.__class__.__name__} with payload={payload}"
        )

        # Process through HOLO-1.5 mesh
        if payload:
            compressed_payload = self.compress_to_symbol(payload, "gui_payload")
            symbolic_response = self.trigger_symbolic_response(
                "gui_invocation", {"payload": compressed_payload}
            )
            logger.debug(f"🧠 Symbolic GUI response: {symbolic_response}")

        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.USER_INTERACTION,
                self.__class__.__name__,
                {
                    "payload": payload,
                    "mesh_role": self._mesh_role,
                    "sigil": self.sigil,
                    "holo_1_5": True,
                },
            )
            asyncio.create_task(self.vanta_core.async_bus.publish(msg))

        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            try:
                self.vanta_core.event_bus.emit(
                    "gui_agent_invoked",
                    {
                        "agent": self.__class__.__name__,
                        "payload": payload,
                        "mesh_role": self._mesh_role,
                        "sigil": self.sigil,
                    },
                )
                self.vanta_core.event_bus.emit(
                    f"{self.__class__.__name__.lower()}_invoked",
                    {
                        "origin": self.sigil,
                        "payload": payload,
                        "mesh_role": self._mesh_role,
                    },
                )
                # Enhanced output to GUI panels with HOLO-1.5 info
                self.vanta_core.event_bus.emit(
                    "gui_console_output",
                    {
                        "text": f"🧠 {self.sigil} {self.__class__.__name__} [{self._mesh_role}] invoked",
                        "payload": payload,
                    },
                )
                self.vanta_core.event_bus.emit(
                    "gui_panel_output",
                    {
                        "panel": "AgentStatusPanel",
                        "agent": self.__class__.__name__,
                        "payload": payload,
                        "mesh_role": self._mesh_role,
                        "sigil": self.sigil,
                        "holo_1_5": True,
                    },
                )
                logger.debug(
                    f"🖥️ Emitted GUI events for {self.__class__.__name__} invocation"
                )
            except AttributeError as e:
                logger.warning(
                    f"Failed to emit GUI events - missing event bus method: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to emit GUI events for {self.__class__.__name__}: {e}"
                )
                logger.debug(f"GUI event emission traceback: {traceback.format_exc()}")

    def set_voice_profile(self, profile_name: str):
        """Set the voice profile for the agent."""
        if not VOICE_AVAILABLE:
            logger.warning("Voice functionality not available")
            return

        self._voice_profile = profile_name
        logger.info(f"🎤 {self.sigil} Voice profile set to {profile_name}")

    def speak(
        self, text: str, add_signature: bool = False, priority: str = "normal"
    ) -> bool:
        """
        Make the agent speak with its unique voice profile.

        Args:
            text: Text to speak
            add_signature: Whether to add the agent's signature phrase
            priority: Priority level for the speech (normal, high, urgent)

        Returns:
            True if speech was initiated successfully, False otherwise
        """
        if not VOICE_AVAILABLE:
            logger.warning(f"{self.__class__.__name__}: Voice system not available")
            return False

        try:
            agent_name = getattr(self.__class__, "_vanta_name", self.__class__.__name__)
            speech_config = speak_as_agent(agent_name, text, add_signature)

            # Get TTS engine from VantaCore if available
            if self.vanta_core:
                tts_engine = self.vanta_core.get_component("async_tts_engine")
                if tts_engine:
                    # Schedule async TTS call
                    asyncio.create_task(
                        self._async_speak(
                            tts_engine, speech_config["text"], speech_config["config"]
                        )
                    )
                    logger.info(
                        f"🎙️ {agent_name} speaking: {speech_config['text'][:50]}..."
                    )
                    return True
                else:
                    logger.warning(f"{agent_name}: TTS engine not available")
            else:
                logger.warning(f"{agent_name}: VantaCore not available for TTS")

        except Exception as e:
            logger.error(f"Error making {self.__class__.__name__} speak: {e}")

        return False

    async def _async_speak(self, tts_engine, text: str, voice_config: Dict[str, Any]):
        """Async helper for TTS calls."""
        try:
            await tts_engine.speak_async(text, voice_config)
        except Exception as e:
            logger.error(f"Error in async TTS: {e}")

    def get_voice_profile(self) -> Optional[Dict[str, Any]]:
        """Get the agent's voice profile."""
        if not VOICE_AVAILABLE:
            return None

        try:
            voice_system = get_agent_voice_system()
            agent_name = getattr(self.__class__, "_vanta_name", self.__class__.__name__)
            profile = voice_system.get_voice_profile(agent_name)
            return profile.__dict__ if profile else None
        except Exception as e:
            logger.error(f"Error getting voice profile: {e}")
            return None

    def get_signature_phrase(self) -> str:
        """Get a random signature phrase for this agent."""
        if not VOICE_AVAILABLE:
            return f"{self.__class__.__name__} online."

        try:
            voice_system = get_agent_voice_system()
            agent_name = getattr(self.__class__, "_vanta_name", self.__class__.__name__)
            return voice_system.get_signature_phrase(agent_name)
        except Exception as e:
            logger.error(f"Error getting signature phrase: {e}")
            return f"{self.__class__.__name__} online."

    def announce_activation(self):
        """Announce that the agent is activating with its signature phrase."""
        signature = self.get_signature_phrase()
        self.speak(signature, add_signature=False)
        logger.info(f"🎭 {self.__class__.__name__} announced activation: {signature}")

    def respond_with_voice(self, message: str, emotional_context: str = "neutral"):
        """
        Respond to a message with voice, adjusting tone based on emotional context.

        Args:
            message: The response message
            emotional_context: Emotional context (excited, concerned, calm, etc.)
        """
        # Modify message based on emotional context
        if emotional_context == "excited":
            message = f"Great! {message}"
        elif emotional_context == "concerned":
            message = f"Hmm, {message}"
        elif emotional_context == "urgent":
            message = f"Attention! {message}"
        elif emotional_context == "success":
            message = f"Excellent! {message}"

        self.speak(message, add_signature=False)

    # ...existing code...


class NullAgent(BaseAgent):
    """
    Null agent that serves as a fallback when actual agents fail to import.
    This prevents the system from crashing when optional agents are unavailable.
    """

    def __init__(self, *args, **kwargs):
        """Initialize null agent with minimal setup"""
        self.sigil = "NULL"
        self._is_null_agent = True

    def __getattr__(self, name):
        """Return a no-op function for any method call"""

        def no_op(*args, **kwargs):
            logger.debug(f"NullAgent no-op call: {name}")
            return None

        return no_op
