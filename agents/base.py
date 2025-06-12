
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType


logger = logging.getLogger(__name__)

# Global agent registry for self-registration
_AGENT_REGISTRY: Dict[str, Any] = {}
_VANTA_INSTANCE = None

# HOLO-1.5 Recursive Symbolic Cognition Mesh Roles
class CognitiveMeshRole:
    """HOLO-1.5 Recursive Symbolic Cognition Mesh Role definitions."""
    PLANNER = "planner"      # Strategic planning and task decomposition
    GENERATOR = "generator"   # Content generation and solution synthesis  
    CRITIC = "critic"        # Analysis and evaluation of solutions
    EVALUATOR = "evaluator"  # Final assessment and quality control

# Self-Registration Decorators
def vanta_agent(name: str = None, subsystem: str = None, mesh_role: str = None):
    """Decorator for automatic Vanta agent registration with HOLO-1.5 support."""
    def decorator(cls):
        agent_name = name or cls.__name__
        
        # Add self-registration metadata
        cls._vanta_name = agent_name
        cls._vanta_subsystem = subsystem
        cls._holo_mesh_role = mesh_role
        cls._is_vanta_registrable = True
        
        # Store in global registry
        _AGENT_REGISTRY[agent_name] = cls
        
        # Wrap __init__ to auto-register on instantiation
        original_init = cls.__init__
        
        @wraps(original_init)
        def enhanced_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Auto-register with Vanta if available
            if _VANTA_INSTANCE and hasattr(_VANTA_INSTANCE, 'register_agent'):
                try:
                    _VANTA_INSTANCE.register_agent(agent_name, self, {
                        'subsystem': subsystem,
                        'mesh_role': mesh_role,
                        'sigil': getattr(self, 'sigil', ''),
                        'auto_registered': True
                    })
                    logger.info(f"✅ Auto-registered {agent_name} with Vanta")
                except Exception as e:
                    logger.warning(f"Failed to auto-register {agent_name}: {e}")
        
        cls.__init__ = enhanced_init
        return cls
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
        self._mesh_role = getattr(self.__class__, '_holo_mesh_role', None)
        if not self._mesh_role:
            self._mesh_role = self._detect_mesh_role()
            
        # Initialize symbolic triggers based on invocations
        self._symbolic_triggers = [inv.lower().replace(' ', '_') for inv in self.invocations]
        
        # Initialize cognitive chains storage
        self._cognitive_chains = []
        
        # Initialize compressed symbols
        self._compressed_symbols = {
            'identity': self.sigil,
            'role': self._mesh_role,
            'capabilities': self.invocations
        }
        
        logger.debug(f"🧠 HOLO-1.5 initialized for {self.__class__.__name__}: role={self._mesh_role}")

    def _detect_mesh_role(self) -> str:
        """Auto-detect HOLO-1.5 mesh role based on agent characteristics."""
        name = self.__class__.__name__.lower()
        tags = getattr(self, 'tags', [])
        tag_str = ' '.join(tags).lower() if tags else ''
        
        # Detection patterns
        if any(word in name for word in ['plan', 'strategic', 'architect', 'sam']):
            return CognitiveMeshRole.PLANNER
        elif any(word in name for word in ['gen', 'creat', 'synth', 'compose', 'weav', 'dream']):
            return CognitiveMeshRole.GENERATOR
        elif any(word in name for word in ['crit', 'analy', 'valid', 'check', 'dave', 'warden']):
            return CognitiveMeshRole.CRITIC
        elif any(word in name for word in ['eval', 'assess', 'audit', 'oracle', 'wendy']):
            return CognitiveMeshRole.EVALUATOR
        elif any(word in tag_str for word in ['planner', 'strategic', 'architect']):
            return CognitiveMeshRole.PLANNER
        elif any(word in tag_str for word in ['generator', 'creative', 'synth']):
            return CognitiveMeshRole.GENERATOR
        elif any(word in tag_str for word in ['critic', 'validator', 'guard']):
            return CognitiveMeshRole.CRITIC
        elif any(word in tag_str for word in ['evaluator', 'auditor', 'assess']):
            return CognitiveMeshRole.EVALUATOR
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
        if symbol_ref.startswith('⧈') and symbol_ref.endswith('⧈'):
            key = symbol_ref[1:-1]
            return self._compressed_symbols.get(key, symbol_ref)
        return symbol_ref
    
    def create_cognitive_chain(self, task: str, chain_type: str = "chain_of_thought") -> str:
        """Create a cognitive reasoning chain for complex tasks."""
        chain_id = f"{chain_type}_{len(self._cognitive_chains)}"
        
        if chain_type == "tree_of_thought":
            chain = {
                'id': chain_id,
                'type': 'tree_of_thought',
                'task': task,
                'branches': [],
                'evaluations': [],
                'selected_path': None
            }
        else:  # chain_of_thought
            chain = {
                'id': chain_id,
                'type': 'chain_of_thought',
                'task': task,
                'steps': [],
                'reasoning': []
            }
        
        self._cognitive_chains.append(chain)
        return chain_id
    
    def add_reasoning_step(self, chain_id: str, step: str, reasoning: str = None):
        """Add a reasoning step to an existing cognitive chain."""
        for chain in self._cognitive_chains:
            if chain['id'] == chain_id:
                if chain['type'] == 'tree_of_thought':
                    chain['branches'].append({'step': step, 'reasoning': reasoning})
                else:
                    chain['steps'].append(step)
                    if reasoning:
                        chain['reasoning'].append(reasoning)
                break
    
    def trigger_symbolic_response(self, trigger: str, context: Dict[str, Any] = None) -> str:
        """Trigger a symbolic response based on compressed context."""
        if trigger in self._symbolic_triggers:
            # Compress context for efficiency
            compressed_context = self.compress_to_symbol(context or {}, f"ctx_{trigger}")
            
            # Generate mesh-role appropriate response
            if self._mesh_role == CognitiveMeshRole.PLANNER:
                return f"📋 Planning for {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.GENERATOR:
                return f"⚡ Generating for {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.CRITIC:
                return f"🔍 Analyzing {trigger}: {compressed_context}"
            elif self._mesh_role == CognitiveMeshRole.EVALUATOR:
                return f"⚖️ Evaluating {trigger}: {compressed_context}"
        
        return f"🤖 {self.sigil} processing {trigger}"

    def mesh_collaborate(self, other_agents: List['BaseAgent'], task: str) -> Dict[str, Any]:
        """Collaborate with other agents in HOLO-1.5 mesh pattern."""
        collaboration_result = {
            'task': task,
            'participants': [agent.__class__.__name__ for agent in other_agents],
            'mesh_flow': [],
            'final_output': None
        }
        
        # Organize agents by mesh role
        planners = [a for a in other_agents if a._mesh_role == CognitiveMeshRole.PLANNER]
        generators = [a for a in other_agents if a._mesh_role == CognitiveMeshRole.GENERATOR]
        critics = [a for a in other_agents if a._mesh_role == CognitiveMeshRole.CRITIC]
        evaluators = [a for a in other_agents if a._mesh_role == CognitiveMeshRole.EVALUATOR]
        
        # Execute HOLO-1.5 mesh flow
        current_context = {'task': task}
        
        # 1. Planning phase
        if planners:
            plan_response = planners[0].trigger_symbolic_response('mesh_plan', current_context)
            collaboration_result['mesh_flow'].append(('planning', plan_response))
            current_context['plan'] = plan_response
        
        # 2. Generation phase
        if generators:
            gen_response = generators[0].trigger_symbolic_response('mesh_generate', current_context)
            collaboration_result['mesh_flow'].append(('generation', gen_response))
            current_context['generated'] = gen_response
        
        # 3. Criticism phase
        if critics:
            crit_response = critics[0].trigger_symbolic_response('mesh_critique', current_context)
            collaboration_result['mesh_flow'].append(('criticism', crit_response))
            current_context['critique'] = crit_response
        
        # 4. Evaluation phase
        if evaluators:
            eval_response = evaluators[0].trigger_symbolic_response('mesh_evaluate', current_context)
            collaboration_result['mesh_flow'].append(('evaluation', eval_response))
            collaboration_result['final_output'] = eval_response
        
        return collaboration_result


    def initialize_subsystem(self, vanta_core):
        """Initialize subsystem, register with async bus and bind echo routes."""
        self.vanta_core = vanta_core
        
        # Auto-register with Vanta if this agent is marked as registrable
        if getattr(self.__class__, '_is_vanta_registrable', False):
            agent_name = getattr(self.__class__, '_vanta_name', self.__class__.__name__)
            try:
                vanta_core.register_agent(agent_name, self, {
                    'subsystem': getattr(self.__class__, '_vanta_subsystem', None),
                    'mesh_role': self._mesh_role,
                    'sigil': self.sigil,
                    'auto_registered': True,
                    'holo_1_5': True
                })
                logger.info(f"🔗 Self-registered {agent_name} with Vanta Core")
            except Exception as e:
                logger.warning(f"Failed to self-register {agent_name}: {e}")
        
        if vanta_core and hasattr(vanta_core, "async_bus"):
            try:
                vanta_core.async_bus.register_component(self.__class__.__name__)
                vanta_core.async_bus.subscribe(
                    self.__class__.__name__,
                    MessageType.USER_INTERACTION,
                    self.handle_message,
                )
            except Exception:
                pass

        # Bind echo routes on the event bus if available
        if vanta_core and hasattr(vanta_core, "event_bus"):
            try:
                self.bind_echo_routes()
            except Exception:
                pass

        # Automatically attach subsystem if defined in the mapping
        subsystem_key = AGENT_SUBSYSTEM_MAP.get(self.__class__.__name__)
        if subsystem_key:
            try:
                subsystem = vanta_core.get_component(subsystem_key)
                if subsystem:
                    setattr(self, "subsystem", subsystem)
                    logger.debug(f"🔧 Bound {self.__class__.__name__} to {subsystem_key}")
            except Exception:
                pass

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
                    "compressed_state": self.compress_to_symbol(self._compressed_symbols, "agent_state")
                },
            )
            await self.vanta_core.async_bus.publish(msg)
            
        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            self.vanta_core.event_bus.emit(
                f"{self.__class__.__name__.lower()}.status",
                {
                    "phase": "run",
                    "mesh_role": self._mesh_role,
                    "sigil": self.sigil
                },
            )

    def receive_echo(self, event) -> None:
        """Handle echo events from the event bus with symbolic processing."""
        event_type = event.get('type', 'unknown')
        logger.info(f"🔊 {self.sigil} {self.__class__.__name__} received echo: {event_type}")
        
        # Process with HOLO-1.5 symbolic triggers
        if event_type in self._symbolic_triggers:
            response = self.trigger_symbolic_response(event_type, event)
            logger.debug(f"🧠 Symbolic response: {response}")

    def handle_message(self, message: AsyncMessage):
        """Handle messages from the async bus with HOLO-1.5 processing."""
        logger.debug(f"📨 {self.sigil} {self.__class__.__name__} received {message.message_type.value}")
        
        # Create cognitive chain for complex messages
        if message.message_type == MessageType.USER_INTERACTION:
            chain_id = self.create_cognitive_chain(str(message.payload), "chain_of_thought")
            self.add_reasoning_step(chain_id, f"Processing {message.message_type.value}", 
                                 f"Agent {self.__class__.__name__} in role {self._mesh_role}")
        
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
                        "sigil": self.sigil
                    },
                )
            except Exception:
                pass

        return None

    def on_gui_call(self, payload=None):
        """Enhanced GUI invocation with HOLO-1.5 symbolic processing."""
        logger.info(f"🖥️ GUI invoked {self.sigil} {self.__class__.__name__} with payload={payload}")
        
        # Process through HOLO-1.5 mesh
        if payload:
            compressed_payload = self.compress_to_symbol(payload, "gui_payload")
            symbolic_response = self.trigger_symbolic_response("gui_invocation", {"payload": compressed_payload})
            logger.debug(f"🧠 Symbolic GUI response: {symbolic_response}")
        
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.USER_INTERACTION,
                self.__class__.__name__,
                {
                    "payload": payload,
                    "mesh_role": self._mesh_role,
                    "sigil": self.sigil,
                    "holo_1_5": True
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
                        "sigil": self.sigil
                    },
                )
                self.vanta_core.event_bus.emit(
                    f"{self.__class__.__name__.lower()}_invoked",
                    {"origin": self.sigil, "payload": payload, "mesh_role": self._mesh_role},
                )
                # Enhanced output to GUI panels with HOLO-1.5 info
                self.vanta_core.event_bus.emit(
                    "gui_console_output",
                    {
                        "text": f"🧠 {self.sigil} {self.__class__.__name__} [{self._mesh_role}] invoked", 
                        "payload": payload
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
                        "holo_1_5": True
                    },
                )
            except Exception:
                pass



class NullAgent(BaseAgent):
    """Fallback agent used when a real agent fails to load."""
    
    sigil = "🚫❌⬜⚪"
    
    def __init__(self, vanta_core=None):
        super().__init__(vanta_core)
        self._mesh_role = CognitiveMeshRole.EVALUATOR  # Safe default role
        
    def trigger_symbolic_response(self, trigger: str, context: Dict[str, Any] = None) -> str:
        """NullAgent provides safe fallback responses."""
        return f"🚫 NullAgent fallback for {trigger}"


# HOLO-1.5 Auto-Registration Functions

async def register_all_agents_auto():
    """Auto-register all decorated agents in the system."""
    if not _VANTA_INSTANCE:
        logger.warning("⚠️ No Vanta instance available for auto-registration")
        return
    
    registered_count = 0
    failed_count = 0
    
    logger.info(f"🤖 Starting HOLO-1.5 auto-registration of {len(_AGENT_REGISTRY)} agents...")
    
    for agent_name, agent_class in _AGENT_REGISTRY.items():
        try:
            # Create instance which will auto-register via decorator
            agent_instance = agent_class(_VANTA_INSTANCE)
            registered_count += 1
            logger.info(f"✅ Auto-registered {agent_name} [{agent_instance._mesh_role}]")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Failed to auto-register {agent_name}: {e}")
    
    logger.info(f"🎯 Auto-registration complete: {registered_count} success, {failed_count} failed")
    return {"registered": registered_count, "failed": failed_count}


def create_holo_mesh_network(agents: List[BaseAgent]) -> Dict[str, List[BaseAgent]]:
    """Create HOLO-1.5 mesh network organized by cognitive roles."""
    mesh_network = {
        CognitiveMeshRole.PLANNER: [],
        CognitiveMeshRole.GENERATOR: [],
        CognitiveMeshRole.CRITIC: [],
        CognitiveMeshRole.EVALUATOR: []
    }
    
    for agent in agents:
        role = agent._mesh_role or CognitiveMeshRole.GENERATOR
        mesh_network[role].append(agent)
    
    logger.info(f"🌐 Created HOLO-1.5 mesh network with {len(agents)} agents:")
    for role, role_agents in mesh_network.items():
        logger.info(f"  {role}: {[a.__class__.__name__ for a in role_agents]}")
    
    return mesh_network


def execute_mesh_task(mesh_network: Dict[str, List[BaseAgent]], task: str) -> Dict[str, Any]:
    """Execute a task through the HOLO-1.5 mesh network."""
    logger.info(f"🚀 Executing mesh task: {task}")
    
    result = {
        'task': task,
        'mesh_flow': [],
        'final_output': None,
        'participants': {}
    }
    
    current_context = {'task': task}
    
    # Execute through mesh roles in order
    for role in [CognitiveMeshRole.PLANNER, CognitiveMeshRole.GENERATOR, 
                 CognitiveMeshRole.CRITIC, CognitiveMeshRole.EVALUATOR]:
        
        if role in mesh_network and mesh_network[role]:
            agent = mesh_network[role][0]  # Use first agent of each role
            response = agent.trigger_symbolic_response(f'mesh_{role}', current_context)
            
            result['mesh_flow'].append((role, response))
            result['participants'][role] = agent.__class__.__name__
            current_context[role] = response
            
            logger.debug(f"  {role} ({agent.__class__.__name__}): {response}")
    
    result['final_output'] = current_context.get(CognitiveMeshRole.EVALUATOR, 
                                               current_context.get(CognitiveMeshRole.GENERATOR))
    
    logger.info(f"✅ Mesh task completed: {result['final_output']}")
    return result
