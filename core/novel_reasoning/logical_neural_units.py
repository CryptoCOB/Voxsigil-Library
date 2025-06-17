"""
Logical Neural Units (LNUs) for Symbolic Reasoning

Implements differentiable symbolic logic operations within neural networks.
Addresses the "pattern matching vs. genuine reasoning" challenge by embedding
formal logic directly into neural computation.

Key Features:
- Symbolic logic operations (AND, OR, NOT, IMPLIES, FORALL, EXISTS)
- Variable binding and unification
- Compositional reasoning with rule chaining
- Differentiable logic gates for gradient-based learning
- Integration with HOLO-1.5 cognitive mesh

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

try:
    from ...agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    HOLO_AVAILABLE = False
    def vanta_agent(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        PROCESSOR = "processor"
        REASONER = "reasoner" 
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


class LogicOperation(Enum):
    """Symbolic logic operations supported by LNUs"""
    AND = "and"
    OR = "or" 
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    FORALL = "forall"
    EXISTS = "exists"
    EQUALS = "equals"
    UNIFY = "unify"


@dataclass
class LogicalState:
    """Represents a logical state with truth values and variable bindings"""
    truth_values: torch.Tensor  # [batch_size, num_propositions]
    variable_bindings: Dict[str, torch.Tensor] = field(default_factory=dict)
    confidence: torch.Tensor = None  # Uncertainty in truth values
    symbolic_depth: int = 0  # For HOLO-1.5 tracking


@dataclass 
class LogicalRule:
    """Represents a logical inference rule"""
    name: str
    premises: List[str]  # Symbolic premise patterns
    conclusion: str      # Symbolic conclusion pattern
    operation: LogicOperation
    weight: float = 1.0  # Learnable rule weight


class DifferentiableLogicGate(nn.Module):
    """
    Differentiable implementation of logic gates using smooth approximations
    
    Uses sigmoid-based approximations to maintain differentiability while
    preserving logical semantics.
    """
    
    def __init__(self, operation: LogicOperation, temperature: float = 5.0):
        super().__init__()
        self.operation = operation
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Learnable bias for each operation
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Apply logic operation to input tensors"""
        if self.operation == LogicOperation.AND:
            return self._smooth_and(*inputs)
        elif self.operation == LogicOperation.OR:
            return self._smooth_or(*inputs)
        elif self.operation == LogicOperation.NOT:
            return self._smooth_not(inputs[0])
        elif self.operation == LogicOperation.IMPLIES:
            return self._smooth_implies(inputs[0], inputs[1])
        elif self.operation == LogicOperation.IFF:
            return self._smooth_iff(inputs[0], inputs[1])
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
    
    def _smooth_and(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of logical AND"""
        # Product with sigmoid sharpening
        result = torch.ones_like(inputs[0])
        for inp in inputs:
            result = result * torch.sigmoid(self.temperature * inp)
        return torch.sigmoid(self.temperature * (result - 0.5)) + self.bias
    
    def _smooth_or(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of logical OR"""
        # 1 - product of negations
        result = torch.ones_like(inputs[0])
        for inp in inputs:
            result = result * (1 - torch.sigmoid(self.temperature * inp))
        return 1 - result + self.bias
    
    def _smooth_not(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of logical NOT"""
        return 1 - torch.sigmoid(self.temperature * x) + self.bias
    
    def _smooth_implies(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of logical IMPLIES (p -> q)"""
        # Equivalent to (NOT p) OR q
        not_p = self._smooth_not(p)
        return self._smooth_or(not_p, q)
    
    def _smooth_iff(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of logical IFF (p <-> q)"""
        # (p -> q) AND (q -> p)
        p_implies_q = self._smooth_implies(p, q)
        q_implies_p = self._smooth_implies(q, p)
        return self._smooth_and(p_implies_q, q_implies_p)


class VariableBinding(nn.Module):
    """
    Neural variable binding mechanism for symbolic reasoning
    
    Implements attention-based variable binding that can unify variables
    across different logical expressions.
    """
    
    def __init__(self, hidden_dim: int, num_variables: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_variables = num_variables
        
        # Variable embedding and binding networks
        self.variable_embeddings = nn.Parameter(torch.randn(num_variables, hidden_dim))
        self.binding_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.unification_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, propositions: torch.Tensor, 
                variable_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bind variables in logical propositions
        
        Args:
            propositions: [batch_size, seq_len, hidden_dim]
            variable_indices: [batch_size, seq_len] - indices of variables
            
        Returns:
            bound_propositions: Propositions with variable bindings
            binding_weights: Variable binding weights
        """
        batch_size, seq_len, _ = propositions.shape
        
        # Get variable embeddings for each position
        var_embeds = self.variable_embeddings[variable_indices]  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention-based binding
        bound_props, attention_weights = self.binding_attention(
            propositions, var_embeds, var_embeds
        )
        
        # Compute unification scores between variables
        binding_weights = {}
        for i in range(self.num_variables):
            for j in range(i + 1, self.num_variables):
                var_i = self.variable_embeddings[i].unsqueeze(0)
                var_j = self.variable_embeddings[j].unsqueeze(0)
                concat_vars = torch.cat([var_i, var_j], dim=-1)
                unify_score = self.unification_network(concat_vars)
                binding_weights[f"var_{i}_var_{j}"] = unify_score
        
        return bound_props, binding_weights


class LogicalNeuralUnit(nn.Module):
    """
    Core Logical Neural Unit implementing symbolic reasoning operations
    
    Combines differentiable logic gates with variable binding to perform
    symbolic reasoning within neural networks.
    """
    
    def __init__(self, hidden_dim: int, num_propositions: int, num_variables: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_propositions = num_propositions
        self.num_variables = num_variables
        
        # Core components
        self.variable_binder = VariableBinding(hidden_dim, num_variables)
        
        # Logic gates for different operations
        self.logic_gates = nn.ModuleDict({
            op.value: DifferentiableLogicGate(op) 
            for op in LogicOperation if op not in [LogicOperation.FORALL, LogicOperation.EXISTS]
        })
        
        # Quantifier networks
        self.forall_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.exists_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Proposition encoder
        self.proposition_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, logical_state: LogicalState, 
                rule: LogicalRule) -> LogicalState:
        """
        Apply logical reasoning rule to current state
        
        Args:
            logical_state: Current logical state with truth values
            rule: Logical rule to apply
            
        Returns:
            new_state: Updated logical state after applying rule
        """
        # Extract premise truth values
        premise_values = []
        for premise in rule.premises:
            # Simple premise matching (can be made more sophisticated)
            prop_idx = hash(premise) % self.num_propositions
            premise_values.append(logical_state.truth_values[:, prop_idx:prop_idx+1])
        
        # Apply logical operation
        if rule.operation in self.logic_gates:
            result = self.logic_gates[rule.operation.value](*premise_values)
        elif rule.operation == LogicOperation.FORALL:
            result = self._apply_forall(premise_values[0], logical_state)
        elif rule.operation == LogicOperation.EXISTS:
            result = self._apply_exists(premise_values[0], logical_state)
        else:
            raise ValueError(f"Unsupported rule operation: {rule.operation}")
        
        # Update truth values
        conclusion_idx = hash(rule.conclusion) % self.num_propositions
        new_truth_values = logical_state.truth_values.clone()
        new_truth_values[:, conclusion_idx:conclusion_idx+1] = result * rule.weight
        
        # Increase symbolic depth
        new_state = LogicalState(
            truth_values=new_truth_values,
            variable_bindings=logical_state.variable_bindings.copy(),
            confidence=logical_state.confidence,
            symbolic_depth=logical_state.symbolic_depth + 1
        )
        
        return new_state
    
    def _apply_forall(self, premise: torch.Tensor, state: LogicalState) -> torch.Tensor:
        """Apply universal quantifier"""
        # For universal quantification, all instances must be true
        return self.forall_network(premise) * torch.min(state.truth_values, dim=1, keepdim=True)[0]
    
    def _apply_exists(self, premise: torch.Tensor, state: LogicalState) -> torch.Tensor:
        """Apply existential quantifier"""
        # For existential quantification, at least one instance must be true
        return self.exists_network(premise) * torch.max(state.truth_values, dim=1, keepdim=True)[0]


@vanta_agent(role=CognitiveMeshRole.REASONER)
class LogicalReasoningEngine(BaseAgent):
    """
    High-level logical reasoning engine using LNUs
    
    Orchestrates multiple LNUs to perform complex symbolic reasoning
    with support for rule chaining and abductive inference.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_propositions = config.get("num_propositions", 64)
        self.num_variables = config.get("num_variables", 16)
        self.max_reasoning_steps = config.get("max_reasoning_steps", 10)
        
        # Core reasoning components
        self.lnu = LogicalNeuralUnit(
            self.hidden_dim, 
            self.num_propositions, 
            self.num_variables
        )
        
        # Rule base
        self.rules: List[LogicalRule] = []
        self._initialize_default_rules()
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "reasoning_steps": 0,
            "rule_applications": 0,
            "symbolic_depth": 0,
            "confidence_level": 0.0
        }
    
    async def async_init(self):
        """Initialize the reasoning engine"""
        if HOLO_AVAILABLE:
            await super().async_init()
        logger.info("LogicalReasoningEngine initialized with HOLO-1.5 integration")
    
    def _initialize_default_rules(self):
        """Initialize with basic logical inference rules"""
        self.rules = [
            # Modus Ponens: P, P->Q ⊢ Q
            LogicalRule("modus_ponens", ["P", "P->Q"], "Q", LogicOperation.IMPLIES),
            
            # Modus Tollens: ¬Q, P->Q ⊢ ¬P  
            LogicalRule("modus_tollens", ["¬Q", "P->Q"], "¬P", LogicOperation.NOT),
            
            # Hypothetical Syllogism: P->Q, Q->R ⊢ P->R
            LogicalRule("hypothetical_syllogism", ["P->Q", "Q->R"], "P->R", LogicOperation.IMPLIES),
            
            # Disjunctive Syllogism: P∨Q, ¬P ⊢ Q
            LogicalRule("disjunctive_syllogism", ["P∨Q", "¬P"], "Q", LogicOperation.OR),
            
            # Conjunction: P, Q ⊢ P∧Q
            LogicalRule("conjunction", ["P", "Q"], "P∧Q", LogicOperation.AND),
        ]
    
    def add_rule(self, rule: LogicalRule):
        """Add a new logical rule to the rule base"""
        self.rules.append(rule)
        logger.info(f"Added logical rule: {rule.name}")
    
    async def reason(self, initial_state: LogicalState, 
                     goal_propositions: Optional[List[str]] = None) -> LogicalState:
        """
        Perform multi-step logical reasoning
        
        Args:
            initial_state: Initial logical state
            goal_propositions: Optional goal propositions to reason towards
            
        Returns:
            final_state: Final logical state after reasoning
        """
        current_state = initial_state
        reasoning_trace = []
        
        for step in range(self.max_reasoning_steps):
            # Try to apply each rule
            best_rule = None
            best_result = None
            best_score = -float('inf')
            
            for rule in self.rules:
                try:
                    # Apply rule and score the result
                    result_state = self.lnu(current_state, rule)
                    score = self._score_state(result_state, goal_propositions)
                    
                    if score > best_score:
                        best_score = score
                        best_rule = rule
                        best_result = result_state
                        
                except Exception as e:
                    logger.debug(f"Failed to apply rule {rule.name}: {e}")
                    continue
            
            # Apply best rule if found
            if best_rule and best_result:
                current_state = best_result
                reasoning_trace.append({
                    "step": step,
                    "rule": best_rule.name,
                    "score": best_score
                })
                
                self.cognitive_metrics["rule_applications"] += 1
                self.cognitive_metrics["symbolic_depth"] = current_state.symbolic_depth
                
                # Check if goal is reached
                if goal_propositions and self._goal_reached(current_state, goal_propositions):
                    break
            else:
                # No applicable rules found
                break
        
        self.cognitive_metrics["reasoning_steps"] = len(reasoning_trace)
        self.cognitive_metrics["confidence_level"] = float(torch.mean(current_state.truth_values))
        
        logger.info(f"Reasoning completed in {len(reasoning_trace)} steps")
        return current_state
    
    def _score_state(self, state: LogicalState, goals: Optional[List[str]]) -> float:
        """Score a logical state based on goal proximity and confidence"""
        base_score = float(torch.mean(state.truth_values))
        
        if goals:
            # Bonus for achieving goal propositions
            goal_score = 0.0
            for goal in goals:
                goal_idx = hash(goal) % self.num_propositions
                goal_score += float(state.truth_values[0, goal_idx])
            base_score += goal_score / len(goals)
        
        return base_score
    
    def _goal_reached(self, state: LogicalState, goals: List[str]) -> bool:
        """Check if goal propositions are satisfied"""
        threshold = 0.8
        for goal in goals:
            goal_idx = hash(goal) % self.num_propositions
            if state.truth_values[0, goal_idx] < threshold:
                return False
        return True
    
    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with more reasoning steps and symbolic depth
        steps_load = min(self.cognitive_metrics["reasoning_steps"] / self.max_reasoning_steps, 1.0)
        depth_load = min(self.cognitive_metrics["symbolic_depth"] / 10.0, 1.0)
        confidence_load = 1.0 - self.cognitive_metrics["confidence_level"]
        
        return (steps_load * 0.4 + depth_load * 0.4 + confidence_load * 0.2)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # LNU has high symbolic depth due to formal logic operations
        return max(self.cognitive_metrics["symbolic_depth"], 5)
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "LogicalReasoningEngine",
            "cognitive_metrics": self.cognitive_metrics,
            "num_rules": len(self.rules),
            "lnu_parameters": sum(p.numel() for p in self.lnu.parameters())
        }


# Factory functions
def create_logical_state(truth_values: torch.Tensor, 
                        variables: Optional[Dict[str, torch.Tensor]] = None) -> LogicalState:
    """Create a logical state with optional variable bindings"""
    return LogicalState(
        truth_values=truth_values,
        variable_bindings=variables or {},
        confidence=torch.ones_like(truth_values)
    )


async def create_reasoning_engine(config: Dict[str, Any]) -> LogicalReasoningEngine:
    """Factory function to create and initialize LogicalReasoningEngine"""
    engine = LogicalReasoningEngine(config)
    await engine.async_init()
    return engine


# Export main classes
__all__ = [
    "LogicalNeuralUnit",
    "LogicalReasoningEngine", 
    "LogicalState",
    "LogicalRule",
    "LogicOperation",
    "DifferentiableLogicGate",
    "VariableBinding",
    "create_logical_state",
    "create_reasoning_engine"
]
