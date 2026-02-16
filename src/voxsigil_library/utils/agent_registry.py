"""
Multi-Agent Registry & Orchestration

Manages agent relationships, dependency graphs, and consensus mechanisms
for scaling beyond single-agent workflows.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# MODELS
# ============================================================================


class AgentTier(str, Enum):
    """Trust tier for agent permissions."""
    OBSERVER = "observer"      # Can only read/emit events
    CONTRIBUTOR = "contributor"  # Can submit intents (requires approval)
    TRUSTED = "trusted"        # Intents auto-approved (1 human review)
    AUTONOMOUS = "autonomous"  # Full autonomy (post-execution audit)
    MODERATOR = "moderator"    # Can approve/deny other agent intents


class AgentRole(str, Enum):
    """Functional role in the system."""
    FORECASTER = "forecaster"
    ORACLE = "oracle"
    AUDITOR = "auditor"
    MODERATOR = "moderator"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"


class ConsensusStrategy(str, Enum):
    """How consensus is reached for critical decisions."""
    SIMPLE_MAJORITY = "simple_majority"        # >50% of voters
    SUPERMAJORITY = "supermajority"            # >=66% of voters
    UNANIMOUS = "unanimous"                    # 100% agreement
    WEIGHTED = "weighted"                      # Weight by reputation
    QUORUM_WITH_VETO = "quorum_with_veto"     # Requires quorum + no vetoes


class RegisteredAgent(BaseModel):
    """Agent registration in the multi-agent system."""
    agent_id: str
    agent_name: str
    agent_type: str
    sigil_public_key: str
    
    # Classification
    tier: AgentTier = AgentTier.OBSERVER
    roles: List[AgentRole] = Field(default_factory=list)
    
    # Capabilities
    can_ingest: bool = True
    can_control: bool = False
    can_approve_intents: bool = False
    can_veto: bool = False
    
    # Trust metrics
    reputation_score: float = 0.0  # 0.0 - 1.0
    events_emitted: int = 0
    intents_submitted: int = 0
    intents_approved: int = 0
    intents_executed_successfully: int = 0
    intents_failed: int = 0
    
    # Relationships
    trusts: List[str] = Field(default_factory=list)  # Agent IDs this agent trusts
    trusted_by: List[str] = Field(default_factory=list)  # Agent IDs that trust this
    delegates_to: Optional[str] = None  # Agent ID this delegates decisions to
    
    # Metadata
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: Optional[datetime] = None
    is_active: bool = True
    
    # Tags for grouping
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRelationship(BaseModel):
    """Relationship between two agents."""
    relationship_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent_id: str
    to_agent_id: str
    relationship_type: str  # "trusts", "delegates", "monitors", etc.
    strength: float = 1.0  # 0.0 - 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentDependencyGraph(BaseModel):
    """Dependency graph for intent execution ordering."""
    graph_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Nodes are intent IDs
    nodes: Set[str] = Field(default_factory=set)
    
    # Edges are dependencies (from_intent -> depends_on -> to_intent)
    edges: List[tuple[str, str]] = Field(default_factory=list)
    
    def add_intent(self, intent_id: str, depends_on: List[str] = None):
        """Add intent to graph with dependencies."""
        self.nodes.add(intent_id)
        if depends_on:
            for dep in depends_on:
                self.edges.append((intent_id, dep))
    
    def get_dependencies(self, intent_id: str) -> List[str]:
        """Get all intents this intent depends on."""
        return [to_id for from_id, to_id in self.edges if from_id == intent_id]
    
    def get_dependents(self, intent_id: str) -> List[str]:
        """Get all intents that depend on this intent."""
        return [from_id for from_id, to_id in self.edges if to_id == intent_id]
    
    def is_ready(self, intent_id: str, executed: Set[str]) -> bool:
        """Check if intent's dependencies are all executed."""
        deps = self.get_dependencies(intent_id)
        return all(dep in executed for dep in deps)
    
    def topological_sort(self) -> List[str]:
        """Return intents in execution order (DAG)."""
        # Simple topological sort using Kahn's algorithm
        in_degree = {node: 0 for node in self.nodes}
        for from_id, to_id in self.edges:
            in_degree[from_id] += 1
        
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for from_id, to_id in self.edges:
                if to_id == node:
                    in_degree[from_id] -= 1
                    if in_degree[from_id] == 0:
                        queue.append(from_id)
        
        if len(result) != len(self.nodes):
            raise ValueError("Circular dependency detected in intent graph")
        
        return result


class ConsensusRound(BaseModel):
    """Consensus voting round for critical decision."""
    round_id: str = Field(default_factory=lambda: str(uuid4()))
    proposal_id: str  # Intent ID or other proposal
    strategy: ConsensusStrategy
    
    # Voting pool
    eligible_voters: List[str]  # Agent IDs
    votes_for: List[str] = Field(default_factory=list)
    votes_against: List[str] = Field(default_factory=list)
    vetoes: List[str] = Field(default_factory=list)
    
    # Weighting (for weighted strategies)
    voter_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Status
    is_complete: bool = False
    result: Optional[bool] = None  # True = approved, False = denied
    decided_at: Optional[datetime] = None
    
    # Thresholds
    quorum_threshold: float = 0.5  # % of eligible voters required
    approval_threshold: float = 0.66  # For supermajority
    
    def cast_vote(self, agent_id: str, vote: bool):
        """Cast vote from agent."""
        if agent_id not in self.eligible_voters:
            raise ValueError(f"Agent {agent_id} not eligible to vote")
        
        if vote:
            if agent_id not in self.votes_for:
                self.votes_for.append(agent_id)
        else:
            if agent_id not in self.votes_against:
                self.votes_against.append(agent_id)
        
        self._check_completion()
    
    def cast_veto(self, agent_id: str):
        """Cast veto (if agent has veto power)."""
        if agent_id not in self.vetoes:
            self.vetoes.append(agent_id)
        
        if self.strategy == ConsensusStrategy.QUORUM_WITH_VETO and self.vetoes:
            self.is_complete = True
            self.result = False
            self.decided_at = datetime.now(timezone.utc)
    
    def _check_completion(self):
        """Check if consensus reached."""
        total_votes = len(self.votes_for) + len(self.votes_against)
        quorum = len(self.eligible_voters) * self.quorum_threshold
        
        # Check if quorum met
        if total_votes < quorum:
            return
        
        if self.strategy == ConsensusStrategy.SIMPLE_MAJORITY:
            if len(self.votes_for) > len(self.votes_against):
                self.result = True
            else:
                self.result = False
            self.is_complete = True
        
        elif self.strategy == ConsensusStrategy.SUPERMAJORITY:
            approval_ratio = len(self.votes_for) / total_votes
            if approval_ratio >= self.approval_threshold:
                self.result = True
            elif total_votes == len(self.eligible_voters):
                self.result = False
            else:
                return  # Wait for more votes
            self.is_complete = True
        
        elif self.strategy == ConsensusStrategy.UNANIMOUS:
            if total_votes == len(self.eligible_voters):
                self.result = len(self.votes_against) == 0
                self.is_complete = True
        
        elif self.strategy == ConsensusStrategy.WEIGHTED:
            total_weight = sum(self.voter_weights.get(v, 1.0) for v in self.votes_for + self.votes_against)
            for_weight = sum(self.voter_weights.get(v, 1.0) for v in self.votes_for)
            
            if for_weight / total_weight > 0.5:
                self.result = True
                self.is_complete = True
            elif total_votes == len(self.eligible_voters):
                self.result = False
                self.is_complete = True
        
        if self.is_complete:
            self.decided_at = datetime.now(timezone.utc)


# ============================================================================
# REGISTRY
# ============================================================================


class AgentRegistry:
    """Multi-agent registry and orchestration."""
    
    def __init__(self):
        self.agents: Dict[str, RegisteredAgent] = {}
        self.relationships: Dict[str, AgentRelationship] = {}
        self.dependency_graphs: Dict[str, IntentDependencyGraph] = {}
        self.consensus_rounds: Dict[str, ConsensusRound] = {}
    
    def register_agent(self, agent: RegisteredAgent) -> RegisteredAgent:
        """Register new agent in the system."""
        self.agents[agent.agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(
        self,
        tier: Optional[AgentTier] = None,
        role: Optional[AgentRole] = None,
        is_active: bool = True
    ) -> List[RegisteredAgent]:
        """List agents with filters."""
        results = []
        for agent in self.agents.values():
            if tier and agent.tier != tier:
                continue
            if role and role not in agent.roles:
                continue
            if agent.is_active != is_active:
                continue
            results.append(agent)
        return results
    
    def upgrade_agent_tier(self, agent_id: str, new_tier: AgentTier):
        """Upgrade agent to higher trust tier."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.tier = new_tier
            
            # Update permissions based on tier
            if new_tier in [AgentTier.TRUSTED, AgentTier.AUTONOMOUS]:
                agent.can_control = True
            if new_tier in [AgentTier.AUTONOMOUS, AgentTier.MODERATOR]:
                agent.can_approve_intents = True
            if new_tier == AgentTier.MODERATOR:
                agent.can_veto = True
    
    def create_relationship(self, from_agent_id: str, to_agent_id: str, rel_type: str):
        """Create relationship between agents."""
        rel = AgentRelationship(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            relationship_type=rel_type
        )
        self.relationships[rel.relationship_id] = rel
        
        # Update agent trust lists
        if rel_type == "trusts":
            from_agent = self.get_agent(from_agent_id)
            to_agent = self.get_agent(to_agent_id)
            if from_agent and to_agent:
                from_agent.trusts.append(to_agent_id)
                to_agent.trusted_by.append(from_agent_id)
        
        return rel
    
    def get_trusted_agents(self, agent_id: str) -> List[RegisteredAgent]:
        """Get all agents this agent trusts."""
        agent = self.get_agent(agent_id)
        if not agent:
            return []
        return [self.get_agent(aid) for aid in agent.trusts if self.get_agent(aid)]
    
    def create_consensus_round(
        self,
        proposal_id: str,
        strategy: ConsensusStrategy,
        eligible_voters: List[str]
    ) -> ConsensusRound:
        """Start consensus round for proposal."""
        round_obj = ConsensusRound(
            proposal_id=proposal_id,
            strategy=strategy,
            eligible_voters=eligible_voters
        )
        self.consensus_rounds[round_obj.round_id] = round_obj
        return round_obj
    
    def get_consensus_round(self, round_id: str) -> Optional[ConsensusRound]:
        """Get consensus round by ID."""
        return self.consensus_rounds.get(round_id)


# Global registry
registry = AgentRegistry()


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("Multi-Agent Registry Example\n")
    
    # 1. Register agents
    forecaster = RegisteredAgent(
        agent_id="agent-001",
        agent_name="btc-forecaster",
        agent_type="llm",
        sigil_public_key="0x123...",
        tier=AgentTier.CONTRIBUTOR,
        roles=[AgentRole.FORECASTER],
        reputation_score=0.85
    )
    registry.register_agent(forecaster)
    print(f"1. Registered {forecaster.agent_name} (tier: {forecaster.tier.value})")
    
    oracle = RegisteredAgent(
        agent_id="agent-002",
        agent_name="price-oracle",
        agent_type="oracle",
        sigil_public_key="0x456...",
        tier=AgentTier.TRUSTED,
        roles=[AgentRole.ORACLE],
        reputation_score=0.95
    )
    registry.register_agent(oracle)
    print(f"2. Registered {oracle.agent_name} (tier: {oracle.tier.value})")
    
    # 3. Create trust relationship
    rel = registry.create_relationship(forecaster.agent_id, oracle.agent_id, "trusts")
    print(f"\n3. {forecaster.agent_name} now trusts {oracle.agent_name}")
    
    # 4. Create consensus round
    consensus = registry.create_consensus_round(
        proposal_id="intent-abc",
        strategy=ConsensusStrategy.SUPERMAJORITY,
        eligible_voters=[forecaster.agent_id, oracle.agent_id]
    )
    print(f"\n4. Consensus round created: {consensus.round_id}")
    print(f"   Strategy: {consensus.strategy.value}")
    print(f"   Eligible voters: {len(consensus.eligible_voters)}")
    
    # 5. Cast votes
    consensus.cast_vote(forecaster.agent_id, vote=True)
    consensus.cast_vote(oracle.agent_id, vote=True)
    print(f"\n5. Votes cast:")
    print(f"   For: {len(consensus.votes_for)}")
    print(f"   Against: {len(consensus.votes_against)}")
    print(f"   Complete: {consensus.is_complete}")
    print(f"   Result: {consensus.result}")
    
    # 6. Dependency graph
    graph = IntentDependencyGraph()
    graph.add_intent("intent-1")
    graph.add_intent("intent-2", depends_on=["intent-1"])
    graph.add_intent("intent-3", depends_on=["intent-1"])
    graph.add_intent("intent-4", depends_on=["intent-2", "intent-3"])
    
    print(f"\n6. Dependency graph created")
    print(f"   Nodes: {len(graph.nodes)}")
    print(f"   Edges: {len(graph.edges)}")
    
    execution_order = graph.topological_sort()
    print(f"   Execution order: {execution_order}")
