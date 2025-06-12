"""
Ensemble Integration for Novel LLM Paradigms

This module implements the orchestration and integration systems that
coordinate all novel reasoning components into a cohesive ensemble
for ARC-like task solving:

- ARC Ensemble Orchestrator: Main ensemble coordinator
- Agent Contracts: Standardized agent interface definitions
- Pipeline Stages: Multi-stage processing pipeline management
- Result Fusion: Multi-agent result combination and consensus

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration.
"""

from .arc_ensemble_orchestrator import (
    ARCEnsembleOrchestrator, AgentContract, SPLREncoderAgent, AKOrNBinderAgent,
    LNUReasonerAgent, ConsensusBuilder, ProcessingStage, EnsembleStrategy,
    ProcessingResult, EnsembleState, create_arc_ensemble
)

__all__ = [
    "ARCEnsembleOrchestrator",
    "AgentContract",
    "SPLREncoderAgent",
    "AKOrNBinderAgent", 
    "LNUReasonerAgent",
    "ConsensusBuilder",
    "ProcessingStage",
    "EnsembleStrategy",
    "ProcessingResult",
    "EnsembleState",
    "create_arc_ensemble"
]

# Version and compatibility info
__version__ = "1.0.0"
__holo_compatible__ = "1.5.0"
__paradigms__ = [
    "ensemble_orchestration",
    "multi_agent_coordination",
    "pipeline_management",
    "result_fusion",
    "consensus_building",
    "dynamic_resource_allocation",
    "meta_cognitive_orchestration"
]
