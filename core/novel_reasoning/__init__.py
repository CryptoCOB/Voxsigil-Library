"""
Novel Reasoning Components for VoxSigil Library

This module implements cutting-edge reasoning paradigms for ARC-like tasks:
- Logical Neural Units (LNUs): Symbolic reasoning within neural networks
- Kuramoto Oscillatory Neurons (AKOrN): Bio-inspired dynamics for object binding
- Spiking Neural Networks with SPLR: Event-driven grid processing
- Relation Based Patterns (RBP/ERBP): Abstract pattern learning
- Graph Neural Networks (GNNs): Native relational reasoning

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration.
"""

from .logical_neural_units import (
    LogicalNeuralUnit, LogicalReasoningEngine, LogicalState, LogicalRule,
    LogicOperation, DifferentiableLogicGate, VariableBinding,
    create_logical_state, create_reasoning_engine
)

from .kuramoto_oscillatory import (
    AKOrNBindingNetwork, KuramotoOscillator, SpatialCouplingNetwork,
    ObjectSegmentationHead, OscillatorState, BindingResult,
    compute_phase_synchrony, extract_synchrony_clusters, create_akorn_network
)

from .spiking_neural_networks import (
    SPLRSpikingNetwork, LIFNeuron, SparsePlasticityRule, SynapticLayer,
    GridToSpikeEncoder, SpikeEvent, SpikeTrain, LIFNeuronState,
    spike_train_statistics, create_splr_network
)

__all__ = [
    # Logical Neural Units
    "LogicalNeuralUnit",
    "LogicalReasoningEngine", 
    "LogicalState",
    "LogicalRule",
    "LogicOperation",
    "DifferentiableLogicGate",
    "VariableBinding",
    "create_logical_state",
    "create_reasoning_engine",
    
    # Kuramoto Oscillatory Networks
    "AKOrNBindingNetwork",
    "KuramotoOscillator", 
    "SpatialCouplingNetwork",
    "ObjectSegmentationHead",
    "OscillatorState",
    "BindingResult",
    "compute_phase_synchrony",
    "extract_synchrony_clusters",
    "create_akorn_network",
    
    # Spiking Neural Networks
    "SPLRSpikingNetwork",
    "LIFNeuron",
    "SparsePlasticityRule", 
    "SynapticLayer",
    "GridToSpikeEncoder",
    "SpikeEvent",
    "SpikeTrain",
    "LIFNeuronState",
    "spike_train_statistics",
    "create_splr_network"
]

# Version and compatibility info
__version__ = "1.0.0"
__holo_compatible__ = "1.5.0"
__paradigms__ = [
    "symbolic_reasoning",
    "bio_inspired_dynamics", 
    "event_driven_processing",
    "relational_reasoning",
    "abstract_pattern_learning",
    "oscillatory_binding",
    "spike_timing_plasticity"
]
