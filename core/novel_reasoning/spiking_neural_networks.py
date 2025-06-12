"""
Spiking Neural Networks with Sparse Plasticity Learning Rule (SPLR)

Implements event-driven spiking neural networks for efficient processing
of ARC grid patterns. Uses bio-inspired spike timing dependent plasticity
and sparse plasticity rules for learning temporal patterns.

Key Features:
- Leaky Integrate-and-Fire (LIF) neurons with adaptive thresholds
- Sparse Plasticity Learning Rule (SPLR) for efficient learning
- Event-driven computation for grid pattern processing
- Temporal spike pattern encoding and decoding
- Integration with HOLO-1.5 cognitive mesh

Based on neuromorphic computing principles and spike-based learning.

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from enum import Enum
import numpy as np
import logging
import math

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
        ENCODER = "encoder"
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


@dataclass
class SpikeEvent:
    """Represents a spike event in the network"""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeTrain:
    """Collection of spike events for a neuron or population"""
    events: List[SpikeEvent]
    duration: float
    neuron_ids: Optional[List[int]] = None
    
    def get_spike_times(self, neuron_id: int) -> List[float]:
        """Get spike times for a specific neuron"""
        return [event.timestamp for event in self.events if event.neuron_id == neuron_id]
    
    def get_firing_rate(self, neuron_id: int) -> float:
        """Compute average firing rate for a neuron"""
        spike_count = sum(1 for event in self.events if event.neuron_id == neuron_id)
        return spike_count / self.duration if self.duration > 0 else 0.0


@dataclass
class LIFNeuronState:
    """State of a Leaky Integrate-and-Fire neuron"""
    membrane_potential: torch.Tensor  # [batch_size, num_neurons]
    spike_output: torch.Tensor        # [batch_size, num_neurons] 
    threshold: torch.Tensor           # [batch_size, num_neurons]
    refractory_time: torch.Tensor     # [batch_size, num_neurons]
    adaptation_current: torch.Tensor  # [batch_size, num_neurons]


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with adaptive threshold
    
    Implements the LIF dynamics:
    τ_m * dV/dt = -(V - V_rest) + R * I_input - R * I_adapt
    τ_adapt * dI_adapt/dt = -I_adapt + b * Σ_spikes
    """
    
    def __init__(self, num_neurons: int, dt: float = 0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        
        # Neuron parameters (learnable)
        self.tau_membrane = nn.Parameter(torch.full((num_neurons,), 20.0))  # Membrane time constant
        self.tau_adaptation = nn.Parameter(torch.full((num_neurons,), 100.0))  # Adaptation time constant
        self.resistance = nn.Parameter(torch.full((num_neurons,), 1.0))     # Membrane resistance
        self.rest_potential = nn.Parameter(torch.full((num_neurons,), -70.0))  # Resting potential
        self.spike_threshold = nn.Parameter(torch.full((num_neurons,), -50.0))  # Spike threshold
        self.reset_potential = nn.Parameter(torch.full((num_neurons,), -80.0))  # Reset potential
        self.adaptation_strength = nn.Parameter(torch.full((num_neurons,), 0.1))  # Adaptation strength
        self.refractory_period = nn.Parameter(torch.full((num_neurons,), 2.0))   # Refractory period
        
        # Initialize state
        self.register_buffer('state_initialized', torch.tensor(False))
    
    def init_state(self, batch_size: int) -> LIFNeuronState:
        """Initialize neuron state"""
        device = self.tau_membrane.device
        
        state = LIFNeuronState(
            membrane_potential=self.rest_potential.unsqueeze(0).expand(batch_size, -1).clone(),
            spike_output=torch.zeros(batch_size, self.num_neurons, device=device),
            threshold=self.spike_threshold.unsqueeze(0).expand(batch_size, -1).clone(),
            refractory_time=torch.zeros(batch_size, self.num_neurons, device=device),
            adaptation_current=torch.zeros(batch_size, self.num_neurons, device=device)
        )
        
        return state
    
    def forward(self, input_current: torch.Tensor, state: LIFNeuronState) -> LIFNeuronState:
        """
        Forward pass through LIF neurons
        
        Args:
            input_current: [batch_size, num_neurons] - Input current
            state: Current neuron state
            
        Returns:
            new_state: Updated neuron state after integration
        """
        batch_size = input_current.shape[0]
        
        # Update membrane potential (only for non-refractory neurons)
        non_refractory = state.refractory_time <= 0
        
        # Membrane dynamics
        membrane_decay = torch.exp(-self.dt / self.tau_membrane)
        adaptation_decay = torch.exp(-self.dt / self.tau_adaptation)
        
        # Input current including adaptation
        total_current = input_current - state.adaptation_current
        
        # Update membrane potential
        new_potential = (
            state.membrane_potential * membrane_decay + 
            self.resistance * total_current * (1 - membrane_decay)
        )
        
        # Apply only to non-refractory neurons
        new_potential = torch.where(
            non_refractory,
            new_potential,
            state.membrane_potential
        )
        
        # Check for spikes
        spikes = (new_potential >= state.threshold) & non_refractory
        
        # Reset spiking neurons
        new_potential = torch.where(
            spikes,
            self.reset_potential.unsqueeze(0).expand_as(new_potential),
            new_potential
        )
        
        # Update refractory time
        new_refractory = torch.where(
            spikes,
            self.refractory_period.unsqueeze(0).expand_as(state.refractory_time),
            torch.clamp(state.refractory_time - self.dt, min=0)
        )
        
        # Update adaptation current
        spike_contribution = spikes.float() * self.adaptation_strength.unsqueeze(0)
        new_adaptation = state.adaptation_current * adaptation_decay + spike_contribution
        
        # Create new state
        new_state = LIFNeuronState(
            membrane_potential=new_potential,
            spike_output=spikes.float(),
            threshold=state.threshold,
            refractory_time=new_refractory,
            adaptation_current=new_adaptation
        )
        
        return new_state


class SparsePlasticityRule(nn.Module):
    """
    Sparse Plasticity Learning Rule (SPLR) for efficient synaptic learning
    
    Implements sparse, spike-timing dependent plasticity that only updates
    a subset of synapses based on recent activity patterns.
    """
    
    def __init__(self, num_pre: int, num_post: int, sparsity_factor: float = 0.1):
        super().__init__()
        self.num_pre = num_pre
        self.num_post = num_post
        self.sparsity_factor = sparsity_factor
        
        # STDP parameters
        self.a_plus = nn.Parameter(torch.tensor(0.01))  # LTP amplitude
        self.a_minus = nn.Parameter(torch.tensor(-0.005))  # LTD amplitude
        self.tau_plus = nn.Parameter(torch.tensor(20.0))  # LTP time constant
        self.tau_minus = nn.Parameter(torch.tensor(20.0))  # LTD time constant
        
        # Eligibility trace parameters
        self.tau_eligibility = nn.Parameter(torch.tensor(50.0))  # Eligibility trace time constant
        
        # Sparsity parameters
        self.activity_threshold = nn.Parameter(torch.tensor(0.1))  # Minimum activity for plasticity
        self.max_weight_change = nn.Parameter(torch.tensor(0.001))  # Maximum weight change per step
    
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                weights: torch.Tensor, eligibility_trace: torch.Tensor,
                dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparse plasticity rule
        
        Args:
            pre_spikes: [batch_size, num_pre] - Presynaptic spikes
            post_spikes: [batch_size, num_post] - Postsynaptic spikes  
            weights: [num_pre, num_post] - Current synaptic weights
            eligibility_trace: [num_pre, num_post] - Eligibility traces
            dt: Time step
            
        Returns:
            weight_updates: Weight changes to apply
            new_eligibility: Updated eligibility traces
        """
        batch_size = pre_spikes.shape[0]
        
        # Update eligibility traces
        trace_decay = torch.exp(-dt / self.tau_eligibility)
        
        # STDP contribution to eligibility
        pre_expanded = pre_spikes.unsqueeze(-1)  # [batch, num_pre, 1]
        post_expanded = post_spikes.unsqueeze(-2)  # [batch, 1, num_post]
        
        # LTP component (post spike after pre spike)
        ltp_contribution = pre_expanded * post_expanded * self.a_plus
        
        # LTD component (pre spike after post spike) 
        ltd_contribution = pre_expanded * post_expanded * self.a_minus
        
        # Combined STDP signal
        stdp_signal = ltp_contribution + ltd_contribution
        
        # Update eligibility trace
        batch_trace_update = torch.mean(stdp_signal, dim=0)  # Average over batch
        new_eligibility = eligibility_trace * trace_decay + batch_trace_update
        
        # Sparse plasticity: only update highly active synapses
        activity_level = torch.abs(new_eligibility)
        sparse_mask = activity_level > self.activity_threshold
        
        # Compute weight updates
        weight_updates = torch.zeros_like(weights)
        weight_updates = torch.where(
            sparse_mask,
            torch.clamp(new_eligibility, -self.max_weight_change, self.max_weight_change),
            weight_updates
        )
        
        # Apply sparsity factor
        num_updates = int(sparse_mask.sum())
        target_updates = int(self.sparsity_factor * weights.numel())
        
        if num_updates > target_updates:
            # Keep only top-k most active synapses
            flat_activity = activity_level.view(-1)
            _, top_indices = torch.topk(flat_activity, target_updates)
            
            sparse_mask_flat = torch.zeros_like(flat_activity, dtype=torch.bool)
            sparse_mask_flat[top_indices] = True
            sparse_mask = sparse_mask_flat.view_as(weights)
            
            weight_updates = torch.where(sparse_mask, weight_updates, torch.zeros_like(weight_updates))
        
        return weight_updates, new_eligibility


class SynapticLayer(nn.Module):
    """
    Synaptic layer connecting spiking neuron populations
    
    Implements sparse synaptic connectivity with SPLR learning.
    """
    
    def __init__(self, num_pre: int, num_post: int, connectivity: float = 0.1):
        super().__init__()
        self.num_pre = num_pre
        self.num_post = num_post
        self.connectivity = connectivity
        
        # Initialize sparse connectivity
        num_connections = int(connectivity * num_pre * num_post)
        self.register_buffer('connection_mask', self._create_sparse_mask(num_connections))
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(num_pre, num_post) * 0.1)
        
        # Plasticity rule
        self.plasticity = SparsePlasticityRule(num_pre, num_post)
        
        # Eligibility traces
        self.register_buffer('eligibility_trace', torch.zeros(num_pre, num_post))
        
        # Synaptic delays (in time steps)
        self.register_buffer('delays', torch.randint(1, 5, (num_pre, num_post)))
        
    def _create_sparse_mask(self, num_connections: int) -> torch.Tensor:
        """Create sparse connectivity mask"""
        mask = torch.zeros(self.num_pre, self.num_post)
        indices = torch.randperm(self.num_pre * self.num_post)[:num_connections]
        
        flat_mask = mask.view(-1)
        flat_mask[indices] = 1.0
        
        return mask
    
    def forward(self, pre_spikes: torch.Tensor, learn: bool = True) -> torch.Tensor:
        """
        Forward pass through synaptic layer
        
        Args:
            pre_spikes: [batch_size, num_pre] - Presynaptic spikes
            learn: Whether to apply plasticity
            
        Returns:
            post_current: [batch_size, num_post] - Postsynaptic current
        """
        # Apply connectivity mask to weights
        effective_weights = self.weights * self.connection_mask
        
        # Compute synaptic current
        post_current = torch.matmul(pre_spikes, effective_weights)
        
        return post_current
    
    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, dt: float = 0.1):
        """Update synaptic weights using SPLR"""
        weight_updates, new_eligibility = self.plasticity(
            pre_spikes, post_spikes, self.weights, self.eligibility_trace, dt
        )
        
        # Apply updates only to connected synapses
        masked_updates = weight_updates * self.connection_mask
        self.weights.data += masked_updates
        self.eligibility_trace.data = new_eligibility


class GridToSpikeEncoder(nn.Module):
    """
    Encoder that converts ARC grid patterns to spike trains
    
    Uses spatial-temporal encoding to represent grid patterns
    as sequences of spike events.
    """
    
    def __init__(self, grid_size: int, num_neurons_per_cell: int = 4, 
                 encoding_duration: float = 50.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_neurons_per_cell = num_neurons_per_cell
        self.encoding_duration = encoding_duration
        self.total_neurons = grid_size * grid_size * num_neurons_per_cell
        
        # Color encoding neurons (one per color value)
        self.max_colors = 10  # Typical ARC has 10 colors
        self.color_encoders = nn.ModuleList([
            nn.Linear(1, num_neurons_per_cell) for _ in range(self.max_colors)
        ])
        
        # Spatial encoding
        self.spatial_encoder = nn.Conv2d(1, num_neurons_per_cell, kernel_size=3, padding=1)
        
    def forward(self, grid: torch.Tensor) -> SpikeTrain:
        """
        Convert grid to spike train
        
        Args:
            grid: [batch_size, height, width] - ARC grid
            
        Returns:
            spike_train: Encoded spike patterns
        """
        batch_size, height, width = grid.shape
        
        events = []
        
        # Encode each cell of the grid
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    color_value = int(grid[b, h, w].item())
                    base_neuron_id = (h * width + w) * self.num_neurons_per_cell
                    
                    # Generate spikes based on color value
                    if color_value > 0 and color_value < self.max_colors:
                        # Different colors spike at different times and patterns
                        spike_times = self._generate_color_pattern(color_value, self.encoding_duration)
                        
                        for i, spike_time in enumerate(spike_times):
                            if i < self.num_neurons_per_cell:
                                events.append(SpikeEvent(
                                    neuron_id=base_neuron_id + i,
                                    timestamp=spike_time,
                                    metadata={'batch': b, 'position': (h, w), 'color': color_value}
                                ))
        
        return SpikeTrain(events=events, duration=self.encoding_duration)
    
    def _generate_color_pattern(self, color_value: int, duration: float) -> List[float]:
        """Generate spike pattern for a specific color"""
        # Different colors have different temporal patterns
        patterns = {
            1: [5.0, 15.0, 25.0, 35.0],      # Regular pattern
            2: [3.0, 7.0, 11.0, 15.0],       # Fast pattern  
            3: [10.0, 20.0, 30.0, 40.0],     # Slow pattern
            4: [2.0, 8.0, 18.0, 32.0],       # Irregular pattern
            5: [5.0, 10.0, 20.0, 45.0],      # Burst pattern
            # Add more patterns for other colors
        }
        
        base_pattern = patterns.get(color_value, [10.0, 20.0, 30.0, 40.0])
        
        # Add some noise for variability
        noisy_pattern = [t + np.random.normal(0, 1.0) for t in base_pattern]
        
        # Keep within duration bounds
        return [max(0, min(t, duration - 1)) for t in noisy_pattern]


@vanta_agent(role=CognitiveMeshRole.ENCODER)
class SPLRSpikingNetwork(BaseAgent):
    """
    Main Spiking Neural Network with Sparse Plasticity Learning Rule
    
    Orchestrates multiple LIF neuron populations with SPLR learning
    for processing ARC grid patterns in an event-driven manner.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Network parameters
        self.grid_size = config.get("grid_size", 30)
        self.num_layers = config.get("num_layers", 3)
        self.neurons_per_layer = config.get("neurons_per_layer", [400, 200, 100])
        self.dt = config.get("dt", 0.1)
        self.simulation_time = config.get("simulation_time", 100.0)
        
        # Input encoding
        self.grid_encoder = GridToSpikeEncoder(self.grid_size)
        
        # Network layers
        self.neuron_layers = nn.ModuleList([
            LIFNeuron(self.neurons_per_layer[i]) for i in range(self.num_layers)
        ])
        
        # Synaptic connections
        layer_sizes = [self.grid_encoder.total_neurons] + self.neurons_per_layer
        self.synaptic_layers = nn.ModuleList([
            SynapticLayer(layer_sizes[i], layer_sizes[i+1])
            for i in range(self.num_layers)
        ])
        
        # Output decoder
        self.output_decoder = nn.Linear(self.neurons_per_layer[-1], self.grid_size * self.grid_size)
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "spike_rate": 0.0,
            "network_synchrony": 0.0,
            "plasticity_updates": 0,
            "encoding_efficiency": 0.0
        }
        
    async def async_init(self):
        """Initialize the spiking network"""
        if HOLO_AVAILABLE:
            await super().async_init()
        logger.info("SPLR Spiking Network initialized with HOLO-1.5 integration")
    
    def forward(self, grid_input: torch.Tensor, learn: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through spiking network
        
        Args:
            grid_input: [batch_size, height, width] - Input ARC grid
            learn: Whether to apply plasticity learning
            
        Returns:
            output: Processed grid representation
            network_state: Network state information
        """
        batch_size = grid_input.shape[0]
        
        # Encode grid to spikes
        spike_train = self.grid_encoder(grid_input)
        
        # Initialize neuron states
        neuron_states = [layer.init_state(batch_size) for layer in self.neuron_layers]
        
        # Simulation variables
        num_steps = int(self.simulation_time / self.dt)
        all_spikes = [[] for _ in range(self.num_layers)]
        input_spikes = self._spike_train_to_tensor(spike_train, batch_size, num_steps)
        
        # Run simulation
        for step in range(num_steps):
            current_time = step * self.dt
            
            # Get input spikes for this time step
            if step < input_spikes.shape[1]:
                layer_input = input_spikes[:, step]  # [batch_size, num_input_neurons]
            else:
                layer_input = torch.zeros(batch_size, self.grid_encoder.total_neurons)
            
            # Propagate through layers
            for layer_idx, (neuron_layer, synaptic_layer) in enumerate(zip(self.neuron_layers, self.synaptic_layers)):
                # Compute synaptic current
                synaptic_current = synaptic_layer(layer_input, learn=learn)
                
                # Update neurons
                neuron_states[layer_idx] = neuron_layer(synaptic_current, neuron_states[layer_idx])
                
                # Store spikes
                all_spikes[layer_idx].append(neuron_states[layer_idx].spike_output.clone())
                
                # Apply plasticity if learning
                if learn and layer_idx > 0:
                    prev_spikes = all_spikes[layer_idx-1][-1] if all_spikes[layer_idx-1] else layer_input
                    current_spikes = neuron_states[layer_idx].spike_output
                    synaptic_layer.update_plasticity(prev_spikes, current_spikes, self.dt)
                    self.cognitive_metrics["plasticity_updates"] += 1
                
                # Set input for next layer
                layer_input = neuron_states[layer_idx].spike_output
        
        # Decode final layer activity to output
        final_activity = torch.stack(all_spikes[-1], dim=1)  # [batch, time, neurons]
        mean_activity = torch.mean(final_activity, dim=1)    # [batch, neurons]
        output = self.output_decoder(mean_activity)          # [batch, grid_size^2]
        output = output.view(batch_size, self.grid_size, self.grid_size)
        
        # Update cognitive metrics
        total_spikes = sum(torch.sum(torch.stack(layer_spikes)) for layer_spikes in all_spikes)
        self.cognitive_metrics["spike_rate"] = float(total_spikes) / (num_steps * sum(self.neurons_per_layer))
        self.cognitive_metrics["network_synchrony"] = self._compute_synchrony(all_spikes)
        self.cognitive_metrics["encoding_efficiency"] = len(spike_train.events) / (self.grid_size * self.grid_size)
        
        # Prepare network state
        network_state = {
            "spike_trains": all_spikes,
            "neuron_states": neuron_states,
            "simulation_steps": num_steps,
            "total_spikes": int(total_spikes)
        }
        
        return output, network_state
    
    def _spike_train_to_tensor(self, spike_train: SpikeTrain, batch_size: int, num_steps: int) -> torch.Tensor:
        """Convert spike train to tensor representation"""
        tensor = torch.zeros(batch_size, num_steps, self.grid_encoder.total_neurons)
        
        for event in spike_train.events:
            time_step = int(event.timestamp / self.dt)
            if 0 <= time_step < num_steps:
                batch_idx = event.metadata.get('batch', 0)
                if batch_idx < batch_size:
                    tensor[batch_idx, time_step, event.neuron_id] = event.amplitude
        
        return tensor
    
    def _compute_synchrony(self, all_spikes: List[List[torch.Tensor]]) -> float:
        """Compute network synchronization measure"""
        if not all_spikes or not all_spikes[0]:
            return 0.0
        
        # Simple synchrony measure: variance of spike timing
        layer_synchrony = []
        for layer_spikes in all_spikes:
            if layer_spikes:
                spike_times = torch.stack(layer_spikes, dim=0)  # [time, batch, neurons]
                spike_variance = torch.var(torch.sum(spike_times, dim=-1))  # Variance of population activity
                layer_synchrony.append(float(spike_variance))
        
        return np.mean(layer_synchrony) if layer_synchrony else 0.0
    
    async def process_grid_sequence(self, grid_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process a sequence of grids with temporal continuity"""
        outputs = []
        
        for i, grid in enumerate(grid_sequence):
            # Learn from previous grids in sequence
            learn = i > 0  # Don't learn on first grid
            
            output, _ = self.forward(grid, learn=learn)
            outputs.append(output)
            
            logger.debug(f"Processed grid {i+1}/{len(grid_sequence)} with "
                        f"spike rate {self.cognitive_metrics['spike_rate']:.3f}")
        
        return outputs
    
    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with more spikes and plasticity updates
        spike_load = min(self.cognitive_metrics["spike_rate"] / 0.1, 1.0)  # Normalize to expected range
        plasticity_load = min(self.cognitive_metrics["plasticity_updates"] / 1000.0, 1.0)
        synchrony_load = min(self.cognitive_metrics["network_synchrony"] / 10.0, 1.0)
        
        return (spike_load * 0.4 + plasticity_load * 0.3 + synchrony_load * 0.3)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # SNN has moderate symbolic depth - it encodes spatial patterns
        # but reasoning happens in downstream components
        base_depth = 2
        layer_bonus = min(len(self.neuron_layers), 3)
        efficiency_bonus = 1 if self.cognitive_metrics["encoding_efficiency"] > 0.5 else 0
        return base_depth + layer_bonus + efficiency_bonus
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "SPLRSpikingNetwork",
            "cognitive_metrics": self.cognitive_metrics,
            "network_architecture": {
                "num_layers": self.num_layers,
                "neurons_per_layer": self.neurons_per_layer,
                "simulation_time": self.simulation_time
            },
            "total_parameters": sum(p.numel() for p in self.parameters())
        }


# Utility functions
def spike_train_statistics(spike_train: SpikeTrain) -> Dict[str, float]:
    """Compute statistics for a spike train"""
    if not spike_train.events:
        return {"num_spikes": 0, "mean_rate": 0.0, "std_isi": 0.0}
    
    # Basic statistics
    num_spikes = len(spike_train.events)
    mean_rate = num_spikes / spike_train.duration
    
    # Inter-spike interval statistics
    spike_times = sorted([event.timestamp for event in spike_train.events])
    if len(spike_times) > 1:
        isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
        std_isi = np.std(isis)
    else:
        std_isi = 0.0
    
    return {
        "num_spikes": num_spikes,
        "mean_rate": mean_rate,
        "std_isi": std_isi
    }


async def create_splr_network(config: Dict[str, Any]) -> SPLRSpikingNetwork:
    """Factory function to create and initialize SPLR network"""
    network = SPLRSpikingNetwork(config)
    await network.async_init()
    return network


# Export main classes
__all__ = [
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
