"""
Kuramoto Oscillatory Neurons (AKOrN) for Bio-inspired Object Binding

Implements bio-inspired neural dynamics based on Kuramoto oscillator models
for solving the binding problem in visual reasoning tasks. Addresses object
segmentation and feature binding challenges in ARC-like tasks.

Key Features:
- Kuramoto phase oscillators for temporal binding
- Spatial-temporal synchronization patterns
- Dynamic object segmentation via phase coherence  
- Frequency adaptation for multi-object scenes
- Integration with HOLO-1.5 cognitive mesh

Based on research in neural synchrony and binding-by-synchrony theory.

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
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
        BINDER = "binder"
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


@dataclass
class OscillatorState:
    """State of Kuramoto oscillators for object binding"""
    phases: torch.Tensor          # [batch_size, height, width, num_oscillators]
    frequencies: torch.Tensor     # [batch_size, height, width, num_oscillators] 
    amplitudes: torch.Tensor      # [batch_size, height, width, num_oscillators]
    coupling_strengths: torch.Tensor  # [batch_size, height, width, num_oscillators, num_oscillators]
    synchrony_measure: torch.Tensor = None  # Global synchrony metric
    object_masks: torch.Tensor = None       # Extracted object segmentations


@dataclass
class BindingResult:
    """Result of oscillatory binding process"""
    bound_objects: List[torch.Tensor]  # List of object masks
    object_features: List[torch.Tensor]  # Features for each object
    synchrony_patterns: torch.Tensor   # Synchronization patterns
    binding_confidence: torch.Tensor  # Confidence in binding
    temporal_dynamics: torch.Tensor   # Phase evolution over time


class KuramotoOscillator(nn.Module):
    """
    Single Kuramoto oscillator with learnable parameters
    
    Implements the classical Kuramoto model:
    dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)
    """
    
    def __init__(self, num_oscillators: int, coupling_strength: float = 1.0):
        super().__init__()
        self.num_oscillators = num_oscillators
        
        # Learnable natural frequencies
        self.natural_frequencies = nn.Parameter(
            torch.randn(num_oscillators) * 0.1
        )
        
        # Learnable coupling matrix (asymmetric allowed)
        self.coupling_matrix = nn.Parameter(
            torch.eye(num_oscillators) * coupling_strength + 
            torch.randn(num_oscillators, num_oscillators) * 0.01
        )
        
        # Phase integration parameters
        self.dt = 0.01  # Integration time step
        self.integration_steps = 10  # Number of integration steps
    
    def forward(self, initial_phases: torch.Tensor, 
                external_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate Kuramoto dynamics forward in time
        
        Args:
            initial_phases: [batch_size, height, width, num_oscillators]
            external_input: Optional external driving input
            
        Returns:
            final_phases: Phases after integration
            phase_trajectory: Complete phase evolution
        """
        batch_size, height, width, num_osc = initial_phases.shape
        
        phases = initial_phases
        trajectory = [phases.clone()]
        
        for step in range(self.integration_steps):
            # Compute phase differences
            phase_diffs = self._compute_phase_differences(phases)
            
            # Kuramoto coupling term
            coupling_term = torch.sum(
                self.coupling_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0) * 
                torch.sin(phase_diffs), dim=-1
            ) / num_osc
            
            # External input term
            external_term = 0
            if external_input is not None:
                external_term = external_input
            
            # Phase derivative
            dpdt = (self.natural_frequencies.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 
                   coupling_term + external_term)
            
            # Integrate using Euler method
            phases = phases + self.dt * dpdt
            
            # Keep phases in [0, 2π]
            phases = torch.fmod(phases, 2 * math.pi)
            
            trajectory.append(phases.clone())
        
        phase_trajectory = torch.stack(trajectory, dim=0)  # [time, batch, height, width, num_osc]
        
        return phases, phase_trajectory
    
    def _compute_phase_differences(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute pairwise phase differences"""
        # phases: [batch, height, width, num_osc]
        phases_expanded = phases.unsqueeze(-1)  # [batch, height, width, num_osc, 1]
        phases_transposed = phases.unsqueeze(-2)  # [batch, height, width, 1, num_osc]
        
        return phases_expanded - phases_transposed  # [batch, height, width, num_osc, num_osc]


class SpatialCouplingNetwork(nn.Module):
    """
    Neural network for computing spatial coupling between oscillators
    
    Learns spatial relationships and connectivity patterns for
    oscillator coupling based on visual features.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Coupling strength predictor
        self.coupling_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Distance-based coupling decay
        self.distance_decay = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial coupling strengths
        
        Args:
            features: [batch_size, feature_dim, height, width]
            
        Returns:
            coupling_matrix: [batch_size, height, width, height, width]
        """
        batch_size, _, height, width = features.shape
        
        # Encode spatial features
        encoded = self.spatial_encoder(features)  # [batch, hidden_dim, height, width]
        
        # Compute pairwise feature similarities
        encoded_flat = encoded.view(batch_size, self.hidden_dim, -1)  # [batch, hidden_dim, h*w]
        similarities = torch.bmm(
            encoded_flat.transpose(1, 2),  # [batch, h*w, hidden_dim]
            encoded_flat                   # [batch, hidden_dim, h*w]
        )  # [batch, h*w, h*w]
        
        # Add spatial distance decay
        coords = torch.stack(torch.meshgrid(
            torch.arange(height, device=features.device),
            torch.arange(width, device=features.device)
        ), dim=0).float()  # [2, height, width]
        
        coords_flat = coords.view(2, -1).T  # [h*w, 2]
        distances = torch.cdist(coords_flat, coords_flat)  # [h*w, h*w]
        
        distance_weights = torch.exp(-self.distance_decay * distances)
        distance_weights = distance_weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine similarity and distance
        coupling_strengths = torch.sigmoid(similarities) * distance_weights
        
        # Reshape to spatial grid
        coupling_matrix = coupling_strengths.view(batch_size, height, width, height, width)
        
        return coupling_matrix


class ObjectSegmentationHead(nn.Module):
    """
    Extract object segmentations from oscillator synchrony patterns
    
    Uses phase coherence and synchronization to identify distinct objects
    in the visual scene.
    """
    
    def __init__(self, num_oscillators: int, max_objects: int = 8):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.max_objects = max_objects
        
        # Synchrony analysis network
        self.synchrony_analyzer = nn.Sequential(
            nn.Conv2d(num_oscillators, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, max_objects, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Coherence threshold for object detection
        self.coherence_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, oscillator_state: OscillatorState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract object masks from oscillator phases
        
        Args:
            oscillator_state: Current state of all oscillators
            
        Returns:
            object_masks: [batch_size, max_objects, height, width]
            coherence_map: [batch_size, height, width] - local phase coherence
        """
        phases = oscillator_state.phases
        batch_size, height, width, num_osc = phases.shape
        
        # Compute local phase coherence (order parameter)
        complex_phases = torch.exp(1j * phases)  # Convert to complex representation
        mean_complex = torch.mean(complex_phases, dim=-1)  # [batch, height, width]
        coherence_map = torch.abs(mean_complex)  # Local synchrony measure
        
        # Compute phase derivatives for motion detection
        phase_input = phases.permute(0, 3, 1, 2)  # [batch, num_osc, height, width]
        
        # Extract object segmentations
        object_masks = self.synchrony_analyzer(phase_input)  # [batch, max_objects, height, width]
        
        # Apply coherence thresholding
        coherence_mask = (coherence_map > self.coherence_threshold).float()
        coherence_mask = coherence_mask.unsqueeze(1)  # [batch, 1, height, width]
        
        object_masks = object_masks * coherence_mask
        
        return object_masks, coherence_map


@vanta_agent(role=CognitiveMeshRole.BINDER)
class AKOrNBindingNetwork(BaseAgent):
    """
    Kuramoto Oscillatory Network for Object Binding (AKOrN)
    
    Main network that orchestrates multiple Kuramoto oscillators for
    object binding and segmentation in visual reasoning tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Network parameters
        self.num_oscillators = config.get("num_oscillators", 8)
        self.feature_dim = config.get("feature_dim", 64)
        self.max_objects = config.get("max_objects", 8)
        self.integration_steps = config.get("integration_steps", 20)
        
        # Core components
        self.oscillator = KuramotoOscillator(self.num_oscillators)
        self.spatial_coupling = SpatialCouplingNetwork(self.feature_dim)
        self.segmentation_head = ObjectSegmentationHead(self.num_oscillators, self.max_objects)
        
        # Feature preprocessing
        self.feature_preprocessor = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.LayerNorm([self.feature_dim]),
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.num_oscillators, kernel_size=1)
        )
        
        # Phase initialization network
        self.phase_initializer = nn.Sequential(
            nn.Conv2d(self.feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.num_oscillators, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1], will be scaled to [0, 2π]
        )
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "synchrony_level": 0.0,
            "binding_confidence": 0.0,
            "num_detected_objects": 0,
            "oscillation_stability": 0.0
        }
    
    async def async_init(self):
        """Initialize the AKOrN network"""
        if HOLO_AVAILABLE:
            await super().async_init()
        logger.info("AKOrN Binding Network initialized with HOLO-1.5 integration")
    
    def forward(self, visual_features: torch.Tensor) -> BindingResult:
        """
        Perform oscillatory binding on visual features
        
        Args:
            visual_features: [batch_size, feature_dim, height, width]
            
        Returns:
            binding_result: Complete binding analysis result
        """
        batch_size, feature_dim, height, width = visual_features.shape
        
        # Initialize oscillator phases from visual features
        initial_phases = self.phase_initializer(visual_features)  # [batch, num_osc, height, width]
        initial_phases = initial_phases * 2 * math.pi  # Scale to [0, 2π]
        initial_phases = initial_phases.permute(0, 2, 3, 1)  # [batch, height, width, num_osc]
        
        # Compute spatial coupling matrix
        coupling_matrix = self.spatial_coupling(visual_features)
        
        # Create oscillator state
        oscillator_state = OscillatorState(
            phases=initial_phases,
            frequencies=self.oscillator.natural_frequencies.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(initial_phases),
            amplitudes=torch.ones_like(initial_phases),
            coupling_strengths=coupling_matrix.view(batch_size, height, width, -1, 1).expand(-1, -1, -1, -1, self.num_oscillators)
        )
        
        # Run oscillator dynamics
        final_phases, phase_trajectory = self.oscillator(
            initial_phases, 
            self.feature_preprocessor(visual_features).permute(0, 2, 3, 1)
        )
        
        # Update oscillator state
        oscillator_state.phases = final_phases
        
        # Extract object segmentations
        object_masks, coherence_map = self.segmentation_head(oscillator_state)
        
        # Post-process object masks to get individual objects
        bound_objects = []
        object_features = []
        
        for obj_idx in range(self.max_objects):
            mask = object_masks[:, obj_idx:obj_idx+1]  # [batch, 1, height, width]
            
            # Filter out weak masks
            if torch.max(mask) > 0.3:
                bound_objects.append(mask)
                
                # Extract features for this object
                masked_features = visual_features * mask
                obj_features = torch.mean(masked_features, dim=(2, 3))  # [batch, feature_dim]
                object_features.append(obj_features)
        
        # Compute binding confidence and synchrony metrics
        synchrony_level = torch.mean(coherence_map)
        binding_confidence = torch.mean(torch.max(object_masks, dim=1)[0])
        
        # Update cognitive metrics
        self.cognitive_metrics.update({
            "synchrony_level": float(synchrony_level),
            "binding_confidence": float(binding_confidence),
            "num_detected_objects": len(bound_objects),
            "oscillation_stability": float(torch.std(phase_trajectory[-5:]))  # Stability of final phases
        })
        
        # Create binding result
        result = BindingResult(
            bound_objects=bound_objects,
            object_features=object_features,
            synchrony_patterns=coherence_map,
            binding_confidence=binding_confidence,
            temporal_dynamics=phase_trajectory
        )
        
        return result
    
    async def bind_objects(self, visual_features: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None) -> BindingResult:
        """
        High-level interface for object binding
        
        Args:
            visual_features: Input visual feature map
            attention_mask: Optional attention mask to focus binding
            
        Returns:
            binding_result: Results of oscillatory binding
        """
        if attention_mask is not None:
            # Apply attention mask to features
            visual_features = visual_features * attention_mask
        
        result = self.forward(visual_features)
        
        logger.info(f"AKOrN binding detected {len(result.bound_objects)} objects "
                   f"with synchrony level {self.cognitive_metrics['synchrony_level']:.3f}")
        
        return result
    
    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with more objects and lower synchrony
        object_load = min(self.cognitive_metrics["num_detected_objects"] / self.max_objects, 1.0)
        synchrony_load = 1.0 - self.cognitive_metrics["synchrony_level"]
        stability_load = min(self.cognitive_metrics["oscillation_stability"], 1.0)
        
        return (object_load * 0.3 + synchrony_load * 0.4 + stability_load * 0.3)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # AKOrN has moderate symbolic depth - it performs object binding
        # which is a form of symbolic grouping and representation
        base_depth = 3
        object_bonus = min(self.cognitive_metrics["num_detected_objects"], 3)
        return base_depth + object_bonus
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "AKOrNBindingNetwork",
            "cognitive_metrics": self.cognitive_metrics,
            "oscillator_parameters": {
                "num_oscillators": self.num_oscillators,
                "integration_steps": self.integration_steps
            },
            "network_parameters": sum(p.numel() for p in self.parameters())
        }


# Utility functions
def compute_phase_synchrony(phases: torch.Tensor) -> torch.Tensor:
    """Compute global phase synchrony measure (Kuramoto order parameter)"""
    complex_phases = torch.exp(1j * phases)
    mean_complex = torch.mean(complex_phases, dim=-1)
    return torch.abs(mean_complex)


def extract_synchrony_clusters(phases: torch.Tensor, threshold: float = 0.7) -> List[torch.Tensor]:
    """Extract clusters of synchronized oscillators"""
    batch_size, height, width, num_osc = phases.shape
    
    # Compute pairwise phase differences
    phase_diffs = phases.unsqueeze(-1) - phases.unsqueeze(-2)
    
    # Check for synchronization (small phase differences)
    sync_matrix = (torch.abs(torch.sin(phase_diffs)) < (1 - threshold)).float()
    
    # Extract connected components (simplified)
    clusters = []
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                sync_local = sync_matrix[b, h, w]
                if torch.sum(sync_local) > 1:  # At least 2 synchronized oscillators
                    clusters.append(sync_local)
    
    return clusters


async def create_akorn_network(config: Dict[str, Any]) -> AKOrNBindingNetwork:
    """Factory function to create and initialize AKOrN network"""
    network = AKOrNBindingNetwork(config)
    await network.async_init()
    return network


# Export main classes
__all__ = [
    "AKOrNBindingNetwork",
    "KuramotoOscillator", 
    "SpatialCouplingNetwork",
    "ObjectSegmentationHead",
    "OscillatorState",
    "BindingResult", 
    "compute_phase_synchrony",
    "extract_synchrony_clusters",
    "create_akorn_network"
]
