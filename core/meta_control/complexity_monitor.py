"""
Complexity Monitor for Real-time Complexity Assessment

Implements continuous monitoring of problem complexity and computational
requirements to enable dynamic adaptation of reasoning strategies.
Addresses the "complexity cliff" problem where LLMs fail to adapt
computational strategies to problem difficulty.

Key Features:
- Real-time complexity tracking during problem solving
- Multi-dimensional complexity assessment (visual, logical, temporal)
- Dynamic strategy adaptation based on complexity evolution
- Resource requirement prediction and adjustment
- Integration with HOLO-1.5 cognitive mesh for complexity awareness

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from enum import Enum
import numpy as np
import logging
import time
from collections import deque
import threading

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
        MONITOR = "monitor"
        ASSESSOR = "assessor"
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


logger = logging.getLogger(__name__)


class ComplexityDimension(Enum):
    """Different dimensions of problem complexity"""
    VISUAL = "visual"                # Visual pattern complexity
    LOGICAL = "logical"              # Logical reasoning complexity
    TEMPORAL = "temporal"            # Temporal sequence complexity
    SPATIAL = "spatial"              # Spatial relationship complexity
    COMPOSITIONAL = "compositional"  # Compositional/hierarchical complexity
    RELATIONAL = "relational"        # Relational reasoning complexity


class ComplexityTrend(Enum):
    """Trends in complexity over time"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"


@dataclass
class ComplexityMeasurement:
    """Single complexity measurement across dimensions"""
    timestamp: float
    dimensions: Dict[ComplexityDimension, float] = field(default_factory=dict)
    overall_complexity: float = 0.0
    confidence: float = 0.0
    computational_load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_dimension_score(self, dimension: ComplexityDimension) -> float:
        """Get complexity score for a specific dimension"""
        return self.dimensions.get(dimension, 0.0)


@dataclass
class ComplexityProfile:
    """Complete complexity profile over time"""
    measurements: List[ComplexityMeasurement] = field(default_factory=list)
    trends: Dict[ComplexityDimension, ComplexityTrend] = field(default_factory=dict)
    peak_complexity: float = 0.0
    average_complexity: float = 0.0
    complexity_variance: float = 0.0
    adaptation_points: List[float] = field(default_factory=list)  # Times when strategy adapted
    
    def add_measurement(self, measurement: ComplexityMeasurement):
        """Add a new complexity measurement"""
        self.measurements.append(measurement)
        self._update_statistics()
        self._update_trends()
    
    def _update_statistics(self):
        """Update complexity statistics"""
        if not self.measurements:
            return
        
        complexities = [m.overall_complexity for m in self.measurements]
        self.peak_complexity = max(complexities)
        self.average_complexity = np.mean(complexities)
        self.complexity_variance = np.var(complexities)
    
    def _update_trends(self):
        """Update complexity trends for each dimension"""
        if len(self.measurements) < 3:
            return
        
        window_size = min(5, len(self.measurements))
        recent_measurements = self.measurements[-window_size:]
        
        for dimension in ComplexityDimension:
            scores = [m.get_dimension_score(dimension) for m in recent_measurements]
            
            if len(scores) >= 3:
                # Simple trend detection using linear regression slope
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0]
                
                if slope > 0.05:
                    self.trends[dimension] = ComplexityTrend.INCREASING
                elif slope < -0.05:
                    self.trends[dimension] = ComplexityTrend.DECREASING
                else:
                    self.trends[dimension] = ComplexityTrend.STABLE


class VisualComplexityAnalyzer(nn.Module):
    """
    Analyzes visual pattern complexity in real-time
    
    Assesses complexity of visual patterns using multiple metrics:
    - Spatial frequency analysis
    - Pattern regularity/irregularity
    - Color distribution complexity
    - Shape complexity
    """
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.Conv2d(input_channels, 16, kernel_size=7, padding=3)
        ])
        
        # Complexity assessment network
        self.complexity_analyzer = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Spatial frequency analyzer
        self.frequency_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze visual complexity
        
        Args:
            visual_input: [batch_size, channels, height, width]
            
        Returns:
            complexity_metrics: Dictionary of complexity measures
        """
        # Multi-scale feature extraction
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(visual_input))
        
        combined_features = torch.cat(features, dim=1)
        
        # Overall visual complexity
        visual_complexity = self.complexity_analyzer(combined_features)
        
        # Spatial frequency complexity
        frequency_complexity = self.frequency_analyzer(visual_input)
        
        # Additional complexity metrics
        batch_size = visual_input.shape[0]
        
        # Color distribution complexity (for colored inputs)
        color_complexity = self._compute_color_complexity(visual_input)
        
        # Pattern regularity
        regularity_complexity = self._compute_pattern_regularity(visual_input)
        
        return {
            "overall": visual_complexity.squeeze(),
            "frequency": frequency_complexity.squeeze(),
            "color": color_complexity,
            "regularity": regularity_complexity
        }
    
    def _compute_color_complexity(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Compute color distribution complexity"""
        batch_size = visual_input.shape[0]
        
        # Simple color complexity based on unique values
        complexity_scores = []
        for b in range(batch_size):
            img = visual_input[b].flatten()
            unique_values = torch.unique(img)
            complexity = len(unique_values) / img.numel()  # Normalized unique value count
            complexity_scores.append(complexity)
        
        return torch.tensor(complexity_scores, device=visual_input.device)
    
    def _compute_pattern_regularity(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Compute pattern regularity/irregularity complexity"""
        batch_size = visual_input.shape[0]
        
        # Pattern regularity using spatial gradients
        grad_x = torch.abs(visual_input[:, :, :, 1:] - visual_input[:, :, :, :-1])
        grad_y = torch.abs(visual_input[:, :, 1:, :] - visual_input[:, :, :-1, :])
        
        # Variance of gradients indicates irregularity/complexity
        grad_variance_x = torch.var(grad_x.view(batch_size, -1), dim=1)
        grad_variance_y = torch.var(grad_y.view(batch_size, -1), dim=1)
        
        regularity_complexity = (grad_variance_x + grad_variance_y) / 2
        
        return torch.clamp(regularity_complexity, 0, 1)


class LogicalComplexityAnalyzer(nn.Module):
    """
    Analyzes logical reasoning complexity
    
    Assesses the complexity of logical reasoning required:
    - Number of inference steps required
    - Depth of logical dependencies
    - Branching factor in reasoning tree
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Logical structure analyzer
        self.structure_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Inference depth predictor
        self.depth_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Branching complexity predictor
        self.branching_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, logical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze logical complexity
        
        Args:
            logical_features: [batch_size, feature_dim] - Abstract logical features
            
        Returns:
            complexity_metrics: Dictionary of logical complexity measures
        """
        # Overall logical structure complexity
        structure_complexity = self.structure_analyzer(logical_features)
        
        # Inference depth complexity
        depth_complexity = self.depth_predictor(logical_features)
        
        # Branching complexity
        branching_complexity = self.branching_predictor(logical_features)
        
        # Combined logical complexity
        combined_complexity = (structure_complexity + depth_complexity + branching_complexity) / 3
        
        return {
            "overall": combined_complexity.squeeze(),
            "structure": structure_complexity.squeeze(),
            "depth": depth_complexity.squeeze(),
            "branching": branching_complexity.squeeze()
        }


class ResourceRequirementPredictor(nn.Module):
    """
    Predicts computational resource requirements based on complexity
    
    Estimates resource needs for different processing components
    based on assessed complexity levels.
    """
    
    def __init__(self, complexity_dim: int = 6):  # Number of complexity dimensions
        super().__init__()
        self.complexity_dim = complexity_dim
        
        # Resource prediction networks for different components
        self.memory_predictor = nn.Sequential(
            nn.Linear(complexity_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.compute_predictor = nn.Sequential(
            nn.Linear(complexity_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.time_predictor = nn.Sequential(
            nn.Linear(complexity_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, complexity_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict resource requirements
        
        Args:
            complexity_scores: [batch_size, complexity_dim] - Complexity across dimensions
            
        Returns:
            resource_predictions: Dictionary of resource requirement predictions
        """
        memory_req = self.memory_predictor(complexity_scores)
        compute_req = self.compute_predictor(complexity_scores)
        time_req = self.time_predictor(complexity_scores)
        
        return {
            "memory": memory_req.squeeze(),
            "compute": compute_req.squeeze(),
            "time": time_req.squeeze()
        }


@vanta_agent(role=CognitiveMeshRole.MONITOR)
class ComplexityMonitor(BaseAgent):
    """
    Main Complexity Monitor for real-time complexity assessment
    
    Orchestrates multiple complexity analyzers to provide continuous
    monitoring of problem complexity across different dimensions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Monitor parameters
        self.monitoring_interval = config.get("monitoring_interval", 0.5)  # seconds
        self.complexity_history_size = config.get("complexity_history_size", 100)
        self.adaptation_threshold = config.get("adaptation_threshold", 0.3)
        
        # Complexity analyzers
        self.visual_analyzer = VisualComplexityAnalyzer()
        self.logical_analyzer = LogicalComplexityAnalyzer()
        self.resource_predictor = ResourceRequirementPredictor()
        
        # Complexity tracking
        self.complexity_profile = ComplexityProfile()
        self.complexity_history = deque(maxlen=self.complexity_history_size)
        self.current_complexity = 0.0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Adaptation callbacks
        self.adaptation_callbacks: List[Callable] = []
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "average_complexity": 0.0,
            "complexity_variance": 0.0,
            "adaptation_frequency": 0.0,
            "prediction_accuracy": 0.0
        }
    
    async def async_init(self):
        """Initialize the complexity monitor"""
        if HOLO_AVAILABLE:
            await super().async_init()
        logger.info("Complexity Monitor initialized with HOLO-1.5 integration")
    
    async def start_monitoring(self, initial_input: torch.Tensor):
        """
        Start continuous complexity monitoring
        
        Args:
            initial_input: Initial problem input for baseline assessment
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        # Perform initial complexity assessment
        await self.assess_complexity(initial_input)
        
        self.monitoring_active = True
        self._stop_monitoring.clear()
        
        logger.info("Started complexity monitoring")
    
    def stop_monitoring(self):
        """Stop complexity monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Stopped complexity monitoring")
    
    async def assess_complexity(self, current_input: torch.Tensor,
                              logical_features: Optional[torch.Tensor] = None) -> ComplexityMeasurement:
        """
        Assess current complexity across all dimensions
        
        Args:
            current_input: Current problem state (visual)
            logical_features: Current logical reasoning features
            
        Returns:
            complexity_measurement: Current complexity assessment
        """
        timestamp = time.time()
        
        # Visual complexity analysis
        visual_metrics = self.visual_analyzer(current_input.unsqueeze(0) if len(current_input.shape) == 3 else current_input)
        visual_complexity = float(visual_metrics["overall"].mean())
        
        # Logical complexity analysis (if features available)
        logical_complexity = 0.0
        if logical_features is not None:
            logical_metrics = self.logical_analyzer(logical_features)
            logical_complexity = float(logical_metrics["overall"].mean())
        
        # Compute dimensional complexities
        dimension_scores = {
            ComplexityDimension.VISUAL: visual_complexity,
            ComplexityDimension.LOGICAL: logical_complexity,
            ComplexityDimension.SPATIAL: float(visual_metrics.get("regularity", torch.tensor(0.0)).mean()),
            ComplexityDimension.TEMPORAL: self._compute_temporal_complexity(),
            ComplexityDimension.COMPOSITIONAL: self._compute_compositional_complexity(current_input),
            ComplexityDimension.RELATIONAL: logical_complexity * 0.8  # Approximation
        }
        
        # Overall complexity (weighted average)
        weights = {
            ComplexityDimension.VISUAL: 0.25,
            ComplexityDimension.LOGICAL: 0.3,
            ComplexityDimension.SPATIAL: 0.15,
            ComplexityDimension.TEMPORAL: 0.1,
            ComplexityDimension.COMPOSITIONAL: 0.1,
            ComplexityDimension.RELATIONAL: 0.1
        }
        
        overall_complexity = sum(dimension_scores[dim] * weights[dim] for dim in ComplexityDimension)
        
        # Resource requirements prediction
        complexity_vector = torch.tensor([dimension_scores[dim] for dim in ComplexityDimension]).unsqueeze(0)
        resource_predictions = self.resource_predictor(complexity_vector)
        computational_load = float(resource_predictions["compute"].mean())
        
        # Create complexity measurement
        measurement = ComplexityMeasurement(
            timestamp=timestamp,
            dimensions=dimension_scores,
            overall_complexity=overall_complexity,
            confidence=0.8,  # TODO: Implement confidence estimation
            computational_load=computational_load,
            metadata={
                "visual_metrics": visual_metrics,
                "resource_predictions": resource_predictions
            }
        )
        
        # Update tracking
        self.complexity_profile.add_measurement(measurement)
        self.complexity_history.append(overall_complexity)
        self.current_complexity = overall_complexity
        
        # Update cognitive metrics
        self.cognitive_metrics["average_complexity"] = self.complexity_profile.average_complexity
        self.cognitive_metrics["complexity_variance"] = self.complexity_profile.complexity_variance
        
        # Check for adaptation triggers
        await self._check_adaptation_triggers(measurement)
        
        logger.debug(f"Complexity assessed: overall={overall_complexity:.3f}, "
                    f"visual={visual_complexity:.3f}, logical={logical_complexity:.3f}")
        
        return measurement
    
    def _compute_temporal_complexity(self) -> float:
        """Compute temporal complexity based on complexity history"""
        if len(self.complexity_history) < 3:
            return 0.0
        
        # Temporal complexity based on variance and trend changes
        recent_history = list(self.complexity_history)[-10:]
        variance = np.var(recent_history)
        
        # Trend changes (direction changes in complexity)
        trend_changes = 0
        for i in range(1, len(recent_history) - 1):
            if ((recent_history[i] > recent_history[i-1]) and 
                (recent_history[i+1] < recent_history[i])) or \
               ((recent_history[i] < recent_history[i-1]) and 
                (recent_history[i+1] > recent_history[i])):
                trend_changes += 1
        
        temporal_complexity = min(variance + trend_changes / len(recent_history), 1.0)
        return float(temporal_complexity)
    
    def _compute_compositional_complexity(self, visual_input: torch.Tensor) -> float:
        """Compute compositional/hierarchical complexity"""
        # Simplified compositional complexity based on spatial patterns
        if len(visual_input.shape) == 3:
            visual_input = visual_input.unsqueeze(0)
        
        # Multi-scale pattern detection
        scales = [2, 4, 8]
        scale_complexities = []
        
        for scale in scales:
            if visual_input.shape[-1] >= scale and visual_input.shape[-2] >= scale:
                # Downsample and compute pattern complexity
                downsampled = F.avg_pool2d(visual_input, kernel_size=scale, stride=scale)
                pattern_variance = torch.var(downsampled)
                scale_complexities.append(float(pattern_variance))
        
        if scale_complexities:
            # Compositional complexity as variance across scales
            compositional_complexity = np.var(scale_complexities)
            return min(compositional_complexity, 1.0)
        
        return 0.0
    
    async def _check_adaptation_triggers(self, measurement: ComplexityMeasurement):
        """Check if complexity changes warrant strategy adaptation"""
        if len(self.complexity_history) < 5:
            return
        
        # Check for significant complexity increase
        recent_avg = np.mean(list(self.complexity_history)[-3:])
        older_avg = np.mean(list(self.complexity_history)[-8:-3])
        
        complexity_change = recent_avg - older_avg
        
        if abs(complexity_change) > self.adaptation_threshold:
            logger.info(f"Complexity change detected: {complexity_change:.3f}, triggering adaptation")
            
            # Record adaptation point
            self.complexity_profile.adaptation_points.append(measurement.timestamp)
            
            # Trigger adaptation callbacks
            for callback in self.adaptation_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(measurement, complexity_change)
                    else:
                        callback(measurement, complexity_change)
                except Exception as e:
                    logger.error(f"Error in adaptation callback: {e}")
            
            # Update adaptation frequency metric
            if len(self.complexity_profile.adaptation_points) > 1:
                time_between_adaptations = (
                    self.complexity_profile.adaptation_points[-1] - 
                    self.complexity_profile.adaptation_points[-2]
                )
                self.cognitive_metrics["adaptation_frequency"] = 1.0 / time_between_adaptations
    
    def add_adaptation_callback(self, callback: Callable):
        """Add callback function to be called when adaptation is triggered"""
        self.adaptation_callbacks.append(callback)
    
    def get_complexity_trend(self, dimension: Optional[ComplexityDimension] = None) -> ComplexityTrend:
        """Get complexity trend for overall or specific dimension"""
        if dimension is None:
            # Overall trend based on complexity history
            if len(self.complexity_history) < 5:
                return ComplexityTrend.STABLE
            
            recent = np.mean(list(self.complexity_history)[-3:])
            older = np.mean(list(self.complexity_history)[-6:-3])
            
            if recent > older + 0.1:
                return ComplexityTrend.INCREASING
            elif recent < older - 0.1:
                return ComplexityTrend.DECREASING
            else:
                return ComplexityTrend.STABLE
        else:
            return self.complexity_profile.trends.get(dimension, ComplexityTrend.STABLE)
    
    def predict_resource_needs(self, time_horizon: float = 5.0) -> Dict[str, float]:
        """Predict resource needs for the near future"""
        if not self.complexity_history:
            return {"memory": 0.5, "compute": 0.5, "time": 0.5}
        
        # Simple trend-based prediction
        current_complexity = self.current_complexity
        trend = self.get_complexity_trend()
        
        # Adjust prediction based on trend
        if trend == ComplexityTrend.INCREASING:
            predicted_complexity = min(current_complexity * 1.2, 1.0)
        elif trend == ComplexityTrend.DECREASING:
            predicted_complexity = max(current_complexity * 0.8, 0.0)
        else:
            predicted_complexity = current_complexity
        
        # Convert to resource predictions (simplified model)
        return {
            "memory": predicted_complexity * 0.8,
            "compute": predicted_complexity,
            "time": predicted_complexity * 1.2
        }
    
    async def get_cognitive_load(self) -> float:
        """Calculate cognitive load for HOLO-1.5"""
        # Higher load with higher complexity and more variance
        complexity_load = self.cognitive_metrics["average_complexity"]
        variance_load = min(self.cognitive_metrics["complexity_variance"], 1.0)
        adaptation_load = min(self.cognitive_metrics["adaptation_frequency"] / 0.1, 1.0)
        
        return (complexity_load * 0.5 + variance_load * 0.3 + adaptation_load * 0.2)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # Complexity monitor has moderate symbolic depth
        base_depth = 3
        dimension_bonus = len(ComplexityDimension)  # Tracks multiple dimensions
        history_bonus = 1 if len(self.complexity_history) > 10 else 0
        return min(base_depth + dimension_bonus // 2 + history_bonus, 8)
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "ComplexityMonitor",
            "cognitive_metrics": self.cognitive_metrics,
            "current_complexity": self.current_complexity,
            "complexity_history_size": len(self.complexity_history),
            "adaptation_points": len(self.complexity_profile.adaptation_points),
            "monitoring_active": self.monitoring_active
        }


# Utility functions
async def create_complexity_monitor(config: Dict[str, Any]) -> ComplexityMonitor:
    """Factory function to create and initialize ComplexityMonitor"""
    monitor = ComplexityMonitor(config)
    await monitor.async_init()
    return monitor


def analyze_complexity_dimensions(measurement: ComplexityMeasurement) -> Dict[str, Any]:
    """Analyze complexity measurement across dimensions"""
    analysis = {
        "dominant_dimension": max(measurement.dimensions.items(), key=lambda x: x[1]),
        "balanced_complexity": np.std(list(measurement.dimensions.values())) < 0.2,
        "high_complexity_dimensions": [
            dim for dim, score in measurement.dimensions.items() if score > 0.7
        ]
    }
    return analysis


# Export main classes
__all__ = [
    "ComplexityMonitor",
    "VisualComplexityAnalyzer",
    "LogicalComplexityAnalyzer", 
    "ResourceRequirementPredictor",
    "ComplexityMeasurement",
    "ComplexityProfile",
    "ComplexityDimension",
    "ComplexityTrend",
    "create_complexity_monitor",
    "analyze_complexity_dimensions"
]
