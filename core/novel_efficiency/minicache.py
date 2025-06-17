#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCache Implementation - KV Cache Compression with Outlier Token Detection

This module implements the MiniCache algorithm for compressing transformer KV caches
using angular distance similarity detection and outlier token preservation.

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration and BLT semantic hashing.
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# HOLO-1.5 Core Imports
try:
    from ...agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    from ...Vanta.core.base import BaseCore
    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    def vanta_agent(role=None, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        PROCESSOR = "PROCESSOR"
    
    class BaseCore:
        def __init__(self, vanta_core=None, config=None):
            self.vanta_core = vanta_core
            self.config = config or {}
    
    HOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a KV cache entry with metadata."""
    key: torch.Tensor
    value: torch.Tensor
    layer_idx: int
    similarity_hash: Optional[int] = None
    outlier_score: float = 0.0
    timestamp: float = 0.0


class OutlierTokenDetector:
    """Detects outlier tokens that should not be compressed."""
    
    def __init__(self, threshold: float = 2.0, window_size: int = 100):
        self.threshold = threshold
        self.window_size = window_size
        self.history = []
    
    def is_outlier(self, key_vector: torch.Tensor, context_keys: List[torch.Tensor]) -> bool:
        """Determine if a key vector is an outlier based on angular distance."""
        if len(context_keys) < 3:
            return False
            
        # Calculate angular distances to recent keys
        similarities = []
        for ctx_key in context_keys[-self.window_size:]:
            cos_sim = torch.cosine_similarity(
                key_vector.unsqueeze(0), 
                ctx_key.unsqueeze(0), 
                dim=-1
            ).item()
            similarities.append(cos_sim)
        
        # Check if significantly different from recent patterns
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        if std_sim > 0:
            z_score = abs((cos_sim - mean_sim) / std_sim)
            return z_score > self.threshold
            
        return False


class KVCacheCompressor:
    """Core algorithm for KV cache compression using angular distance similarity."""
    
    def __init__(self, similarity_threshold: float = 0.95, compression_ratio: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.compression_ratio = compression_ratio
        self.outlier_detector = OutlierTokenDetector()
        
    def compress_layer_cache(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compress KV cache for a single layer using similarity clustering.
        
        Args:
            keys: Key tensor [batch_size, seq_len, hidden_dim]
            values: Value tensor [batch_size, seq_len, hidden_dim] 
            layer_idx: Layer index for tracking
            
        Returns:
            Compressed keys, compressed values, compression metadata
        """
        batch_size, seq_len, hidden_dim = keys.shape
        
        if seq_len <= 2:
            return keys, values, {"compression_ratio": 1.0, "outliers_preserved": 0}
        
        # Target compressed sequence length
        target_len = max(2, int(seq_len * self.compression_ratio))
        
        compressed_keys = []
        compressed_values = []
        outliers_preserved = 0
        
        for batch_idx in range(batch_size):
            batch_keys = keys[batch_idx]  # [seq_len, hidden_dim]
            batch_values = values[batch_idx]  # [seq_len, hidden_dim]
            
            # Always preserve first and last tokens
            preserved_indices = [0, seq_len - 1]
            preserved_keys = [batch_keys[0], batch_keys[-1]]
            preserved_values = [batch_values[0], batch_values[-1]]
            
            # Find outliers that must be preserved
            context_keys = [batch_keys[i] for i in range(seq_len)]
            for i in range(1, seq_len - 1):
                if self.outlier_detector.is_outlier(batch_keys[i], context_keys):
                    preserved_indices.append(i)
                    preserved_keys.append(batch_keys[i])
                    preserved_values.append(batch_values[i])
                    outliers_preserved += 1
            
            # If we already have enough preserved tokens, use them
            if len(preserved_indices) >= target_len:
                # Sort by position and take first target_len
                sorted_indices = sorted(preserved_indices[:target_len])
                final_keys = torch.stack([batch_keys[i] for i in sorted_indices])
                final_values = torch.stack([batch_values[i] for i in sorted_indices])
            else:
                # Need to find additional representative tokens
                remaining_target = target_len - len(preserved_indices)
                available_indices = [i for i in range(1, seq_len - 1) if i not in preserved_indices]
                
                if remaining_target > 0 and available_indices:
                    # Use clustering to find representative tokens
                    selected_indices = self._select_representative_tokens(
                        batch_keys, available_indices, remaining_target
                    )
                    
                    # Combine preserved and selected
                    all_indices = sorted(preserved_indices + selected_indices)
                    final_keys = torch.stack([batch_keys[i] for i in all_indices])
                    final_values = torch.stack([batch_values[i] for i in all_indices])
                else:
                    final_keys = torch.stack(preserved_keys)
                    final_values = torch.stack(preserved_values)
            
            compressed_keys.append(final_keys)
            compressed_values.append(final_values)
        
        # Stack back to batch format
        max_len = max(k.size(0) for k in compressed_keys)
        
        # Pad sequences to same length
        padded_keys = []
        padded_values = []
        
        for k, v in zip(compressed_keys, compressed_values):
            if k.size(0) < max_len:
                pad_len = max_len - k.size(0)
                k_padded = torch.cat([k, torch.zeros(pad_len, hidden_dim, device=k.device)])
                v_padded = torch.cat([v, torch.zeros(pad_len, hidden_dim, device=v.device)])
            else:
                k_padded = k
                v_padded = v
            
            padded_keys.append(k_padded)
            padded_values.append(v_padded)
        
        final_keys = torch.stack(padded_keys)
        final_values = torch.stack(padded_values)
        
        metadata = {
            "original_length": seq_len,
            "compressed_length": max_len,
            "compression_ratio": max_len / seq_len,
            "outliers_preserved": outliers_preserved,
            "layer_idx": layer_idx
        }
        
        return final_keys, final_values, metadata
    
    def _select_representative_tokens(
        self, 
        keys: torch.Tensor, 
        available_indices: List[int], 
        target_count: int
    ) -> List[int]:
        """Select representative tokens using similarity-based clustering."""
        if target_count >= len(available_indices):
            return available_indices
        
        # Calculate pairwise similarities
        available_keys = keys[available_indices]  # [n_available, hidden_dim]
        similarities = torch.cosine_similarity(
            available_keys.unsqueeze(1), 
            available_keys.unsqueeze(0), 
            dim=-1
        )  # [n_available, n_available]
        
        # Greedy selection to maximize diversity
        selected_local_indices = [0]  # Start with first available
        
        for _ in range(target_count - 1):
            # Find token most dissimilar to already selected
            selected_keys = available_keys[selected_local_indices]
            remaining_indices = [i for i in range(len(available_indices)) 
                               if i not in selected_local_indices]
            
            if not remaining_indices:
                break
            
            best_idx = None
            min_max_similarity = float('inf')
            
            for candidate_idx in remaining_indices:
                candidate_key = available_keys[candidate_idx]
                
                # Calculate max similarity to any selected key
                max_sim = max(
                    torch.cosine_similarity(
                        candidate_key.unsqueeze(0), 
                        sel_key.unsqueeze(0), 
                        dim=-1
                    ).item()
                    for sel_key in selected_keys
                )
                
                # Select candidate with minimum max similarity (most diverse)
                if max_sim < min_max_similarity:
                    min_max_similarity = max_sim
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_local_indices.append(best_idx)
        
        # Convert back to original indices
        return [available_indices[i] for i in selected_local_indices]


@vanta_agent(
    name="minicache_optimizer",
    subsystem="efficiency",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="KV cache compression with outlier token detection and angular distance similarity",
    capabilities=["cache_compression", "memory_optimization", "outlier_detection", "similarity_clustering"],
    cognitive_load=2.0,
    symbolic_depth=2,
    collaboration_patterns=["memory_efficiency", "adaptive_compression", "resource_optimization"]
)
class MiniCacheWrapper(BaseCore if HOLO_AVAILABLE else object):
    """
    MiniCache wrapper for transformer KV cache compression.
    
    Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration
    for adaptive compression based on cognitive load and symbolic depth.
    """
    
    def __init__(self, vanta_core=None, config=None):
        if HOLO_AVAILABLE:
            super().__init__(vanta_core, config)
        
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.95)
        self.compression_ratio = self.config.get("compression_ratio", 0.7)
        self.adaptive_compression = self.config.get("adaptive_compression", True)
        
        # Core compression components
        self.compressor = KVCacheCompressor(
            similarity_threshold=self.similarity_threshold,
            compression_ratio=self.compression_ratio
        )
        
        # Cache storage and statistics
        self.cache_layers = {}
        self.compression_stats = {
            "total_compressions": 0,
            "memory_saved_bytes": 0,
            "avg_compression_ratio": 0.0,
            "outliers_preserved": 0
        }
        
        # HOLO-1.5 cognitive metrics
        if HOLO_AVAILABLE:
            self.cognitive_metrics = {
                "compression_efficiency": 0.0,
                "memory_utilization": 0.0,
                "adaptive_ratio": self.compression_ratio,
                "outlier_detection_accuracy": 0.0
            }
            self._vanta_initialized = False
            self.monitoring_task = None
    
    async def async_init(self):
        """Initialize HOLO-1.5 cognitive mesh integration."""
        if not HOLO_AVAILABLE:
            logger.info("HOLO-1.5 not available, running in standalone mode")
            return
        
        try:
            await self.register_cognitive_capabilities()
            await self.start_cognitive_monitoring()
            self._vanta_initialized = True
            logger.info("🧠 MiniCacheWrapper HOLO-1.5 initialization complete")
        except Exception as e:
            logger.warning(f"HOLO-1.5 initialization failed: {e}")
            self._vanta_initialized = False
    
    async def register_cognitive_capabilities(self):
        """Register compression capabilities with VantaCore mesh."""
        if not HOLO_AVAILABLE or not hasattr(self, 'vanta_core') or not self.vanta_core:
            return
        
        capabilities = {
            "cache_compression": {
                "similarity_threshold": self.similarity_threshold,
                "compression_ratio": self.compression_ratio,
                "outlier_detection": True,
                "adaptive_optimization": self.adaptive_compression
            },
            "memory_optimization": {
                "dynamic_ratio_adjustment": True,
                "cognitive_load_awareness": True,
                "memory_pressure_response": True
            },
            "outlier_detection": {
                "angular_distance_analysis": True,
                "context_awareness": True,
                "preservation_guarantee": True
            }
        }
        
        await self.vanta_core.register_capabilities("minicache_optimizer", capabilities)
    
    async def start_cognitive_monitoring(self):
        """Start background cognitive monitoring and adaptive optimization."""
        if not HOLO_AVAILABLE:
            return
        
        async def monitor_loop():
            while True:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                self._update_cognitive_metrics()
                self._adapt_compression_parameters()
                
                # Generate cognitive trace for mesh learning
                if hasattr(self, 'vanta_core') and self.vanta_core:
                    trace = self._generate_cognitive_trace()
                    await self.vanta_core.emit_cognitive_trace(trace)
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
    
    def compress_kv_cache(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        layer_idx: int,
        force_compression: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Main interface for KV cache compression.
        
        Args:
            keys: Key tensor to compress
            values: Value tensor to compress
            layer_idx: Layer index for tracking
            force_compression: Force compression regardless of heuristics
            
        Returns:
            Compressed keys, compressed values, compression metadata
        """
        start_time = time.time()
        
        # Adaptive compression ratio based on cognitive load
        if HOLO_AVAILABLE and self._vanta_initialized and self.adaptive_compression:
            adaptive_ratio = self._calculate_adaptive_ratio()
            self.compressor.compression_ratio = adaptive_ratio
        
        # Perform compression
        compressed_keys, compressed_values, metadata = self.compressor.compress_layer_cache(
            keys, values, layer_idx
        )
        
        # Update statistics
        self._update_compression_stats(keys, compressed_keys, metadata, time.time() - start_time)
        
        # Store in cache if needed
        if layer_idx not in self.cache_layers:
            self.cache_layers[layer_idx] = []
        
        self.cache_layers[layer_idx].append({
            "compressed_keys": compressed_keys,
            "compressed_values": compressed_values,
            "metadata": metadata,
            "timestamp": time.time()
        })
        
        return compressed_keys, compressed_values, metadata
    
    def _calculate_adaptive_ratio(self) -> float:
        """Calculate adaptive compression ratio based on cognitive state."""
        base_ratio = self.compression_ratio
        
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return base_ratio
        
        # Adjust based on memory utilization
        memory_util = self.cognitive_metrics.get("memory_utilization", 0.5)
        if memory_util > 0.8:
            # High memory pressure - compress more aggressively
            adaptive_ratio = max(0.4, base_ratio - 0.2)
        elif memory_util < 0.3:
            # Low memory pressure - compress less aggressively
            adaptive_ratio = min(0.9, base_ratio + 0.1)
        else:
            adaptive_ratio = base_ratio
        
        # Adjust based on compression efficiency
        efficiency = self.cognitive_metrics.get("compression_efficiency", 0.5)
        if efficiency > 0.8:
            # High efficiency - can compress more
            adaptive_ratio = max(0.4, adaptive_ratio - 0.05)
        elif efficiency < 0.4:
            # Low efficiency - compress less
            adaptive_ratio = min(0.9, adaptive_ratio + 0.05)
        
        return adaptive_ratio
    
    def _update_compression_stats(
        self, 
        original_keys: torch.Tensor, 
        compressed_keys: torch.Tensor, 
        metadata: Dict[str, Any],
        processing_time: float
    ):
        """Update compression statistics and cognitive metrics."""
        original_size = original_keys.numel() * original_keys.element_size()
        compressed_size = compressed_keys.numel() * compressed_keys.element_size()
        memory_saved = original_size - compressed_size
        
        # Update basic stats
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["memory_saved_bytes"] += memory_saved
        
        # Update running average of compression ratio
        current_ratio = metadata["compression_ratio"]
        total_compressions = self.compression_stats["total_compressions"]
        prev_avg = self.compression_stats["avg_compression_ratio"]
        
        self.compression_stats["avg_compression_ratio"] = (
            (prev_avg * (total_compressions - 1) + current_ratio) / total_compressions
        )
        
        self.compression_stats["outliers_preserved"] += metadata["outliers_preserved"]
        
        # Update cognitive metrics if HOLO available
        if HOLO_AVAILABLE and self._vanta_initialized:
            self._update_cognitive_metrics_from_compression(
                memory_saved, current_ratio, processing_time, metadata
            )
    
    def _update_cognitive_metrics_from_compression(
        self, 
        memory_saved: int, 
        compression_ratio: float, 
        processing_time: float,
        metadata: Dict[str, Any]
    ):
        """Update cognitive metrics based on compression results."""
        # Compression efficiency (memory saved per processing time)
        efficiency = memory_saved / max(processing_time, 0.001) / 1e6  # MB/s
        self.cognitive_metrics["compression_efficiency"] = (
            self.cognitive_metrics["compression_efficiency"] * 0.9 + 
            min(efficiency / 100.0, 1.0) * 0.1  # Normalize to 0-1 range
        )
        
        # Memory utilization estimate
        total_memory_saved = self.compression_stats["memory_saved_bytes"]
        utilization_improvement = min(total_memory_saved / 1e9, 1.0)  # Normalize to GB
        self.cognitive_metrics["memory_utilization"] = max(
            0.0, 1.0 - utilization_improvement
        )
        
        # Update adaptive ratio tracking
        self.cognitive_metrics["adaptive_ratio"] = compression_ratio
        
        # Outlier detection accuracy (heuristic based on preserved outliers)
        outliers_ratio = metadata["outliers_preserved"] / max(metadata["original_length"], 1)
        self.cognitive_metrics["outlier_detection_accuracy"] = (
            self.cognitive_metrics["outlier_detection_accuracy"] * 0.95 +
            min(outliers_ratio * 10, 1.0) * 0.05  # Boost signal for outlier detection
        )
    
    def _update_cognitive_metrics(self):
        """Update cognitive metrics during monitoring loop."""
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return
        
        # Calculate overall compression performance
        if self.compression_stats["total_compressions"] > 0:
            avg_ratio = self.compression_stats["avg_compression_ratio"]
            memory_saved_gb = self.compression_stats["memory_saved_bytes"] / 1e9
            
            # Update memory utilization based on total savings
            self.cognitive_metrics["memory_utilization"] = max(0.0, 1.0 - memory_saved_gb)
    
    def _adapt_compression_parameters(self):
        """Adapt compression parameters based on cognitive state."""
        if not HOLO_AVAILABLE or not self._vanta_initialized or not self.adaptive_compression:
            return
        
        # Adjust outlier detection sensitivity based on accuracy
        accuracy = self.cognitive_metrics.get("outlier_detection_accuracy", 0.5)
        if accuracy < 0.3:
            # Lower threshold to catch more outliers
            self.compressor.outlier_detector.threshold = max(1.5, 
                self.compressor.outlier_detector.threshold - 0.1)
        elif accuracy > 0.8:
            # Raise threshold to be more selective
            self.compressor.outlier_detector.threshold = min(3.0,
                self.compressor.outlier_detector.threshold + 0.1)
    
    def _generate_cognitive_trace(self) -> Dict[str, Any]:
        """Generate cognitive trace for mesh learning."""
        return {
            "component": "MiniCacheWrapper",
            "role": "PROCESSOR",
            "timestamp": time.time(),
            "metrics": self.cognitive_metrics.copy(),
            "stats": self.compression_stats.copy(),
            "cognitive_state": {
                "compression_efficiency": self.cognitive_metrics.get("compression_efficiency", 0.0),
                "memory_optimization": self.cognitive_metrics.get("memory_utilization", 0.0),
                "adaptive_performance": self.cognitive_metrics.get("adaptive_ratio", 0.7)
            }
        }
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        stats = self.compression_stats.copy()
        
        if HOLO_AVAILABLE and self._vanta_initialized:
            stats["cognitive_metrics"] = self.cognitive_metrics.copy()
            stats["adaptive_compression_enabled"] = self.adaptive_compression
        
        # Add computed metrics
        if stats["total_compressions"] > 0:
            stats["avg_memory_saved_per_compression"] = (
                stats["memory_saved_bytes"] / stats["total_compressions"]
            )
            stats["avg_outliers_per_compression"] = (
                stats["outliers_preserved"] / stats["total_compressions"]
            )
        
        return stats
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.compression_stats = {
            "total_compressions": 0,
            "memory_saved_bytes": 0,
            "avg_compression_ratio": 0.0,
            "outliers_preserved": 0
        }
        
        if HOLO_AVAILABLE and self._vanta_initialized:
            self.cognitive_metrics = {
                "compression_efficiency": 0.0,
                "memory_utilization": 0.0,
                "adaptive_ratio": self.compression_ratio,
                "outlier_detection_accuracy": 0.0
            }
    
    def shutdown(self):
        """Clean shutdown of monitoring tasks."""
        if hasattr(self, 'monitoring_task') and self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("MiniCacheWrapper shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    # Create sample KV tensors
    batch_size, seq_len, hidden_dim = 2, 100, 512
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    values = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Initialize MiniCache
    cache_wrapper = MiniCacheWrapper(config={
        "similarity_threshold": 0.95,
        "compression_ratio": 0.7,
        "adaptive_compression": True
    })
    
    # Compress cache
    compressed_keys, compressed_values, metadata = cache_wrapper.compress_kv_cache(
        keys, values, layer_idx=0
    )
    
    print(f"Original shape: {keys.shape}")
    print(f"Compressed shape: {compressed_keys.shape}")
    print(f"Compression ratio: {metadata['compression_ratio']:.3f}")
    print(f"Outliers preserved: {metadata['outliers_preserved']}")
    
    # Show statistics
    stats = cache_wrapper.get_compression_stats()
    print(f"\nCompression Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
