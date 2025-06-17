"""
DeltaNet Linear Attention Implementation

This module implements DeltaNet-inspired linear attention mechanisms for O(L) complexity
scaling, addressing the quadratic complexity bottleneck in transformer architectures.

Key Features:
- Linear-time attention computation using delta rule operators
- Memory-efficient attention with constant memory footprint
- HOLO-1.5 cognitive mesh integration with symbolic processing capabilities
- Hardware-optimized implementations for efficient scaling

References:
- DeltaNet: Conditional Computation for Efficient Attention
- Linear Attention Mechanisms for Transformer Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass

# HOLO-1.5 Cognitive Mesh Integration
try:
    from ...agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    HOLO_AVAILABLE = True
    
    # Define VantaAgentCapability locally
    class VantaAgentCapability:
        LINEAR_ATTENTION = "linear_attention"
        DELTA_RULE = "delta_rule"
        CONSTANT_MEMORY = "constant_memory"
        HARDWARE_OPTIMIZATION = "hardware_optimization"
        
except ImportError:
    # Fallback for non-HOLO environments
    def vanta_agent(role=None, cognitive_load=0, symbolic_depth=0, capabilities=None):
        def decorator(cls):
            cls._holo_role = role
            cls._holo_cognitive_load = cognitive_load
            cls._holo_symbolic_depth = symbolic_depth
            cls._holo_capabilities = capabilities or []
            return cls
        return decorator
    
    class CognitiveMeshRole:
        PROCESSOR = "PROCESSOR"
    
    class VantaAgentCapability:
        LINEAR_ATTENTION = "linear_attention"
        DELTA_RULE = "delta_rule"
        CONSTANT_MEMORY = "constant_memory"
        HARDWARE_OPTIMIZATION = "hardware_optimization"
    
    class BaseAgent:
        pass
    
    HOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LinearAttentionConfig:
    """Configuration for linear attention mechanisms."""
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    causal: bool = True
    feature_map: str = "elu"  # Options: elu, relu, exp, polynomial
    feature_dimension: int = 64
    delta_rule_strength: float = 0.1
    adaptive_threshold: bool = True
    cognitive_load_scaling: bool = True
    
class DeltaRuleOperator(nn.Module):
    """Delta rule-based update operator for linear attention."""
    
    def __init__(self, d_model: int, strength: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.strength = strength
        
        # Learnable delta rule parameters
        self.alpha = nn.Parameter(torch.ones(1) * strength)
        self.beta = nn.Parameter(torch.ones(1) * (1 - strength))
        
        # State tracking for adaptive updates
        self.register_buffer('cumulative_state', torch.zeros(d_model))
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor) -> torch.Tensor:
        """Apply delta rule updates to attention computation."""
        batch_size, seq_len, d_model = queries.shape
        
        # Compute attention updates using delta rule
        # Delta = α * (target - current) where target is KV interaction
        kv_interaction = torch.einsum('bld,bmd->blm', keys, values)
        current_state = self.cumulative_state.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Delta rule update
        delta = self.alpha * (kv_interaction - current_state)
        updated_state = current_state + delta
        
        # Apply to queries
        output = torch.einsum('bld,bld->bld', queries, updated_state)
        
        # Update cumulative state (detached to prevent gradient flow)
        self.cumulative_state = self.beta * self.cumulative_state + \
                               self.alpha * kv_interaction.mean(dim=(0, 1)).detach()
        
        return output

class LinearAttentionKernel(nn.Module):
    """Linear attention kernel with feature map functions."""
    
    def __init__(self, config: LinearAttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dimension = config.feature_dimension
        
        # Feature map projection
        if config.feature_map == "polynomial":
            self.feature_proj = nn.Linear(config.d_model, config.feature_dimension)
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map function to input."""
        if self.config.feature_map == "elu":
            return F.elu(x) + 1
        elif self.config.feature_map == "relu":
            return F.relu(x)
        elif self.config.feature_map == "exp":
            return torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        elif self.config.feature_map == "polynomial":
            x_proj = self.feature_proj(x)
            return torch.cat([x_proj, x_proj**2], dim=-1)
        else:
            return F.elu(x) + 1  # Default to ELU
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute linear attention using feature maps."""
        batch_size, seq_len, d_model = queries.shape
        
        # Apply feature maps
        phi_q = self.feature_map(queries)  # [B, L, F]
        phi_k = self.feature_map(keys)     # [B, L, F]
        
        if self.config.causal:
            # Causal linear attention using cumulative sums
            output = self._causal_linear_attention(phi_q, phi_k, values, mask)
        else:
            # Non-causal linear attention
            output = self._non_causal_linear_attention(phi_q, phi_k, values, mask)
            
        return output
    
    def _causal_linear_attention(self, phi_q: torch.Tensor, phi_k: torch.Tensor,
                                values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Causal linear attention with O(L) complexity."""
        batch_size, seq_len, feature_dim = phi_q.shape
        d_value = values.shape[-1]
        
        # Initialize cumulative state
        S = torch.zeros(batch_size, feature_dim, d_value, device=phi_q.device, dtype=phi_q.dtype)
        Z = torch.zeros(batch_size, feature_dim, device=phi_q.device, dtype=phi_q.dtype)
        
        outputs = []
        
        for i in range(seq_len):
            # Update cumulative sums
            S = S + torch.einsum('bf,bd->bfd', phi_k[:, i], values[:, i])
            Z = Z + phi_k[:, i]
            
            # Compute attention output
            numerator = torch.einsum('bf,bfd->bd', phi_q[:, i], S)
            denominator = torch.einsum('bf,bf->b', phi_q[:, i], Z).clamp(min=1e-8)
            
            output_i = numerator / denominator.unsqueeze(-1)
            
            if mask is not None and mask.dim() > 1:
                output_i = output_i * mask[:, i].unsqueeze(-1)
                
            outputs.append(output_i)
        
        return torch.stack(outputs, dim=1)
    
    def _non_causal_linear_attention(self, phi_q: torch.Tensor, phi_k: torch.Tensor,
                                   values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Non-causal linear attention with O(L) complexity."""
        # Global sums for non-causal attention
        KV = torch.einsum('blf,bld->bfd', phi_k, values)  # [B, F, D]
        K_sum = phi_k.sum(dim=1)  # [B, F]
        
        # Compute attention outputs
        numerator = torch.einsum('blf,bfd->bld', phi_q, KV)
        denominator = torch.einsum('blf,bf->bl', phi_q, K_sum).clamp(min=1e-8)
        
        output = numerator / denominator.unsqueeze(-1)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
            
        return output

@vanta_agent(
    name="deltanet_attention",
    subsystem="efficiency", 
    mesh_role=CognitiveMeshRole.PROCESSOR,
    capabilities=[
        VantaAgentCapability.LINEAR_ATTENTION,
        VantaAgentCapability.DELTA_RULE,
        VantaAgentCapability.CONSTANT_MEMORY,
        VantaAgentCapability.HARDWARE_OPTIMIZATION,
        "efficient_scaling",
        "memory_conservation"
    ],
    cognitive_load=3.0,
    symbolic_depth=3
)
class DeltaNetAttention(BaseAgent if HOLO_AVAILABLE else object, nn.Module):
    """Delta rule-based linear attention mechanism with cognitive mesh integration."""
    
    def __init__(self, config: LinearAttentionConfig):
        if HOLO_AVAILABLE:
            BaseAgent.__init__(self)
        nn.Module.__init__(self)
        
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        # Multi-head projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Linear attention kernel
        self.linear_kernel = LinearAttentionKernel(config)
        
        # Delta rule operator
        self.delta_operator = DeltaRuleOperator(
            self.d_head, 
            strength=config.delta_rule_strength
        )
        
        # Dropout and normalization
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Cognitive load monitoring
        self.cognitive_metrics = {
            'attention_entropy': 0.0,
            'memory_usage': 0.0,
            'computational_efficiency': 0.0,
            'delta_update_magnitude': 0.0
        }
        
        # HOLO-1.5 initialization
        if HOLO_AVAILABLE:
            self._vanta_initialized = False
            self.monitoring_task = None
        
    async def async_init(self):
        """Initialize HOLO-1.5 cognitive mesh integration."""
        if not HOLO_AVAILABLE:
            logger.info("HOLO-1.5 not available, skipping cognitive mesh initialization")
            return

        try:
            # Register capabilities with cognitive mesh
            await self._register_capabilities()
            await self._start_cognitive_monitoring()
            self._vanta_initialized = True
            logger.info("DeltaNetAttention HOLO-1.5 cognitive mesh initialization complete")
            
        except Exception as e:
            logger.warning(f"HOLO-1.5 initialization failed: {e}")
            self._vanta_initialized = False
        
    async def _register_capabilities(self):
        """Register DeltaNet capabilities with the cognitive mesh."""
        capabilities = {
            "linear_attention": {
                "complexity": "O(L)",
                "memory_efficient": True,
                "supports_causal": True,
                "supports_streaming": True
            },
            "delta_rule_learning": {
                "adaptive_updates": True,
                "memory_consolidation": True,
                "online_learning": True
            },
            "hardware_optimization": {
                "vectorizable": True,
                "cache_friendly": True,
                "parallel_computation": True
            }
        }
        
        if hasattr(self, 'vanta_core') and self.vanta_core:
            await self.vanta_core.register_capabilities("deltanet_attention", capabilities)
    
    async def _start_cognitive_monitoring(self):
        """Start background cognitive monitoring."""
        if not HOLO_AVAILABLE:
            return
            
        monitoring_config = {
            "metrics": ["attention_entropy", "memory_usage", "computational_efficiency", "delta_update_magnitude"],
            "adaptive_thresholds": True,
            "learning_rate": 0.1
        }
        
        if hasattr(self, 'vanta_core') and self.vanta_core:
            await self.vanta_core.start_monitoring("deltanet_attention", monitoring_config)
        
    def _register_capabilities(self):
        """Register DeltaNet capabilities with the cognitive mesh."""
        capabilities = {
            "linear_attention": {
                "complexity": "O(L)",
                "memory_efficient": True,
                "supports_causal": True,
                "supports_streaming": True
            },
            "delta_rule_learning": {
                "adaptive_updates": True,
                "memory_consolidation": True,
                "online_learning": True
            },
            "hardware_optimization": {
                "vectorizable": True,
                "cache_friendly": True,
                "parallel_computation": True
            }
        }
        
        self.register_capability("efficient_attention", capabilities)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with linear attention computation."""
        batch_size, seq_len, d_model = query.shape
        
        # Project to multi-head space
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for multi-head computation
        Q = Q.transpose(1, 2)  # [B, H, L, D]
        K = K.transpose(1, 2)  # [B, H, L, D]
        V = V.transpose(1, 2)  # [B, H, L, D]
        
        # Apply linear attention per head
        attention_outputs = []
        total_entropy = 0.0
        
        for h in range(self.n_heads):
            # Extract head-specific tensors
            q_h = Q[:, h]  # [B, L, D]
            k_h = K[:, h]  # [B, L, D]
            v_h = V[:, h]  # [B, L, D]
            
            # Apply delta rule if enabled
            if hasattr(self, 'delta_operator'):
                q_h = self.delta_operator(q_h, k_h, v_h)
            
            # Linear attention computation
            attn_output = self.linear_kernel(q_h, k_h, v_h, attn_mask)
            attention_outputs.append(attn_output)
            
            # Track attention entropy for cognitive metrics
            if self.config.cognitive_load_scaling:
                with torch.no_grad():
                    phi_q = self.linear_kernel.feature_map(q_h)
                    phi_k = self.linear_kernel.feature_map(k_h)
                    attention_weights = torch.einsum('bld,bmd->blm', phi_q, phi_k)
                    attention_weights = F.softmax(attention_weights, dim=-1)
                    entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
                    total_entropy += entropy.item()
        
        # Concatenate multi-head outputs
        attention_output = torch.stack(attention_outputs, dim=1)  # [B, H, L, D]
        attention_output = attention_output.transpose(1, 2).contiguous()  # [B, L, H, D]
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        # Final output projection
        output = self.out_proj(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        # Update cognitive metrics
        self._update_cognitive_metrics(total_entropy / self.n_heads, seq_len)
        
        # Generate cognitive trace for mesh learning
        self._generate_cognitive_trace(output, attention_output)
        
        return output, None  # Linear attention doesn't compute explicit weights
    
    def _update_cognitive_metrics(self, attention_entropy: float, seq_len: int):
        """Update cognitive performance metrics."""
        # Attention entropy (measure of focus vs. diffusion)
        self.cognitive_metrics['attention_entropy'] = attention_entropy
        
        # Memory efficiency (linear vs quadratic)
        quadratic_memory = seq_len ** 2
        linear_memory = seq_len * self.config.feature_dimension
        efficiency = 1.0 - (linear_memory / max(quadratic_memory, 1))
        self.cognitive_metrics['memory_usage'] = linear_memory
        self.cognitive_metrics['computational_efficiency'] = efficiency
          # Delta update magnitude
        if hasattr(self.delta_operator, 'alpha'):
            self.cognitive_metrics['delta_update_magnitude'] = self.delta_operator.alpha.item()
    
    def _generate_cognitive_trace(self, final_output: torch.Tensor, attention_output: torch.Tensor):
        """Generate cognitive trace for mesh learning."""
        trace = {
            "module": "deltanet_attention",
            "operation": "linear_attention_computation",
            "metrics": self.cognitive_metrics.copy(),
            "efficiency_gains": {
                "memory_reduction": f"{self.cognitive_metrics['computational_efficiency']:.2%}",
                "complexity_class": "O(L) vs O(L²)",
                "delta_learning": self.cognitive_metrics['delta_update_magnitude']
            },
            "output_statistics": {
                "output_norm": torch.norm(final_output).item(),
                "attention_norm": torch.norm(attention_output).item(),
                "sequence_length": final_output.shape[1]
            }
        }
        
        # Emit cognitive trace if HOLO is available
        if HOLO_AVAILABLE and hasattr(self, 'vanta_core') and self.vanta_core:
            self.vanta_core.emit_cognitive_trace("linear_attention_computation", trace)
    
    def get_computational_stats(self) -> Dict[str, Any]:
        """Return computational efficiency statistics."""
        return {
            "attention_type": "linear",
            "complexity": "O(L)",
            "memory_efficient": True,
            "delta_rule_enabled": hasattr(self, 'delta_operator'),
            "feature_map": self.config.feature_map,
            "cognitive_metrics": self.cognitive_metrics.copy(),
            "parameters": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "feature_dimension": self.config.feature_dimension,
                "delta_strength": self.config.delta_rule_strength
            }
        }
    
    def enable_streaming_mode(self):
        """Enable streaming mode for real-time processing."""
        self.config.causal = True
        logger.info("DeltaNet attention enabled for streaming mode")
    
    def optimize_for_length(self, typical_length: int):
        """Optimize configuration for typical sequence length."""
        # Adaptive feature dimension based on sequence length
        if typical_length > 2048:
            self.config.feature_dimension = min(128, self.config.d_model // 4)
        elif typical_length > 512:
            self.config.feature_dimension = min(64, self.config.d_model // 8)
        else:
            self.config.feature_dimension = min(32, self.config.d_model // 16)
            
        # Recreate linear kernel with new configuration
        self.linear_kernel = LinearAttentionKernel(self.config)
        
        logger.info(f"DeltaNet optimized for length {typical_length}, "
                   f"feature_dim={self.config.feature_dimension}")
    
    def calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on attention metrics."""
        base_load = 3.0  # Base cognitive load
        
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return base_load

        # Adjust based on computational efficiency and memory usage
        efficiency = self.cognitive_metrics.get("computational_efficiency", 0.0)
        attention_entropy = self.cognitive_metrics.get("attention_entropy", 0.0)
        
        load_adjustment = 0
        
        # Lower load for higher efficiency
        if efficiency > 0.8:
            load_adjustment -= 0.5
        elif efficiency < 0.4:
            load_adjustment += 0.5
            
        # Adjust based on attention patterns
        if attention_entropy > 3.0:  # High entropy = diffuse attention
            load_adjustment += 0.3
        elif attention_entropy < 1.0:  # Low entropy = focused attention
            load_adjustment -= 0.2

        return max(1, min(6, base_load + load_adjustment))

    def calculate_symbolic_depth(self) -> int:
        """Calculate current symbolic depth based on delta rule learning."""
        base_depth = 3  # Base symbolic depth
        
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return base_depth

        # Adjust based on delta learning magnitude and efficiency
        delta_magnitude = self.cognitive_metrics.get("delta_update_magnitude", 0.0)
        efficiency = self.cognitive_metrics.get("computational_efficiency", 0.0)
        
        depth_adjustment = 0
        if delta_magnitude > 0.1:
            depth_adjustment += 1
        if efficiency > 0.9:
            depth_adjustment += 1

        return max(1, min(6, base_depth + depth_adjustment))

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive status for mesh coordination."""
        if not HOLO_AVAILABLE or not self._vanta_initialized:
            return {
                "cognitive_load": 3.0,
                "symbolic_depth": 3,
                "mesh_role": "PROCESSOR",
                "vanta_initialized": False
            }

        return {
            "cognitive_load": self.calculate_cognitive_load(),
            "symbolic_depth": self.calculate_symbolic_depth(),
            "attention_entropy": self.cognitive_metrics.get("attention_entropy", 0.0),
            "computational_efficiency": self.cognitive_metrics.get("computational_efficiency", 0.0),
            "memory_usage": self.cognitive_metrics.get("memory_usage", 0.0),
            "delta_learning_active": self.cognitive_metrics.get("delta_update_magnitude", 0.0) > 0.01,
            "linear_attention_enabled": True,
            "mesh_role": "PROCESSOR",
            "vanta_initialized": self._vanta_initialized
        }

    def shutdown(self):
        """Clean shutdown of the attention module and monitoring tasks."""
        if hasattr(self, 'monitoring_task') and self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        logger.info("DeltaNetAttention shutdown complete")

# Factory function for easy instantiation
def create_deltanet_attention(d_model: int = 512, n_heads: int = 8, **kwargs) -> DeltaNetAttention:
    """Factory function to create DeltaNet attention with default configuration."""
    config = LinearAttentionConfig(
        d_model=d_model,
        n_heads=n_heads,
        **kwargs
    )
    
    attention = DeltaNetAttention(config)
    
    # Initialize HOLO-1.5 integration if available
    if HOLO_AVAILABLE:
        import asyncio
        try:
            # Try to run async init if we're in an async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(attention.async_init())
            else:
                asyncio.run(attention.async_init())
        except RuntimeError:
            # No event loop, will need to be initialized later
            logger.info("DeltaNet created, async_init() must be called manually")
    
    return attention
