#!/usr/bin/env python3
"""
Hybrid Multimodal Student Architecture
Shared trunk + modality-specific adapters + lightweight BLT integration
Optimized for 3×RTX 3060 (12GB each) setup
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Import our modality-aware NAS
from research.nas_search_space import (
    get_modality_optimized_architecture,
    create_multimodal_architecture,
    MODALITY_REGISTRY
)

# Import BLT student interface
from modules.blt_student_interface import BLTStudentInterface

logger = logging.getLogger(__name__)

@dataclass
class ModalityAdapterConfig:
    """Configuration for modality-specific adapters"""
    modality: str
    adapter_type: str  # "lora", "conv_stem", "cross_attn", etc.
    rank: int = 16  # For LoRA adapters
    width: int = 512  # For conv/linear adapters
    dropout: float = 0.1
    targets: List[str] = None  # Which layers to adapt

class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for efficient fine-tuning"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

class ModalityAdapter(nn.Module):
    """Modality-specific adapter wrapper"""
    
    def __init__(self, config: ModalityAdapterConfig, trunk_dim: int):
        super().__init__()
        self.config = config
        self.modality = config.modality
        
        if config.adapter_type == "lora":
            self.adapter = LoRAAdapter(trunk_dim, trunk_dim, rank=config.rank)
        elif config.adapter_type == "conv_stem":
            # For image/audio input preprocessing
            if config.modality == "image":
                self.adapter = nn.Sequential(
                    nn.Conv2d(3, config.width, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(config.width),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(config.width, trunk_dim)
                )
            elif config.modality == "audio":
                self.adapter = nn.Sequential(
                    nn.Conv1d(1, config.width, kernel_size=3, padding=1),
                    nn.BatchNorm1d(config.width),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(config.width, trunk_dim)
                )
            else:
                self.adapter = nn.Linear(config.width, trunk_dim)
        else:
            # Default linear adapter
            self.adapter = nn.Linear(trunk_dim, trunk_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

class HybridMultimodalStudent(nn.Module):
    """
    Hybrid multimodal student with shared trunk + modality adapters
    Integrates with modality-aware NAS and BLT interface
    """
    
    def __init__(
        self,
        trunk_architecture: Dict[str, Any],
        adapters_config: List[ModalityAdapterConfig],
        heads_config: Dict[str, Dict[str, Any]],
        blt_interface: Optional[BLTStudentInterface] = None,
        enable_lazy_loading: bool = True
    ):
        super().__init__()
        
        self.trunk_architecture = trunk_architecture
        self.enable_lazy_loading = enable_lazy_loading
        self.current_modality = None
        self.loaded_adapters = {}
        self.loaded_heads = {}
        
        # Build shared trunk from NAS architecture
        self.trunk = self._build_trunk(trunk_architecture)
        self.trunk_dim = trunk_architecture.get('modality_config', {}).get('input_dim', 768)
        
        # Store adapter and head configurations
        self.adapters_config = {cfg.modality: cfg for cfg in adapters_config}
        self.heads_config = heads_config
        
        # Initialize BLT interface
        self.blt = blt_interface or BLTStudentInterface()
        
        # Lazy loading containers
        self.adapters = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        
        # Load default adapters if not lazy loading
        if not enable_lazy_loading:
            self._load_all_adapters()
            self._load_all_heads()
        
        logger.info("✅ HybridMultimodalStudent initialized:")
        logger.info(f"   • Trunk: {trunk_architecture['num_layers']} layers")
        logger.info(f"   • Adapters: {list(self.adapters_config.keys())}")
        logger.info(f"   • Heads: {list(self.heads_config.keys())}")
        logger.info(f"   • Lazy loading: {enable_lazy_loading}")
    
    def _build_trunk(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build shared trunk from NAS architecture"""
        from research.nas_search_space import NASSearchSpace
        
        search_space = NASSearchSpace()
        trunk = search_space.build_model(architecture, device="cpu")  # Build on CPU first
        
        # Add gradient checkpointing for memory efficiency
        if hasattr(trunk, 'gradient_checkpointing_enable'):
            trunk.gradient_checkpointing_enable()
        
        return trunk
    
    def _load_adapter(self, modality: str):
        """Load adapter for specific modality"""
        if modality in self.loaded_adapters:
            return
        
        if modality not in self.adapters_config:
            logger.warning(f"No adapter config for modality: {modality}")
            return
        
        config = self.adapters_config[modality]
        adapter = ModalityAdapter(config, self.trunk_dim)
        
        self.adapters[modality] = adapter
        self.loaded_adapters[modality] = True
        
        logger.info(f"📥 Loaded {modality} adapter ({config.adapter_type})")
    
    def _load_head(self, modality: str):
        """Load output head for specific modality"""
        if modality in self.loaded_heads:
            return
        
        if modality not in self.heads_config:
            logger.warning(f"No head config for modality: {modality}")
            return
        
        head_config = self.heads_config[modality]
        head_type = head_config.get('type', 'linear')
        
        if head_type == 'lm':
            # Language modeling head
            vocab_size = head_config.get('vocab_size', 32000)
            self.heads[modality] = nn.Linear(self.trunk_dim, vocab_size)
        elif head_type == 'image_latent':
            # Image latent space projection
            latent_dim = head_config.get('latent_dim', 512)
            self.heads[modality] = nn.Sequential(
                nn.Linear(self.trunk_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, latent_dim)
            )
        elif head_type == 'audio_codec':
            # Audio codec output
            codec_dim = head_config.get('codec_dim', 1024)
            self.heads[modality] = nn.Linear(self.trunk_dim, codec_dim)
        else:
            # Default linear head
            output_dim = head_config.get('output_dim', self.trunk_dim)
            self.heads[modality] = nn.Linear(self.trunk_dim, output_dim)
        
        self.loaded_heads[modality] = True
        logger.info(f"📥 Loaded {modality} head ({head_type})")
    
    def _unload_adapter(self, modality: str):
        """Unload adapter to free memory"""
        if modality in self.adapters:
            del self.adapters[modality]
            self.loaded_adapters.pop(modality, None)
            torch.cuda.empty_cache()
            logger.info(f"📤 Unloaded {modality} adapter")
    
    def _unload_head(self, modality: str):
        """Unload head to free memory"""
        if modality in self.heads:
            del self.heads[modality]
            self.loaded_heads.pop(modality, None)
            torch.cuda.empty_cache()
            logger.info(f"📤 Unloaded {modality} head")
    
    def _load_all_adapters(self):
        """Load all adapters (for non-lazy loading)"""
        for modality in self.adapters_config:
            self._load_adapter(modality)
    
    def _load_all_heads(self):
        """Load all heads (for non-lazy loading)"""
        for modality in self.heads_config:
            self._load_head(modality)
    
    def set_modality(self, modality: str, unload_others: bool = True):
        """Set current modality and manage adapter/head loading"""
        if self.current_modality == modality:
            return
        
        # Unload other modalities if requested (for memory efficiency)
        if unload_others and self.enable_lazy_loading:
            for other_modality in list(self.loaded_adapters.keys()):
                if other_modality != modality:
                    self._unload_adapter(other_modality)
            
            for other_modality in list(self.loaded_heads.keys()):
                if other_modality != modality:
                    self._unload_head(other_modality)
        
        # Load required components for new modality
        if self.enable_lazy_loading:
            self._load_adapter(modality)
            self._load_head(modality)
        
        self.current_modality = modality
        logger.info(f"🔄 Switched to modality: {modality}")
    
    async def forward(
        self,
        x: torch.Tensor,
        modality: str,
        task_type: str = "inference",
        log_interaction: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with modality awareness and BLT integration
        
        Args:
            x: Input tensor
            modality: Input modality (text, image, audio, etc.)
            task_type: Task type for BLT compression
            log_interaction: Whether to log to BLT interface
            
        Returns:
            Model output
        """
        # Set modality and load required components
        self.set_modality(modality)
        
        # BLT compression (optional)
        if self.blt and isinstance(x, str):
            compressed = await self.blt.compress_input(x, modality)
            x = torch.tensor(compressed, dtype=torch.float32).unsqueeze(0)
        
        # Apply modality adapter if available
        if modality in self.adapters:
            x = self.adapters[modality](x)
        
        # Pass through shared trunk
        trunk_output = self.trunk(x)
        
        # Apply modality head if available
        if modality in self.heads:
            output = self.heads[modality](trunk_output)
        else:
            output = trunk_output
        
        # BLT expansion (optional)
        if self.blt:
            expanded_output = await self.blt.expand_output(output, modality)
            
            # Log interaction
            if log_interaction:
                self.blt.log_interaction(x, expanded_output, importance=0.5)
            
            return expanded_output
        
        return output
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage stats"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        return {
            "gpu_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "loaded_adapters": list(self.loaded_adapters.keys()),
            "loaded_heads": list(self.loaded_heads.keys()),
            "current_modality": self.current_modality
        }
    
    def save_checkpoint(self, path: Union[str, Path], save_adapters: bool = True, save_heads: bool = True):
        """Save model checkpoint with optional adapter/head saving"""
        checkpoint = {
            "trunk_state_dict": self.trunk.state_dict(),
            "trunk_architecture": self.trunk_architecture,
            "adapters_config": self.adapters_config,
            "heads_config": self.heads_config,
            "current_modality": self.current_modality
        }
        
        if save_adapters and self.adapters:
            checkpoint["adapters_state_dict"] = {
                modality: adapter.state_dict()
                for modality, adapter in self.adapters.items()
            }
        
        if save_heads and self.heads:
            checkpoint["heads_state_dict"] = {
                modality: head.state_dict()
                for modality, head in self.heads.items()
            }
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], blt_interface: Optional[BLTStudentInterface] = None):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location="cpu")
        
        # Reconstruct adapter configs
        adapters_config = []
        for modality, config_dict in checkpoint["adapters_config"].items():
            if hasattr(config_dict, 'modality'):  # If it's already a config object
                adapters_config.append(config_dict)
            else:  # If it's a dictionary
                config = ModalityAdapterConfig(
                    modality=modality,
                    adapter_type=config_dict.get('adapter_type', 'lora'),
                    rank=config_dict.get('rank', 16),
                    width=config_dict.get('width', 512),
                    dropout=config_dict.get('dropout', 0.1),
                    targets=config_dict.get('targets', None)
                )
                adapters_config.append(config)
        
        # Create model
        model = cls(
            trunk_architecture=checkpoint["trunk_architecture"],
            adapters_config=adapters_config,
            heads_config=checkpoint["heads_config"],
            blt_interface=blt_interface
        )
        
        # Load state dicts
        model.trunk.load_state_dict(checkpoint["trunk_state_dict"])
        
        if "adapters_state_dict" in checkpoint:
            for modality, state_dict in checkpoint["adapters_state_dict"].items():
                model._load_adapter(modality)
                model.adapters[modality].load_state_dict(state_dict)
        
        if "heads_state_dict" in checkpoint:
            for modality, state_dict in checkpoint["heads_state_dict"].items():
                model._load_head(modality)
                model.heads[modality].load_state_dict(state_dict)
        
        model.current_modality = checkpoint.get("current_modality")
        
        logger.info(f"📂 Loaded checkpoint from {path}")
        return model

# Utility functions for easy model creation

def create_hybrid_student(
    complexity: str = "medium",
    modalities: List[str] = ["text", "image", "audio"],
    enable_blt: bool = True,
    enable_lazy_loading: bool = True
) -> HybridMultimodalStudent:
    """
    Create a hybrid multimodal student optimized for 3×RTX 3060
    
    Args:
        complexity: Model complexity (small, medium, large)
        modalities: List of modalities to support
        enable_blt: Whether to enable BLT interface
        enable_lazy_loading: Whether to use lazy loading for memory efficiency
        
    Returns:
        Configured HybridMultimodalStudent
    """
    # Create multimodal trunk architecture
    trunk_arch = create_multimodal_architecture(
        primary_modality="text",
        secondary_modalities=[m for m in modalities if m != "text"],
        fusion_strategy="attention"
    )
    
    # Create adapter configurations
    adapters_config = []
    for modality in modalities:
        if modality in ["text", "code"]:
            config = ModalityAdapterConfig(
                modality=modality,
                adapter_type="lora",
                rank=16,
                targets=["q", "k", "v", "o", "mlp"]
            )
        elif modality == "image":
            config = ModalityAdapterConfig(
                modality=modality,
                adapter_type="conv_stem",
                width=512
            )
        elif modality == "audio":
            config = ModalityAdapterConfig(
                modality=modality,
                adapter_type="conv_stem",
                width=512
            )
        else:
            config = ModalityAdapterConfig(
                modality=modality,
                adapter_type="lora",
                rank=16
            )
        
        adapters_config.append(config)
    
    # Create head configurations
    heads_config = {}
    for modality in modalities:
        if modality in ["text", "code"]:
            heads_config[modality] = {"type": "lm", "vocab_size": 32000}
        elif modality == "image":
            heads_config[modality] = {"type": "image_latent", "latent_dim": 512}
        elif modality == "audio":
            heads_config[modality] = {"type": "audio_codec", "codec_dim": 1024}
        else:
            heads_config[modality] = {"type": "linear", "output_dim": 768}
    
    # Create BLT interface if enabled
    blt_interface = None
    if enable_blt:
        from modules.blt_student_interface import create_blt_interface
        blt_interface = create_blt_interface(modality="multimodal", lightweight=True)
    
    # Create model
    model = HybridMultimodalStudent(
        trunk_architecture=trunk_arch,
        adapters_config=adapters_config,
        heads_config=heads_config,
        blt_interface=blt_interface,
        enable_lazy_loading=enable_lazy_loading
    )
    
    return model