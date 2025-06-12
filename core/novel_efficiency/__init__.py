"""
Novel Efficiency Components for VoxSigil Library

This module implements memory optimization, architectural efficiency, and data management paradigms:
- MiniCache: KV cache compression with outlier token detection
- DeltaNet: Linear attention mechanisms for O(L) complexity  
- Adaptive Memory: Dynamic memory management and resource allocation
- Dataset Manager: Licensing compliance and data governance for academic datasets

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh integration.
"""

from .minicache import MiniCacheWrapper, KVCacheCompressor, OutlierTokenDetector
from .deltanet_attention import DeltaNetAttention, LinearAttentionConfig, DeltaRuleOperator
from .adaptive_memory import AdaptiveMemoryManager, MemoryPool, ResourceOptimizer
from .dataset_manager import (
    DatasetManager, DatasetRegistry, DatasetMetadata, DatasetLicense,
    LogicalGroundTruth, LicenseType, DatasetType, ComplianceLevel,
    create_dataset_manager
)
try:
    from .minicache_blt import BLTMiniCacheWrapper, SemanticHashCache
    BLT_AVAILABLE = True
except ImportError:
    BLT_AVAILABLE = False

__all__ = [
    # MiniCache Components
    "MiniCacheWrapper",
    "KVCacheCompressor", 
    "OutlierTokenDetector",
    
    # DeltaNet Components
    "DeltaNetAttention",
    "LinearAttentionConfig",
    "DeltaRuleOperator",
    
    # Adaptive Memory Components
    "AdaptiveMemoryManager",
    "MemoryPool",
    "ResourceOptimizer",
    
    # Dataset Management Components
    "DatasetManager",
    "DatasetRegistry",
    "DatasetMetadata",
    "DatasetLicense",
    "LogicalGroundTruth",
    "LicenseType",
    "DatasetType",
    "ComplianceLevel",
    "create_dataset_manager"
]

# Conditionally add BLT components if available
if BLT_AVAILABLE:
    __all__.extend([
        "BLTMiniCacheWrapper",
        "SemanticHashCache"
    ])

# Version and compatibility info
__version__ = "1.0.0"
__holo_compatible__ = "1.5.0"
__paradigms__ = [
    "memory_optimization",
    "architectural_efficiency", 
    "linear_attention",
    "adaptive_resource_management",
    "dataset_management",
    "licensing_compliance"
]
