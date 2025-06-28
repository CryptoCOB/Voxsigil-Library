"""
Adaptive Memory Manager for Novel LLM Paradigms
Addresses GPU budget reality and long-term memory compaction risks

Implementation of adaptive memory management with:
- Real-time GPU memory monitoring
- Dynamic paradigm allocation
- Graceful degradation strategies
- Memory compaction for bio-inspired layers
- Budget-aware resource allocation

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import asyncio
import gc
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    from ...agents.base import BaseAgent, CognitiveMeshRole, vanta_agent

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

    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def async_init(self):
            pass


class MemoryPriority(Enum):
    """Memory allocation priority levels"""

    CRITICAL = 0  # Core attention mechanisms
    HIGH = 1  # Active paradigms
    MEDIUM = 2  # Cached computations
    LOW = 3  # Historical data
    ARCHIVE = 4  # Long-term storage


class ParadigmState(Enum):
    """Paradigm operational states"""

    ACTIVE = "active"
    STANDBY = "standby"
    COMPRESSED = "compressed"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


@dataclass
class MemoryBudget:
    """GPU memory budget configuration"""

    total_gb: float = 16.0
    reserve_gb: float = 2.0  # Emergency reserve

    # Paradigm allocations (percentages of available memory)
    deltanet_percent: float = 25.0
    minicache_percent: float = 20.0
    lnu_percent: float = 15.0
    akorn_percent: float = 15.0
    splr_percent: float = 10.0
    gnn_percent: float = 10.0
    rbp_percent: float = 5.0

    # Dynamic adjustment bounds
    min_allocation_gb: float = 0.5
    max_allocation_gb: float = 8.0


@dataclass
class MemoryMetrics:
    """Real-time memory usage metrics"""

    timestamp: float = field(default_factory=time.time)
    gpu_used_gb: float = 0.0
    gpu_total_gb: float = 0.0
    gpu_utilization: float = 0.0

    # Paradigm-specific usage
    paradigm_usage: Dict[str, float] = field(default_factory=dict)

    # System health
    fragmentation_ratio: float = 0.0
    allocation_failures: int = 0
    compaction_events: int = 0

    # Performance impact
    allocation_latency_ms: float = 0.0
    deallocation_latency_ms: float = 0.0


@dataclass
class ParadigmConfig:
    """Configuration for paradigm memory management"""

    name: str
    priority: MemoryPriority
    state: ParadigmState = ParadigmState.STANDBY

    # Memory bounds
    min_memory_gb: float = 0.1
    max_memory_gb: float = 4.0
    target_memory_gb: float = 1.0

    # Degradation strategy
    can_compress: bool = True
    can_suspend: bool = True
    compression_ratio: float = 0.3  # Expected compression ratio

    # Performance requirements
    max_latency_ms: float = 100.0
    min_accuracy: float = 0.85


class ResourceOptimizer:
    """
    Optimizes resource allocation across Novel LLM paradigms.
    Implements dynamic scaling and efficiency improvements.
    """

    def __init__(self, memory_manager: "AdaptiveMemoryManager"):
        self.memory_manager = memory_manager
        self.optimization_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def optimize_allocation(
        self, paradigm_demands: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize memory allocation across paradigms based on demands and performance.

        Args:
            paradigm_demands: Dictionary of paradigm names to memory demands (MB)

        Returns:
            Optimized allocation dictionary
        """
        total_demand = sum(paradigm_demands.values())
        available_memory = self.memory_manager.get_available_memory()

        if total_demand <= available_memory:
            # Sufficient memory - allocate as requested
            return paradigm_demands.copy()

        # Insufficient memory - optimize allocation
        optimization_factor = available_memory / total_demand

        # Priority-based scaling
        priority_weights = {
            "logical_neural_units": 1.2,  # Higher priority for reasoning
            "akonr_binding": 1.1,
            "spiking_networks": 0.9,
            "deltanet_attention": 1.0,
            "adaptive_memory": 0.8,
        }

        optimized_allocation = {}
        total_weighted_demand = 0

        # Calculate weighted demands
        for paradigm, demand in paradigm_demands.items():
            weight = priority_weights.get(paradigm, 1.0)
            weighted_demand = demand * weight
            total_weighted_demand += weighted_demand

        # Allocate based on weighted priorities
        for paradigm, demand in paradigm_demands.items():
            weight = priority_weights.get(paradigm, 1.0)
            weighted_demand = demand * weight
            allocation_ratio = weighted_demand / total_weighted_demand
            optimized_allocation[paradigm] = available_memory * allocation_ratio

        # Log optimization
        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "original_demands": paradigm_demands.copy(),
                "optimized_allocation": optimized_allocation.copy(),
                "optimization_factor": optimization_factor,
                "available_memory": available_memory,
            }
        )

        self.logger.info(
            f"Optimized allocation: {optimization_factor:.2f}x scaling applied"
        )
        return optimized_allocation

    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get resource optimization efficiency metrics."""
        if not self.optimization_history:
            return {"optimization_count": 0, "avg_efficiency": 1.0}

        recent_optimizations = self.optimization_history[-10:]  # Last 10 optimizations
        avg_efficiency = sum(
            opt["optimization_factor"] for opt in recent_optimizations
        ) / len(recent_optimizations)

        return {
            "optimization_count": len(self.optimization_history),
            "avg_efficiency": avg_efficiency,
            "recent_efficiency": recent_optimizations[-1]["optimization_factor"]
            if recent_optimizations
            else 1.0,
        }


class MemoryPool:
    """Smart memory pool with compaction and defragmentation"""

    def __init__(self, size_gb: float):
        self.size_bytes = int(size_gb * 1024**3)
        self.allocated_blocks: Dict[str, Tuple[int, int]] = {}  # id -> (start, size)
        self.free_blocks: List[Tuple[int, int]] = [(0, self.size_bytes)]
        self.fragmentation_threshold = 0.3

    def allocate(self, size_bytes: int, block_id: str) -> Optional[int]:
        """Allocate memory block with defragmentation if needed"""
        # Try to find suitable free block
        for i, (start, size) in enumerate(self.free_blocks):
            if size >= size_bytes:
                # Allocate at start of free block
                self.allocated_blocks[block_id] = (start, size_bytes)

                # Update free blocks
                remaining_size = size - size_bytes
                if remaining_size > 0:
                    self.free_blocks[i] = (start + size_bytes, remaining_size)
                else:
                    del self.free_blocks[i]

                return start

        # No suitable block found - try compaction
        if self.get_fragmentation_ratio() > self.fragmentation_threshold:
            self.compact()
            return self.allocate(size_bytes, block_id)  # Retry after compaction

        return None  # Allocation failed

    def deallocate(self, block_id: str) -> bool:
        """Deallocate memory block and merge adjacent free blocks"""
        if block_id not in self.allocated_blocks:
            return False

        start, size = self.allocated_blocks[block_id]
        del self.allocated_blocks[block_id]

        # Add to free blocks and merge adjacent ones
        self.free_blocks.append((start, size))
        self.free_blocks.sort()

        # Merge adjacent free blocks
        merged = []
        for block_start, block_size in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == block_start:
                # Merge with previous block
                merged[-1] = (merged[-1][0], merged[-1][1] + block_size)
            else:
                merged.append((block_start, block_size))

        self.free_blocks = merged
        return True

    def compact(self):
        """Compact allocated blocks to reduce fragmentation"""
        if not self.allocated_blocks:
            self.free_blocks = [(0, self.size_bytes)]
            return

        # Sort allocated blocks by start position
        sorted_blocks = sorted(self.allocated_blocks.items(), key=lambda x: x[1][0])

        # Move blocks to beginning of pool
        current_pos = 0
        new_allocations = {}

        for block_id, (_, size) in sorted_blocks:
            new_allocations[block_id] = (current_pos, size)
            current_pos += size

        # Update allocations
        self.allocated_blocks = new_allocations

        # Update free space
        remaining_size = self.size_bytes - current_pos
        if remaining_size > 0:
            self.free_blocks = [(current_pos, remaining_size)]
        else:
            self.free_blocks = []

    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio"""
        if not self.free_blocks:
            return 0.0

        total_free = sum(size for _, size in self.free_blocks)
        largest_free = max(size for _, size in self.free_blocks)

        if total_free == 0:
            return 0.0

        return 1.0 - (largest_free / total_free)

    def get_usage_stats(self) -> Dict[str, float]:
        """Get memory pool usage statistics"""
        total_allocated = sum(size for _, size in self.allocated_blocks.values())
        total_free = sum(size for _, size in self.free_blocks)

        return {
            "allocated_gb": total_allocated / (1024**3),
            "free_gb": total_free / (1024**3),
            "utilization": total_allocated / self.size_bytes,
            "fragmentation": self.get_fragmentation_ratio(),
            "num_free_blocks": len(self.free_blocks),
            "num_allocated_blocks": len(self.allocated_blocks),
        }


@vanta_agent(role=CognitiveMeshRole.MONITOR)
class AdaptiveMemoryManager(BaseAgent):
    """
    Adaptive Memory Manager for Novel LLM Paradigms

    Provides real-time GPU memory monitoring, dynamic paradigm allocation,
    graceful degradation, and memory compaction for bio-inspired layers.
    """

    def __init__(self, budget: Optional[MemoryBudget] = None):
        super().__init__()
        self.budget = budget or MemoryBudget()
        self.paradigms: Dict[str, ParadigmConfig] = {}
        self.memory_pool: Optional[MemoryPool] = None
        self.metrics_history: List[MemoryMetrics] = []
        self.degradation_callbacks: Dict[str, List[Callable]] = {}

        # Monitoring state
        self.monitoring_active = False
        self.alert_thresholds = {
            "high_usage": 0.85,
            "critical_usage": 0.95,
            "fragmentation": 0.4,
        }

        # Performance tracking
        self.allocation_times: List[float] = []
        self.compaction_times: List[float] = []

        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "memory_efficiency": 0.0,
            "allocation_success_rate": 1.0,
            "paradigm_balance": 0.0,
            "degradation_frequency": 0.0,
        }

        self.logger = logging.getLogger(__name__)

    async def async_init(self):
        """Initialize memory manager and register with HOLO-1.5 mesh"""
        if HOLO_AVAILABLE:
            await super().async_init()
            await self.register_capabilities(
                [
                    "memory_monitoring",
                    "paradigm_allocation",
                    "graceful_degradation",
                    "memory_compaction",
                ]
            )

        # Initialize memory pool
        available_memory = self.budget.total_gb - self.budget.reserve_gb
        self.memory_pool = MemoryPool(available_memory)

        # Register default paradigms
        await self._register_default_paradigms()

        # Start monitoring
        await self.start_monitoring()

        self.logger.info(
            f"AdaptiveMemoryManager initialized with {available_memory:.1f}GB"
        )

    async def _register_default_paradigms(self):
        """Register default paradigm configurations"""
        paradigm_configs = [
            ParadigmConfig(
                name="deltanet",
                priority=MemoryPriority.HIGH,
                target_memory_gb=self.budget.total_gb
                * self.budget.deltanet_percent
                / 100,
                max_latency_ms=50.0,
            ),
            ParadigmConfig(
                name="minicache",
                priority=MemoryPriority.CRITICAL,
                target_memory_gb=self.budget.total_gb
                * self.budget.minicache_percent
                / 100,
                can_suspend=False,  # Critical component
            ),
            ParadigmConfig(
                name="lnu",
                priority=MemoryPriority.HIGH,
                target_memory_gb=self.budget.total_gb * self.budget.lnu_percent / 100,
                compression_ratio=0.4,
            ),
            ParadigmConfig(
                name="akorn",
                priority=MemoryPriority.MEDIUM,
                target_memory_gb=self.budget.total_gb * self.budget.akorn_percent / 100,
                compression_ratio=0.2,  # Bio-inspired layers compress well
            ),
            ParadigmConfig(
                name="splr",
                priority=MemoryPriority.MEDIUM,
                target_memory_gb=self.budget.total_gb * self.budget.splr_percent / 100,
                can_compress=True,
            ),
            ParadigmConfig(
                name="gnn",
                priority=MemoryPriority.HIGH,
                target_memory_gb=self.budget.total_gb * self.budget.gnn_percent / 100,
                max_latency_ms=75.0,
            ),
            ParadigmConfig(
                name="rbp",
                priority=MemoryPriority.LOW,
                target_memory_gb=self.budget.total_gb * self.budget.rbp_percent / 100,
                can_suspend=True,
            ),
        ]

        for config in paradigm_configs:
            await self.register_paradigm(config)

    async def register_paradigm(self, config: ParadigmConfig):
        """Register a paradigm for memory management"""
        self.paradigms[config.name] = config
        self.degradation_callbacks[config.name] = []

        self.logger.info(
            f"Registered paradigm: {config.name} "
            f"(target: {config.target_memory_gb:.1f}GB, "
            f"priority: {config.priority.name})"
        )

    def register_degradation_callback(self, paradigm: str, callback: Callable):
        """Register callback for paradigm degradation events"""
        if paradigm in self.degradation_callbacks:
            self.degradation_callbacks[paradigm].append(callback)

    async def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        self.logger.info("Memory monitoring started")

    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        self.logger.info("Memory monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]

                # Check for alerts and trigger degradation if needed
                await self._check_alerts(metrics)

                # Update cognitive metrics
                await self._update_cognitive_metrics(metrics)

                await asyncio.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Longer delay on error

    async def _collect_metrics(self) -> MemoryMetrics:
        """Collect current memory usage metrics"""
        metrics = MemoryMetrics()

        # GPU metrics
        if torch.cuda.is_available():
            metrics.gpu_used_gb = torch.cuda.memory_allocated() / (1024**3)
            metrics.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            metrics.gpu_utilization = metrics.gpu_used_gb / metrics.gpu_total_gb

        # Memory pool metrics
        if self.memory_pool:
            pool_stats = self.memory_pool.get_usage_stats()
            metrics.fragmentation_ratio = pool_stats["fragmentation"]

        # Paradigm-specific usage (would be filled by actual paradigm agents)
        for paradigm_name in self.paradigms.keys():
            # Placeholder - actual implementations would report their usage
            metrics.paradigm_usage[paradigm_name] = 0.0

        return metrics

    async def _check_alerts(self, metrics: MemoryMetrics):
        """Check for memory alerts and trigger degradation if needed"""
        # High usage alert
        if metrics.gpu_utilization > self.alert_thresholds["high_usage"]:
            await self._handle_high_usage(metrics)

        # Critical usage alert
        if metrics.gpu_utilization > self.alert_thresholds["critical_usage"]:
            await self._handle_critical_usage(metrics)

        # Fragmentation alert
        if metrics.fragmentation_ratio > self.alert_thresholds["fragmentation"]:
            await self._handle_fragmentation(metrics)

    async def _handle_high_usage(self, metrics: MemoryMetrics):
        """Handle high memory usage situation"""
        self.logger.warning(
            f"High memory usage detected: {metrics.gpu_utilization:.1%}"
        )

        # Try to compress low-priority paradigms
        for paradigm_name, config in self.paradigms.items():
            if config.priority.value >= 2 and config.can_compress:
                await self._compress_paradigm(paradigm_name)

    async def _handle_critical_usage(self, metrics: MemoryMetrics):
        """Handle critical memory usage situation"""
        self.logger.error(
            f"Critical memory usage detected: {metrics.gpu_utilization:.1%}"
        )

        # Suspend low-priority paradigms
        for paradigm_name, config in self.paradigms.items():
            if config.priority.value >= 3 and config.can_suspend:
                await self._suspend_paradigm(paradigm_name)

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    async def _handle_fragmentation(self, metrics: MemoryMetrics):
        """Handle memory fragmentation"""
        self.logger.info(
            f"Memory fragmentation detected: {metrics.fragmentation_ratio:.1%}"
        )

        if self.memory_pool:
            start_time = time.time()
            self.memory_pool.compact()
            compaction_time = (time.time() - start_time) * 1000
            self.compaction_times.append(compaction_time)

            self.logger.info(f"Memory compaction completed in {compaction_time:.1f}ms")

    async def _compress_paradigm(self, paradigm_name: str):
        """Compress a paradigm to reduce memory usage"""
        config = self.paradigms[paradigm_name]
        if config.state == ParadigmState.COMPRESSED:
            return

        # Notify paradigm of compression request
        callbacks = self.degradation_callbacks.get(paradigm_name, [])
        for callback in callbacks:
            try:
                await callback("compress")
            except Exception as e:
                self.logger.error(
                    f"Error in compression callback for {paradigm_name}: {e}"
                )

        config.state = ParadigmState.COMPRESSED
        self.logger.info(f"Compressed paradigm: {paradigm_name}")

    async def _suspend_paradigm(self, paradigm_name: str):
        """Suspend a paradigm to free memory"""
        config = self.paradigms[paradigm_name]
        if config.state == ParadigmState.SUSPENDED:
            return

        # Notify paradigm of suspension request
        callbacks = self.degradation_callbacks.get(paradigm_name, [])
        for callback in callbacks:
            try:
                await callback("suspend")
            except Exception as e:
                self.logger.error(
                    f"Error in suspension callback for {paradigm_name}: {e}"
                )

        config.state = ParadigmState.SUSPENDED
        self.logger.info(f"Suspended paradigm: {paradigm_name}")

    async def _update_cognitive_metrics(self, metrics: MemoryMetrics):
        """Update cognitive metrics for HOLO-1.5 mesh coordination"""
        # Memory efficiency: how well we're using available memory
        if metrics.gpu_total_gb > 0:
            self.cognitive_metrics["memory_efficiency"] = min(
                metrics.gpu_utilization / self.alert_thresholds["high_usage"], 1.0
            )

        # Allocation success rate
        recent_failures = sum(
            1 for m in self.metrics_history[-60:] if m.allocation_failures > 0
        )
        self.cognitive_metrics["allocation_success_rate"] = max(
            1.0 - recent_failures / 60.0, 0.0
        )

        # Paradigm balance: how evenly memory is distributed
        if metrics.paradigm_usage:
            usage_values = list(metrics.paradigm_usage.values())
            if usage_values:
                usage_variance = sum(
                    (x - sum(usage_values) / len(usage_values)) ** 2
                    for x in usage_values
                ) / len(usage_values)
                self.cognitive_metrics["paradigm_balance"] = max(
                    0.0, 1.0 - usage_variance
                )

        # Degradation frequency: how often we need to degrade performance
        recent_degradations = sum(
            1 for m in self.metrics_history[-300:] if m.compaction_events > 0
        )
        self.cognitive_metrics["degradation_frequency"] = recent_degradations / 300.0

    async def allocate_paradigm_memory(
        self, paradigm_name: str, size_gb: float
    ) -> Optional[str]:
        """Allocate memory for a paradigm"""
        if paradigm_name not in self.paradigms:
            self.logger.error(f"Unknown paradigm: {paradigm_name}")
            return None

        if not self.memory_pool:
            self.logger.error("Memory pool not initialized")
            return None

        size_bytes = int(size_gb * 1024**3)
        block_id = f"{paradigm_name}_{int(time.time() * 1000)}"

        start_time = time.time()
        allocation_start = self.memory_pool.allocate(size_bytes, block_id)
        allocation_time = (time.time() - start_time) * 1000
        self.allocation_times.append(allocation_time)

        if allocation_start is not None:
            self.logger.debug(
                f"Allocated {size_gb:.1f}GB for {paradigm_name} "
                f"in {allocation_time:.1f}ms"
            )
            return block_id
        else:
            self.logger.warning(
                f"Failed to allocate {size_gb:.1f}GB for {paradigm_name}"
            )
            return None

    async def deallocate_paradigm_memory(self, block_id: str) -> bool:
        """Deallocate paradigm memory"""
        if not self.memory_pool:
            return False

        start_time = time.time()
        success = self.memory_pool.deallocate(block_id)
        deallocation_time = (time.time() - start_time) * 1000

        if success:
            self.logger.debug(
                f"Deallocated block {block_id} in {deallocation_time:.1f}ms"
            )

        return success

    async def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status"""
        current_metrics = await self._collect_metrics()
        pool_stats = self.memory_pool.get_usage_stats() if self.memory_pool else {}

        paradigm_states = {
            name: {
                "state": config.state.value,
                "priority": config.priority.name,
                "target_memory_gb": config.target_memory_gb,
                "usage_gb": current_metrics.paradigm_usage.get(name, 0.0),
            }
            for name, config in self.paradigms.items()
        }

        return {
            "timestamp": current_metrics.timestamp,
            "gpu_metrics": {
                "used_gb": current_metrics.gpu_used_gb,
                "total_gb": current_metrics.gpu_total_gb,
                "utilization": current_metrics.gpu_utilization,
            },
            "pool_stats": pool_stats,
            "paradigm_states": paradigm_states,
            "cognitive_metrics": self.cognitive_metrics,
            "performance": {
                "avg_allocation_time_ms": sum(self.allocation_times[-100:])
                / len(self.allocation_times[-100:])
                if self.allocation_times
                else 0.0,
                "avg_compaction_time_ms": sum(self.compaction_times[-10:])
                / len(self.compaction_times[-10:])
                if self.compaction_times
                else 0.0,
            },
        }

    async def get_cognitive_load(self) -> float:
        """Calculate current cognitive load for HOLO-1.5"""
        if not self.metrics_history:
            return 0.0

        latest_metrics = self.metrics_history[-1]

        # Base load from memory utilization
        memory_load = latest_metrics.gpu_utilization

        # Additional load from fragmentation
        fragmentation_load = latest_metrics.fragmentation_ratio * 0.3

        # Load from recent degradation events
        recent_degradations = sum(
            1 for m in self.metrics_history[-60:] if m.compaction_events > 0
        )
        degradation_load = min(recent_degradations / 10.0, 0.5)

        return min(memory_load + fragmentation_load + degradation_load, 1.0)

    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # Memory management has moderate symbolic depth
        # - Resource allocation decisions
        # - Priority-based scheduling
        # - Degradation strategy selection
        return 3

    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "AdaptiveMemoryManager",
            "cognitive_metrics": self.cognitive_metrics,
            "memory_status": await self.get_memory_status(),
            "paradigm_count": len(self.paradigms),
            "monitoring_active": self.monitoring_active,
        }

    @asynccontextmanager
    async def memory_context(self, paradigm_name: str, size_gb: float):
        """Context manager for paradigm memory allocation"""
        block_id = await self.allocate_paradigm_memory(paradigm_name, size_gb)
        try:
            yield block_id
        finally:
            if block_id:
                await self.deallocate_paradigm_memory(block_id)

    async def save_configuration(self, filepath: Path):
        """Save memory manager configuration"""
        config_data = {
            "budget": {
                "total_gb": self.budget.total_gb,
                "reserve_gb": self.budget.reserve_gb,
                "allocations": {
                    "deltanet_percent": self.budget.deltanet_percent,
                    "minicache_percent": self.budget.minicache_percent,
                    "lnu_percent": self.budget.lnu_percent,
                    "akorn_percent": self.budget.akorn_percent,
                    "splr_percent": self.budget.splr_percent,
                    "gnn_percent": self.budget.gnn_percent,
                    "rbp_percent": self.budget.rbp_percent,
                },
            },
            "paradigms": {
                name: {
                    "priority": config.priority.name,
                    "target_memory_gb": config.target_memory_gb,
                    "can_compress": config.can_compress,
                    "can_suspend": config.can_suspend,
                    "compression_ratio": config.compression_ratio,
                }
                for name, config in self.paradigms.items()
            },
            "alert_thresholds": self.alert_thresholds,
        }

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Configuration saved to {filepath}")

    async def load_configuration(self, filepath: Path):
        """Load memory manager configuration"""
        with open(filepath, "r") as f:
            config_data = json.load(f)

        # Update budget
        budget_data = config_data.get("budget", {})
        if budget_data:
            self.budget.total_gb = budget_data.get("total_gb", self.budget.total_gb)
            self.budget.reserve_gb = budget_data.get(
                "reserve_gb", self.budget.reserve_gb
            )

            allocations = budget_data.get("allocations", {})
            for key, value in allocations.items():
                if hasattr(self.budget, key):
                    setattr(self.budget, key, value)

        # Update alert thresholds
        if "alert_thresholds" in config_data:
            self.alert_thresholds.update(config_data["alert_thresholds"])

        self.logger.info(f"Configuration loaded from {filepath}")


# Factory function for easy instantiation
async def create_adaptive_memory_manager(
    total_gpu_gb: float = 16.0, reserve_gb: float = 2.0
) -> AdaptiveMemoryManager:
    """Factory function to create and initialize AdaptiveMemoryManager"""
    budget = MemoryBudget(total_gb=total_gpu_gb, reserve_gb=reserve_gb)
    manager = AdaptiveMemoryManager(budget)
    await manager.async_init()
    return manager


# Export main classes
__all__ = [
    "AdaptiveMemoryManager",
    "MemoryBudget",
    "MemoryMetrics",
    "ParadigmConfig",
    "MemoryPriority",
    "ParadigmState",
    "MemoryPool",
    "ResourceOptimizer",
    "create_adaptive_memory_manager",
]
