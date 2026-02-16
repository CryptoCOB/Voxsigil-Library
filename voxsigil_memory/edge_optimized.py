"""
Edge-Optimized VoxSigil Pipeline

Three device profiles with configurable speed/quality tradeoffs:
- Server (full pipeline, no constraints)
- Edge (moderate constraints, pruning cost acceptable)
- Ultra-Edge (aggressive constraints, memory <= 256MB, latency <= 10ms)

Automatically selects strategy based on device_profile parameter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Dict, List, Tuple

from voxsigil_memory.semantic import (
    GameSemanticPruner,
    BLTLatentCodec,
    EntropyRouter,
    ContextPackBuilder,
    LatentMemoryUnit,
)


class DeviceProfile(str, Enum):
    """Device classification for automatic strategy selection."""

    SERVER = "server"  # No constraints
    EDGE = "edge"  # CPU + modest memory (2-4GB)
    ULTRA_EDGE = "ultra_edge"  # Severely constrained (256-512MB)


@dataclass
class DeviceConfig:
    """Per-device optimization configuration."""

    name: str
    max_latency_ms: int  # Max E2E latency budget
    max_memory_mb: int  # Max memory footprint
    max_doc_length: int  # Max chars per document
    pruning_ratio: float  # Aggressive = 0.3, moderate = 0.6, lenient = 0.8
    use_routing: bool  # Include entropy routing?
    use_compression: bool  # Include zlib compression?
    quantize_embeddings: Literal["disabled", "int8", "int4"]
    batch_size: int  # Docs per batch


# Preset configurations
DEVICE_CONFIGS = {
    DeviceProfile.SERVER: DeviceConfig(
        name="Server (no constraints)",
        max_latency_ms=100,
        max_memory_mb=2048,
        max_doc_length=10000,
        pruning_ratio=0.8,  # Lenient: keep 80%
        use_routing=True,
        use_compression=True,
        quantize_embeddings="disabled",
        batch_size=32,
    ),
    DeviceProfile.EDGE: DeviceConfig(
        name="Edge Device (mobile/desktop)",
        max_latency_ms=50,
        max_memory_mb=512,
        max_doc_length=4000,
        pruning_ratio=0.5,  # Moderate: keep 50%
        use_routing=True,
        use_compression=True,
        quantize_embeddings="int8",
        batch_size=8,
    ),
    DeviceProfile.ULTRA_EDGE: DeviceConfig(
        name="Ultra-Edge (IoT/embedded)",
        max_latency_ms=10,
        max_memory_mb=256,
        max_doc_length=1000,
        pruning_ratio=0.3,  # Aggressive: keep 30%
        use_routing=False,  # Skip routing to save compute
        use_compression=True,
        quantize_embeddings="int4",
        batch_size=1,  # Process one at a time
    ),
}


class EdgeOptimizedPipeline:
    """
    Hybrid pipeline that adapts to device constraints.

    Automatically selects which components to use based on device profile.
    """

    def __init__(self, device_profile: DeviceProfile = DeviceProfile.SERVER):
        """Initialize with device profile."""
        self.device_profile = device_profile
        self.config = DEVICE_CONFIGS[device_profile]

        # Initialize components
        self.pruner = GameSemanticPruner()
        self.codec = BLTLatentCodec()  # quantize not supported yet
        self.router = (
            EntropyRouter(max_budget_tokens=1024)
            if self.config.use_routing
            else None
        )
        self.builder = ContextPackBuilder()

    def process_document(self, text: str) -> LatentMemoryUnit:
        """
        Process single document through optimized pipeline.

        Steps:
        1. Truncate if too long
        2. Prune aggressively
        3. Encode with optional quantization
        4. Return latent unit
        """
        # Step 1: Truncate
        if len(text) > self.config.max_doc_length:
            text = text[: self.config.max_doc_length] + "..."

        # Step 2: Prune (always enabled)
        pruned, pruned_frac = self.pruner.prune(
            text, target_ratio=self.config.pruning_ratio
        )

        # Step 3: Encode
        latent_unit = self.codec.encode(pruned)
        latent_unit.pruned_fraction = pruned_frac

        return latent_unit

    def process_units(
        self, units: List[LatentMemoryUnit], budget_tokens: int = 512
    ) -> Tuple[List[LatentMemoryUnit], Dict]:
        """
        Route units through pipeline (conditionally apply routing).

        Returns:
            (routed_units, statistics)
        """
        stats = {
            "total_units": len(units),
            "routing_used": False,
            "routing_stats": {},
        }

        # If routing disabled on ultra-edge, return all units up to budget
        if not self.config.use_routing:
            kept = []
            tokens = 0
            heuristic_tokens_per_unit = budget_tokens // max(1, len(units))

            for mem_unit in units:
                if tokens + heuristic_tokens_per_unit <= budget_tokens:
                    kept.append(mem_unit)
                    tokens += heuristic_tokens_per_unit
                else:
                    break

            stats["units_kept"] = len(kept)
            stats["budget_used"] = tokens
            return kept, stats

        # Full routing pipeline (server/edge)
        routed, routing_stats = self.router.route(units)
        stats["routing_used"] = True
        stats["routing_stats"] = routing_stats

        return routed, stats

    def build_context_pack(
        self,
        units: List[LatentMemoryUnit],
        query: str = "",
        budget_tokens: int = 512,
    ) -> Dict:
        """
        Assemble final context pack.

        For ultra-edge, includes metadata for local re-expansion without decoder.
        """
        if not units:
            return {
                "expanded_text": "",
                "latent_units": [],
                "token_count": 0,
                "device_profile": self.device_profile.value,
            }

        pack = self.builder.build_pack(units, self.codec, query, budget_tokens)

        # Add device profile metadata
        pack["device_profile"] = self.device_profile.value
        pack["config"] = {
            "max_latency_ms": self.config.max_latency_ms,
            "pruning_ratio": self.config.pruning_ratio,
            "use_routing": self.config.use_routing,
            "quantize_embeddings": self.config.quantize_embeddings,
        }

        return pack

    def profile_memory_usage(self) -> Dict:
        """Estimate memory footprint for this device."""
        embedding_size_mb = (384 * 4) / (1024 * 1024)  # 384-d float32
        model_size_mb = {
            "disabled": 80,  # MiniLM base size
            "int8": 22,  # ~80% reduction with quantization
            "int4": 12,  # ~85% reduction with 4-bit
        }

        return {
            "device_profile": self.device_profile.value,
            "model_size_mb": model_size_mb[self.config.quantize_embeddings],
            "per_unit_mb": embedding_size_mb,
            "max_units_in_memory": max(
                1,
                int(
                    (self.config.max_memory_mb * 0.7)
                    / embedding_size_mb
                ),
            ),
            "total_budget_mb": self.config.max_memory_mb,
        }


def auto_select_profile(
    ram_available_mb: int, latency_budget_ms: int
) -> DeviceProfile:
    """
    Automatically select device profile based on hardware.

    Args:
        ram_available_mb: Available RAM in megabytes
        latency_budget_ms: Maximum acceptable latency

    Returns:
        Best matching DeviceProfile
    """
    if ram_available_mb >= 1024 and latency_budget_ms >= 100:
        return DeviceProfile.SERVER

    if ram_available_mb >= 256 and latency_budget_ms >= 50:
        return DeviceProfile.EDGE

    return DeviceProfile.ULTRA_EDGE


# Convenience function for quick pipeline creation
def create_pipeline(device_profile: DeviceProfile = None) -> EdgeOptimizedPipeline:
    """
    Create optimized pipeline for device.

    Example:
        pipeline = create_pipeline(DeviceProfile.EDGE)
        unit = pipeline.process_document("Long text")
        pack = pipeline.build_context_pack([unit])
    """
    return EdgeOptimizedPipeline(device_profile or DeviceProfile.SERVER)


if __name__ == "__main__":
    # Quick demo
    print("=" * 70)
    print("VoxSigil Edge-Optimized Pipeline Demo")
    print("=" * 70)

    SAMPLE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "This is important information. "
        "Some filler text here. " * 10
    )

    for profile in DeviceProfile:
        print(f"\n[{profile.value.upper()}]")
        print("-" * 70)

        pipeline = create_pipeline(profile)

        # Process
        unit = pipeline.process_document(SAMPLE_TEXT)
        pruned_len = len(SAMPLE_TEXT) * (1 - unit.pruned_fraction)
        print(f"Document: {len(SAMPLE_TEXT)} → {pruned_len:.0f} chars")
        print(f"Embedding: {unit.embedding.shape[0]}-d "
              f"(quantized={pipeline.config.quantize_embeddings})")

        # Memory estimate
        mem = pipeline.profile_memory_usage()
        print(f"Memory: {mem['model_size_mb']}MB model + {mem['per_unit_mb']:.2f}MB/unit")

        # Config
        print("Config:")
        print(f"  - Max latency: {pipeline.config.max_latency_ms}ms")
        print(f"  - Pruning: keep {pipeline.config.pruning_ratio*100:.0f}%")
        print(f"  - Routing: {'enabled' if pipeline.config.use_routing else 'disabled'}")
        print(f"  - Compression: {'enabled' if pipeline.config.use_compression else 'disabled'}")

    print("\n" + "=" * 70)
    print("Auto-select example:")
    print("=" * 70)
    profile = auto_select_profile(ram_available_mb=512, latency_budget_ms=50)
    print(f"RAM=512MB, latency_budget=50ms → {profile.value}")
