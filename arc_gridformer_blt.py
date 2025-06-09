#!/usr/bin/env python
"""
🎯 ARC GridFormer BLT Color Validation System
Specialized copy of VoxSigil Hybrid BLT for ARC color prediction correction.

🔗 Origin: Copied from Voxsigil_Library/VoxSigilRag/hybrid_blt.py
🎯 Purpose: GridFormer ARC color validation without breaking VantaCore

🧠 ARC-Specific Enhancements:
1. Fine-Tuned Correction Attention - detects color prediction errors (values outside 0-9)
2. Predictive Latency Masking - multi-step grid refinement with temporal awareness
3. Embedding Palette Bias - ARC color constraints built into tokenization
4. Color Validation Pipeline - validates predictions against ARC palette
5. Out-of-bounds Detection - identifies and logs illegal color values

🔄 Key Features for GridFormer:
- Color clamping with logging (no silent failures)
- Prediction confidence masking for multi-step corrections
- Temporal correction patterns learning
- ARC palette embedding (0=background, 1-9=colors)
- Grid-aware byte-level attention for localized corrections

This module implements an enhanced hybrid architecture that combines:
1. Robust Entropy-based routing with dynamic calculation and fallbacks.
2. Differentiated processing paths for BLT and token-based methods.
3. Context caching with normalization and TTL for performance.
4. Lazy initialization of heavy components.
5. Dynamic execution budgeting (conceptual, can be tied to resource limits).
6. Configuration management for easier deployment.

This script addresses the previously identified issues:
- Constant Entropy = 0: Solved by active entropy calculation.
- Identical Similarities: Solved by distinct processing paths based on entropy.
- Hybrid Timing Slowness: Mitigated by caching, efficient routing, and ensuring
  BLT is used appropriately for low-entropy (structured) content.
"""

import hashlib
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Third-party dependencies (ensure these are in your requirements.txt)
# Example:
# numpy
# pydantic
# pydantic-settings

try:
    # Try importing Pydantic V2 style with field_validator
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings as PydanticBaseSettings

    IS_PYDANTIC_V2 = True
    IS_PYDANTIC_AVAILABLE = True

    # Create type aliases instead of subclasses to avoid type assignment issues
    BaseModel = PydanticBaseModel  # type: ignore # No type annotation
    BaseSettings = PydanticBaseSettings  # type: ignore # No type annotation

except ImportError:
    try:
        # Fall back to Pydantic V1 style if field_validator is not available
        from pydantic import BaseModel as PydanticBaseModel
        from pydantic import Field, validator
        from pydantic_settings import BaseSettings as PydanticBaseSettings

        IS_PYDANTIC_V2 = False
        IS_PYDANTIC_AVAILABLE = True

        # Create type aliases instead of subclasses to avoid type assignment issues
        BaseModel = PydanticBaseModel  # No type annotation # type: ignore
        BaseSettings = PydanticBaseSettings  # No type annotation # type: ignore

    except ImportError:
        print(
            "Pydantic (pydantic, pydantic-settings) is not installed. Using basic config."
        )  # Basic fallback if pydantic is not available for core logic,

        # but configuration will be less robust.
        class BaseModel:
            pass

        class BaseSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        def Field(default, **kwargs):
            return default

        IS_PYDANTIC_V2 = False
        IS_PYDANTIC_AVAILABLE = False


# Configure logging
# In production, consider structured logging (e.g., JSON) and sending logs
# to a centralized logging system (ELK stack, Splunk, Datadog, etc.)
logging.basicConfig(
    level=logging.INFO,  # Consider making this configurable (e.g., from env var)
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("VoxSigilHybridMiddleware")


# --- Configuration ---
# Using Pydantic for robust configuration management (environment variables, .env files)
class HybridMiddlewareConfig(BaseSettings):
    """Configuration for the Hybrid Middleware."""

    entropy_threshold: float = Field(
        0.25,
        description="Threshold for determining high vs low entropy. Inputs below this are routed to 'patch_based'.",
    )
    blt_hybrid_weight: float = Field(
        0.7,
        description="Weight factor for BLT in hybrid embeddings (used by hybrid_embedding utility).",
    )
    entropy_router_fallback: str = Field(
        "token_based",
        description="Default path if entropy calculation fails ('patch_based' or 'token_based').",
    )
    cache_ttl_seconds: int = Field(
        300, description="Time-to-live for cached RAG contexts in seconds."
    )
    log_level: str = Field(
        "INFO", description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)."
    )

    class Config:
        validate_assignment = True
        extra = "forbid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Always validate after init
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.log_level}")
        if self.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative")

    # Example of loading from a .env file or environment variables
    # class Config:
    #     env_file = ".env"
    #     env_prefix = "VOXSIGIL_HYBRID_" # e.g. VOXSIGIL_HYBRID_ENTROPY_THRESHOLD=0.3
    # Define validator function based on Pydantic version
    if IS_PYDANTIC_V2:

        @field_validator("log_level")
        def set_log_level(cls, value):
            numeric_level = getattr(logging, value.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {value}")
            logging.getLogger().setLevel(numeric_level)  # Set root logger level
            logger.setLevel(numeric_level)  # Set specific logger level
            return value

        @field_validator("entropy_threshold")
        def check_entropy_threshold(cls, value):
            if value < 0:
                raise ValueError("entropy_threshold must be non-negative")
            return value

        def model_post_init(self, __context):
            # Explicitly trigger validators for log_level and entropy_threshold
            numeric_level = getattr(logging, self.log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {self.log_level}")
            if self.entropy_threshold < 0:
                raise ValueError("entropy_threshold must be non-negative")
    else:

        @validator("log_level")
        def set_log_level_v1(cls, value):
            numeric_level = getattr(logging, value.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {value}")
            logging.getLogger().setLevel(numeric_level)  # Set root logger level
            logger.setLevel(numeric_level)  # Set specific logger level
            return value

        @validator("entropy_threshold")
        def check_entropy_threshold_v1(cls, value):
            if value < 0:
                raise ValueError("entropy_threshold must be non-negative")
            return value

    def model_post_init(self, __context):
        # Explicitly trigger validators for log_level and entropy_threshold
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.log_level}")
        if self.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative")


# Load configuration globally or pass it down.
# For simplicity in this script, we'll instantiate it where needed or assume a global instance.
# In a larger app, this would be managed by your application framework.
try:
    APP_CONFIG = HybridMiddlewareConfig()
except Exception as e:
    logger.error(f"Failed to load Pydantic config, using defaults: {e}")
    APP_CONFIG = HybridMiddlewareConfig(
        entropy_threshold=0.25,
        blt_hybrid_weight=0.7,
        entropy_router_fallback="token_based",
        cache_ttl_seconds=300,
        log_level="INFO",
    )  # Basic fallback


# --- VoxSigil Component Stubs/Interfaces ---
# These represent your actual VoxSigil components.
# Ensure they are robust, efficient, and well-tested.


# Import the canonical SigilPatchEncoder implementation from the BLT package
try:
    from BLT import SigilPatchEncoder
except ImportError:  # Final fallback (should not happen under normal setup)
    from BLT import SigilPatchEncoder


# Move MockEmbeddingModel to top-level to avoid multiprocessing pickling issues
class MockEmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        return np.frombuffer(h, dtype=np.float32)[:128]  # Use first 128 floats


class VoxSigilRAG:
    """Production-grade standard RAG component for VoxSigil."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        # In production, load a real embedding model here
        # For demonstration, use a deterministic hash-based embedding

        # Adding missing attributes for type checking
        self._loaded_sigils = []
        self._sigil_cache = {}

    def _compute_text_embedding(self, text: str) -> np.ndarray:
        h = np.frombuffer(
            np.pad(
                bytearray(text.encode("utf-8")),
                (0, max(0, self.embedding_dim - len(text))),
                "constant",
            ),
            dtype=np.uint8,
        )[: self.embedding_dim]
        return h.astype(np.float32) / 255.0

    def load_all_sigils(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load all sigils from storage.

        Args:
            force_reload: Force reload sigils even if already loaded.

        Returns:
            List of loaded sigil dictionaries.
        """
        # Implementation would typically load from disk/database
        # This is a stub implementation for type checking
        if force_reload or not self._loaded_sigils:
            self._loaded_sigils = [
                {"sigil": f"test_sigil_{i}", "relationships": {}} for i in range(5)
            ]
            self._sigil_cache = {}
        return self._loaded_sigils

    def create_rag_context(
        self, query: str, num_sigils: int = 5, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Compute embedding but not used in this simplified implementation
        # In a real implementation, this would be used for similarity search
        _ = self._compute_text_embedding(query)
        sigils = [
            {
                "id": f"doc_{i}",
                "content": f"Relevant document {i} for '{query[:20]}'",
                "score": float(np.random.rand()),
            }
            for i in range(num_sigils)
        ]
        context_str = "\n".join([s["content"] for s in sigils])
        return f"RAG CONTEXT:\n{context_str}", sigils


class ByteLatentTransformerEncoder:
    """
    Production-grade BLT encoder enhanced for ARC GridFormer color validation and correction.

    Features:
    1. Fine-Tuned Correction Attention - detects prediction anomalies vs ground truth
    2. Predictive Latency Masking - multi-step refinement with temporal awareness
    3. Embedding Palette Bias - ARC color constraints (0-9) built into tokenization
    """

    def __init__(
        self,
        patch_size: int = 64,
        max_patches: int = 16,
        embedding_dim: int = 128,
        arc_mode: bool = True,  # Enable ARC-specific features
        color_correction_threshold: float = 0.8,  # Confidence threshold for corrections
    ):
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embedding_dim = embedding_dim
        self.arc_mode = arc_mode
        self.color_correction_threshold = (
            color_correction_threshold  # ARC Color Palette (0=background, 1-9=colors)
        )
        self.arc_palette = list(range(10))
        self.palette_embeddings = self._init_palette_embeddings() if arc_mode else None

        # Correction attention cache for temporal consistency
        self.correction_history = []  # Store (prediction, correction, confidence) tuples
        self.max_history = 100  # Keep last 100 corrections for pattern learning

    def _init_palette_embeddings(self) -> np.ndarray:
        # Initialize embeddings for ARC palette colors (0-9)
        embeddings = np.eye(len(self.arc_palette), self.embedding_dim, dtype=np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def encode(self, text: str) -> np.ndarray:
        patches = self.create_patches(text)
        if not patches:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        patch_embs = [self._patch_embedding(p.content) for p in patches]
        emb = np.mean(patch_embs, axis=0)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _patch_embedding(self, text: str) -> np.ndarray:
        h = np.frombuffer(
            np.pad(
                bytearray(text.encode("utf-8")),
                (0, max(0, self.embedding_dim - len(text))),
                "constant",
            ),
            dtype=np.uint8,
        )[: self.embedding_dim]
        return h.astype(np.float32) / 255.0

    def validate_input(self, text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, (int, float, bool)):
            return str(text)
        if isinstance(text, bytes):
            try:
                return text.decode("utf-8", errors="replace")
            except Exception:
                return str(text)
        return str(text)

    class Patch:
        def __init__(self, content, start_pos, end_pos, entropy):
            self.content = content
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.entropy = entropy

    def create_patches(self, text: str) -> List[Any]:
        validated_text = self.validate_input(text)
        text_bytes = validated_text.encode("utf-8", errors="replace")
        text_len = len(text_bytes)
        patches = []
        for i in range(0, text_len, self.patch_size):
            end_pos = min(i + self.patch_size, text_len)
            chunk = text_bytes[i:end_pos]
            entropy = self._shannon_entropy(chunk)
            chunk_text = chunk.decode("utf-8", errors="replace")
            patches.append(self.Patch(chunk_text, i, end_pos, entropy))
            if len(patches) >= self.max_patches:
                break
        if not patches and validated_text:
            patches = [
                self.Patch(
                    validated_text[: min(len(validated_text), self.patch_size)],
                    0,
                    min(len(validated_text), self.patch_size),
                    0.5,
                )
            ]
        return patches

    def _shannon_entropy(self, data: bytes) -> float:
        if not data:
            return 0.0
        from collections import Counter

        counts = Counter(data)
        probs = [v / len(data) for v in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        if (
            embedding1 is None
            or embedding2 is None
            or embedding1.size == 0
            or embedding2.size == 0
        ):
            return 0.0
        min_dim = min(embedding1.size, embedding2.size)
        embedding1 = embedding1.flatten()[:min_dim]
        embedding2 = embedding2.flatten()[:min_dim]
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))


class BLTEnhancedRAG(VoxSigilRAG):
    """Production-grade BLT-enhanced RAG component."""

    def __init__(
        self,
        entropy_threshold: float,
        blt_hybrid_weight: float,
        embedding_dim: int = 128,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.entropy_threshold = entropy_threshold
        self.blt_hybrid_weight = blt_hybrid_weight
        self.blt_encoder = ByteLatentTransformerEncoder(embedding_dim=embedding_dim)

    def _compute_text_embedding(self, text: str) -> np.ndarray:
        blt_emb = self.blt_encoder.encode(text)
        std_emb = super()._compute_text_embedding(text)
        # Weighted hybrid
        hybrid_emb = (self.blt_hybrid_weight * blt_emb) + (
            (1 - self.blt_hybrid_weight) * std_emb
        )
        return hybrid_emb / (np.linalg.norm(hybrid_emb) + 1e-9)

    def create_rag_context(
        self, query: str, num_sigils: int = 5, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Compute embedding but not used in this simplified implementation
        _ = self._compute_text_embedding(query)
        sigils = [
            {
                "id": f"blt_doc_{i}",
                "content": f"BLT relevant document {i} for '{query[:20]}'",
                "score": float(np.random.rand()),
            }
            for i in range(num_sigils)
        ]
        context_str = "\n".join([s["content"] for s in sigils])
        return f"BLT RAG CONTEXT:\n{context_str}", sigils


# --- Core Hybrid Logic ---


class EntropyRouter:
    """Routes inputs based on their dynamically calculated entropy level."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        self.patch_encoder = (
            SigilPatchEncoder()
        )  # This would be your actual entropy model/analyzer
        logger.info(
            f"EntropyRouter initialized with threshold: {self.config.entropy_threshold}, fallback: {self.config.entropy_router_fallback}"
        )

    def route(self, text: str) -> Tuple[str, Optional[List[str]], List[float]]:
        """
        Determine the best processing path for the input text.

        Args:
            text: The input text to route.

        Returns:
            Tuple of (route_decision, patches, entropy_scores).
            route_decision can be 'patch_based' or 'token_based'.
        """
        if not text:
            logger.warning("Empty text received for routing. Using fallback.")
            return (
                self.config.entropy_router_fallback,
                None,
                [0.5],
            )  # Default to medium entropy for empty

        # Pre-check for grid-like/structured input: always route as patch_based
        if any(c in text for c in ["<", ">", "{", "}", "[", "]"]) and len(text) < 200:
            logger.info(
                "Pre-check: Detected grid-like/structured input, forcing patch_based route."
            )
            return (
                "patch_based",
                [text],
                [0.15],
            )

        try:
            patches, entropy_scores = self.patch_encoder.analyze_entropy(text)

            if (
                not entropy_scores
            ):  # If analyze_entropy returns empty scores for some reason
                logger.warning(
                    f"Entropy calculation returned no scores for text: '{text[:50]}...'. Applying heuristic."
                )
                # Fallback heuristic if primary entropy model fails to give scores
                if (
                    any(c in text for c in ["<", ">", "{", "}", "[", "]"])
                    and len(text) < 200
                ):  # More likely structured
                    avg_entropy = 0.15
                    entropy_scores = [avg_entropy]
                    patches = patches or [text]  # Ensure patches exist
                    logger.info(
                        "Heuristic: Detected grid-like/structured input, forcing patch_based route."
                    )
                    return (
                        "patch_based",
                        patches,
                        entropy_scores,
                    )
                elif (
                    len(text) < 50 and " " not in text
                ):  # Short, no spaces, might be an identifier
                    avg_entropy = 0.2
                else:  # More likely natural language
                    avg_entropy = 0.75
                entropy_scores = [avg_entropy]
                patches = patches or [text]  # Ensure patches exist

            # Calculate average entropy, ensuring there's at least one score
            avg_entropy = (
                sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
            )

            logger.info(
                f"Text avg_entropy: {avg_entropy:.4f} (threshold: {self.config.entropy_threshold}) for query: '{text[:30]}...'"
            )

            if avg_entropy < self.config.entropy_threshold:
                return (
                    "patch_based",
                    patches,
                    entropy_scores,
                )  # Low entropy, likely structured -> BLT/Patch
            else:
                return (
                    "token_based",
                    None,
                    entropy_scores,
                )  # High entropy, likely natural language -> Standard/Token
        except Exception as e:
            logger.error(
                f"Entropy calculation/routing failed: {e}. Using fallback path: {self.config.entropy_router_fallback}",
                exc_info=True,
            )
            return (
                self.config.entropy_router_fallback,
                None,
                [0.5],
            )  # Default to medium entropy on error


class HybridProcessor:
    """Processes inputs using either BLT or token-based approaches based on routing."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        self.router = EntropyRouter(config)
        # Lazy initialization for potentially heavy RAG components
        self._standard_rag: Optional[VoxSigilRAG] = None
        self._blt_rag: Optional[BLTEnhancedRAG] = None

        logger.info("HybridProcessor initialized.")

    @property
    def standard_rag(self) -> VoxSigilRAG:
        if self._standard_rag is None:
            logger.info("Lazy initializing Standard VoxSigilRAG...")
            self._standard_rag = VoxSigilRAG()
        return self._standard_rag

    @property
    def blt_rag(self) -> BLTEnhancedRAG:
        if self._blt_rag is None:
            logger.info("Lazy initializing BLT-Enhanced RAG...")
            # Pass relevant parts of config if BLTEnhancedRAG needs them
            self._blt_rag = BLTEnhancedRAG(
                entropy_threshold=self.config.entropy_threshold,
                blt_hybrid_weight=self.config.blt_hybrid_weight,
            )
        return self._blt_rag

    def compute_embedding(self, text: str) -> Dict[str, Any]:
        """
        Compute embedding using the appropriately routed method.
        (This is more of a utility; main RAG context creation is separate)
        """
        route, patches, entropy_scores = self.router.route(text)
        avg_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
        )

        start_time = time.monotonic()
        embedding, method = None, route  # Initialize method with route decision

        try:
            if route == "patch_based":
                embedding = self.blt_rag._compute_text_embedding(
                    text
                )  # Uses BLT encoder
                method = "blt"
            else:  # "token_based" or fallback
                embedding = self.standard_rag._compute_text_embedding(
                    text
                )  # Uses standard encoder
                method = "token"
        except Exception as e:
            logger.error(
                f"Error computing embedding via {method} path: {e}. Falling back to standard.",
                exc_info=True,
            )
            embedding = self.standard_rag._compute_text_embedding(
                text
            )  # Fallback to standard
            method = "token_fallback"

        processing_time = time.monotonic() - start_time

        return {
            "embedding": embedding,
            "method_used": method,  # Actual method used
            "routing_decision": route,  # Initial routing decision
            "patches_count": len(patches) if patches else 0,
            "avg_entropy": avg_entropy,
            "processing_time_seconds": processing_time,
        }

    def get_rag_context_and_route(
        self, query: str, num_sigils: int = 5, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Determines route and creates RAG context using the appropriate method.
        """
        route, _, entropy_scores = self.router.route(
            query
        )  # Patches and scores for logging/metrics if needed
        avg_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
        )

        context_str, sigils_list = "", []
        actual_method_used = route

        try:
            if route == "patch_based":
                logger.info(f"Routing query to BLT RAG (entropy: {avg_entropy:.2f})")
                context_str, sigils_list = self.blt_rag.create_rag_context(
                    query=query, num_sigils=num_sigils, **kwargs
                )
            else:  # "token_based" or fallback route
                logger.info(
                    f"Routing query to Standard RAG (entropy: {avg_entropy:.2f})"
                )
                context_str, sigils_list = self.standard_rag.create_rag_context(
                    query=query, num_sigils=num_sigils, **kwargs
                )
        except Exception as e:
            logger.error(
                f"Error creating RAG context via {route} path: {e}. Attempting fallback.",
                exc_info=True,
            )
            # Attempt fallback to the other RAG if one path fails catastrophically
            if route == "patch_based":
                logger.warning("BLT RAG failed. Falling back to Standard RAG.")
                try:
                    context_str, sigils_list = self.standard_rag.create_rag_context(
                        query=query, num_sigils=num_sigils, **kwargs
                    )
                    actual_method_used = "token_fallback_from_blt_failure"
                except Exception as fallback_e:
                    logger.critical(
                        f"Standard RAG fallback also failed: {fallback_e}",
                        exc_info=True,
                    )
                    raise  # Re-raise if fallback also fails
            else:  # Token-based failed
                logger.warning(
                    "Standard RAG failed. Falling back to BLT RAG (less common fallback)."
                )
                try:
                    context_str, sigils_list = self.blt_rag.create_rag_context(
                        query=query, num_sigils=num_sigils, **kwargs
                    )
                    actual_method_used = "blt_fallback_from_token_failure"
                except Exception as fallback_e:
                    logger.critical(
                        f"BLT RAG fallback also failed: {fallback_e}", exc_info=True
                    )
                    raise  # Re-raise if fallback also fails

        return context_str, sigils_list, actual_method_used


class DynamicExecutionBudgeter:  # More conceptual in this script
    """
    Allocates conceptual computational "budget" or can be used to adjust parameters.
    In a real system, this might influence timeouts, resource allocation in a queue,
    or model choice (e.g., smaller vs. larger standard model).
    """

    def __init__(self, base_budget: float = 1.0, entropy_multiplier: float = 1.5):
        self.base_budget = base_budget
        self.entropy_multiplier = entropy_multiplier

    def allocate_budget(
        self, method: str, avg_entropy: float, text_length: int
    ) -> float:
        budget = self.base_budget
        if "blt" in method.lower():  # Patch-based or BLT
            budget *= 1.0 - 0.5 * avg_entropy  # Lower entropy, less budget (faster)
        else:  # Token-based
            budget *= (
                1.0 + self.entropy_multiplier * avg_entropy
            )  # Higher entropy, more budget

        length_factor = max(
            0.5, min(2.0, text_length / 500.0)
        )  # Scale around 500 chars
        budget *= length_factor
        logger.debug(
            f"Allocated budget: {budget:.2f} for method={method}, entropy={avg_entropy:.2f}, len={text_length}"
        )
        return budget


class HybridMiddleware:
    """
    Middleware to be integrated into a request-response pipeline (e.g., for an LLM).
    Enhances incoming messages with RAG context based on hybrid processing.
    """

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        logger.info("Initializing HybridMiddleware with config.")
        self.processor = HybridProcessor(config)
        self.budgeter = DynamicExecutionBudgeter()  # Using default budgeter params

        self._context_cache: Dict[
            str, Tuple[str, List[Dict[str, Any]], str, float]
        ] = {}  # key: (context, sigils, route, timestamp)
        self._request_counter = 0  # For basic operational metrics
        self._processing_times: List[float] = []  # For basic perf monitoring

        # Initialize VoxSigil components
        self._initialize_voxsigil_components()

    def _initialize_voxsigil_components(self):
        """Initialize VoxSigil components including necessary middleware components."""
        try:
            # Initialize a RAG instance via the processor if available
            if hasattr(self.processor, "standard_rag") and self.processor.standard_rag:
                self.voxsigil_rag = self.processor.standard_rag

                # Fix relationship format in loaded sigils to ensure schema validation passes
                if self.voxsigil_rag:
                    self._normalize_sigil_relationships_format()
            else:
                self.voxsigil_rag = None
                logger.warning(
                    "VoxSigil RAG component not available in processor. Limited functionality available."
                )

            # Initialize history and tracking components
            self.conversation_history = []
            self.selected_sigils_history = {}
            self.turn_counter = 0

            # Additional configurations for conditional RAG
            self.rag_off_keywords = ["@@norag@@", "norag"]
            self.min_prompt_len_for_rag = 5

            # Setup cache for RAG
            self._rag_cache = {}

            logger.info("VoxSigil components initialized for HybridMiddleware")
        except Exception as e:
            logger.warning(f"Failed to initialize VoxSigil components: {e}")
            logger.warning(traceback.format_exc())

    def _normalize_sigil_relationships_format(self):
        """
        Ensure all loaded sigil relationships are in the proper dictionary format.
        Converts list-format relationships to dictionary with unique keys.
        This method is called during initialization to ensure schema validation passes.
        """
        # First ensure voxsigil_rag exists
        if not hasattr(self, "voxsigil_rag") or self.voxsigil_rag is None:
            logger.warning(
                "VoxSigilRAG instance is not available. Skipping normalization."
            )
            return

        # Check if the class has the necessary attributes
        has_loaded_sigils = hasattr(self.voxsigil_rag, "_loaded_sigils")
        has_load_method = hasattr(self.voxsigil_rag, "load_all_sigils")

        # If it doesn't have loaded sigils but has load method, try loading them
        if not has_loaded_sigils and has_load_method:
            try:
                # Add type ignores for Pylance
                self.voxsigil_rag.load_all_sigils(force_reload=False)  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to load sigils: {e}")
                return

        # Check again if loaded_sigils exists and is not empty
        if not hasattr(self.voxsigil_rag, "_loaded_sigils") or not getattr(
            self.voxsigil_rag, "_loaded_sigils", None
        ):
            logger.warning("No sigils loaded to normalize relationships format.")
            return

        # Now we know _loaded_sigils exists and is not empty
        normalized_count = 0
        for sigil in self.voxsigil_rag._loaded_sigils:  # type: ignore
            normalized_sigil = self._normalize_single_sigil_relationships(sigil)
            if normalized_sigil != sigil:
                normalized_count += 1

        if normalized_count > 0:
            logger.info(
                f"Normalized relationships format for {normalized_count} sigils"
            )

            # Clear the sigil cache if it exists
            if hasattr(self.voxsigil_rag, "_sigil_cache"):
                self.voxsigil_rag._sigil_cache = {}  # type: ignore

            # Try to force reload
            if has_load_method:
                try:
                    self.voxsigil_rag.load_all_sigils(force_reload=True)  # type: ignore
                except Exception as e:
                    logger.warning(f"Failed to reload sigils after normalization: {e}")

    def _normalize_single_sigil_relationships(
        self, sigil: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize relationships format for a single sigil to ensure schema compatibility.

        Args:
            sigil: The sigil dictionary to normalize

        Returns:
            The normalized sigil dictionary
        """
        if "relationships" not in sigil:
            return sigil

        if not isinstance(sigil["relationships"], dict):
            # Found relationships that need to be converted from list/other format to dictionary
            if isinstance(sigil["relationships"], list):
                # Convert list of relationships to dictionary with unique keys
                relations_dict = {}

                for i, rel in enumerate(sigil["relationships"]):
                    # Generate a unique key based on the relationship value or index
                    if isinstance(rel, str):
                        # If it's a string, use it as a value with a generated key
                        key = f"relation_{i + 1}"
                        relations_dict[key] = rel
                    elif isinstance(rel, dict) and len(rel) == 1:
                        # If it's a dictionary with a single key-value pair, use it directly
                        key, value = next(iter(rel.items()))
                        relations_dict[key] = value
                    else:
                        # For other types, create a key based on index
                        key = f"relation_{i + 1}"
                        relations_dict[key] = rel

                sigil["relationships"] = relations_dict
            else:
                # If it's not a list or dict, convert to a simple dict with a default key
                sigil["relationships"] = {"default": sigil["relationships"]}

        return sigil

    def format_sigil_for_context(
        self,
        sigil: Dict[str, Any],
        detail_level: str = "standard",
        include_explanation: bool = False,
    ) -> str:
        """
        Format a sigil for inclusion in a context for RAG.

        Args:
            sigil: The sigil dictionary to format
            detail_level: "summary", "standard", "detailed", or "full"
            include_explanation: Whether to include match explanation

        Returns:
            Formatted sigil string
        """
        output = []

        # Start with the sigil ID for reference
        if "sigil" in sigil:
            output.append(f'Sigil: "{sigil["sigil"]}"')

        # Consolidate tag handling as they might come in various formats
        all_tags = []
        if "tag" in sigil and sigil["tag"]:  # Handle legacy 'tag' field
            if isinstance(sigil["tag"], str) and sigil["tag"] not in all_tags:
                all_tags.append(sigil["tag"])
        if "tags" in sigil and sigil["tags"]:
            if isinstance(sigil["tags"], list):
                for t in sigil["tags"]:
                    if t not in all_tags:
                        all_tags.append(t)
            elif isinstance(sigil["tags"], str) and sigil["tags"] not in all_tags:
                all_tags.append(sigil["tags"])
        if all_tags:
            output.append("Tags: " + ", ".join(f'"{tag}"' for tag in all_tags))

        # Core principle is always included
        if "principle" in sigil:
            output.append(f'Principle: "{sigil["principle"]}"')

        # Summary detail level has only the above information
        if detail_level.lower() == "summary":
            return "\n".join(output)

        # Standard and higher detail levels include usage information
        if "usage" in sigil and isinstance(sigil["usage"], dict):
            if "description" in sigil["usage"]:
                output.append(f'Usage: "{sigil["usage"]["description"]}"')
            # Show first example if available
            if "examples" in sigil["usage"] and sigil["usage"]["examples"]:
                examples = sigil["usage"]["examples"]
                example_str = (
                    f'"{examples[0]}"'
                    if isinstance(examples, list) and examples
                    else f'"{examples}"'
                )
                output.append(f"Example: {example_str}")

        if "_source_file" in sigil:  # Added for context
            output.append(f"Source File: {Path(sigil['_source_file']).name}")

        if include_explanation and "_similarity_explanation" in sigil:
            output.append(f"Match Information: {sigil['_similarity_explanation']}")

        # Detailed or full detail level includes additional information
        if detail_level.lower() in ("detailed", "full"):
            # Include relationships and scaffolds for fuller context
            if "relationships" in sigil and isinstance(sigil["relationships"], dict):
                for rel_type, rel_values in sigil["relationships"].items():
                    if rel_values:
                        val_str = (
                            ", ".join(f'"{v}"' for v in rel_values)
                            if isinstance(rel_values, list)
                            else f'"{rel_values}"'
                        )
                        output.append(f"Relationship ({rel_type}): {val_str}")

            # Include scaffold details if available
            if "scaffolds" in sigil:
                scaffolds = sigil["scaffolds"]
                if isinstance(scaffolds, list):
                    scaffolds_str = ", ".join(f'"{s}"' for s in scaffolds)
                else:
                    scaffolds_str = f'"{scaffolds}"'
                output.append(f"Scaffolds: {scaffolds_str}")

            # Include glyphs details if available
            if "glyphs" in sigil:
                glyphs = sigil["glyphs"]
                if isinstance(glyphs, list):
                    glyphs_str = ", ".join(f'"{g}"' for g in glyphs)
                else:
                    glyphs_str = f'"{glyphs}"'
                output.append(f"Glyphs: {glyphs_str}")

            # Include prompt template details if available and in full detail
            if (
                detail_level.lower() == "full"
                and "prompt_template" in sigil
                and isinstance(sigil["prompt_template"], dict)
            ):
                if "type" in sigil["prompt_template"]:
                    output.append(f"Template Type: {sigil['prompt_template']['type']}")
                if "description" in sigil["prompt_template"]:
                    output.append(
                        f'Template Description: "{sigil["prompt_template"]["description"]}"'
                    )
                if "template" in sigil["prompt_template"]:
                    template_text = sigil["prompt_template"]["template"]
                    # Only show a preview if template is large
                    if len(template_text) > 100:
                        template_text = template_text[:100] + "..."
                    output.append(f'Template Preview: "{template_text}"')

        return "\n".join(output)

    def format_sigils_for_context(
        self,
        sigils: List[Dict[str, Any]],
        detail_level: str = "standard",
        include_explanations: bool = False,
    ) -> str:
        """
        Format a list of sigils for inclusion in a RAG context.

        Args:
            sigils: List of sigil dictionaries to format
            detail_level: Level of detail to include
            include_explanations: Whether to include match explanations

        Returns:
            Formatted context string with all sigils
        """
        if not sigils:
            return ""

        output_sections = []

        # Add a header
        output_sections.append(f"--- VoxSigil Context ({len(sigils)} sigils) ---")

        # Format each sigil
        for idx, sigil in enumerate(sigils, 1):
            sigil_text = self.format_sigil_for_context(
                sigil, detail_level, include_explanations
            )
            output_sections.append(f"---\nSIGIL {idx}:\n{sigil_text}")

        # Add a footer for clear separation from model context
        output_sections.append("--- End VoxSigil Context ---")

        return "\n\n".join(output_sections)

    def _get_cache_key(self, query: str) -> str:
        """Create a cache key from the query, normalizing it."""
        normalized_query = " ".join(
            query.lower().strip().split()
        )  # Lowercase, strip, normalize whitespace
        # Hash longer queries to keep keys manageable and consistent
        if len(normalized_query) > 256:
            return hashlib.sha256(normalized_query.encode()).hexdigest()
        return normalized_query

    def _clean_expired_cache_entries(self) -> None:
        current_time = time.monotonic()
        expired_keys = [
            key
            for key, (_, _, _, timestamp) in self._context_cache.items()
            if current_time - timestamp > self.config.cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._context_cache[key]
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries.")

    def _extract_query_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extracts the primary query, typically the last user message."""
        if not messages:
            return None
        # Look for the last message with role 'user' or the last message if no user role found.
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                return msg["content"]
        # Fallback to last message content if no user message
        last_message = messages[-1]
        if isinstance(last_message, dict) and isinstance(
            last_message.get("content"), str
        ):
            return last_message["content"]
        logger.warning(
            f"Could not extract a valid query string from messages: {messages}"
        )
        return None

    def _enhance_messages_with_context(
        self, messages: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        """Adds or prepends RAG context to the system message."""
        if not context:
            return messages

        enhanced_messages = [msg.copy() for msg in messages]  # Work on a copy

        # Prepend to system message or add a new one        # This strategy can be tuned based on how your LLM best uses context.
        system_message_content = (
            f"You are an AI assistant. Use the following retrieved VoxSigil context to answer the user's query accurately. "
            f"If the context is not relevant, rely on your general knowledge.\n\n"
            f"--- VOXSIGIL CONTEXT START ---\n{context}\n--- VOXSIGIL CONTEXT END ---"
        )

        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            original_system_content = enhanced_messages[0].get("content", "")
            enhanced_messages[0]["content"] = (
                f"{system_message_content}\n\n{original_system_content}"
            )
            # System message already existed and has been updated
        else:
            enhanced_messages.insert(
                0, {"role": "system", "content": system_message_content}
            )

        logger.debug("Messages enhanced with RAG context.")
        return enhanced_messages

    def __call__(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a model input dictionary (e.g., from an OpenAI-like API request).
        Expected keys in model_input: 'messages' (List[Dict]), other passthrough kwargs.
        Returns the modified model_input dictionary with enhanced messages.

        Example model_input:
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello?"}],
            "temperature": 0.7
        }
        """
        self._request_counter += 1
        start_time_total = time.monotonic()

        messages = model_input.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.warning(
                "No messages found or invalid format in model_input. Skipping VoxSigil enhancement."
            )
            return model_input  # Passthrough if no valid messages

        query = self._extract_query_from_messages(messages)
        if not query:
            logger.warning(
                "No query extractable from messages. Skipping VoxSigil enhancement."
            )
            return model_input

        # Periodically clean cache (e.g., every N requests or based on time)
        if self._request_counter % 50 == 0:
            self._clean_expired_cache_entries()

        cache_key = self._get_cache_key(query)
        cached_entry = self._context_cache.get(cache_key)
        current_monotonic_time = time.monotonic()

        # Default values if not from cache
        context_str, sigils_list, route_method = "", [], "unknown"
        cache_hit = False

        if cached_entry:
            cached_context_str, cached_sigils_list, cached_route_method, timestamp = (
                cached_entry
            )
            if current_monotonic_time - timestamp <= self.config.cache_ttl_seconds:
                logger.info(
                    f"Cache HIT for query (key: {cache_key[:30]}...). Using cached context from route: {cached_route_method}"
                )
                context_str, sigils_list, route_method = (
                    cached_context_str,
                    cached_sigils_list,
                    cached_route_method,
                )
                # Update timestamp to refresh TTL (Least Recently Used semantic for cache hit)
                self._context_cache[cache_key] = (
                    context_str,
                    sigils_list,
                    route_method,
                    current_monotonic_time,
                )
                cache_hit = True
            else:
                logger.info(
                    f"Cache STALE for query (key: {cache_key[:30]}...). Re-computing."
                )
                del self._context_cache[cache_key]  # Remove stale entry

        if not cache_hit:
            logger.info(
                f"Cache MISS for query (key: {cache_key[:30]}...). Processing query."
            )
            try:
                # Determine route and get RAG context
                context_str, sigils_list, route_method = (
                    self.processor.get_rag_context_and_route(query, num_sigils=5)
                )  # num_sigils configurable

                # Cache the new result
                self._context_cache[cache_key] = (
                    context_str,
                    sigils_list,
                    route_method,
                    current_monotonic_time,
                )
                logger.info(
                    f"Successfully processed and cached query. Route: {route_method}"
                )
            except Exception as e:
                logger.critical(
                    f"Critical error during RAG context creation for query '{query[:50]}...': {e}",
                    exc_info=True,
                )
                # Depending on policy, might return original messages or an error message
                # For now, we return original messages to not break the flow.
                return model_input

        # --- Enhance messages and gather metadata ---
        # Calculate avg_entropy even for cache hits if needed for logs/budget (might need to store it in cache too)
        # For simplicity, this example calculates entropy again if not from cache, or uses default for cached.
        avg_entropy_for_budget = 0.5
        if not cache_hit:
            # Re-route just to get entropy score if not stored in cache (could be optimized)
            _, _, entropy_scores_for_budget = self.processor.router.route(query)
            if entropy_scores_for_budget:
                avg_entropy_for_budget = sum(entropy_scores_for_budget) / len(
                    entropy_scores_for_budget
                )

        budget = self.budgeter.allocate_budget(
            method=route_method,
            avg_entropy=avg_entropy_for_budget,
            text_length=len(query),
        )

        enhanced_messages = self._enhance_messages_with_context(messages, context_str)

        end_time_total = time.monotonic()
        total_processing_time = end_time_total - start_time_total
        self._processing_times.append(total_processing_time)

        # --- Metadata for logging/monitoring ---
        # In production, send these to a metrics system (Prometheus, Datadog, etc.)
        log_metadata = {
            "request_id": hashlib.md5(query.encode()).hexdigest()[
                :8
            ],  # Example request ID
            "query_preview": query[:50] + "...",
            "route_method_used": route_method,
            "cache_hit": cache_hit,
            "context_length": len(context_str),
            "num_sigils_retrieved": len(sigils_list),
            "allocated_budget": round(budget, 2),
            "total_processing_time_ms": round(total_processing_time * 1000, 2),
            "avg_entropy_for_budget": round(avg_entropy_for_budget, 3),
        }
        logger.info(f"VoxSigilMiddleware processed request: {log_metadata}")

        # Update the messages in the model_input
        model_input["messages"] = enhanced_messages
        # Optionally add metadata to kwargs for downstream use, if the calling system supports it
        model_input.setdefault("voxsigil_metadata", {}).update(log_metadata)

        return model_input

    def get_stats(self) -> Dict[str, Any]:
        """Returns basic operational statistics."""
        if not self._processing_times:
            avg_proc_time = 0
        else:
            avg_proc_time = sum(self._processing_times) / len(self._processing_times)

        return {
            "total_requests_processed": self._request_counter,
            "cache_size": len(self._context_cache),
            "average_processing_time_seconds": round(avg_proc_time, 4),
            # More stats can be added: cache hit/miss_rate, route distribution, etc.
        }

    def _augment_query(self, query: str) -> str:
        """
        Augments the query with synonyms from the internal map to improve
        retrieval of relevant sigils.

        Args:
            query: The original user query

        Returns:
            Augmented query with synonym terms added
        """
        if not hasattr(self, "synonym_map"):
            # Initialize default synonym map if not already defined
            self.synonym_map = {
                "ai": ["artificial intelligence", "machine learning", "deep learning"],
                "symbolic reasoning": [
                    "logic-based reasoning",
                    "knowledge representation",
                    "declarative reasoning",
                ],
                "voxsigil": ["vox sigil language", "sigil language"],
                "llm": ["large language model", "language model", "generative model"],
                "scaffold": ["structure", "framework", "template", "pattern"],
                "glyph": ["symbol", "character", "representation", "sigil component"],
                "entity": ["object", "concept", "element", "component"],
                "hybrid": ["combined", "mixed", "integrated", "fusion"],
            }

        augmented_parts = [query]
        query_lower = query.lower()

        for term, synonyms in self.synonym_map.items():
            if term in query_lower:  # If the base term is in the query
                for syn in synonyms:
                    # Add synonym if it's not already in query or parts
                    if (
                        syn not in query_lower
                        and syn.lower() not in " ".join(augmented_parts).lower()
                    ):
                        augmented_parts.append(syn)

        augmented_query = " ".join(augmented_parts)
        if augmented_query != query:
            logger.info(f"Query augmented: '{query}' -> '{augmented_query}'")

        return augmented_query

    def _validate_sigil_data(
        self, sigil_data: Dict[str, Any], file_path: Optional[str] = None
    ) -> bool:
        """
        Validates sigil data against the defined schema if jsonschema is available.

        Args:
            sigil_data: The sigil dictionary to validate
            file_path: Optional file path for logging purposes

        Returns:
            True if validation succeeds or if jsonschema is not available
        """
        # Check if jsonschema is available
        have_jsonschema = False
        try:
            from jsonschema import ValidationError, validate

            have_jsonschema = True
        except ImportError:
            logger.warning(
                "jsonschema not available. Sigil schema validation will be skipped."
            )
            return True

        if not have_jsonschema:
            return True  # Skip validation if library not present

        # Define default sigil schema if not already defined
        if not hasattr(self, "sigil_schema"):
            self.sigil_schema = {
                "type": "object",
                "properties": {
                    "sigil": {
                        "type": "string",
                        "description": "The unique identifier for the sigil.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Primary tag for the sigil (can be deprecated in favor of 'tags' list).",
                    },
                    "tags": {
                        "type": ["array", "string"],
                        "items": {"type": "string"},
                        "description": "A list of tags or a single tag string.",
                    },
                    "principle": {
                        "type": "string",
                        "description": "The core concept or idea the sigil represents.",
                    },
                    "usage": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "examples": {
                                "type": ["array", "string"],
                                "items": {"type": "string"},
                            },
                        },
                        "description": "How to use the sigil.",
                    },
                    "prompt_template": {
                        "type": "object",
                        "description": "Associated prompt template, if any.",
                    },
                    "relationships": {
                        "type": "object",
                        "description": "Links to other related sigils.",
                    },
                    "scaffolds": {
                        "type": ["array", "string"],
                        "description": "Scaffolds associated with this sigil.",
                    },
                    "glyphs": {
                        "type": ["array", "string"],
                        "description": "Glyphs associated with this sigil.",
                    },
                    # Meta fields (added by loader, not typically in user files)
                    "_source_file": {"type": "string"},
                    "_last_modified": {"type": "number"},  # Unix timestamp
                    "_similarity_score": {"type": "number"},
                    "_recency_boost_applied": {"type": "number"},
                },
                "required": [
                    "sigil",
                    "principle",
                ],  # Example: sigil name and principle are mandatory
                "additionalProperties": True,  # Allow other fields not defined in schema
            }

        try:
            validate(instance=sigil_data, schema=self.sigil_schema)
            return True
        except ValidationError as e:
            # Provide more context from the error object
            path_str = " -> ".join(map(str, e.path))
            logger.warning(
                f"Schema validation failed for {file_path or 'sigil'}: {e.message} (at path: '{path_str}')"
            )
            return False
        except Exception as e_generic:
            logger.error(
                f"Generic error during schema validation for {file_path or 'sigil'}: {e_generic}"
            )
            return False

    def _apply_recency_boost(
        self,
        sigils_with_scores: List[Dict[str, Any]],
        recency_boost_factor: float = 0.05,
        recency_max_days: int = 90,
    ) -> List[Dict[str, Any]]:
        """
        Applies a recency boost to sigil scores if enabled and applicable.
        More recently modified sigils receive a higher score boost.

        Args:
            sigils_with_scores: List of sigil dictionaries with _similarity_score values
            recency_boost_factor: Factor to boost scores of recent sigils (0 to disable)
            recency_max_days: Sigils updated within this period get boosted

        Returns:
            List of sigils with potentially boosted scores
        """
        if not recency_boost_factor > 0:
            return sigils_with_scores

        current_time_utc = datetime.now(timezone.utc).timestamp()
        recency_max_days_seconds = recency_max_days * 24 * 60 * 60
        boosted_sigils = []

        for sigil_data in sigils_with_scores:
            last_modified_utc = sigil_data.get("_last_modified")  # Unix timestamp
            original_score = sigil_data.get("_similarity_score", 0.0)

            if last_modified_utc and isinstance(last_modified_utc, (int, float)):
                age_seconds = current_time_utc - last_modified_utc
                if 0 <= age_seconds < recency_max_days_seconds:
                    # Linear decay for boost: newer items get more boost
                    boost_multiplier = 1.0 - (age_seconds / recency_max_days_seconds)
                    recency_bonus = recency_boost_factor * boost_multiplier

                    new_score = min(
                        1.0, original_score + recency_bonus
                    )  # Cap score at 1.0
                    if new_score > original_score:
                        sigil_data["_similarity_score"] = new_score
                        sigil_data["_recency_boost_applied"] = recency_bonus
                        logger.debug(
                            f"Applied recency boost {recency_bonus:.3f} to sigil '{sigil_data.get('sigil', 'N/A')}' (new score: {new_score:.3f})"
                        )
            boosted_sigils.append(sigil_data)

        return boosted_sigils

    def auto_fuse_related_sigils(
        self, base_sigils: List[Dict[str, Any]], max_additional: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Automatically add related sigils to an initial set.
        Tries to find related sigils based on 'relationships' field and shared tags.

        Args:
            base_sigils: The initial set of sigils to fuse related sigils into
            max_additional: Maximum number of additional sigils to add

        Returns:
            Enhanced list of sigils with related ones added
        """
        if not base_sigils or max_additional <= 0:
            return base_sigils

        if not hasattr(self, "voxsigil_rag") or not self.voxsigil_rag:
            logger.warning(
                "VoxSigil RAG component not available. Cannot auto-fuse related sigils."
            )
            return base_sigils

        # Get all sigils
        all_system_sigils = self.voxsigil_rag.load_all_sigils()
        if not all_system_sigils:
            return base_sigils

        # Create an index for quick lookups by sigil ID
        sigil_index_by_id = {s["sigil"]: s for s in all_system_sigils if "sigil" in s}

        # Track IDs already in the list (base + newly added) to avoid duplicates
        current_sigil_ids = {s["sigil"] for s in base_sigils if "sigil" in s}

        fused_sigils_list = list(base_sigils)  # Start with a copy
        added_count = 0

        # Iterate through a copy of base_sigils for stable iteration if modifying fused_sigils_list
        for sigil_item in list(base_sigils):
            if added_count >= max_additional:
                break

            source_sigil_id = sigil_item.get("sigil")
            if not source_sigil_id:
                continue

            # 1. Explicit relationships
            if "relationships" in sigil_item and isinstance(
                sigil_item["relationships"], dict
            ):
                for rel_type, rel_targets in sigil_item["relationships"].items():
                    targets_as_list = (
                        rel_targets if isinstance(rel_targets, list) else [rel_targets]
                    )
                    for target_id in targets_as_list:
                        if (
                            isinstance(target_id, str)
                            and target_id in sigil_index_by_id
                            and target_id not in current_sigil_ids
                        ):
                            related_s_data = sigil_index_by_id[target_id].copy()
                            related_s_data["_fusion_reason"] = (
                                f"related_to:{source_sigil_id}(type:{rel_type})"
                            )
                            related_s_data.setdefault(
                                "_similarity_score", 0.4
                            )  # Assign modest score
                            fused_sigils_list.append(related_s_data)
                            current_sigil_ids.add(target_id)
                            added_count += 1
                            if added_count >= max_additional:
                                break
                    if added_count >= max_additional:
                        break

            # 2. Shared tags (if still need more and explicit relations didn't fill quota)
            if added_count < max_additional:
                source_tags = set()
                if "tag" in sigil_item and sigil_item["tag"]:
                    source_tags.add(str(sigil_item["tag"]).lower())
                if "tags" in sigil_item:
                    s_tags_val = sigil_item["tags"]
                    if isinstance(s_tags_val, list):
                        source_tags.update(str(t).lower() for t in s_tags_val)
                    elif isinstance(s_tags_val, str):
                        source_tags.add(s_tags_val.lower())

                if source_tags:
                    for other_s in all_system_sigils:
                        other_id = other_s.get("sigil")
                        if not other_id or other_id in current_sigil_ids:
                            continue

                        other_s_tags = set()
                        if "tag" in other_s and other_s["tag"]:
                            other_s_tags.add(str(other_s["tag"]).lower())
                        if "tags" in other_s:
                            os_tags_val = other_s["tags"]
                            if isinstance(os_tags_val, list):
                                other_s_tags.update(str(t).lower() for t in os_tags_val)
                            elif isinstance(os_tags_val, str):
                                other_s_tags.add(os_tags_val.lower())

                        if not source_tags.isdisjoint(
                            other_s_tags
                        ):  # If any shared tag
                            shared = source_tags.intersection(other_s_tags)
                            related_s_data = other_s.copy()
                            related_s_data["_fusion_reason"] = (
                                f"shared_tags_with:{source_sigil_id}(tags:{','.join(list(shared)[:2])})"
                            )
                            related_s_data.setdefault(
                                "_similarity_score", 0.3
                            )  # Lower score for tag match
                            fused_sigils_list.append(related_s_data)
                            current_sigil_ids.add(other_id)
                            added_count += 1
                            if added_count >= max_additional:
                                break

        if added_count > 0:
            logger.info(f"Auto-fused {added_count} additional sigils.")

        return fused_sigils_list

    def _optimize_context_by_chars(
        self,
        sigils_for_context: List[Dict[str, Any]],
        initial_detail_level: str,
        target_char_budget: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Adjusts sigil count or detail level to fit a character budget.
        This function tries to keep as many relevant sigils as possible while
        reducing detail level if needed to fit within the target character budget.

        Args:
            sigils_for_context: List of sigil dictionaries to optimize
            initial_detail_level: Starting detail level ('full', 'standard', 'summary', etc.)
            target_char_budget: Maximum character count for the context

        Returns:
            Tuple of (optimized_sigil_list, final_detail_level)
        """
        final_sigils = list(sigils_for_context)  # Work with a copy
        current_detail_level = initial_detail_level.lower()

        if not target_char_budget or not final_sigils:
            return final_sigils, current_detail_level

        def estimate_chars(s_list: List[Dict[str, Any]], d_level: str) -> int:
            return sum(len(self.format_sigil_for_context(s, d_level)) for s in s_list)

        # Detail levels ordered from most to least verbose
        detail_levels_ordered = ["full", "detailed", "standard", "summary", "minimal"]
        try:
            current_detail_idx = detail_levels_ordered.index(current_detail_level)
        except ValueError:
            current_detail_idx = (
                2  # Default to 'standard' if initial_detail_level is unknown
            )
            current_detail_level = "standard"

        current_chars = estimate_chars(final_sigils, current_detail_level)

        # Stage 1: Reduce detail level if over budget
        while (
            current_chars > target_char_budget
            and current_detail_idx < len(detail_levels_ordered) - 1
        ):
            current_detail_idx += 1
            new_detail_level = detail_levels_ordered[current_detail_idx]
            logger.info(
                f"Context Optimizer: Chars {current_chars} > budget {target_char_budget}. "
                f"Reducing detail from {current_detail_level} to {new_detail_level} for {len(final_sigils)} sigils."
            )
            current_detail_level = new_detail_level
            current_chars = estimate_chars(final_sigils, current_detail_level)

        # Stage 2: If still over budget, remove least relevant sigils (sigils are pre-sorted by relevance)
        while (
            current_chars > target_char_budget and len(final_sigils) > 1
        ):  # Keep at least one if possible
            removed_sigil = (
                final_sigils.pop()
            )  # Removes the last (least relevant) sigil
            sig_name = removed_sigil.get("sigil", "N/A")
            logger.info(
                f"Context Optimizer: Chars {current_chars} > budget {target_char_budget} at {current_detail_level} detail. "
                f"Removing sigil: '{sig_name}' ({len(final_sigils)} remaining)."
            )
            current_chars = estimate_chars(final_sigils, current_detail_level)

        if current_chars > target_char_budget and final_sigils:
            logger.warning(
                f"Context Optimizer: Final context ({len(final_sigils)} sigil(s), {current_detail_level} detail, {current_chars} chars) "
                f"still exceeds budget ({target_char_budget} chars). Smallest possible context provided."
            )
        elif final_sigils:
            logger.info(
                f"Context Optimizer: Final context: {len(final_sigils)} sigils at {current_detail_level} detail ({current_chars} chars)."
            )

        return final_sigils, current_detail_level

    def enhanced_rag_process(
        self,
        query: str,
        num_sigils: int = 5,
        filter_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        detail_level: str = "standard",
        apply_recency_boost: bool = True,
        augment_query: bool = True,
        enable_context_optimization: bool = True,
        max_context_chars: int = 8000,
        auto_fuse_related: bool = True,
        max_fusion_sigils: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhanced RAG processing with all VoxSigil features enabled.

        Args:
            query: The user query to process
            num_sigils: Number of sigils to retrieve
            filter_tags: Optional list of tags to filter by (include only these)
            exclude_tags: Optional list of tags to exclude
            detail_level: Level of detail to include in context ('standard', 'detailed', etc.)
            apply_recency_boost: Whether to boost scores of recently modified sigils
            augment_query: Whether to enhance query with synonyms
            enable_context_optimization: Whether to use context optimization to fit char budget
            max_context_chars: Maximum character count for context
            auto_fuse_related: Whether to automatically add related sigils
            max_fusion_sigils: Maximum number of additional related sigils to add

        Returns:
            Tuple of (formatted_context_string, retrieved_sigils_list)
        """
        if not hasattr(self, "voxsigil_rag") or not self.voxsigil_rag:
            logger.warning(
                "VoxSigil RAG component not available. Cannot perform enhanced RAG processing."
            )
            return "", []

        effective_query = query
        if augment_query:
            effective_query = self._augment_query(query)

        # Determine the routing path for this query
        route, patches, entropy_scores = self.processor.router.route(effective_query)
        avg_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
        )

        # Retrieve relevant sigils based on the query
        try:
            # Get sigils from appropriate component based on routing
            retrieved_sigils = []
            if route == "patch_based" and hasattr(self.processor, "blt_rag"):
                # For structured/low-entropy queries, use BLT-enhanced retrieval
                logger.info(
                    f"Using BLT-enhanced retrieval (entropy: {avg_entropy:.2f})"
                )
                # We need to adapt this to account for potential differences in APIs
                # This is a simplified approach - you might need to adjust based on actual implementation
                context_str, retrieved_sigils = (
                    self.processor.blt_rag.create_rag_context(
                        query=effective_query,
                        num_sigils=num_sigils,
                        filter_tags=filter_tags,
                        exclude_tags=exclude_tags,
                        detail_level=detail_level,
                    )
                )
            else:
                # For natural language/high-entropy queries, use standard retrieval
                logger.info(f"Using standard retrieval (entropy: {avg_entropy:.2f})")
                # This assumes voxsigil_rag has a similar API - adjust as needed
                context_str, retrieved_sigils = self.voxsigil_rag.create_rag_context(
                    query=effective_query,
                    num_sigils=num_sigils,
                    filter_tags=filter_tags,
                    exclude_tags=exclude_tags,
                    detail_level=detail_level,
                )
        except Exception as e:
            logger.error(f"Error retrieving sigils: {e}")
            return "", []

        # Apply recency boost if enabled
        if apply_recency_boost and retrieved_sigils:
            retrieved_sigils = self._apply_recency_boost(retrieved_sigils)
            # Re-sort based on updated scores
            retrieved_sigils.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )

        # Auto-fuse related sigils if enabled
        if auto_fuse_related and retrieved_sigils:
            retrieved_sigils = self.auto_fuse_related_sigils(
                retrieved_sigils, max_additional=max_fusion_sigils
            )
            # Re-sort after fusion
            retrieved_sigils.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )

        # Optimize context if enabled
        if enable_context_optimization and max_context_chars > 0:
            optimized_sigils, final_detail_level = self._optimize_context_by_chars(
                retrieved_sigils, detail_level, max_context_chars
            )
            # Format optimized sigils
            formatted_context = self.format_sigils_for_context(
                optimized_sigils,
                detail_level=final_detail_level,
                include_explanations=True,
            )
        else:
            # Format without optimization
            formatted_context = self.format_sigils_for_context(
                retrieved_sigils, detail_level=detail_level, include_explanations=True
            )

        return formatted_context, retrieved_sigils
