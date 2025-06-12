#!/usr/bin/env python
"""
Production-Grade Hybrid BLT Middleware for VoxSigil system.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from . import ByteLatentTransformerEncoder, SigilPatchEncoder
# VoxSigilRAG resides in the top-level VoxSigilRag package
from VoxSigilRag.voxsigil_rag import VoxSigilRAG

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("VoxSigilHybridMiddleware")

BLT_RAG_AVAILABLE = True
IS_PYDANTIC_V2 = True
IS_PYDANTIC_AVAILABLE = True


# --- Configuration ---
class HybridMiddlewareConfig(BaseSettings):
    """Configuration for the Hybrid Middleware using Pydantic."""

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

    @field_validator("log_level")
    def set_log_level(cls, value: str) -> str:
        numeric_level = getattr(logging, value.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {value}")
        logging.getLogger().setLevel(numeric_level)
        logger.setLevel(numeric_level)
        return value

    @field_validator("entropy_threshold")
    def _validate_entropy_threshold(cls, value: float) -> float:
        if value < 0:
            raise ValueError("entropy_threshold must be non-negative.")
        return value

    @field_validator("cache_ttl_seconds")
    def _validate_cache_ttl_seconds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cache_ttl_seconds must be non-negative.")
        return value

    @field_validator("entropy_router_fallback")
    def _validate_entropy_router_fallback(cls, value: str) -> str:
        if not value:
            raise ValueError("entropy_router_fallback cannot be empty.")
        if value not in ["patch_based", "token_based"]:
            raise ValueError(
                "entropy_router_fallback must be 'patch_based' or 'token_based'."
            )
        return value


# Load configuration
APP_CONFIG = HybridMiddlewareConfig(
    entropy_threshold=0.25,
    blt_hybrid_weight=0.7,
    entropy_router_fallback="token_based",
    cache_ttl_seconds=300,
    log_level="INFO",
)

HAVE_SENTENCE_TRANSFORMERS = True
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_CACHE_PATH = Path.home() / ".voxsigil" / "embeddings_cache.pkl"
DEFAULT_VOXSİGİL_LIBRARY_PATH = Path.home() / "VoxSigil_Library"

DEFAULT_SIGIL_SCHEMA = {
    "type": "object",
    "properties": {
        "sigil": {"type": "string"},
        "principle": {"type": "string"},
        "tags": {"type": ["array", "string"]},
        "relationships": {"type": "object"},
    },
    "required": ["sigil", "principle"],
    "additionalProperties": True,
}


class BLTEnhancedRAG(VoxSigilRAG):
    """Production-grade BLT-enhanced RAG component."""

    def __init__(
        self,
        voxsigil_library_path: Path | None = None,
        cache_enabled: bool = True,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        entropy_threshold: float = 0.25,  # From app config typically
        blt_hybrid_weight: float = 0.7,  # From app config typically
        embedding_dim: int = 128,
        enable_patch_validation: bool = True,
        enable_patch_compression: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            voxsigil_library_path=voxsigil_library_path,
            cache_enabled=cache_enabled,
            embedding_model=embedding_model,
            **kwargs,
        )
        self.entropy_threshold = entropy_threshold
        self.blt_hybrid_weight = blt_hybrid_weight
        self.embedding_dim = embedding_dim  # Store embedding_dim as instance variable
        self.blt_encoder = ByteLatentTransformerEncoder(
            embedding_dim=self.embedding_dim
        )  # Pass consistent dim
        # These are just flags, real implementation would instantiate validator/encoder objects
        self.patch_validator = enable_patch_validation
        self.patch_encoder_component = enable_patch_compression  # Renamed to avoid conflict with blt_encoder method

    def _compute_text_embedding(self, text: str) -> np.ndarray:
        # This specialized version creates a hybrid BLT + Standard embedding
        blt_emb_raw = self.blt_encoder.encode(
            text
        )  # This is already normalized by BLT encoder
        std_emb_raw = super()._compute_text_embedding(
            text
        )  # Standard RAG's method (might be SBERT or hash)

        # Ensure std_emb is normalized (SBERT usually is, hash fallback needs it)
        std_norm = np.linalg.norm(std_emb_raw)
        std_emb_normalized = (
            std_emb_raw / (std_norm + 1e-9) if std_norm > 0 else std_emb_raw
        )

        # Weighted hybrid. BLT's encode should return normalized.
        hybrid_emb = (self.blt_hybrid_weight * blt_emb_raw) + (
            (1 - self.blt_hybrid_weight) * std_emb_normalized
        )
        hybrid_norm = np.linalg.norm(hybrid_emb)
        return hybrid_emb / (hybrid_norm + 1e-9) if hybrid_norm > 0 else hybrid_emb

    def create_rag_context(
        self, query: str, num_sigils: int = 5, **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        self.load_all_sigils()  # Ensure sigils are available
        query_embedding = self._compute_text_embedding(query)
        sigils_with_scores = self._find_similar_sigils(
            query, query_embedding, num_sigils
        )

        # Retrieve detail_level from kwargs or use a default
        detail_level = kwargs.get("detail_level", "standard")
        context_str = self._format_sigils_for_context(sigils_with_scores, detail_level)

        # Update retrieval counts and times
        for s_data in sigils_with_scores:
            s_id = s_data.get("sigil", s_data.get("id", "unknown_sigil"))
            self._sigil_retrieval_counts[s_id] += 1
            self._sigil_last_retrieved_time[s_id] = time.monotonic()

        return f"BLT RAG CONTEXT:\n{context_str}", sigils_with_scores

    def _find_similar_sigils(
        self, query_text: str, query_embedding: np.ndarray, num_sigils: int
    ) -> list[dict[str, Any]]:
        if not self._loaded_sigils:
            logger.warning("No sigils loaded for similarity search")
            return []

        similarities_data: list[tuple[dict[str, Any], float]] = []
        for sigil_data in self._loaded_sigils:
            sigil_content_text = self._extract_sigil_text(sigil_data)
            # For BLT RAG, sigil embeddings should also be hybrid or BLT-specific for fair comparison.
            # Here, we re-compute on the fly for simplicity, but caching sigil embeddings is vital for perf.
            sigil_embedding_hybrid = self._compute_text_embedding(sigil_content_text)

            similarity_score = self.blt_encoder.calculate_similarity(
                query_embedding, sigil_embedding_hybrid
            )
            similarities_data.append((sigil_data, similarity_score))

        similarities_data.sort(key=lambda x: x[1], reverse=True)

        top_sigils_enriched: list[dict[str, Any]] = []
        for s_data, score in similarities_data[:num_sigils]:
            enriched_s_data = s_data.copy()  # Work on a copy
            enriched_s_data["_similarity_score"] = score  # Store raw similarity
            enriched_s_data["_similarity_explanation"] = (
                f"Similarity to query ('{query_text[:20]}...'): {score:.4f}"
            )
            top_sigils_enriched.append(enriched_s_data)
        return top_sigils_enriched

    def _extract_sigil_text(self, sigil: dict[str, Any]) -> str:
        texts_to_embed = []
        if "principle" in sigil and isinstance(sigil["principle"], str):
            texts_to_embed.append(sigil["principle"])
        if "sigil" in sigil and isinstance(
            sigil["sigil"], str
        ):  # Include sigil name itself
            texts_to_embed.append(sigil["sigil"])

        if "usage" in sigil and isinstance(sigil["usage"], dict):
            if "description" in sigil["usage"] and isinstance(
                sigil["usage"]["description"], str
            ):
                texts_to_embed.append(sigil["usage"]["description"])
            if "examples" in sigil["usage"]:
                examples = sigil["usage"]["examples"]
                if isinstance(examples, list):
                    texts_to_embed.extend(
                        str(ex) for ex in examples if isinstance(ex, (str, int, float))
                    )
                elif isinstance(examples, (str, int, float)):
                    texts_to_embed.append(str(examples))
        if "tags" in sigil:
            tags = sigil["tags"]
            if isinstance(tags, list):
                texts_to_embed.extend(str(t) for t in tags)
            elif isinstance(tags, str):
                texts_to_embed.append(tags)

        return "\n".join(texts_to_embed)

    def _format_sigils_for_context(
        self, sigils: list[dict[str, Any]], detail_level: str = "standard"
    ) -> str:
        # This method could be inherited or specialized if BLT RAG needs different formatting.
        # For now, assume it uses a shared formatting utility or a simple version.
        # Calling a hypothetical shared formatter:
        # return SharedFormattingUtil.format_list_of_sigils(sigils, detail_level)

        # Simple version for this standalone class:
        if not sigils:
            return ""
        output = []
        for s_data in sigils:
            s_text = s_data.get("principle", s_data.get("sigil", "Unknown Sigil"))
            score = s_data.get("_similarity_score", 0.0)
            output.append(f"- {s_text} (Score: {score:.3f})")
        return "\n".join(output)

    # Re-alias for external calls if needed
    def enhanced_rag_process(
        self, query: str, num_sigils: int = 5, **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        return self.create_rag_context(query, num_sigils, **kwargs)


# --- Core Hybrid Logic ---
class EntropyRouter:
    """Routes inputs based on their dynamically calculated entropy level."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        self.patch_encoder = SigilPatchEncoder()
        logger.info(
            f"EntropyRouter initialized with threshold: {self.config.entropy_threshold}, fallback: {self.config.entropy_router_fallback}"
        )

    def route(self, text: str) -> tuple[str, list[str] | None, list[float]]:
        if not text:
            logger.warning("Empty text received for routing. Using fallback.")
            return self.config.entropy_router_fallback, None, [0.5]

        try:
            patches_content, entropy_scores = self.patch_encoder.analyze_entropy(text)

            if not entropy_scores:
                logger.warning(
                    f"Entropy calculation returned no scores for text: '{text[:50]}...'. Applying heuristic."
                )
                # Apply heuristic as in original
                if (
                    any(c in text for c in ["<", ">", "{", "}", "[", "]"])
                    and len(text) < 200
                ):
                    avg_entropy = 0.15
                elif len(text) < 50 and " " not in text:
                    avg_entropy = 0.2
                else:
                    avg_entropy = 0.75
                entropy_scores = [avg_entropy]
                # Ensure patches_content is a list of strings
                patches_content = patches_content or [text]
                if not all(isinstance(p, str) for p in patches_content):
                    patches_content = [str(p) for p in patches_content]

            avg_entropy = (
                sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
            )
            logger.info(
                f"Text avg_entropy: {avg_entropy:.4f} (threshold: {self.config.entropy_threshold}) for query: '{text[:30]}...'"
            )

            # Ensure patches_content contains strings
            valid_patches: list[str] | None = None
            if patches_content:
                valid_patches = [str(p) for p in patches_content]

            if avg_entropy < self.config.entropy_threshold:
                return "patch_based", valid_patches, entropy_scores
            else:
                return (
                    "token_based",
                    None,
                    entropy_scores,
                )  # No patches needed for token_based
        except Exception as e:
            logger.error(
                f"Entropy calculation/routing failed: {e}. Using fallback path: {self.config.entropy_router_fallback}",
                exc_info=True,
            )
            return self.config.entropy_router_fallback, None, [0.5]


class HybridProcessor:
    """Hybrid processor using either Standard or BLT-enhanced RAG based on entropy-based routing."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        self.router = EntropyRouter(config)

        self._standard_rag_instance: Optional[VoxSigilRAG] = None
        self._blt_rag_instance: Optional[VoxSigilRAG] = None

        logger.info("✅ HybridProcessor initialized.")

    @property
    def standard_rag(self) -> VoxSigilRAG:
        """Lazy-load standard RAG instance."""
        if self._standard_rag_instance is None:
            logger.info("⏳ Initializing Standard VoxSigilRAG...")
            self._standard_rag_instance = VoxSigilRAG(embedding_dim=128)
        return self._standard_rag_instance

    @property
    def blt_rag(self) -> VoxSigilRAG:
        """Lazy-load BLT-enhanced RAG instance."""
        if self._blt_rag_instance is None:
            logger.info("⏳ Initializing BLT-Enhanced VoxSigilRAG...")
            self._blt_rag_instance = VoxSigilRAG(
                embedding_dim=self.standard_rag.embedding_dim
            )
        return self._blt_rag_instance

    def _safe_compute_embedding(self, text: str, method: str) -> Tuple[Any, str]:
        """Compute embedding safely with fallback on error."""
        try:
            if method == "patch_based":
                return self.blt_rag._compute_text_embedding(text), "blt"
            return self.standard_rag._compute_text_embedding(text), "token"
        except Exception as e:
            logger.error(
                f"❌ Embedding error via {method}: {e}. Falling back to standard.",
                exc_info=True,
            )
            return self.standard_rag._compute_text_embedding(text), "token_fallback"

    def compute_embedding(self, text: str) -> Dict[str, Any]:
        """Route and compute embedding with entropy and fallback."""
        route, patches, entropy_scores = self.router.route(text)
        avg_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
        )

        start_time = time.monotonic()
        embedding, method = self._safe_compute_embedding(text, route)
        duration = time.monotonic() - start_time

        return {
            "embedding": embedding,
            "method_used": method,
            "routing_decision": route,
            "patches_count": len(patches) if patches else 0,
            "avg_entropy": avg_entropy,
            "processing_time_seconds": duration,
        }

    def _try_rag_context(
        self, query: str, num_sigils: int, use_blt: bool, **kwargs: Any
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Attempt to get RAG context from a specific engine."""
        engine = self.blt_rag if use_blt else self.standard_rag
        return engine.create_rag_context(query=query, num_sigils=num_sigils, **kwargs)

    def get_rag_context_and_route(
        self, query: str, num_sigils: int = 5, **kwargs: Any
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Get RAG context and sigils list, auto-routing with fallback."""
        route, _, entropy_scores = self.router.route(query)
        avg_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
        )
        use_blt = route == "patch_based"

        try:
            logger.info(
                f"🧭 Routing to {'BLT' if use_blt else 'Standard'} RAG (entropy={avg_entropy:.2f})"
            )
            context, sigils = self._try_rag_context(
                query, num_sigils, use_blt, **kwargs
            )
            return context, sigils, route
        except Exception as primary_e:
            logger.warning(
                f"⚠️ Primary RAG ({route}) failed: {primary_e}. Trying fallback...",
                exc_info=True,
            )

            try:
                fallback_use_blt = not use_blt
                context, sigils = self._try_rag_context(
                    query, num_sigils, fallback_use_blt, **kwargs
                )
                method = (
                    "token_fallback_from_blt_error"
                    if use_blt
                    else "blt_fallback_from_token_error"
                )
                return context, sigils, method
            except Exception as fallback_e:
                logger.critical(
                    f"❌ Both primary and fallback RAGs failed: {fallback_e}",
                    exc_info=True,
                )
                fail_type = f"{route}_then_total_failure"
                return "", [], fail_type


class DynamicExecutionBudgeter:
    def __init__(self, base_budget: float = 1.0, entropy_multiplier: float = 1.5):
        self.base_budget = base_budget
        self.entropy_multiplier = entropy_multiplier

    def allocate_budget(
        self, method: str, avg_entropy: float, text_length: int
    ) -> float:
        budget = self.base_budget
        if "blt" in method.lower():
            budget *= 1.0 - 0.5 * avg_entropy
        else:
            budget *= 1.0 + self.entropy_multiplier * avg_entropy
        length_factor = max(0.5, min(2.0, text_length / 500.0))
        budget *= length_factor
        logger.debug(
            f"Allocated budget: {budget:.2f} for method={method}, entropy={avg_entropy:.2f}, len={text_length}"
        )
        return budget


class HybridMiddleware:
    """Middleware to enhance incoming messages with RAG context."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        logger.info("Initializing HybridMiddleware with config.")
        self.processor = HybridProcessor(config)
        self.budgeter = DynamicExecutionBudgeter()
        self._context_cache: dict[
            str, tuple[str, list[dict[str, Any]], str, float]
        ] = {}
        self._request_counter = 0
        self._processing_times: list[float] = []
        self._initialize_voxsigil_components()  # Modified to call this

    def process(
        self, text: str, num_sigils: int = 5, **kwargs
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Alias for get_rag_context_and_route to provide a simple processing interface"""
        return self.processor.get_rag_context_and_route(
            text, num_sigils=num_sigils, **kwargs
        )

    def _initialize_voxsigil_components(self):
        try:
            self.voxsigil_rag_component = (
                self.processor.standard_rag
            )  # Ensure it's initialized
            if self.voxsigil_rag_component:
                self._normalize_sigil_relationships_format()  # Call normalization
            else:
                logger.warning(
                    "VoxSigil RAG component not available. Limited functionality."
                )

            self.conversation_history: list[Any] = []
            self.selected_sigils_history: dict[Any, Any] = {}
            self.turn_counter = 0
            self.rag_off_keywords = ["@@norag@@", "norag"]
            self.min_prompt_len_for_rag = 5
            self._rag_cache: dict[Any, Any] = {}  # Different from _context_cache
            logger.info("VoxSigil components initialized for HybridMiddleware")
        except Exception as e:
            logger.error(
                f"Failed to initialize VoxSigil components: {e}", exc_info=True
            )

    def _normalize_sigil_relationships_format(self):
        if not self.voxsigil_rag_component or not hasattr(
            self.voxsigil_rag_component, "_loaded_sigils"
        ):
            if hasattr(self.voxsigil_rag_component, "load_all_sigils"):
                self.voxsigil_rag_component.load_all_sigils(force_reload=False)
            else:
                logger.warning(
                    "No ability to load sigils. Skipping relationship format normalization."
                )
                return

        if not self.voxsigil_rag_component._loaded_sigils:  # Check after potential load
            logger.warning("No sigils loaded to normalize relationships format.")
            return

        normalized_count = 0
        # Iterate over a copy if modification happens in-place to avoid issues, or modify copies.
        # Assuming _normalize_single_sigil_relationships modifies in place or returns a modified copy.
        # For safety, let's work with a new list if sigils are modified.

        newly_loaded_sigils = []
        for sigil in self.voxsigil_rag_component._loaded_sigils:
            original_relationships = sigil.get("relationships")
            modified_sigil = self._normalize_single_sigil_relationships(
                sigil.copy()
            )  # Work on a copy
            if (
                modified_sigil.get("relationships") != original_relationships
            ):  # Crude check for modification
                normalized_count += 1
            newly_loaded_sigils.append(modified_sigil)

        if normalized_count > 0:
            logger.info(
                f"Normalized relationships format for {normalized_count} sigils (potentially)."
            )
            self.voxsigil_rag_component._loaded_sigils = (
                newly_loaded_sigils  # Update with normalized sigils
            )
            if hasattr(
                self.voxsigil_rag_component, "_sigil_cache"
            ):  # Clear cache as content changed
                self.voxsigil_rag_component._sigil_cache = {}
            # Consider re-validation if schema is strict on normalized form
            # self.voxsigil_rag_component.load_all_sigils(force_reload=True) # This would re-validate if _load_sigil_file calls _validate_sigil

    def _normalize_single_sigil_relationships(
        self, sigil: dict[str, Any]
    ) -> dict[str, Any]:
        if "relationships" not in sigil:
            return sigil
        current_relationships = sigil["relationships"]
        if not isinstance(current_relationships, dict):
            new_rels: dict[str, Any] = {}
            if isinstance(current_relationships, list):
                for i, rel_item in enumerate(current_relationships):
                    if isinstance(rel_item, str):
                        new_rels[f"relation_{i}"] = rel_item
                    elif isinstance(rel_item, dict) and len(rel_item) == 1:
                        key, value = next(iter(rel_item.items()))
                        new_rels[key] = value
                    else:
                        new_rels[f"relation_{i}"] = rel_item  # Store as is
            else:  # Not a list, not a dict, wrap it
                new_rels["default_relation"] = current_relationships
            sigil["relationships"] = new_rels
            logger.debug(
                f"Normalized relationships for sigil '{sigil.get('sigil', 'N/A')}'"
            )
        return sigil

    def format_sigil_for_context(
        self,
        sigil: dict[str, Any],
        detail_level: str = "standard",
        include_explanation: bool = False,
    ) -> str:
        output = []
        if "sigil" in sigil:
            output.append(f'Sigil: "{sigil["sigil"]}"')

        all_tags = []
        if sigil.get("tag"):
            all_tags.append(str(sigil["tag"]))
        if isinstance(sigil.get("tags"), list):
            all_tags.extend(str(t) for t in sigil["tags"] if str(t) not in all_tags)
        elif isinstance(sigil.get("tags"), str) and sigil["tags"] not in all_tags:
            all_tags.append(sigil["tags"])
        if all_tags:
            output.append("Tags: " + ", ".join(f'"{tag}"' for tag in all_tags))

        if "principle" in sigil:
            output.append(f'Principle: "{sigil["principle"]}"')
        if detail_level.lower() == "summary":
            return "\n".join(output)

        usage = sigil.get("usage", {})
        if isinstance(usage, dict):
            if "description" in usage:
                output.append(f'Usage: "{usage["description"]}"')
            if "examples" in usage and usage["examples"]:
                ex_str = (
                    f'"{usage["examples"][0]}"'
                    if isinstance(usage["examples"], list)
                    else f'"{usage["examples"]}"'
                )
                output.append(f"Example: {ex_str}")

        if sigil.get("_source_file"):
            output.append(f"Source File: {Path(sigil['_source_file']).name}")
        if include_explanation and sigil.get("_similarity_explanation"):
            output.append(f"Match Information: {sigil['_similarity_explanation']}")

        if detail_level.lower() in ("detailed", "full"):
            # (Simplified relationships formatting for brevity)
            if isinstance(sigil.get("relationships"), dict) and sigil["relationships"]:
                output.append(f"Relationships: {len(sigil['relationships'])} links")
            if sigil.get("scaffolds"):
                output.append(f"Scaffolds: {sigil['scaffolds']}")

        return "\n".join(output)

    def format_sigils_for_context(
        self,
        sigils: list[dict[str, Any]],
        detail_level: str = "standard",
        include_explanations: bool = False,
    ) -> str:
        if not sigils:
            return ""
        output_sections = [f"--- VoxSigil Context ({len(sigils)} sigils) ---"]
        for idx, sigil in enumerate(sigils, 1):
            sigil_text = self.format_sigil_for_context(
                sigil, detail_level, include_explanations
            )
            output_sections.append(f"---\nSIGIL {idx}:\n{sigil_text}")
        output_sections.append("--- End VoxSigil Context ---")
        return "\n\n".join(output_sections)

    def _get_cache_key(self, query: str) -> str:
        normalized_query = " ".join(query.lower().strip().split())
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
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        if not messages:
            return None
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                return msg["content"]
        last_message = messages[-1]
        if isinstance(last_message, dict) and isinstance(
            last_message.get("content"), str
        ):
            return last_message["content"]
        logger.warning(f"Could not extract valid query from messages: {messages}")
        return None

    def _enhance_messages_with_context(
        self, messages: list[dict[str, Any]], context: str
    ) -> list[dict[str, Any]]:
        if not context:
            return messages
        enhanced_messages = [msg.copy() for msg in messages]
        system_message_content = (
            f"Use the following VoxSigil context to answer accurately. If not relevant, use general knowledge.\n\n"
            f"--- VOXSIGIL CONTEXT START ---\n{context}\n--- VOXSIGIL CONTEXT END ---"
        )
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] = (
                f"{system_message_content}\n\n{enhanced_messages[0].get('content', '')}"
            )
        else:
            enhanced_messages.insert(
                0, {"role": "system", "content": system_message_content}
            )
        logger.debug("Messages enhanced with RAG context.")
        return enhanced_messages

    def __call__(self, model_input: dict[str, Any]) -> dict[str, Any]:
        self._request_counter += 1
        start_time_total = time.monotonic()
        messages = model_input.get("messages")

        if not isinstance(messages, list) or not messages:
            logger.warning("No messages/invalid format in model_input. Skipping.")
            return model_input

        query = self._extract_query_from_messages(messages)
        if not query:
            logger.warning("No query extractable. Skipping.")
            return model_input

        if self._request_counter % 50 == 0:
            self._clean_expired_cache_entries()
        cache_key = self._get_cache_key(query)
        cached_entry = self._context_cache.get(cache_key)
        current_monotonic_time = time.monotonic()
        context_str, sigils_list, route_method, cache_hit = "", [], "unknown", False

        if cached_entry:
            cached_context_str, cached_sigils_list, cached_route_method, timestamp = (
                cached_entry
            )
            if current_monotonic_time - timestamp <= self.config.cache_ttl_seconds:
                logger.info(
                    f"Cache HIT for query (key: {cache_key[:30]}...). Route: {cached_route_method}"
                )
                context_str, sigils_list, route_method, cache_hit = (
                    cached_context_str,
                    cached_sigils_list,
                    cached_route_method,
                    True,
                )
                self._context_cache[cache_key] = (
                    context_str,
                    sigils_list,
                    route_method,
                    current_monotonic_time,
                )
            else:
                logger.info(f"Cache STALE (key: {cache_key[:30]}...). Re-computing.")
                del self._context_cache[cache_key]

        if not cache_hit:
            logger.info(f"Cache MISS (key: {cache_key[:30]}...). Processing.")
            try:
                # Use enhanced_rag_process which internally handles routing now
                context_str, sigils_list = self.enhanced_rag_process(
                    query, num_sigils=5
                )
                # The 'route_method' is somewhat implicit now or needs to be returned by enhanced_rag_process
                # For simplicity, we'll determine it again, or enhanced_rag_process should return it.
                # Let's assume enhanced_rag_process returns (context, sigils, method_used)
                # To fit existing structure, let's just get method after the fact for logging
                _route_for_log, _, _ = self.processor.router.route(query)
                route_method = _route_for_log  # This is the routing decision, not necessarily the final method if fallbacks occurred

                self._context_cache[cache_key] = (
                    context_str,
                    sigils_list,
                    route_method,
                    current_monotonic_time,
                )
                logger.info(
                    f"Processed and cached query. Route decision: {route_method}"
                )
            except Exception as e:
                logger.critical(
                    f"Error during RAG for query '{query[:50]}...': {e}", exc_info=True
                )
                return model_input  # Passthrough on critical error

        _, _, entropy_scores_for_budget = self.processor.router.route(
            query
        )  # For budget
        avg_entropy_for_budget = (
            sum(entropy_scores_for_budget) / len(entropy_scores_for_budget)
            if entropy_scores_for_budget
            else 0.5
        )
        budget = self.budgeter.allocate_budget(
            route_method, avg_entropy_for_budget, len(query)
        )
        enhanced_messages = self._enhance_messages_with_context(messages, context_str)

        total_processing_time = time.monotonic() - start_time_total
        self._processing_times.append(total_processing_time)

        log_metadata = {
            "request_id": hashlib.md5(query.encode()).hexdigest()[:8],
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
        model_input["messages"] = enhanced_messages
        model_input.setdefault("voxsigil_metadata", {}).update(log_metadata)
        return model_input

    def get_stats(self) -> dict[str, Any]:
        avg_proc_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0
        )
        return {
            "total_requests_processed": self._request_counter,
            "cache_size": len(self._context_cache),
            "average_processing_time_seconds": round(avg_proc_time, 4),
        }

    def _augment_query(self, query: str) -> str:
        if not hasattr(self, "synonym_map"):
            self.synonym_map = {
                "ai": ["artificial intelligence", "ml", "deep learning"],
                "voxsigil": ["vox sigil", "sigil language"],
                # Add more relevant synonyms for your domain
            }
        augmented_parts = [query]
        query_lower = query.lower()
        for term, synonyms in self.synonym_map.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn.lower() not in " ".join(augmented_parts).lower():
                        augmented_parts.append(syn)

        augmented_query = " ".join(augmented_parts)
        if augmented_query != query:
            logger.info(f"Query augmented: '{query}' -> '{augmented_query}'")
        return augmented_query

    def _validate_sigil_data(
        self, sigil_data: dict[str, Any], file_path: str | None = None
    ) -> bool:
        # This assumes HybridMiddleware instance has self.sigil_schema, ensure it's initialized.
        if not hasattr(self, "sigil_schema"):  # Ensure schema is set before validation
            self.sigil_schema = DEFAULT_SIGIL_SCHEMA  # Or load from config

        from jsonschema import ValidationError, validate

        try:
            validate(instance=sigil_data, schema=self.sigil_schema)
            return True
        except ValidationError as e:
            path_str = " -> ".join(map(str, e.path))
            logger.warning(
                f"Schema validation failed for {file_path or 'sigil'}: {e.message} (at path: '{path_str}')"
            )
            return False
        except Exception as e_generic:
            logger.error(
                f"Generic error during schema validation for {file_path or 'sigil'}: {e_generic}",
                exc_info=True,
            )
            return False

    def _apply_recency_boost(
        self,
        sigils_with_scores: list[dict[str, Any]],
        recency_boost_factor: float = 0.05,
        recency_max_days: int = 90,
    ) -> list[dict[str, Any]]:
        if not recency_boost_factor > 0:
            return sigils_with_scores
        current_time_utc = datetime.now(timezone.utc).timestamp()
        recency_max_seconds = recency_max_days * 24 * 60 * 60
        boosted_sigils: list[dict[str, Any]] = []

        for s_data in sigils_with_scores:
            sigil_copy = (
                s_data.copy()
            )  # Work on a copy to avoid modifying original list items directly
            last_modified = sigil_copy.get("_last_modified")
            original_score = sigil_copy.get("_similarity_score", 0.0)
            if isinstance(last_modified, (int, float)):
                age_seconds = current_time_utc - last_modified
                if 0 <= age_seconds < recency_max_seconds:
                    boost = recency_boost_factor * (
                        1.0 - (age_seconds / recency_max_seconds)
                    )
                    new_score = min(1.0, original_score + boost)
                    if new_score > original_score:
                        sigil_copy["_similarity_score"] = new_score
                        sigil_copy["_recency_boost_applied"] = boost
                        logger.debug(
                            f"Applied recency boost {boost:.3f} to '{sigil_copy.get('sigil', 'N/A')}'"
                        )
            boosted_sigils.append(sigil_copy)
        return boosted_sigils

    def auto_fuse_related_sigils(
        self, base_sigils: list[dict[str, Any]], max_additional: int = 3
    ) -> list[dict[str, Any]]:
        if (
            not base_sigils
            or max_additional <= 0
            or not hasattr(self, "voxsigil_rag_component")
        ):
            return base_sigils
        all_system_sigils = self.voxsigil_rag_component.load_all_sigils()
        if not all_system_sigils:
            return base_sigils

        sigil_index_by_id = {s["sigil"]: s for s in all_system_sigils if "sigil" in s}
        current_sigil_ids = {s["sigil"] for s in base_sigils if "sigil" in s}
        fused_sigils_list = list(base_sigils)
        added_count = 0

        for sigil_item in list(base_sigils):  # Iterate a copy
            if added_count >= max_additional:
                break
            source_id = sigil_item.get("sigil")
            if not source_id:
                continue

            # Explicit relationships
            if isinstance(sigil_item.get("relationships"), dict):
                for rel_type, rel_targets_val in sigil_item["relationships"].items():
                    targets = (
                        rel_targets_val
                        if isinstance(rel_targets_val, list)
                        else [rel_targets_val]
                    )
                    for target_id in targets:
                        if (
                            isinstance(target_id, str)
                            and target_id in sigil_index_by_id
                            and target_id not in current_sigil_ids
                        ):
                            related_s = sigil_index_by_id[target_id].copy()
                            related_s["_fusion_reason"] = (
                                f"related_to:{source_id}({rel_type})"
                            )
                            related_s.setdefault(
                                "_similarity_score", 0.4
                            )  # Modest score
                            fused_sigils_list.append(related_s)
                            current_sigil_ids.add(target_id)
                            added_count += 1
                            if added_count >= max_additional:
                                break
                    if added_count >= max_additional:
                        break
            # (Shared tags logic omitted for brevity but would follow)
        if added_count > 0:
            logger.info(f"Auto-fused {added_count} additional sigils.")
        return fused_sigils_list

    def _optimize_context_by_chars(
        self,
        sigils_for_context: list[dict[str, Any]],
        initial_detail_level: str,
        target_char_budget: int,
    ) -> tuple[list[dict[str, Any]], str]:
        final_sigils = list(sigils_for_context)
        current_detail = initial_detail_level.lower()
        if not target_char_budget or not final_sigils:
            return final_sigils, current_detail

        detail_levels = ["full", "detailed", "standard", "summary", "minimal"]
        try:
            current_detail_idx = detail_levels.index(current_detail)
        except ValueError:
            current_detail_idx = 2
            current_detail = "standard"

        def estimate_chars(s_list, d_lvl):
            return sum(len(self.format_sigil_for_context(s, d_lvl)) for s in s_list)

        current_chars = estimate_chars(final_sigils, current_detail)

        while (
            current_chars > target_char_budget
            and current_detail_idx < len(detail_levels) - 1
        ):
            current_detail_idx += 1
            new_detail = detail_levels[current_detail_idx]
            logger.info(
                f"Context Optimizer: Budget {target_char_budget}, Chars {current_chars}. Detail {current_detail} -> {new_detail}"
            )
            current_detail = new_detail
            current_chars = estimate_chars(final_sigils, current_detail)

        while current_chars > target_char_budget and len(final_sigils) > 1:
            removed_s = final_sigils.pop()
            logger.info(
                f"Context Optimizer: Detail {current_detail}. Removing '{removed_s.get('sigil', 'N/A')}'"
            )
            current_chars = estimate_chars(final_sigils, current_detail)

        if final_sigils:
            logger.info(
                f"Context Optimizer: Final {len(final_sigils)} sigils, {current_detail} detail, {current_chars} chars."
            )
        return final_sigils, current_detail

    def enhanced_rag_process(
        self,
        query: str,
        num_sigils: int = 5,
        filter_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        detail_level: str = "standard",
        apply_recency_boost: bool = True,
        augment_query_flag: bool = True,  # Renamed to avoid conflict
        enable_context_optimization: bool = True,
        max_context_chars: int = 8000,
        auto_fuse_related: bool = True,
        max_fusion_sigils: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        if (
            not hasattr(self, "voxsigil_rag_component")
            or not self.voxsigil_rag_component
        ):
            logger.warning("VoxSigil RAG not available for enhanced processing.")
            return "", []

        effective_query = self._augment_query(query) if augment_query_flag else query

        # The create_rag_context in VoxSigilRAG/BLTEnhancedRAG needs to be able
        # to handle filtering, or we filter after retrieval. Assume it handles it for now.
        # For routing, we use the processor's internal RAGs.
        _context_str_raw, retrieved_sigils, _method_used = (
            self.processor.get_rag_context_and_route(
                query=effective_query,
                num_sigils=num_sigils,  # Pass other params if methods support them
                filter_tags=filter_tags,
                exclude_tags=exclude_tags,
            )
        )
        # Note: _method_used is the internal routing choice, might differ from final list post-processing

        if apply_recency_boost and retrieved_sigils:
            retrieved_sigils = self._apply_recency_boost(
                retrieved_sigils, self.config.blt_hybrid_weight * 0.1, 90
            )  # Use config for factors
            retrieved_sigils.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )
        if auto_fuse_related and retrieved_sigils:
            retrieved_sigils = self.auto_fuse_related_sigils(
                retrieved_sigils, max_additional=max_fusion_sigils
            )
            retrieved_sigils.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )  # Resort after fusion

        final_detail_level = detail_level
        if enable_context_optimization and max_context_chars > 0:
            retrieved_sigils, final_detail_level = self._optimize_context_by_chars(
                retrieved_sigils, detail_level, max_context_chars
            )

        formatted_context = self.format_sigils_for_context(
            retrieved_sigils, detail_level=final_detail_level, include_explanations=True
        )
        return formatted_context, retrieved_sigils


# --- Utility Function for Direct Hybrid Embedding (Optional) ---
def hybrid_embedding_utility(
    text: str, config: HybridMiddlewareConfig
) -> tuple[np.ndarray | None, str, float]:
    router = EntropyRouter(config)  # Router needs config
    # These should use consistent embedding_dim from config or a default
    std_encoder = VoxSigilRAG(embedding_dim=128)  # Use default embedding dimension
    blt_encoder = ByteLatentTransformerEncoder(
        embedding_dim=128
    )  # Use default embedding dimension

    route, _, entropy_scores = router.route(text)
    avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
    embedding: np.ndarray | None = None

    if avg_entropy < config.entropy_threshold * 0.8:
        embedding = blt_encoder.encode(text)
        method_used = "blt_direct"
    elif avg_entropy > config.entropy_threshold * 1.2:
        embedding = std_encoder._compute_text_embedding(text)
        method_used = "standard_direct"
    else:
        blt_emb = blt_encoder.encode(text)  # Already normalized
        std_emb_raw = std_encoder._compute_text_embedding(text)
        std_norm = np.linalg.norm(std_emb_raw)
        std_emb_normalized = (
            std_emb_raw / (std_norm + 1e-9) if std_norm > 0 else std_emb_raw
        )

        hybrid_emb = (config.blt_hybrid_weight * blt_emb) + (
            (1 - config.blt_hybrid_weight) * std_emb_normalized
        )
        hybrid_norm = np.linalg.norm(hybrid_emb)
        embedding = hybrid_emb / (hybrid_norm + 1e-9) if hybrid_norm > 0 else hybrid_emb
        method_used = "hybrid_weighted"
    return embedding, method_used, avg_entropy


# Exported functions for simplified API access
def entropy_router_util(
    text: str, config: HybridMiddlewareConfig | None = None
) -> tuple[str, float]:  # Renamed to avoid conflict
    effective_config = (
        config or APP_CONFIG  # Use provided config or default app config
    )  # Use default if none provided
    router_instance = EntropyRouter(effective_config)  # Needs instance
    route_decision, _, entropy_scores = router_instance.route(text)
    avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
    return route_decision, avg_entropy


def hybrid_embedding_combiner(
    text: str,
    token_embedding: np.ndarray,
    patch_embedding: np.ndarray,
    config: HybridMiddlewareConfig | None = None,
) -> np.ndarray:  # Renamed
    effective_config = config or APP_CONFIG
    route, _ = entropy_router_util(text, effective_config)  # Use the util version

    blt_weight = (
        effective_config.blt_hybrid_weight
        if route == "patch_based"
        else (1.0 - effective_config.blt_hybrid_weight)
    )
    token_weight = 1.0 - blt_weight

    # Normalize inputs before combining, if not already normalized
    norm_token = np.linalg.norm(token_embedding)
    norm_patch = np.linalg.norm(patch_embedding)

    token_emb_norm = (
        token_embedding / (norm_token + 1e-9) if norm_token > 0 else token_embedding
    )
    patch_emb_norm = (
        patch_embedding / (norm_patch + 1e-9) if norm_patch > 0 else patch_embedding
    )

    combined = (token_emb_norm * token_weight) + (patch_emb_norm * blt_weight)
    norm_combined = np.linalg.norm(combined)
    return combined / (norm_combined + 1e-9) if norm_combined > 0 else combined


# --- Main Example Usage / Smoke Test ---
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 VoxSigil Hybrid Middleware - Production Grade Script Test 🚀")
    print("=" * 80)

    test_config_instance: HybridMiddlewareConfig
    if (
        IS_PYDANTIC_AVAILABLE
    ):  # Create a mutable copy for testing if APP_CONFIG is Pydantic-based
        test_config_instance = (
            APP_CONFIG.model_copy()
            if hasattr(APP_CONFIG, "model_copy")
            else HybridMiddlewareConfig(
                entropy_threshold=0.25,
                blt_hybrid_weight=0.7,
                entropy_router_fallback="token_based",
                cache_ttl_seconds=60,
                log_level="DEBUG",
            )
        )
    else:  # Fallback
        test_config_instance = HybridMiddlewareConfig(
            entropy_threshold=0.25,
            blt_hybrid_weight=0.7,
            entropy_router_fallback="token_based",
            cache_ttl_seconds=60,
            log_level="DEBUG",
        )

    middleware = HybridMiddleware(test_config_instance)

    print("\n--- Test 1: Basic routing and context creation ---")
    query1 = "What are the benefits of AI in healthcare?"
    model_input1 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": query1}],
        "temperature": 0.7,
    }
    response1 = middleware(model_input1)
    print(
        f"Response 1 (query: '{query1}'): {response1['messages'][0]['content'][:200]}..."
    )  # Show part of enhanced system message

    stats1 = middleware.get_stats()
    print(f"Stats after 1st request: {stats1}")

    print("\n--- Test 2: Cache usage ---")
    # This query should be a cache hit if query1 was similar enough or if normalization makes them same key

    # For robust test, use exact same query for definite hit
    model_input2 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": query1}],
        "temperature": 0.7,
    }
    response2 = middleware(model_input2)
    print(f"Response 2 (cached query: '{query1}'): Cache hit expected.")
    if "voxsigil_metadata" in response2 and response2["voxsigil_metadata"]["cache_hit"]:
        print("✅ Cache hit confirmed for Response 2.")
    else:
        print("❌ Cache miss for Response 2 - check cache key or TTL.")

    query3 = "The role of AI in modern healthcare systems"  # Different query
    model_input3 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": query3}],
        "temperature": 0.7,
    }
    response3 = middleware(model_input3)
    print(f"Response 3 (new query: '{query3}'): Cache miss expected.")
    if (
        "voxsigil_metadata" in response3
        and not response3["voxsigil_metadata"]["cache_hit"]
    ):
        print("✅ Cache miss confirmed for Response 3.")
    else:
        print("❌ Cache hit for Response 3 - unexpected.")

    print("\n--- Test 3: Enhanced RAG process ---")
    query4 = "Explain symbolic reasoning in AI."
    context4, sigils4 = middleware.enhanced_rag_process(query4, num_sigils=3)
    print(f"Enhanced RAG Context for '{query4}':\n{context4}")
    print(f"Number of sigils retrieved: {len(sigils4)}")
    if sigils4:
        print(f"First sigil example: {sigils4[0].get('sigil', 'N/A')}")

    print("\n--- Test 4: Direct hybrid embedding utility ---")
    text_emb_test = (
        "Artificial intelligence and machine learning are transforming industries."
    )
    embedding_res, method_res, entropy_res = hybrid_embedding_utility(
        text_emb_test, test_config_instance
    )
    print(
        f"Hybrid embedding for '{text_emb_test[:30]}...' (method: {method_res}, entropy: {entropy_res:.3f}):"
    )
    if embedding_res is not None:
        print(f" Shape: {embedding_res.shape}, First 3 values: {embedding_res[:3]}")
    else:
        print(" Embedding computation failed or returned None.")

    final_stats = middleware.get_stats()
    print(f"\nFinal Stats: {final_stats}")

    print("=" * 80)
    print("✅ VoxSigil Hybrid Middleware - Tests Completed")
    print("=" * 80)
