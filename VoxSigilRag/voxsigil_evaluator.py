#!/usr/bin/env python
"""
VoxSigil Unified System: RAG, Middleware, and Evaluator.

This script combines functionality for:
1. VoxSigil Retrieval-Augmented Generation (RAG) core.
2. VoxSigil Middleware for runtime RAG injection in conversations.
3. VoxSigil Response Evaluator with enhanced metrics.
"""

import hashlib
import html
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


# --- Centralized Configuration (NEW FEATURE 1) ---
class VoxSigilConfig:
    """Centralized configuration for the VoxSigil system."""

    def __init__(self):
        self.DEFAULT_VOXSİGİL_LIBRARY_PATH = (
            Path(__file__).resolve().parent / "VoxSigil-Library"
        )
        self.DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.DEFAULT_EMBEDDINGS_CACHE_PATH = (
            Path(__file__).resolve().parent / "cache" / "embeddings_cache.npz"
        )
        self.LOG_LEVEL = logging.INFO
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # For Evaluator visual diff
        self.HTML_DIFF_STYLES = """
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .variant {{ flex: 1; min-width: 300px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 5px #eee; }}
        .variant-header {{ background-color: #f0f4f8; color: #2c3e50; padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 1px solid #ddd; border-top-left-radius: 5px; border-top-right-radius: 5px;}}
        .metrics {{ background-color: #f9f9f9; padding: 10px; margin-top: 15px; border-radius: 3px; border: 1px solid #eee;}}
        .metric-header {{font-weight: bold; color: #34495e; margin-bottom: 8px; border-bottom: 1px solid #ececec; padding-bottom: 4px;}}
        .metric {{ margin-bottom: 5px; font-size: 0.9em; }}
        .score {{ font-weight: bold; }}
        .high {{ color: #27ae60; }} /* Green */
        .medium {{ color: #f39c12; }} /* Orange */
        .low {{ color: #c0392b; }} /* Red */
        .highlight {{ background-color: #fff3cd; padding: 0 2px; }} /* Light yellow */
        .sigil-mention {{ background-color: #d4efdf; padding: 0 2px; border-radius: 3px; border: 1px solid #a9dfbf;}} /* Light green */
        .tag-mention {{ background-color: #d6eaf8; padding: 0 2px; border-radius: 3px; border: 1px solid #aed6f1;}} /* Light blue */
        .principle-match {{ background-color: #ebdef0; padding: 0 2px; border-radius: 3px; border: 1px solid #d7bde2;}} /* Light purple */
        .content {{ white-space: pre-wrap; background-color: #fff; padding:10px; border-radius:3px; border: 1px solid #f0f0f0;}}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; font-size: 0.9em;}}
        th {{ background-color: #ecf0f1; color: #34495e; }}
        """


# Instantiate global config
VS_CONFIG = VoxSigilConfig()

# --- Configure Logging ---
logging.basicConfig(level=VS_CONFIG.LOG_LEVEL, format=VS_CONFIG.LOG_FORMAT)
logger = logging.getLogger("VoxSigilSystem")  # Unified logger name

# --- Optional Dependencies (Lazy Loading) ---
HAVE_SENTENCE_TRANSFORMERS = False
sbert_util_for_grounding = None
SentenceTransformer = None
cosine_similarity = None

def _load_sentence_transformers():
    """Lazy load sentence transformers to avoid blocking imports"""
    global HAVE_SENTENCE_TRANSFORMERS, sbert_util_for_grounding, SentenceTransformer, cosine_similarity
    if not HAVE_SENTENCE_TRANSFORMERS:
        try:
            import sentence_transformers.util as sbert_util_for_grounding  # For Middleware/Evaluator grounding
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity  # For Enhanced Evaluator
            HAVE_SENTENCE_TRANSFORMERS = True
            logger.info("Sentence Transformers and scikit-learn loaded successfully.")
        except ImportError as e:
            logger.warning(f"Sentence Transformers not available: {e}")
            HAVE_SENTENCE_TRANSFORMERS = False

# Try to load on import, but don't fail if not available
try:
    _load_sentence_transformers()
except Exception as e:
    logger.warning(f"Failed to load sentence transformers during import: {e}")
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.info("Sentence Transformers and scikit-learn available.")
except ImportError:
    logger.warning(
        "Sentence Transformers or scikit-learn not available. Semantic features will be limited."
    )
    sbert_util_for_grounding = None  # Define for type hinting
    cosine_similarity = None  # Define for type hinting

HAVE_JSONSCHEMA = False
try:
    from jsonschema import ValidationError, validate

    HAVE_JSONSCHEMA = True
    logger.info("jsonschema available for Sigil schema validation.")
except ImportError:
    HAVE_JSONSCHEMA = False

    def validate(instance: Any, schema: Dict[str, Any]) -> None:
        pass

    class ValidationError(Exception):
        pass  # type: ignore

    logger.warning(
        "jsonschema not available. Sigil schema validation will be skipped. Install with: pip install jsonschema"
    )


# --- Custom Exceptions (NEW FEATURE 7 - Partial) ---
class VoxSigilError(Exception):
    """Base exception for VoxSigil system errors."""

    pass


class VoxSigilRAGError(VoxSigilError):
    """Exception for RAG module errors."""

    pass


class VoxSigilMiddlewareError(VoxSigilError):
    """Exception for Middleware errors."""

    pass


class VoxSigilEvaluationError(VoxSigilError):
    """Exception for Evaluator errors."""

    pass


# --- Tokenizer Placeholder (NEW FEATURE 5) ---
_global_tokenizer = None
_tokenizer_name = None


def set_global_tokenizer(
    tokenizer_instance: Optional[Any], name: Optional[str] = None
) -> None:
    """Sets a global tokenizer instance (e.g., from tiktoken)."""
    global _global_tokenizer, _tokenizer_name
    _global_tokenizer = tokenizer_instance
    _tokenizer_name = name
    if tokenizer_instance:
        logger.info(f"Global tokenizer '{name or 'Custom'}' set for VoxSigil system.")
    else:
        logger.info("Global tokenizer cleared.")


def get_global_tokenizer() -> Optional[Any]:
    """Gets the global tokenizer instance."""
    return _global_tokenizer


def count_tokens(text: str) -> int:
    """Counts tokens using the global tokenizer if set, otherwise estimates characters."""
    if _global_tokenizer and hasattr(_global_tokenizer, "encode"):
        try:
            return len(_global_tokenizer.encode(text))
        except Exception as e:
            logger.warning(
                f"Error using global tokenizer ({_tokenizer_name}): {e}. Falling back to char count."
            )
    # Fallback: 1 token ~ 4 chars (very rough estimate)
    return len(text) // 4


# --- VoxSigilRAG Class (from voxsigil_rag.py) ---
DEFAULT_SIGIL_SCHEMA = {  # Copied from RAG, ensure it's used
    "type": "object",
    "properties": {
        "sigil": {"type": "string"},
        "tag": {"type": "string"},
        "tags": {"type": ["array", "string"], "items": {"type": "string"}},
        "principle": {"type": "string"},
        "usage": {"type": "object"},
        "prompt_template": {"type": "object"},
        "relationships": {"type": "object"},
        "_source_file": {"type": "string"},
        "_last_modified": {"type": "number"},
        "_similarity_score": {"type": "number"},
        "_recency_boost_applied": {"type": "number"},
        "creation_date": {"type": "string", "format": "date-time"},  # For NEW FEATURE 6
        "version": {"type": "string"},  # For NEW FEATURE 6
        "author": {"type": "string"},  # For NEW FEATURE 6
    },
    "required": ["sigil", "principle"],
}


class VoxSigilRAG:
    def __init__(
        self,
        voxsigil_library_path: Optional[Path] = None,
        cache_enabled: bool = True,
        embedding_model_name: str = VS_CONFIG.DEFAULT_EMBEDDING_MODEL,
        recency_boost_factor: float = 0.05,
        recency_max_days: int = 90,
        default_max_context_chars: int = 8000,
    ):
        self.voxsigil_library_path = (
            voxsigil_library_path or VS_CONFIG.DEFAULT_VOXSİGİL_LIBRARY_PATH
        )
        self.cache_enabled = cache_enabled
        self._sigil_cache: Dict[str, Dict[str, Any]] = {}
        self._loaded_sigils: Optional[List[Dict[str, Any]]] = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._embeddings_cache_path = VS_CONFIG.DEFAULT_EMBEDDINGS_CACHE_PATH

        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_model_name = embedding_model_name
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"[RAG] Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.error(
                    f"[RAG] Error loading embedding model {embedding_model_name}: {e}"
                )
                raise VoxSigilRAGError(f"Failed to load embedding model: {e}") from e

        if self.cache_enabled:
            self._embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_embeddings_cache()

        self.sigil_schema = DEFAULT_SIGIL_SCHEMA
        self._sigil_retrieval_counts = defaultdict(int)
        self._sigil_last_retrieved_time: Dict[str, float] = {}
        self.recency_boost_factor = recency_boost_factor
        self.recency_max_days_seconds = recency_max_days * 24 * 60 * 60
        self.default_max_context_chars = default_max_context_chars
        self.synonym_map: Dict[str, List[str]] = {
            "ai": ["artificial intelligence", "machine learning"],
            "voxsigil": ["vox sigil language", "sigil prompt language"],
        }
        self.default_rag_params = {  # for evaluate_batch usage
            "num_sigils": 5,
            "min_score_threshold": 0.4,
            "detail_level": "standard",
        }

    def _validate_sigil_data(self, sigil_data: Dict[str, Any], file_path: Path) -> bool:
        if not HAVE_JSONSCHEMA:
            return True
        try:
            validate(instance=sigil_data, schema=self.sigil_schema)
            return True
        except ValidationError as e:
            path_str = " -> ".join(map(str, e.path))
            logger.warning(
                f"[RAG] Schema validation failed for {file_path}: {e.message} (at path: '{path_str}')"
            )
            return False
        except Exception as e_generic:
            logger.error(
                f"[RAG] Generic error during schema validation for {file_path}: {e_generic}"
            )
            return False

    def _load_sigil_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            file_mod_time = file_path.stat().st_mtime
        except FileNotFoundError:
            logger.error(f"[RAG] File not found: {file_path}")
            return None

        cache_key = f"{str(file_path)}::{file_mod_time}"
        if self.cache_enabled and cache_key in self._sigil_cache:
            return self._sigil_cache[cache_key]

        try:
            ext = file_path.suffix.lower()
            with open(file_path, "r", encoding="utf-8") as f:
                if ext in [".yaml", ".yml", ".voxsigil"]:
                    sigil_data = yaml.safe_load(f)
                elif ext == ".json":
                    sigil_data = json.load(f)
                else:
                    return None
            if not isinstance(sigil_data, dict):
                return None
            if not self._validate_sigil_data(sigil_data, file_path):
                return None

            sigil_data["_last_modified"] = file_mod_time
            sigil_data.setdefault(
                "_source_file", str(file_path)
            )  # NEW FEATURE 6 - ensure source
            # NEW FEATURE 6 - Check for other metadata
            for meta_key in ["creation_date", "version", "author"]:
                if meta_key not in sigil_data:
                    sigil_data.setdefault(
                        meta_key, None
                    )  # Default to None if not present

            if self.cache_enabled:
                self._sigil_cache[cache_key] = sigil_data
            return sigil_data
        except Exception as e:
            logger.error(f"[RAG] Error loading sigil {file_path}: {e}")
            return None

    def load_all_sigils(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        if self._loaded_sigils is not None and not force_reload:
            return self._loaded_sigils

        logger.info(f"[RAG] Loading VoxSigil sigils from {self.voxsigil_library_path}")
        sigil_dirs = [
            self.voxsigil_library_path / "core",
            self.voxsigil_library_path / "pglyph",
            self.voxsigil_library_path / "sigils",
            self.voxsigil_library_path / "examples",
            self.voxsigil_library_path / "constructs",
            self.voxsigil_library_path / "loader",
            self.voxsigil_library_path / "memory",
            self.voxsigil_library_path / "tags",
            self.voxsigil_library_path / "scaffolds",
        ]
        sigil_files: List[Path] = []
        for dir_path in sigil_dirs:
            if dir_path.exists() and dir_path.is_dir():
                for ext in ["*.voxsigil", "*.yaml", "*.yml", "*.json"]:
                    sigil_files.extend(list(dir_path.rglob(ext)))

        loaded_sigils_list = [
            s for f in sigil_files if (s := self._load_sigil_file(f)) is not None
        ]

        logger.info(
            f"[RAG] Successfully loaded {len(loaded_sigils_list)} sigils from {len(sigil_files)} files."
        )
        self._loaded_sigils = loaded_sigils_list
        return self._loaded_sigils

    def format_sigil_for_prompt(
        self, sigil: Dict[str, Any], detail_level: str = "standard"
    ) -> str:
        # (Method content from RAG script, condensed for brevity here)
        output = [f'Sigil: "{sigil["sigil"]}"'] if "sigil" in sigil else []
        all_tags = set()
        if "tag" in sigil and sigil["tag"]:
            all_tags.add(str(sigil["tag"]))
        if "tags" in sigil and sigil["tags"]:
            tags_val = sigil["tags"]
            if isinstance(tags_val, list):
                all_tags.update(str(t) for t in tags_val)
            elif isinstance(tags_val, str):
                all_tags.add(tags_val)
        if all_tags:
            output.append(
                "Tags: " + ", ".join(f'"{t}"' for t in sorted(list(all_tags)))
            )
        if "principle" in sigil:
            output.append(f'Principle: "{sigil["principle"]}"')
        if detail_level.lower() == "summary":
            return "\n".join(output)
        # ... (Standard and Full details as in original RAG)
        if "usage" in sigil and isinstance(sigil["usage"], dict):
            if "description" in sigil["usage"]:
                output.append(f'Usage: "{sigil["usage"]["description"]}"')
        if "_source_file" in sigil:
            output.append(f"Source File: {Path(sigil['_source_file']).name}")
        # ... (Full detail logic)
        return "\n".join(output)

    def _augment_query(self, query: str) -> str:  # From RAG script
        # ... (content of _augment_query)
        augmented_parts = [query]
        query_lower = query.lower()
        for term, synonyms in self.synonym_map.items():
            if term in query_lower:
                for syn in synonyms:
                    if (
                        syn not in query_lower
                        and syn.lower() not in " ".join(augmented_parts).lower()
                    ):
                        augmented_parts.append(syn)
        return " ".join(augmented_parts)

    def _apply_recency_boost(
        self, sigils_with_scores: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:  # From RAG script
        # ... (content of _apply_recency_boost)
        if not self.recency_boost_factor > 0:
            return sigils_with_scores
        current_time_utc = datetime.now(timezone.utc).timestamp()
        for item in sigils_with_scores:
            last_modified_utc = item.get("_last_modified")
            original_score = item.get("_similarity_score", 0.0)
            if last_modified_utc and isinstance(last_modified_utc, (int, float)):
                age_seconds = current_time_utc - last_modified_utc
                if 0 <= age_seconds < self.recency_max_days_seconds:
                    boost_multiplier = 1.0 - (
                        age_seconds / self.recency_max_days_seconds
                    )
                    recency_bonus = self.recency_boost_factor * boost_multiplier
                    new_score = min(1.0, original_score + recency_bonus)
                    if new_score > original_score:
                        item["_similarity_score"] = new_score
                        item["_recency_boost_applied"] = recency_bonus
        return sigils_with_scores

    def _optimize_context_by_chars(
        self,
        sigils_for_context: List[Dict[str, Any]],
        initial_detail_level: str,
        target_char_budget: int,
    ) -> Tuple[List[Dict[str, Any]], str]:  # From RAG script
        # ... (content, using self.format_sigil_for_prompt for char estimation)
        final_sigils = list(sigils_for_context)
        current_detail_level = initial_detail_level.lower()
        if not target_char_budget or not final_sigils:
            return final_sigils, current_detail_level

        def estimate_chars(s_list: List[Dict[str, Any]], d_level: str) -> int:
            return sum(len(self.format_sigil_for_prompt(s, d_level)) for s in s_list)

        detail_levels_ordered = ["full", "standard", "summary"]
        current_detail_idx = (
            detail_levels_ordered.index(current_detail_level)
            if current_detail_level in detail_levels_ordered
            else 1
        )
        current_chars = estimate_chars(final_sigils, current_detail_level)

        while (
            current_chars > target_char_budget
            and current_detail_idx < len(detail_levels_ordered) - 1
        ):
            current_detail_idx += 1
            current_detail_level = detail_levels_ordered[current_detail_idx]
            current_chars = estimate_chars(final_sigils, current_detail_level)

        while current_chars > target_char_budget and len(final_sigils) > 1:
            final_sigils.pop()
            current_chars = estimate_chars(final_sigils, current_detail_level)

        return final_sigils, current_detail_level

    # NEW FEATURE 2: Lightweight Retrieval Mode for RAG
    def _lightweight_retrieve_sigils(
        self,
        query: str,
        num_sigils: int,
        filter_tags: Optional[List[str]] = None,
        tag_operator: str = "OR",
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"[RAG] Performing lightweight retrieval for query: '{query[:50]}...'"
        )
        all_loaded_sigils = self.load_all_sigils()
        candidates = []
        query_terms = set(q.lower() for q in query.split() if len(q) > 2)

        # Apply tag filtering first
        if filter_tags:
            norm_filter_tags = {tag.lower() for tag in filter_tags}
            filtered_by_tags = []
            for sigil in all_loaded_sigils:
                s_tags_set = set()
                if "tag" in sigil and sigil["tag"]:
                    s_tags_set.add(str(sigil["tag"]).lower())
                if "tags" in sigil and sigil["tags"]:
                    tags_v = sigil["tags"]
                    if isinstance(tags_v, list):
                        s_tags_set.update(str(t).lower() for t in tags_v)
                    elif isinstance(tags_v, str):
                        s_tags_set.add(tags_v.lower())

                match = False
                if tag_operator.upper() == "AND":
                    if norm_filter_tags.issubset(s_tags_set):
                        match = True
                else:  # OR
                    if not norm_filter_tags.isdisjoint(s_tags_set):
                        match = True
                if match:
                    filtered_by_tags.append(sigil)
            target_pool = filtered_by_tags
        else:
            target_pool = all_loaded_sigils

        # Keyword matching on principle and sigil name
        for sigil in target_pool:
            score = 0.0
            text_to_search = (
                sigil.get("sigil", "") + " " + sigil.get("principle", "")
            ).lower()

            # Basic keyword overlap score
            matched_terms = query_terms.intersection(set(text_to_search.split()))
            score = len(matched_terms)

            # Boost if sigil name itself is in query terms
            if sigil.get("sigil") and sigil.get("sigil", "").lower() in query_terms:
                score += 5  # Arbitrary boost

            if score > 0:
                s_copy = sigil.copy()
                s_copy["_similarity_score"] = score  # Using score as a relevance proxy
                candidates.append(s_copy)

        candidates.sort(key=lambda x: x.get("_similarity_score", 0.0), reverse=True)
        return candidates[:num_sigils]

    def create_rag_context(
        self,
        num_sigils: int = 5,
        filter_tag: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        tag_operator: str = "OR",
        detail_level: str = "standard",
        query: Optional[str] = None,
        min_score_threshold: float = 0.0,
        include_explanations: bool = False,
        exclude_tags: Optional[List[str]] = None,
        exclude_sigil_ids: Optional[List[str]] = None,
        augment_query_flag: bool = True,
        apply_recency_boost_flag: bool = True,
        enable_context_optimizer: bool = False,
        max_context_chars_budget: Optional[int] = None,
        retrieval_mode: str = "semantic",  # NEW FEATURE 2 ("semantic" or "lightweight")
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # (Main logic from RAG script, with retrieval_mode switch)
        all_loaded_sigils_initial = self.load_all_sigils()
        if not all_loaded_sigils_initial:
            return "", []

        current_sigils_pool = all_loaded_sigils_initial  # Start with all
        # ... (Exclusion and Inclusion filter logic from original RAG create_rag_context - condensed for brevity)
        # Apply exclusion filters
        if exclude_tags or exclude_sigil_ids:
            norm_exclude_tags = (
                {tag.lower() for tag in exclude_tags} if exclude_tags else set()
            )
            norm_exclude_ids = (
                {sid.lower() for sid in exclude_sigil_ids}
                if exclude_sigil_ids
                else set()
            )
            temp_pool = []
            for sigil in current_sigils_pool:
                is_excluded = False
                if sigil.get("sigil", "").lower() in norm_exclude_ids:
                    is_excluded = True
                if not is_excluded and norm_exclude_tags:
                    s_tags_set = set()  # ... collect sigil's tags ...
                    if "tags" in sigil:
                        s_tags_set.update(
                            t.lower()
                            for t in (
                                sigil["tags"]
                                if isinstance(sigil["tags"], list)
                                else [sigil["tags"]]
                            )
                        )
                    if "tag" in sigil:
                        s_tags_set.add(str(sigil["tag"]).lower())
                    if not norm_exclude_tags.isdisjoint(s_tags_set):
                        is_excluded = True
                if not is_excluded:
                    temp_pool.append(sigil)
            current_sigils_pool = temp_pool
        # Apply inclusion filters
        if filter_tag and not filter_tags:
            filter_tags = [filter_tag]
        if filter_tags and current_sigils_pool:
            norm_filter_tags = {tag.lower() for tag in filter_tags}
            temp_pool = []
            for sigil in current_sigils_pool:
                s_tags_set = set()  # ... collect sigil's tags ...
                if "tags" in sigil:
                    s_tags_set.update(
                        t.lower()
                        for t in (
                            sigil["tags"]
                            if isinstance(sigil["tags"], list)
                            else [sigil["tags"]]
                        )
                    )
                if "tag" in sigil:
                    s_tags_set.add(str(sigil["tag"]).lower())
                match = False
                if tag_operator.upper() == "AND":
                    if norm_filter_tags.issubset(s_tags_set):
                        match = True
                else:  # OR
                    if not norm_filter_tags.isdisjoint(s_tags_set):
                        match = True
                if match:
                    temp_pool.append(sigil)
            current_sigils_pool = temp_pool

        sigils_with_scores: List[Dict[str, Any]] = []
        effective_query = query
        if query and augment_query_flag:
            effective_query = self._augment_query(query)

        if effective_query and retrieval_mode == "lightweight":  # NEW FEATURE 2
            sigils_with_scores = self._lightweight_retrieve_sigils(
                effective_query, num_sigils * 2, filter_tags, tag_operator
            )  # Get more to allow other sort/boost
            # Lightweight score is already in _similarity_score
        elif (
            effective_query
            and retrieval_mode == "semantic"
            and HAVE_SENTENCE_TRANSFORMERS
            and self.embedding_model
            and current_sigils_pool
        ):
            logger.info(
                f"[RAG] Semantic search with query: '{effective_query[:50]}...' on {len(current_sigils_pool)} candidates."
            )
            query_embedding = self.embedding_model.encode(effective_query)
            for sigil_data in current_sigils_pool:
                sigil_embedding = self._get_sigil_embedding(sigil_data)
                if sigil_embedding is not None:
                    similarity = np.dot(query_embedding, sigil_embedding) / (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(sigil_embedding)
                    )
                    if not np.isnan(similarity) and similarity >= min_score_threshold:
                        s_copy = sigil_data.copy()
                        s_copy["_similarity_score"] = float(similarity)
                        sigils_with_scores.append(s_copy)
        elif (
            current_sigils_pool
        ):  # No semantic/lightweight query, but sigils remain after filtering
            for sigil_data in current_sigils_pool:
                s_copy = sigil_data.copy()
                s_copy["_similarity_score"] = 0.5  # Neutral score
                sigils_with_scores.append(s_copy)

        sigils_with_scores.sort(
            key=lambda x: x.get("_similarity_score", 0.0), reverse=True
        )
        if apply_recency_boost_flag:
            sigils_with_scores = self._apply_recency_boost(sigils_with_scores)
            sigils_with_scores.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )

        selected_for_context = sigils_with_scores[:num_sigils]
        # Auto-fusion (from RAG, slightly adapted for brevity)
        # ... auto_fuse_related_sigils call if needed ...
        final_retrieved_sigils = selected_for_context
        current_detail_level = detail_level
        if enable_context_optimizer and final_retrieved_sigils:
            char_budget = max_context_chars_budget or self.default_max_context_chars
            final_retrieved_sigils, current_detail_level = (
                self._optimize_context_by_chars(
                    final_retrieved_sigils, detail_level, char_budget
                )
            )

        formatted_parts = []
        for s_item in final_retrieved_sigils:
            formatted_text = self.format_sigil_for_prompt(s_item, current_detail_level)
            # ... include_explanations logic from RAG ...
            formatted_parts.append(formatted_text)
        return "\n\n---\n\n".join(formatted_parts), final_retrieved_sigils

    def inject_voxsigil_context(
        self, prompt: str, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:  # Use kwargs
        # (Method content from RAG script, passing kwargs to create_rag_context)
        # All params like num_sigils, query, etc. are passed via kwargs
        search_query = kwargs.get("query", prompt)
        # Remove 'prompt' from kwargs if it's there, as create_rag_context uses 'query'
        kwargs_for_create = kwargs.copy()
        if "prompt" in kwargs_for_create:
            del kwargs_for_create["prompt"]

        sigil_context, retrieved_sigils = self.create_rag_context(
            query=search_query, **kwargs_for_create
        )

        # Simplified prompt structure
        rag_signature = self._generate_rag_signature(retrieved_sigils)  # Assumed helper
        enhanced_prompt = f"VOXSİGİL CONTEXT:\n{sigil_context if sigil_context else 'N/A'}\n{rag_signature}\n\nUSER PROMPT: {prompt}"
        return enhanced_prompt, retrieved_sigils

    def _get_sigil_text_for_embedding(
        self, sigil: Dict[str, Any]
    ) -> str:  # From RAG script
        # ... (content of _get_sigil_text_for_embedding, condensed)
        texts = []
        if sigil.get("sigil"):
            texts.append(f"Sigil: {sigil['sigil']}")
        # ... (tags, principle, usage as in original RAG) ...
        return "\n".join(texts)

    def _get_sigil_embedding(
        self, sigil: Dict[str, Any]
    ) -> Optional[np.ndarray]:  # From RAG script
        # (content from RAG, including robust cache key)
        if not HAVE_SENTENCE_TRANSFORMERS or not self.embedding_model:
            return None
        sigil_id = sigil.get("sigil")
        mod_time = sigil.get("_last_modified", 0)
        cache_key_base = (
            f"id:{sigil_id}"
            if sigil_id
            else f"file:{sigil.get('_source_file', hashlib.md5(str(sigil).encode()).hexdigest())}"
        )
        cache_key = (
            f"{cache_key_base}::mod:{mod_time}::model:{self.embedding_model_name}"
        )
        if self.cache_enabled and cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        text = self._get_sigil_text_for_embedding(sigil)
        if not text.strip():
            return None
        try:
            embedding = self.embedding_model.encode(text)
            if self.cache_enabled:
                self._embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"[RAG] Error embedding sigil {sigil_id or 'N/A'}: {e}")
            return None

    # Methods for cache, analytics, relationships from RAG script
    # _load_embeddings_cache, _save_embeddings_cache, export/import_embeddings_cache
    # precompute_all_embeddings, _update_sigil_analytics, get_sigil_analytics
    # find_related_sigils, auto_fuse_related_sigils, _generate_rag_signature
    # These would be included here, condensed for brevity in this combined view.
    # For example:
    def _load_embeddings_cache(self):
        logger.debug("[RAG] _load_embeddings_cache called")  # Placeholder

    def _save_embeddings_cache(self):
        logger.debug("[RAG] _save_embeddings_cache called")  # Placeholder

    def precompute_all_embeddings(
        self, force_recompute: bool = False, batch_size: int = 32
    ) -> int:
        logger.info(
            f"[RAG] Precomputing embeddings (force: {force_recompute}, batch: {batch_size})"
        )
        # ... (actual implementation) ...
        return 0

    def _generate_rag_signature(self, retrieved_sigils: List[Dict[str, Any]]) -> str:
        if not retrieved_sigils:
            return "/* RAG Sig: None */"
        ids = [s.get("sigil", "unk") for s in retrieved_sigils[:3]]
        return f"/* RAG Sig: {','.join(ids)}{'...' if len(retrieved_sigils) > 3 else ''} */"


# --- VoxSigilMiddleware Class (from voxsigil_middleware.py) ---
class VoxSigilMiddleware:
    def __init__(
        self,
        voxsigil_rag_instance: Optional[VoxSigilRAG] = None,
        conversation_history_size: int = 5,
        rag_off_keywords: Optional[List[str]] = None,
        min_prompt_len_for_rag: int = 5,  # In words
        enable_intent_detection: bool = False,
    ):
        self.voxsigil_rag = voxsigil_rag_instance or VoxSigilRAG()
        self.conversation_history: deque[Dict[str, Any]] = deque(
            maxlen=conversation_history_size
        )
        self.selected_sigils_history: Dict[int, List[Dict[str, Any]]] = {}
        self.turn_counter = 0
        self.rag_off_keywords = (
            [kw.lower() for kw in rag_off_keywords]
            if rag_off_keywords
            else ["@@norag@@", "stop rag"]
        )
        self.min_prompt_len_for_rag = min_prompt_len_for_rag
        self._rag_cache: Dict[Tuple[Any, ...], Tuple[str, List[Dict[str, Any]]]] = {}
        self.enable_intent_detection = enable_intent_detection
        self.intent_to_tags_map: Dict[str, List[str]] = {
            "explain_concept": ["definition", "principle"],
            "find_example": ["example", "usage"],
        }
        self.default_rag_params = {  # For middleware specific defaults
            "num_sigils": 3,
            "min_score_threshold": 0.35,
            "detail_level": "standard",
            "include_explanations": True,
            "retrieval_mode": "semantic",
        }

    def _detect_intent(self, text: str) -> Optional[str]:  # From Middleware
        # ... (content as in middleware)
        if not self.enable_intent_detection:
            return None
        text_lower = text.lower()
        if "explain" in text_lower or "what is" in text_lower:
            return "explain_concept"
        if "example" in text_lower:
            return "find_example"
        return None

    def _get_focused_history_for_rag_query(
        self, current_prompt: str
    ) -> str:  # From Middleware
        # ... (content as in middleware, condensed)
        if not self.conversation_history:
            return ""
        history_list = list(self.conversation_history)
        parts = []
        if len(history_list) > 0 and history_list[-1]["role"] == "user":
            parts.append(f"Prev User: {history_list[-1]['content'][:100]}...")
        if len(history_list) > 1 and history_list[-2]["role"] == "assistant":
            parts.append(f"Prev AI: {history_list[-2]['content'][:100]}...")
        return "\n".join(parts)

    def _parse_dynamic_rag_config(
        self, text: str
    ) -> Tuple[str, Dict[str, Any]]:  # From Middleware
        config: Dict[str, Any] = {}
        match = re.search(r"@@voxsigil_config:({.*?})@@", text, re.IGNORECASE)
        if match:
            try:
                config_str = match.group(1)
                config = json.loads(config_str)
                text = text.replace(match.group(0), "").strip()
            except json.JSONDecodeError:
                logger.warning(
                    f"[Middleware] Invalid JSON in @@voxsigil_config: {config_str}"
                )
        return text, config

    def _determine_rag_strategy(
        self, user_prompt: str, current_rag_params: Dict[str, Any]
    ) -> str:  # NEW FEATURE 3
        """Determines RAG strategy (e.g., 'semantic', 'lightweight', 'off')."""
        # Simple example: if prompt is very short or contains "quick answer"
        if len(user_prompt.split()) < 3 or "quick answer" in user_prompt.lower():
            logger.info(
                "[Middleware] Switching to lightweight RAG strategy due to short/simple prompt."
            )
            return "lightweight"
        if "complex analysis" in user_prompt.lower():
            logger.info(
                "[Middleware] Ensuring semantic RAG strategy for complex query."
            )
            return "semantic"  # Ensure semantic if not already
        return current_rag_params.get("retrieval_mode", "semantic")

    def preprocess_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.turn_counter += 1
        current_turn_id = self.turn_counter
        original_messages = request.get("messages", [])
        if not original_messages:
            return request

        # Add new user message to history
        last_msg = original_messages[-1]
        if last_msg.get("role") == "user":
            if (
                not self.conversation_history
                or self.conversation_history[-1] != last_msg
            ):
                self.conversation_history.append(last_msg.copy())

        user_message_content = (
            last_msg.get("content", "") if last_msg.get("role") == "user" else ""
        )
        user_message_index = len(original_messages) - 1 if user_message_content else -1

        if not user_message_content:
            self.selected_sigils_history[current_turn_id] = []
            return request

        current_rag_params = self.default_rag_params.copy()
        if "voxsigil_rag_config" in request:  # Per-request config
            current_rag_params.update(request["voxsigil_rag_config"])

        user_message_content, dynamic_config = self._parse_dynamic_rag_config(
            user_message_content
        )
        current_rag_params.update(dynamic_config)
        if user_message_index != -1:
            original_messages[user_message_index]["content"] = (
                user_message_content  # Update if cleaned
            )

        # NEW FEATURE 3: Dynamic RAG Strategy
        current_rag_params["retrieval_mode"] = self._determine_rag_strategy(
            user_message_content, current_rag_params
        )

        # Conditional RAG
        if (
            any(
                keyword in user_message_content.lower()
                for keyword in self.rag_off_keywords
            )
            or len(user_message_content.split()) < self.min_prompt_len_for_rag
            or current_rag_params["retrieval_mode"] == "off"
        ):
            logger.info(f"[Middleware] RAG disabled for turn {current_turn_id}.")
            self.selected_sigils_history[current_turn_id] = []
            return request

        history_context_for_rag = self._get_focused_history_for_rag_query(
            user_message_content
        )
        rag_query_text = (
            f"{history_context_for_rag}\n\nCURRENT QUERY: {user_message_content}"
            if history_context_for_rag
            else user_message_content
        )

        detected_intent = self._detect_intent(user_message_content)
        if detected_intent and detected_intent in self.intent_to_tags_map:
            intent_tags = self.intent_to_tags_map[detected_intent]
            current_rag_params["filter_tags"] = list(
                set((current_rag_params.get("filter_tags") or []) + intent_tags)
            )

        cache_key_params = tuple(sorted(current_rag_params.items()))
        rag_cache_key = (rag_query_text, cache_key_params)

        if rag_cache_key in self._rag_cache:
            enhanced_prompt, retrieved_sigils = self._rag_cache[rag_cache_key]
        else:
            try:
                enhanced_prompt, retrieved_sigils = (
                    self.voxsigil_rag.inject_voxsigil_context(
                        prompt=user_message_content,
                        query=rag_query_text,
                        **current_rag_params,
                    )
                )
                self._rag_cache[rag_cache_key] = (enhanced_prompt, retrieved_sigils)
            except Exception as e:
                logger.error(
                    f"[Middleware] Error during RAG injection: {e}. Using original prompt.",
                    exc_info=True,
                )
                enhanced_prompt = user_message_content
                retrieved_sigils = []

        self.selected_sigils_history[current_turn_id] = retrieved_sigils
        if user_message_index != -1:
            original_messages[user_message_index]["content"] = enhanced_prompt
        request["messages"] = original_messages
        return request

    def postprocess_response(
        self, response: Dict[str, Any], scoring_enabled: bool = True
    ) -> Dict[str, Any]:
        # (Method content from Middleware, condensed)
        current_turn_id = self.turn_counter
        llm_message_content = ""
        if response.get("choices") and response["choices"][0].get("message"):
            assistant_message = response["choices"][0]["message"]
            if assistant_message.get("role") == "assistant":
                self.conversation_history.append(assistant_message.copy())
            llm_message_content = assistant_message.get("content", "")
            # Expose basic response stats for downstream analysis
            response.setdefault("voxsigil_metadata", {})
            response["voxsigil_metadata"]["assistant_response_length"] = len(
                llm_message_content
            )
            response["voxsigil_metadata"]["assistant_response_tokens_est"] = (
                count_tokens(llm_message_content)
            )

        response.setdefault("voxsigil_metadata", {})["turn_id"] = current_turn_id
        retrieved_sigils = self.selected_sigils_history.get(current_turn_id, [])
        response["voxsigil_metadata"]["retrieved_sigils"] = [
            {"sigil": s.get("sigil", "N/A"), "score": s.get("_similarity_score", 0)}
            for s in retrieved_sigils
        ]
        # Grounding scores call would be here, if evaluator instance is available
        # For simplicity, grounding is not directly called here to avoid circular dependency during init.
        # Can be done by an external orchestrator.
        return response

    def wrap_llm_api(
        self, llm_api_call: Callable[..., Any]
    ) -> Callable[..., Any]:  # From Middleware
        def wrapped_llm_api(*args: Any, **kwargs: Any) -> Any:
            request_dict = (
                kwargs.get("request")
                if "request" in kwargs and isinstance(kwargs.get("request"), dict)
                else (
                    args[0]
                    if args and isinstance(args[0], dict) and "messages" in args[0]
                    else None
                )
            )
            if not request_dict:
                return llm_api_call(*args, **kwargs)

            processed_request = self.preprocess_request(request_dict.copy())

            if "request" in kwargs and isinstance(kwargs.get("request"), dict):
                kwargs["request"] = processed_request
            elif args and isinstance(args[0], dict):
                args = (processed_request,) + args[1:]

            raw_response = llm_api_call(*args, **kwargs)
            return (
                self.postprocess_response(raw_response.copy())
                if isinstance(raw_response, dict)
                else raw_response
            )

        return wrapped_llm_api


# --- VoxSigilResponseEvaluator Class (Enhanced version) ---
class VoxSigilResponseEvaluator:
    def __init__(
        self,
        voxsigil_rag_instance: Optional[VoxSigilRAG] = None,
        embedding_model_name: str = VS_CONFIG.DEFAULT_EMBEDDING_MODEL,  # Use config
        default_score_weights: Optional[Dict[str, float]] = None,
        # NEW FEATURE 4: Evaluation Profiles
        evaluation_profiles: Optional[Dict[str, Dict[str, float]]] = None,
        active_profile_name: str = "default",
    ):
        self.voxsigil_rag = voxsigil_rag_instance or VoxSigilRAG()

        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_model_name = embedding_model_name
        if HAVE_SENTENCE_TRANSFORMERS and self.embedding_model_name:
            try:
                logger.info(
                    f"[Evaluator] Loading embedding model: {embedding_model_name}"
                )
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.error(f"[Evaluator] Error loading embedding model: {e}")
                # No hard fail, semantic similarity will be skipped

        self.default_score_weights = (
            default_score_weights
            or {  # Default profile weights
                "tag_inclusion": 0.20,
                "tag_paraphrasing": 0.10,
                "principle_adherence": 0.25,
                "sigil_mentions": 0.10,
                "structural_integrity": 0.05,
                "lexical_overlap": 0.05,
                "semantic_similarity": 0.15,
                "contradiction_penalty_factor": 0.5,  # Added from other evaluator
            }
        )
        self.evaluation_profiles = evaluation_profiles or {
            "default": self.default_score_weights.copy()
        }
        self.active_profile_name = active_profile_name
        self._current_response_embedding_cache: Optional[np.ndarray] = None

    def set_active_evaluation_profile(self, profile_name: str) -> bool:  # NEW FEATURE 4
        if profile_name in self.evaluation_profiles:
            self.active_profile_name = profile_name
            logger.info(
                f"[Evaluator] Active evaluation profile set to: '{profile_name}'"
            )
            return True
        logger.warning(f"[Evaluator] Evaluation profile '{profile_name}' not found.")
        return False

    def get_current_score_weights(self) -> Dict[str, float]:  # NEW FEATURE 4
        return self.evaluation_profiles.get(
            self.active_profile_name, self.default_score_weights
        )

    def _get_response_embedding_eval(
        self, response_text: str
    ) -> Optional[np.ndarray]:  # Renamed for clarity
        if self._current_response_embedding_cache is not None:
            return self._current_response_embedding_cache
        if self.embedding_model:
            try:
                self._current_response_embedding_cache = self.embedding_model.encode(
                    response_text
                )
                return self._current_response_embedding_cache
            except Exception as e:
                logger.warning(f"[Evaluator] Embedding failed for response: {e}")
        return None

    # Evaluation sub-methods from Enhanced Evaluator, adapted
    def _evaluate_tag_inclusion(
        self,
        response_lower: str,
        relevant_sigils: List[Dict[str, Any]],
        expected_tags: Optional[List[str]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        # (Logic from Enhanced Evaluator's _evaluate_tag_inclusion, ensuring details are built locally)
        tag_matches_details = []
        all_expected_tags = set(t.lower() for t in (expected_tags or []))

        # Collect all tags from relevant_sigils if expected_tags is not comprehensive
        if not all_expected_tags:
            for item in relevant_sigils:
                sigil = item.get("sigil", {})
                if "tag" in sigil and sigil["tag"]:
                    all_expected_tags.add(str(sigil["tag"]).lower())
                if "tags" in sigil and sigil["tags"]:
                    tags_val = sigil["tags"]
                    if isinstance(tags_val, list):
                        all_expected_tags.update(str(t).lower() for t in tags_val)
                    elif isinstance(tags_val, str):
                        all_expected_tags.add(tags_val.lower())

        if not all_expected_tags:
            return 0.0, []

        matched_count = 0
        for tag_lower in all_expected_tags:
            if tag_lower in response_lower:
                matched_count += 1
                tag_matches_details.append(
                    {"tag": tag_lower, "status": "matched_explicitly"}
                )
            else:
                tag_matches_details.append({"tag": tag_lower, "status": "not_matched"})

        score = matched_count / len(all_expected_tags) if all_expected_tags else 0.0
        return score, tag_matches_details

    def _evaluate_principle_adherence(
        self,
        response_text: str,
        response_lower: str,
        relevant_sigils: List[Dict[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        # (Logic from Enhanced Evaluator's _evaluate_principle_adherence, adapted)
        principle_match_details_list = []
        all_sigil_principle_scores = []

        response_embedding_eval = self._get_response_embedding_eval(
            response_text
        )  # Use evaluator's specific embed method

        for item in relevant_sigils:
            sigil = item.get("sigil", {})
            sigil_id = sigil.get("sigil", "unknown")
            principle_text = sigil.get("principle", "")
            if not principle_text:
                continue

            current_sigil_score = 0.0
            method_used = "keyword"
            current_match_detail = {
                "sigil": sigil_id,
                "score": 0.0,
                "method": method_used,
                "matches": [],
            }

            if self.embedding_model and response_embedding_eval is not None:
                try:
                    principle_embedding = self.embedding_model.encode(principle_text)
                    if (
                        cosine_similarity
                    ):  # Check if sklearn's cosine_similarity is imported
                        sim = cosine_similarity(
                            [response_embedding_eval], [principle_embedding]
                        )[0][0]
                        current_sigil_score = (sim + 1) / 2  # Normalize
                        method_used = "semantic"
                        current_match_detail["matches"].append(
                            {"type": "semantic", "value": round(current_sigil_score, 3)}
                        )
                except Exception:  # Fallback on error
                    pass  # Keyword check will run

            if method_used == "keyword":  # Fallback or primary
                key_phrases = [
                    p.strip().lower()
                    for p in principle_text.split(".")
                    if len(p.strip()) > 5
                ][:3]
                if key_phrases:
                    matched_phrases = sum(
                        1 for kp in key_phrases if kp in response_lower
                    )
                    current_sigil_score = matched_phrases / len(key_phrases)
                    current_match_detail["matches"].append(
                        {"type": "keyword", "value": round(current_sigil_score, 3)}
                    )

            current_match_detail["score"] = current_sigil_score
            current_match_detail["method"] = method_used
            all_sigil_principle_scores.append(current_sigil_score)
            principle_match_details_list.append(current_match_detail)

        overall_score = (
            np.mean(all_sigil_principle_scores) if all_sigil_principle_scores else 0.0
        )
        return overall_score, principle_match_details_list

    # ... Other _evaluate methods from EnhancedEvaluator (sigil_mentions, structural_integrity, lexical_overlap, semantic_similarity) ...
    # Need to be adapted to build and return their own details, not modify a global one.
    def _evaluate_sigil_mentions(
        self, response_lower: str, relevant_sigils: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        mention_details = []
        mention_count = 0
        for item in relevant_sigils:
            sigil = item.get("sigil", {})
            sigil_id = sigil.get("sigil", "unknown")
            if sigil_id != "unknown" and sigil_id.lower() in response_lower:
                mention_count += 1
                mention_details.append({"sigil": sigil_id, "status": "mentioned"})
        score = mention_count / len(relevant_sigils) if relevant_sigils else 0.0
        return score, mention_details

    def _evaluate_structural_integrity(
        self, response: str
    ) -> Tuple[float, List[Dict[str, Any]]]:  # Basic version
        patterns = []
        score = 0.0
        if re.search(r"\n\s*\d+\.\s+", response):
            patterns.append(
                {
                    "type": "numbered_list",
                    "count": len(re.findall(r"\n\s*\d+\.\s+", response)),
                }
            )
            score = max(score, 0.5)
        if re.search(r"\n\s*[\*\-]\s+", response):
            patterns.append(
                {
                    "type": "bullet_list",
                    "count": len(re.findall(r"\n\s*[\*\-]\s+", response)),
                }
            )
            score = max(score, 0.5)
        if len(patterns) > 1:
            score = 0.8
        if not patterns and len(response.split("\n")) > 5:
            score = 0.2  # Some structure if long and multiline
        return score, patterns

    def _evaluate_lexical_overlap(
        self, response_lower: str, relevant_sigils: List[Dict[str, Any]]
    ) -> float:
        sigil_texts_combined = " ".join(
            item.get("sigil", {}).get("principle", "")
            + " "
            + item.get("sigil", {}).get("sigil", "")
            for item in relevant_sigils
        ).lower()
        if not sigil_texts_combined:
            return 0.0

        sigil_words = set(w for w in sigil_texts_combined.split() if len(w) > 3)
        response_words = set(w for w in response_lower.split() if len(w) > 3)
        if not sigil_words:
            return 0.0

        overlap = len(sigil_words.intersection(response_words))
        return overlap / len(sigil_words)

    def _evaluate_semantic_similarity(
        self, response: str, relevant_sigils: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        if not self.embedding_model or not cosine_similarity:
            return 0.0, {}
        response_embedding = self._get_response_embedding_eval(response)
        if response_embedding is None:
            return 0.0, {}

        similarities = {}
        all_sim_scores = []
        for item in relevant_sigils:
            sigil = item.get("sigil", {})
            sigil_id = sigil.get("sigil", "unknown")
            sigil_text = (
                sigil.get("principle", "") + " " + " ".join(sigil.get("tags", []))
            )  # Combine principle and tags for semantics
            if not sigil_text.strip():
                continue

            try:
                sigil_embedding = self.embedding_model.encode(sigil_text)
                sim = cosine_similarity([response_embedding], [sigil_embedding])[0][0]
                similarities[sigil_id] = float(sim)
                all_sim_scores.append(max(0, sim))  # Use non-negative for averaging
            except Exception as e:
                logger.debug(
                    f"[Evaluator] Semantic sim error for sigil {sigil_id}: {e}"
                )
                similarities[sigil_id] = 0.0

        overall_semantic_score = np.mean(all_sim_scores) if all_sim_scores else 0.0
        return overall_semantic_score, similarities

    def evaluate_response_enhanced(
        self,
        response_text: str,
        query: str,
        expected_tags: Optional[List[str]] = None,
        relevant_sigils_override: Optional[
            List[Dict[str, Any]]
        ] = None,  # Renamed from relevant_sigils
        detailed_report: bool = True,
    ) -> Dict[str, Any]:
        self._current_response_embedding_cache = None  # Reset for this call
        response_lower = response_text.lower()

        # Fetch relevant sigils if not provided
        if relevant_sigils_override is None:
            _, retrieved_items = self.voxsigil_rag.create_rag_context(
                query=query, num_sigils=5
            )  # Default 5
            # The RAG returns sigils wrapped, unwrap them for evaluator.
            # Assuming RAG returns a list of dicts, where each dict has a 'sigil' key that is the actual sigil content.
            # If RAG already returns the list of sigil content dicts, then this adjustment is not needed.
            # For now, assume `retrieved_items` is List[Dict[str, Any]] where each dict is a sigil's data.
            current_relevant_sigils = retrieved_items
        else:
            current_relevant_sigils = relevant_sigils_override

        # Initialize scores & details
        scores: Dict[str, float] = {
            "tag_inclusion": 0.0,
            "tag_paraphrasing": 0.0,
            "principle_adherence": 0.0,
            "sigil_mentions": 0.0,
            "structural_integrity": 0.0,
            "lexical_overlap": 0.0,
            "semantic_similarity": 0.0,
            "contradiction_score": 0.0,  # From other evaluator version
            "total_score": 0.0,
        }
        details: Dict[str, Any] = {
            "per_sigil_scores": {}
        }  # Changed from sigil_scores to per_sigil_scores

        # Collect expected tags if not provided
        actual_expected_tags = list(expected_tags) if expected_tags else []
        if not actual_expected_tags and current_relevant_sigils:
            for item in current_relevant_sigils:
                # item is assumed to be the sigil data dict directly
                if "tag" in item and item["tag"]:
                    actual_expected_tags.append(str(item["tag"]))
                if "tags" in item and item["tags"]:
                    val = item["tags"]
                    if isinstance(val, list):
                        actual_expected_tags.extend(str(t) for t in val)
                    elif isinstance(val, str):
                        actual_expected_tags.append(val)
        actual_expected_tags = list(set(actual_expected_tags))  # Unique

        # --- Call evaluation sub-methods ---
        scores["tag_inclusion"], details["tag_matches_report"] = (
            self._evaluate_tag_inclusion(
                response_lower, current_relevant_sigils, actual_expected_tags
            )
        )
        scores["principle_adherence"], details["principle_adherence_report"] = (
            self._evaluate_principle_adherence(
                response_text, response_lower, current_relevant_sigils
            )
        )
        scores["sigil_mentions"], details["sigil_mentions_report"] = (
            self._evaluate_sigil_mentions(response_lower, current_relevant_sigils)
        )
        scores["structural_integrity"], details["structural_patterns_report"] = (
            self._evaluate_structural_integrity(response_text)
        )
        scores["lexical_overlap"] = self._evaluate_lexical_overlap(
            response_lower, current_relevant_sigils
        )

        if self.embedding_model:
            scores["semantic_similarity"], details["semantic_similarity_per_sigil"] = (
                self._evaluate_semantic_similarity(
                    response_text, current_relevant_sigils
                )
            )
            # Tag paraphrasing using semantic similarity to tag concepts
            if actual_expected_tags:
                tag_concepts = [tag.replace("_", " ") for tag in actual_expected_tags]
                response_embedding_eval = self._get_response_embedding_eval(
                    response_text
                )
                if (
                    response_embedding_eval is not None
                    and self.embedding_model
                    and cosine_similarity
                ):
                    try:
                        tag_embeddings = self.embedding_model.encode(tag_concepts)
                        sims = cosine_similarity(
                            [response_embedding_eval], tag_embeddings
                        )[0]
                        scores["tag_paraphrasing"] = (
                            float(np.mean(sims)) if len(sims) > 0 else 0.0
                        )
                    except Exception as e:
                        logger.debug(f"Tag paraphrasing embedding failed: {e}")
            else:
                scores["tag_paraphrasing"] = 0.0
        else:  # Fallback for tag_paraphrasing
            scores["tag_paraphrasing"] = (
                scores["lexical_overlap"] * 0.5
            )  # Simple fallback

        # Calculate overall contradiction (max over sigils)
        # And collect per-sigil details
        max_contradiction = 0.0
        for sigil_item in current_relevant_sigils:
            sigil_id = sigil_item.get("sigil", "unknown")
            contradiction = self._calculate_contradiction_for_sigil(
                response_lower, sigil_item
            )  # From other evaluator
            max_contradiction = max(max_contradiction, contradiction)
            if sigil_id not in details["per_sigil_scores"]:
                details["per_sigil_scores"][sigil_id] = {}
            details["per_sigil_scores"][sigil_id]["contradiction"] = contradiction
        scores["contradiction_score"] = max_contradiction

        current_weights = self.get_current_score_weights()
        scores["total_score"] = sum(
            scores.get(metric, 0.0) * current_weights.get(metric, 0.0)
            for metric in current_weights
            if metric != "contradiction_penalty_factor"
        )
        # Apply contradiction penalty
        penalty = scores.get("contradiction_score", 0.0) * current_weights.get(
            "contradiction_penalty_factor", 0.5
        )
        scores["total_score"] = max(0.0, scores["total_score"] - penalty)

        # Round all scores
        for k in scores:
            scores[k] = round(scores[k], 4)

        return (
            {"scores": scores, "details": details}
            if detailed_report
            else {"scores": scores}
        )

    def generate_visual_diff(
        self,
        responses_map: Dict[str, str],
        query_for_rag: str,
        title: str = "VoxSigil Response Comparison",
    ) -> str:
        # (Method content from Enhanced Evaluator, using self.evaluate_response_enhanced)
        html_parts = [
            f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title><style>{VS_CONFIG.HTML_DIFF_STYLES}</style></head><body><h1>{title}</h1><div class='container'>"
        ]

        # Fetch relevant_sigils once for the query
        # The RAG call now expects kwargs correctly
        _, common_relevant_sigils = self.voxsigil_rag.create_rag_context(
            query=query_for_rag, num_sigils=5
        )

        for variant_name, response_text in responses_map.items():
            if not response_text:
                continue
            evaluation = self.evaluate_response_enhanced(
                response_text,
                query_for_rag,
                relevant_sigils_override=common_relevant_sigils,
                detailed_report=True,
            )
            scores = evaluation.get("scores", {})
            # Highlight response (basic version for now)
            formatted_response = html.escape(response_text).replace("\n", "<br>")

            html_parts.append(
                f"<div class='variant'><div class='variant-header'><h2>{variant_name} (Score: {scores.get('total_score', 0.0):.2f})</h2></div>"
            )
            html_parts.append(f"<div class='content'>{formatted_response}</div>")
            html_parts.append(
                "<div class='metrics'><h3 class='metric-header'>Metrics</h3><table><tr><th>Metric</th><th>Score</th></tr>"
            )
            for metric, score_val in sorted(scores.items()):
                color_class = (
                    "high"
                    if score_val >= 0.7
                    else ("medium" if score_val >= 0.4 else "low")
                )
                if metric == "contradiction_score" and score_val > 0.1:
                    color_class = "low"  # Higher contradiction is bad
                elif metric == "contradiction_score":
                    color_class = "high"

                html_parts.append(
                    f"<tr><td>{metric}</td><td><span class='score {color_class}'>{score_val:.3f}</span></td></tr>"
                )
            html_parts.append("</table></div></div>")  # Close metrics and variant

        html_parts.append("</div></body></html>")
        return "".join(html_parts)

    def save_comparison_to_file(
        self,
        responses_map: Dict[str, str],
        query_for_rag: str,
        output_path: Union[str, Path],
        title: str = "VoxSigil Response Comparison",
    ) -> str:
        # (Method content from Enhanced Evaluator)
        html_content = self.generate_visual_diff(responses_map, query_for_rag, title)
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"[Evaluator] Comparison saved to {output_p}")
        return str(output_p)

    def evaluate_batch_enhanced(
        self,
        prompt_response_pairs: List[Dict[str, Any]],
        output_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        # (New batch evaluation method tailored for the enhanced evaluator)
        all_results = []
        summary_scores_collection: Dict[str, List[float]] = defaultdict(list)

        for i, pair in enumerate(prompt_response_pairs):
            prompt = pair.get("prompt", pair.get("query", ""))
            response = pair.get("response", "")
            eval_id = pair.get("id", f"eval_{i}")

            if not prompt or not response:
                logger.warning(
                    f"[Evaluator] Skipping batch item {eval_id}: missing prompt or response."
                )
                all_results.append(
                    {"id": eval_id, "error": "Missing prompt or response", "scores": {}}
                )
                continue

            # Use relevant_sigils_override if provided in pair_data, else RAG fetches them based on query
            relevant_sigils = pair.get("relevant_sigils")  # Could be None

            logger.info(
                f"[Evaluator] Evaluating batch item: {eval_id} (Query: {prompt[:30]}...)"
            )
            evaluation = self.evaluate_response_enhanced(
                response,
                prompt,
                relevant_sigils_override=relevant_sigils,
                detailed_report=True,
            )

            current_result = {
                "id": eval_id,
                "prompt": prompt,
                "response_snippet": response[:100]
                + "...",  # Snippet for brevity in main results list
                "scores": evaluation.get("scores", {}),
                # Optionally include full details if needed, but can make JSON large
                # 'details': evaluation.get('details',{})
            }
            all_results.append(current_result)

            # Collect scores for summary
            for metric_name, score_value in evaluation.get("scores", {}).items():
                if isinstance(score_value, (int, float)):  # Ensure it's a number
                    summary_scores_collection[metric_name].append(score_value)

        summary_statistics: Dict[str, Dict[str, Union[float, int]]] = {}
        for metric, values in summary_scores_collection.items():
            if values:
                summary_statistics[metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "median": round(float(np.median(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                    "count": len(values),
                }

        batch_report = {
            "results": all_results,
            "summary_statistics": summary_statistics,
        }
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(batch_report, f, indent=2)
            logger.info(f"[Evaluator] Batch evaluation report saved to {output_file}")
        return batch_report


# --- Main Execution / Demo ---
if __name__ == "__main__":
    logger.info("--- VoxSigil Unified System: Demo ---")

    # --- Setup Test Environment & RAG ---
    test_lib_path = Path("VoxSigil-System-Test-Library")
    core_sigils_path = test_lib_path / "core"
    core_sigils_path.mkdir(parents=True, exist_ok=True)

    sigil_alpha_data = {
        "sigil": "α_Concept",
        "principle": "Alpha defines foundational concepts with clarity.",
        "tags": ["concept", "foundation", "alpha_tag"],
        "version": "1.0",
        "author": "System",
    }
    sigil_beta_data = {
        "sigil": "β_Example",
        "principle": "Beta provides practical examples for illustration.",
        "tags": ["example", "practical", "beta_tag"],
        "version": "1.1",
        "author": "System",
    }
    with open(core_sigils_path / "alpha.voxsigil", "w") as f:
        yaml.dump(sigil_alpha_data, f)
    with open(core_sigils_path / "beta.yaml", "w") as f:
        yaml.dump(sigil_beta_data, f)

    # Initialize RAG - it's used by Middleware and Evaluator
    vs_rag = VoxSigilRAG(voxsigil_library_path=test_lib_path)
    vs_rag.load_all_sigils(force_reload=True)
    if HAVE_SENTENCE_TRANSFORMERS:
        vs_rag.precompute_all_embeddings(force_recompute=True)

    # --- Middleware Demo ---
    logger.info("\n--- Middleware Demo ---")
    vs_middleware = VoxSigilMiddleware(
        voxsigil_rag_instance=vs_rag, conversation_history_size=3
    )

    def mock_llm_api_call(request: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt_content = request["messages"][-1]["content"]
        logger.info(
            f"  [Mock LLM] Received enhanced prompt (first 150 chars): '{user_prompt_content[:150]}...'"
        )
        response_content = "Acknowledging enhanced prompt about concepts. The α_Concept sigil is indeed about clarity. Examples use β_Example."
        if (
            "@@voxsigil_config" in user_prompt_content
        ):  # Check if the original user prompt had it
            response_content += " Dynamic config noted."
        return {
            "choices": [{"message": {"role": "assistant", "content": response_content}}]
        }

    wrapped_api = vs_middleware.wrap_llm_api(mock_llm_api_call)

    test_request_mw = {
        "messages": [
            {
                "role": "user",
                "content": 'Explain foundational concepts with examples. @@voxsigil_config:{"num_sigils":1}@@',
            }
        ]
    }
    response_mw = wrapped_api(request=test_request_mw.copy())
    print(
        f"  Middleware Demo Response (metadata): {json.dumps(response_mw.get('voxsigil_metadata'), indent=2)}"
    )
    print(
        f"  Middleware Demo Response (content): {response_mw['choices'][0]['message']['content']}"
    )

    # --- Evaluator Demo ---
    logger.info("\n--- Evaluator Demo ---")
    # Custom profiles for evaluator
    eval_profiles = {
        "default": vs_rag.default_rag_params,  # Default from RAG
        "strict_principle": {
            **vs_rag.default_rag_params,
            "principle_adherence": 0.7,
            "tag_inclusion": 0.1,
        },
        "tag_focused": {
            **vs_rag.default_rag_params,
            "tag_inclusion": 0.6,
            "principle_adherence": 0.2,
        },
    }

    vs_evaluator = VoxSigilResponseEvaluator(
        voxsigil_rag_instance=vs_rag, evaluation_profiles=eval_profiles
    )
    vs_evaluator.set_active_evaluation_profile(
        "strict_principle"
    )  # Use a custom profile

    eval_query = "Describe foundational AI concepts using examples, referring to α_Concept and β_Example."
    eval_response_good = "The α_Concept sigil helps define foundational ideas with clarity. For instance, β_Example provides practical illustrations for these alpha_tag concepts."
    eval_response_poor = "AI is complex. Things are hard to explain."

    # Evaluate good response
    # RAG to get relevant_sigils (as Middleware does for the LLM call, Evaluator needs them)
    _, relevant_sigils_for_eval = vs_rag.create_rag_context(
        query=eval_query, num_sigils=2
    )

    evaluation_good = vs_evaluator.evaluate_response_enhanced(
        eval_response_good,
        eval_query,
        relevant_sigils_override=relevant_sigils_for_eval,
    )
    print(
        f"  Evaluation for GOOD response (profile: {vs_evaluator.active_profile_name}):"
    )
    print(f"    Overall Score: {evaluation_good['scores'].get('total_score')}")
    print(f"    Tag Inclusion: {evaluation_good['scores'].get('tag_inclusion')}")
    print(
        f"    Principle Adherence: {evaluation_good['scores'].get('principle_adherence')}"
    )

    # --- Visual Diff Demo (Evaluator) ---
    logger.info("\n--- Visual Diff Demo ---")
    responses_for_diff = {
        "Good Response (Example Query)": eval_response_good,
        "Poor Response (Example Query)": eval_response_poor,
        "LLM Simulated Response from Middleware": response_mw["choices"][0]["message"][
            "content"
        ],
    }
    diff_output_path = Path("voxsigil_response_comparison.html")
    vs_evaluator.save_comparison_to_file(
        responses_for_diff,
        eval_query,
        diff_output_path,
        title="Demo VoxSigil Response Comparison",
    )

    # --- Batch Evaluation Demo (Evaluator) ---
    logger.info("\n--- Batch Evaluation Demo ---")
    batch_eval_data = [
        {
            "id": "good_eval",
            "query": eval_query,
            "response": eval_response_good,
            "relevant_sigils": relevant_sigils_for_eval,
        },
        {
            "id": "poor_eval",
            "query": eval_query,
            "response": eval_response_poor,
            "relevant_sigils": relevant_sigils_for_eval,
        },  # Re-use relevant sigils
        {
            "id": "middleware_eval",
            "query": test_request_mw["messages"][0]["content"],
            "response": response_mw["choices"][0]["message"]["content"],
            "relevant_sigils": response_mw.get("voxsigil_metadata", {}).get(
                "retrieved_sigils", []
            ),  # Use what middleware retrieved
        },
    ]
    batch_report_path = Path("voxsigil_batch_eval_report.json")
    vs_evaluator.evaluate_batch_enhanced(batch_eval_data, output_file=batch_report_path)

    # --- Tokenizer Demo (if available) ---
    logger.info("\n--- Tokenizer Demo ---")
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")  # Example tokenizer
        set_global_tokenizer(tokenizer, "cl100k_base")
        text_sample = "This is a sample text for token counting."
        logger.info(
            f"    '{text_sample}' -> Tokens: {count_tokens(text_sample)} (using {_tokenizer_name})"
        )
    except ImportError:
        logger.warning("    tiktoken not installed. Skipping global tokenizer demo.")
        logger.info(
            f"    '{text_sample}' -> Est. Tokens (chars/4): {count_tokens(text_sample)}"
        )

    # --- Cleanup ---
    logger.info("\n--- Cleaning up test files ---")
    try:
        import shutil

        if test_lib_path.exists():
            shutil.rmtree(test_lib_path)
        if diff_output_path.exists():
            os.remove(diff_output_path)
        if batch_report_path.exists():
            os.remove(batch_report_path)
        logger.info("    Cleanup complete.")
    except Exception as e:
        logger.error(f"    Error during cleanup: {e}")

    logger.info("--- VoxSigil Unified System Demo Finished ---")
