#!/usr/bin/env python
"""
VoxSigil RAG-style Injection Module for ARC.

This module provides functionality for loading VoxSigil sigils from the VoxSigil-Library
and injecting them as retrieval-augmented generation (RAG) context into model prompts.
"""

import os
import yaml
import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import time 
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for optional dependencies
HAVE_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
    logger.info("Sentence Transformers available for semantic search")
except ImportError:
    logger.warning("Sentence Transformers not available - semantic search disabled")
    logger.warning("Install with: pip install sentence-transformers")

HAVE_JSONSCHEMA = False
try:
    from jsonschema import validate, ValidationError
    HAVE_JSONSCHEMA = True
    logger.info("jsonschema available for Sigil schema validation")
except ImportError:
    HAVE_JSONSCHEMA = False
    # Mock validate function if jsonschema is not available
    def validate(instance: Any, schema: Dict[str, Any]) -> None: pass 
    class ValidationError(Exception): pass # type: ignore
    logger.warning("jsonschema not available. Sigil schema validation will be skipped. Install with: pip install jsonschema")

# Define paths
DEFAULT_VOXSİGİL_LIBRARY_PATH = Path(__file__).resolve().parent.parent / "voxsigil-Library"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Common default, good for CPU
DEFAULT_EMBEDDINGS_CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "embeddings_cache.npz"

# FEATURE 10: Sigil Schema (Example)
DEFAULT_SIGIL_SCHEMA = {
    "type": "object",
    "properties": {
        "sigil": {"type": "string", "description": "The unique identifier for the sigil."},
        "tag": {"type": "string", "description": "Primary tag for the sigil (can be deprecated in favor of 'tags' list)."},
        "tags": {
            "type": ["array", "string"], 
            "items": {"type": "string"},
            "description": "A list of tags or a single tag string."
        },
        "principle": {"type": "string", "description": "The core concept or idea the sigil represents."},
        "usage": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "examples": {"type": ["array", "string"], "items": {"type": "string"}}
            },
            "description": "How to use the sigil."
        },
        "prompt_template": {"type": "object", "description": "Associated prompt template, if any."},
        "relationships": {"type": "object", "description": "Links to other related sigils."},
        # Meta fields (added by loader, not typically in user files)
        "_source_file": {"type": "string"},
        "_last_modified": {"type": "number"}, # Unix timestamp
        "_similarity_score": {"type": "number"},
        "_recency_boost_applied": {"type": "number"},
    },
    "required": ["sigil", "principle"], # Example: sigil name and principle are mandatory
    "additionalProperties": True # Allow other fields not defined in schema
}


class VoxSigilRAG:
    """
    VoxSigil RAG Injection handler for enhancing LLM prompts with VoxSigil sigils.
    """

    def __init__(self, voxsigil_library_path: Optional[Path] = None, cache_enabled: bool = True,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 # FEATURE 7: RecencyBooster parameters
                 recency_boost_factor: float = 0.05, # Small boost for recency
                 recency_max_days: int = 90, # Sigils modified in last 90 days get boost
                 # FEATURE 1: DynamicContextOptimizer default budget
                 default_max_context_chars: int = 8000): # Approx 2000 tokens
        """
        Initialize the VoxSigil RAG handler.

        Args:
            voxsigil_library_path: Path to the VoxSigil Library directory.
            cache_enabled: Whether to cache loaded sigils and embeddings.
            embedding_model: Name of the sentence-transformer model for embeddings.
            recency_boost_factor: Factor to boost scores of recent sigils (0 to disable).
            recency_max_days: Sigils updated within this period get max boost.
            default_max_context_chars: Default character budget for DynamicContextOptimizer.
        """
        self.voxsigil_library_path = voxsigil_library_path or DEFAULT_VOXSİGİL_LIBRARY_PATH
        self.cache_enabled = cache_enabled
        self._sigil_cache: Dict[str, Dict[str, Any]] = {} # For sigil file content
        self._loaded_sigils: Optional[List[Dict[str, Any]]] = None
        
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._embeddings_cache_path = DEFAULT_EMBEDDINGS_CACHE_PATH
        
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Embedding model '{embedding_model}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading embedding model '{embedding_model}': {e}")
                logger.warning("Vector similarity search will be disabled.")
        
        if self.cache_enabled:
            self._embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_embeddings_cache()

        # FEATURE 10: Assign schema
        self.sigil_schema = DEFAULT_SIGIL_SCHEMA
        
        # FEATURE 2: SigilAnalytics
        self._sigil_retrieval_counts = defaultdict(int)
        self._sigil_last_retrieved_time: Dict[str, float] = {}

        # FEATURE 7: RecencyBooster parameters
        self.recency_boost_factor = recency_boost_factor
        self.recency_max_days_seconds = recency_max_days * 24 * 60 * 60

        # FEATURE 1: DynamicContextOptimizer parameter
        self.default_max_context_chars = default_max_context_chars
        
        # FEATURE 5: QueryAugmenter - Simple example synonym map
        self.synonym_map: Dict[str, List[str]] = {
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "symbolic reasoning": ["logic-based reasoning", "knowledge representation", "declarative reasoning"],
            "voxsigil": ["vox sigil language", "sigil language"],
            # Add more domain-specific synonyms as needed
        }
        
    def load_all_sigils(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Load all VoxSigil sigils from the library path.
        
        Args:
            force_reload: Force reloading even if already cached/loaded.
            
        Returns:
            List of sigil dictionaries.
        """
        if self._loaded_sigils is not None and not force_reload:
            return self._loaded_sigils
            
        logger.info(f"Loading VoxSigil sigils from {self.voxsigil_library_path}")
        sigil_dirs = [ # Standard directory names within the VoxSigil-Library
            self.voxsigil_library_path / "core",
            self.voxsigil_library_path / "pglyph", # From original
            self.voxsigil_library_path / "sigils", # From original
            self.voxsigil_library_path / "examples", # From original
            self.voxsigil_library_path / "constructs",
            self.voxsigil_library_path / "loader", # From original
            self.voxsigil_library_path / "memory", # From original
            self.voxsigil_library_path / "tags", # From original
            self.voxsigil_library_path / "scaffolds", # From original
        ]
        
        sigil_files = []
        for dir_path in sigil_dirs:
            if dir_path.exists() and dir_path.is_dir():
                
                for ext in ['*.voxsigil', '*.yaml', '*.yml', '*.json']:
                    sigil_files.extend(list(dir_path.rglob(ext))) # rglob for recursive
        
        logger.info(f"Found {len(sigil_files)} potential sigil files.")
        
        loaded_sigils_list = []
        for file_path in sigil_files:
            sigil_content = self._load_sigil_file(file_path)
            if sigil_content:
                # Ensure essential meta-fields are present
                sigil_content.setdefault('_source_file', str(file_path))
                if '_last_modified' not in sigil_content: # Should be added by _load_sigil_file
                     try:
                        sigil_content['_last_modified'] = file_path.stat().st_mtime
                     except FileNotFoundError:
                        logger.warning(f"Could not stat file {file_path} during load_all_sigils fallback.")
                        sigil_content['_last_modified'] = time.time()

                loaded_sigils_list.append(sigil_content)
        
        logger.info(f"Successfully loaded {len(loaded_sigils_list)} sigils.")
        self._loaded_sigils = loaded_sigils_list
        if force_reload and HAVE_SENTENCE_TRANSFORMERS and self.embedding_model:
            logger.info("Force reload: Consider running precompute_all_embeddings(force_recompute=True) if embeddings need update.")
        return self._loaded_sigils

    # FEATURE 10: Sigil Schema Validator
    def _validate_sigil_data(self, sigil_data: Dict[str, Any], file_path: Path) -> bool:
        """Validates sigil data against the defined schema if jsonschema is available."""
        if not HAVE_JSONSCHEMA:
            return True # Skip validation if library not present
        try:
            validate(instance=sigil_data, schema=self.sigil_schema)
            return True
        except ValidationError as e:
            # Provide more context from the error object
            path_str = " -> ".join(map(str, e.path))
            logger.warning(f"Schema validation failed for {file_path}: {e.message} (at path: '{path_str}')")
            return False
        except Exception as e_generic: 
            logger.error(f"Generic error during schema validation for {file_path}: {e_generic}")
            return False

    def _load_sigil_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single VoxSigil sigil file with cache invalidation and schema validation."""
        # FEATURE 4: SmartCacheManager - File Modification Time for Sigil Content Cache
        try:
            file_mod_time = file_path.stat().st_mtime
        except FileNotFoundError:
            logger.error(f"File not found during stat: {file_path}")
            return None
        
        # Cache key includes path and modification time to detect changes
        cache_key = f"{str(file_path)}::{file_mod_time}" 

        if self.cache_enabled and cache_key in self._sigil_cache:
            return self._sigil_cache[cache_key]

        try:
            ext = file_path.suffix.lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                if ext in ['.yaml', '.yml', '.voxsigil']: # .voxsigil treated as YAML
                    sigil_data = yaml.safe_load(f)
                elif ext == '.json':
                    sigil_data = json.load(f)
                else:
                    logger.warning(f"Unsupported file extension: {ext} for file {file_path}")
                    return None

            if not isinstance(sigil_data, dict): 
                logger.warning(f"File {file_path} did not parse into a dictionary. Content type: {type(sigil_data)}. Skipping.")
                return None
            
            # Normalize relationships format to ensure schema validation passes
            if 'relationships' in sigil_data and not isinstance(sigil_data['relationships'], dict):
                # Convert relationships from list or other format to dictionary to pass schema validation
                if isinstance(sigil_data['relationships'], list):
                    # Convert list of relationships to dictionary with unique keys
                    relations_dict = {}
                    for i, rel in enumerate(sigil_data['relationships']):
                        if isinstance(rel, str):
                            # If it's a string, use it as a value with a generated key
                            key = f"relation_{i+1}"
                            relations_dict[key] = rel
                        elif isinstance(rel, dict) and len(rel) == 1:
                            # If it's a dictionary with a single key-value pair, use it directly
                            key, value = next(iter(rel.items()))
                            relations_dict[key] = value
                        else:
                            # For other types, create a key based on index
                            key = f"relation_{i+1}"
                            relations_dict[key] = rel
                    sigil_data['relationships'] = relations_dict
                    logger.debug(f"Converted relationships from list to dictionary in sigil: {file_path}")
                else:
                    # If it's not a list or dict, convert to a simple dict with a default key
                    sigil_data['relationships'] = {"default": sigil_data['relationships']}
                    logger.debug(f"Converted non-dict relationships to dictionary in sigil: {file_path}")
            
            # FEATURE 10: Sigil Schema Validation
            if not self._validate_sigil_data(sigil_data, file_path):
                logger.warning(f"Skipping sigil from {file_path} due to schema validation errors.")
                return None 
            
            sigil_data['_last_modified'] = file_mod_time # Store mod time for FEATURE 4 / FEATURE 7

            if self.cache_enabled:
                self._sigil_cache[cache_key] = sigil_data
            return sigil_data
        except yaml.YAMLError as ye:
            logger.error(f"YAML parsing error in sigil {file_path}: {ye}")
            return None
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error in sigil {file_path}: {je}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading sigil {file_path}: {e}")
            return None
            
    def format_sigil_for_prompt(self, sigil: Dict[str, Any], detail_level: str = "standard") -> str:
        """
        Format a sigil for inclusion in a prompt.
        
        Args:
            sigil: The sigil dictionary to format.
            detail_level: "summary", "standard", or "full".
            
        Returns:
            Formatted sigil string.
        """
        output = []
        
        if 'sigil' in sigil: output.append(f"Sigil: \"{sigil['sigil']}\"")
        
        # Consolidate tag handling
        all_tags = []
        if 'tag' in sigil and sigil['tag']: # Handle legacy 'tag' field
            if isinstance(sigil['tag'], str) and sigil['tag'] not in all_tags:
                all_tags.append(sigil['tag'])
        if 'tags' in sigil and sigil['tags']:
            if isinstance(sigil['tags'], list):
                for t in sigil['tags']:
                    if t not in all_tags:
                        all_tags.append(t)
            elif isinstance(sigil['tags'], str) and sigil['tags'] not in all_tags:
                all_tags.append(sigil['tags'])
        if all_tags:
            formatted_tags = ", ".join(f'"{tag}"' for tag in all_tags)
            output.append(f"Tags: {formatted_tags}")

        if 'principle' in sigil: output.append(f"Principle: \"{sigil['principle']}\"")
        
        if detail_level.lower() == "summary":
            return '\n'.join(output)
        
        if 'usage' in sigil and isinstance(sigil['usage'], dict):
            if 'description' in sigil['usage']:
                output.append(f"Usage: \"{sigil['usage']['description']}\"")
            # Show first example if available
            if 'examples' in sigil['usage'] and sigil['usage']['examples']:
                examples = sigil['usage']['examples']
                example_str = f"\"{examples[0]}\"" if isinstance(examples, list) and examples else f"\"{examples}\""
                output.append(f"Example: {example_str}")
        
        if '_source_file' in sigil: # Added for context
            output.append(f"Source File: {Path(sigil['_source_file']).name}")
        
        if detail_level.lower() == "full":
            if 'relationships' in sigil and isinstance(sigil['relationships'], dict):
                for rel_type, rel_values in sigil['relationships'].items():
                    if rel_values:
                        val_str = ', '.join(f'\"{v}\"' for v in rel_values) if isinstance(rel_values, list) else f"\"{rel_values}\""
                        output.append(f"Relationship ({rel_type}): {val_str}")
            
            if 'prompt_template' in sigil and isinstance(sigil['prompt_template'], dict):
                if 'type' in sigil['prompt_template']:
                    output.append(f"Template Type: {sigil['prompt_template']['type']}")
                if 'description' in sigil['prompt_template']:
                    output.append(f"Template Description: \"{sigil['prompt_template']['description']}\"")
        return '\n'.join(output)

    # FEATURE 5: QueryAugmenter
    def _augment_query(self, query: str) -> str:
        """Augments the query with synonyms from the internal map."""
        augmented_parts = [query]
        query_lower = query.lower()
        for term, synonyms in self.synonym_map.items():
            if term in query_lower: # If the base term is in the query
                for syn in synonyms:
                    # Add synonym if it's not already in query or parts
                    if syn not in query_lower and syn.lower() not in " ".join(augmented_parts).lower():
                         augmented_parts.append(syn)
        
        augmented_query = " ".join(augmented_parts)
        if augmented_query != query:
            logger.info(f"Query augmented: '{query}' -> '{augmented_query}'")
        return augmented_query

    # FEATURE 7: RecencyBooster
    def _apply_recency_boost(self, sigils_with_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies a recency boost to sigil scores if enabled and applicable."""
        if not self.recency_boost_factor > 0:
            return sigils_with_scores
            
        current_time_utc = datetime.now(timezone.utc).timestamp()
        boosted_sigils = []

        for item in sigils_with_scores:
            sigil_data = item # item is already the sigil dictionary
            last_modified_utc = sigil_data.get('_last_modified') # Unix timestamp
            original_score = sigil_data.get('_similarity_score', 0.0)
            
            if last_modified_utc and isinstance(last_modified_utc, (int, float)):
                age_seconds = current_time_utc - last_modified_utc
                if 0 <= age_seconds < self.recency_max_days_seconds:
                    # Linear decay for boost: newer items get more boost
                    boost_multiplier = 1.0 - (age_seconds / self.recency_max_days_seconds)
                    recency_bonus = self.recency_boost_factor * boost_multiplier
                    
                    new_score = min(1.0, original_score + recency_bonus) # Cap score at 1.0
                    if new_score > original_score:
                        sigil_data['_similarity_score'] = new_score
                        sigil_data['_recency_boost_applied'] = recency_bonus
                        logger.debug(f"Applied recency boost {recency_bonus:.3f} to sigil '{sigil_data.get('sigil', 'N/A')}' (new score: {new_score:.3f})")
            boosted_sigils.append(sigil_data)
        return boosted_sigils

    # FEATURE 1: DynamicContextOptimizer
    def _optimize_context_by_chars(
        self, sigils_for_context: List[Dict[str, Any]], 
        initial_detail_level: str,
        target_char_budget: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Adjusts sigil count or detail level to fit a character budget.
        Note: This uses character count as a proxy for tokens. For precise token counting,
        a library like 'tiktoken' would be needed along with the specific model's tokenizer.
        """
        final_sigils = list(sigils_for_context) # Work with a copy
        current_detail_level = initial_detail_level.lower()
        
        if not target_char_budget or not final_sigils:
            return final_sigils, current_detail_level

        def estimate_chars(s_list: List[Dict[str, Any]], d_level: str) -> int:
            return sum(len(self.format_sigil_for_prompt(s, d_level)) for s in s_list)

        # Detail levels ordered from most to least verbose
        detail_levels_ordered = ["full", "standard", "summary"]
        try:
            current_detail_idx = detail_levels_ordered.index(current_detail_level)
        except ValueError:
            current_detail_idx = 1 # Default to 'standard' if initial_detail_level is unknown
            current_detail_level = "standard"

        current_chars = estimate_chars(final_sigils, current_detail_level)

        # Stage 1: Reduce detail level if over budget
        while current_chars > target_char_budget and current_detail_idx < len(detail_levels_ordered) - 1:
            current_detail_idx += 1
            new_detail_level = detail_levels_ordered[current_detail_idx]
            logger.info(
                f"Context Optimizer: Chars {current_chars} > budget {target_char_budget}. "
                f"Reducing detail from {current_detail_level} to {new_detail_level} for {len(final_sigils)} sigils."
            )
            current_detail_level = new_detail_level
            current_chars = estimate_chars(final_sigils, current_detail_level)
        
        # Stage 2: If still over budget, remove least relevant sigils (sigils are pre-sorted by relevance)
        while current_chars > target_char_budget and len(final_sigils) > 1: # Keep at least one if possible
            removed_sigil = final_sigils.pop() # Removes the last (least relevant) sigil
            sig_name = removed_sigil.get('sigil', 'N/A')
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

    def _compute_text_embedding(self, text: str) -> np.ndarray:
        """
        Compute the embedding vector for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.embedding_model:
            raise ValueError("Embedding model is not initialized")
        
        # Use the standard embedding model to encode the text
        return self.embedding_model.encode(text)

    def create_rag_context(self, num_sigils: int = 5, filter_tag: Optional[str] = None, 
                           filter_tags: Optional[List[str]] = None, tag_operator: str = "OR", 
                           detail_level: str = "standard", query: Optional[str] = None, 
                           min_score_threshold: float = 0.0, # Min semantic score
                           include_explanations: bool = False,
                           # FEATURE 6: ExclusionFilter parameters
                           exclude_tags: Optional[List[str]] = None,
                           exclude_sigil_ids: Optional[List[str]] = None,
                           # FEATURE 5: Query Augmentation toggle
                           augment_query_flag: bool = True,
                           # FEATURE 7: Recency Boost toggle
                           apply_recency_boost_flag: bool = True,
                           # FEATURE 1: Context Optimizer parameters
                           enable_context_optimizer: bool = False,
                           max_context_chars_budget: Optional[int] = None
                           ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a RAG context string using loaded sigils, with new features.
        """
        all_loaded_sigils = self.load_all_sigils()
        if not all_loaded_sigils:
            logger.warning("No sigils loaded from the library. RAG context will be empty.")
            return "", []

        # FEATURE 6: Apply exclusion filters first
        current_sigils_pool = []
        norm_exclude_tags = {tag.lower() for tag in exclude_tags} if exclude_tags else set()
        norm_exclude_ids = {sid.lower() for sid in exclude_sigil_ids} if exclude_sigil_ids else set()

        for sigil in all_loaded_sigils:
            is_excluded = False
            if sigil.get('sigil','').lower() in norm_exclude_ids:
                is_excluded = True
            if not is_excluded and norm_exclude_tags:
                current_s_tags_set = set()
                if 'tags' in sigil:
                    s_tags_val = sigil['tags']
                    if isinstance(s_tags_val, list): current_s_tags_set.update(t.lower() for t in s_tags_val)
                    elif isinstance(s_tags_val, str): current_s_tags_set.add(s_tags_val.lower())
                if 'tag' in sigil and sigil['tag']: # legacy
                     current_s_tags_set.add(sigil['tag'].lower())
                if not norm_exclude_tags.isdisjoint(current_s_tags_set): # if any exclusion tag is present
                    is_excluded = True
            
            if not is_excluded:
                current_sigils_pool.append(sigil)
        
        logger.debug(f"Applied exclusion filters: {len(all_loaded_sigils) - len(current_sigils_pool)} excluded. {len(current_sigils_pool)} remaining.")

        # Apply inclusion tag filtering (if any sigils remain)
        if filter_tag and not filter_tags: filter_tags = [filter_tag] # backward compatibility
        
        if filter_tags and current_sigils_pool:
            norm_filter_tags = {tag.lower() for tag in filter_tags}
            tag_filtered_sigils = []
            for sigil in current_sigils_pool:
                current_s_tags_set = set()
                if 'tags' in sigil:
                    s_tags_val = sigil['tags']
                    if isinstance(s_tags_val, list): current_s_tags_set.update(t.lower() for t in s_tags_val)
                    elif isinstance(s_tags_val, str): current_s_tags_set.add(s_tags_val.lower())
                if 'tag' in sigil and sigil['tag']: # legacy
                     current_s_tags_set.add(sigil['tag'].lower())

                match = False
                if tag_operator.upper() == "AND":
                    if norm_filter_tags.issubset(current_s_tags_set): match = True
                else: # OR (default)
                    if not norm_filter_tags.isdisjoint(current_s_tags_set): match = True
                
                if match:
                    tag_filtered_sigils.append(sigil)
            
            current_sigils_pool = tag_filtered_sigils
            logger.debug(f"After inclusion tag filtering: {len(current_sigils_pool)} sigils.")
        
        # Semantic search if query is provided
        sigils_with_scores: List[Dict[str, Any]] = [] # This will hold sigils with their scores
        
        effective_query = query
        if query and augment_query_flag: # FEATURE 5
            effective_query = self._augment_query(query)

        if effective_query and HAVE_SENTENCE_TRANSFORMERS and self.embedding_model and current_sigils_pool:
            logger.info(f"Performing semantic search with query: '{effective_query}' on {len(current_sigils_pool)} candidates.")
            query_embedding = self.embedding_model.encode(effective_query)
            
            for sigil_data in current_sigils_pool:
                sigil_embedding = self._get_sigil_embedding(sigil_data)
                if sigil_embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, sigil_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(sigil_embedding)
                    )
                    if not np.isnan(similarity) and similarity >= min_score_threshold:
                        # Create a copy to store score and other retrieval metadata
                        retrieved_item = sigil_data.copy() 
                        retrieved_item['_similarity_score'] = float(similarity)
                        sigils_with_scores.append(retrieved_item)
            
            logger.info(f"Semantic search yielded {len(sigils_with_scores)} sigils meeting threshold {min_score_threshold:.2f}.")
        elif current_sigils_pool: # No semantic search, but sigils remain after filtering
            logger.info(f"No semantic search performed. Using {len(current_sigils_pool)} sigils from filtering stage. Assigning neutral score.")
            for sigil_data in current_sigils_pool:
                retrieved_item = sigil_data.copy()
                retrieved_item['_similarity_score'] = 0.5 # Neutral score if only filtered
                sigils_with_scores.append(retrieved_item)
        
        # Sort by score (semantic or default)
        sigils_with_scores.sort(key=lambda x: x.get('_similarity_score', 0.0), reverse=True)

        # FEATURE 7: Apply Recency Boost
        if apply_recency_boost_flag and sigils_with_scores:
            sigils_with_scores = self._apply_recency_boost(sigils_with_scores)
            # Re-sort if scores were changed by recency boost
            sigils_with_scores.sort(key=lambda x: x.get('_similarity_score', 0.0), reverse=True)
            logger.debug("Applied recency boost (if applicable) and re-sorted.")

        # FEATURE 3: AdvancedReRanker (Placeholder for future extension)
        # Example: if self.advanced_reranker_module:
        # sigils_with_scores = self.advanced_reranker_module.rerank(sigils_with_scores, effective_query)
        # logger.info("Applied advanced re-ranking.")
        
        # Select top N sigils before potential optimization
        selected_sigils_for_context = sigils_with_scores[:num_sigils]

        # Auto-fusion (from original script, adapted)
        if selected_sigils_for_context and len(selected_sigils_for_context) < num_sigils:
            logger.debug(f"Selected {len(selected_sigils_for_context)} sigils, less than requested {num_sigils}. Attempting auto-fusion.")
            # Pass the current selected sigils and how many more we can add
            fused_sigils = self.auto_fuse_related_sigils(
                selected_sigils_for_context,
                max_additional = num_sigils - len(selected_sigils_for_context)
            )
            if len(fused_sigils) > len(selected_sigils_for_context):
                 selected_sigils_for_context = fused_sigils # Update with fused results
                 logger.info(f"Auto-fused sigils. Total now: {len(selected_sigils_for_context)}")


        # FEATURE 2: Update analytics for the sigils selected so far
        for s_analytic_item in selected_sigils_for_context:
            if s_analytic_item.get('sigil'):
                self._update_sigil_analytics(s_analytic_item['sigil'])
        
        final_retrieved_sigils = selected_sigils_for_context
        current_detail_level = detail_level

        # FEATURE 1: Dynamic Context Optimizer
        if enable_context_optimizer and final_retrieved_sigils:
            char_budget = max_context_chars_budget or self.default_max_context_chars
            logger.info(f"Applying dynamic context optimization with budget: {char_budget} chars.")
            final_retrieved_sigils, current_detail_level = self._optimize_context_by_chars(
                final_retrieved_sigils, detail_level, char_budget
            )
        
        # Format final sigils for prompt
        formatted_sigils_parts = []
        for s_format_item in final_retrieved_sigils:
            formatted_text = self.format_sigil_for_prompt(s_format_item, current_detail_level)
            if include_explanations:
                explanation_parts = []
                if '_similarity_score' in s_format_item:
                    score_val = s_format_item['_similarity_score']
                    explanation_parts.append(f"Relevance: {score_val*100:.1f}%")
                if '_recency_boost_applied' in s_format_item:
                    boost_val = s_format_item['_recency_boost_applied']
                    explanation_parts.append(f"RecencyBoost: +{boost_val*100:.1f}%")
                if explanation_parts:
                    formatted_text += f"\n[Match Info: {'; '.join(explanation_parts)}]"
            formatted_sigils_parts.append(formatted_text)
            
        sigil_context_str = "\n\n---\n\n".join(formatted_sigils_parts) # Use a clear separator
        
        # Save embeddings cache periodically or at end of a significant operation
        if self.cache_enabled and HAVE_SENTENCE_TRANSFORMERS and len(self._embeddings_cache) % 100 > 80: # Heuristic
            self._save_embeddings_cache()
            
        return sigil_context_str, final_retrieved_sigils
        
    def inject_voxsigil_context(self, prompt: str, num_sigils: int = 5, filter_tag: Optional[str] = None,
                            filter_tags: Optional[List[str]] = None, tag_operator: str = "OR",
                            detail_level: str = "standard", query: Optional[str] = None,
                            min_score_threshold: float = 0.0, include_explanations: bool = False,
                            # Pass-through for new features
                            exclude_tags: Optional[List[str]] = None,
                            exclude_sigil_ids: Optional[List[str]] = None,
                            augment_query_flag: bool = True,
                            apply_recency_boost_flag: bool = True,
                            enable_context_optimizer: bool = False,
                            max_context_chars_budget: Optional[int] = None
                            ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Inject VoxSigil context into a prompt using RAG approach with all features.
        """
        search_query = query if query is not None else prompt # Use prompt as query if none specific provided
        
        sigil_context, retrieved_sigils = self.create_rag_context(
            num_sigils=num_sigils, filter_tag=filter_tag, filter_tags=filter_tags, 
            tag_operator=tag_operator, detail_level=detail_level, query=search_query, 
            min_score_threshold=min_score_threshold, include_explanations=include_explanations,
            exclude_tags=exclude_tags, exclude_sigil_ids=exclude_sigil_ids,
            augment_query_flag=augment_query_flag,
            apply_recency_boost_flag=apply_recency_boost_flag,
            enable_context_optimizer=enable_context_optimizer,
            max_context_chars_budget=max_context_chars_budget
        )
        
        rag_signature = self._generate_rag_signature(retrieved_sigils)
        
        enhanced_prompt = f"""You are an AI assistant operating with VoxSigil, a symbolic prompt language for orchestrating complex reasoning.
When working with VoxSigil inputs, pay careful attention to sigils and structured tags that guide your thinking process.

Here is relevant VoxSigil knowledge context retrieved for your task:
--- START VOXSİGİL CONTEXT ---
{sigil_context if sigil_context else "No specific VoxSigil context items were retrieved for this query."}
--- END VOXSİGİL CONTEXT ---

{rag_signature}

Now, respond to the following prompt, making use of the VoxSigil context provided above where appropriate:

User Prompt: {prompt}
"""
        return enhanced_prompt, retrieved_sigils
    
    def create_sigil_index(self) -> Dict[str, Dict[str, Any]]: # Renamed from create_cached_sigil_index
        """
        Create an index of sigils by their ID for quick lookup.
        Loads all sigils if not already loaded.
        """
        sigils = self.load_all_sigils()
        sigil_index: Dict[str, Dict[str, Any]] = {}
        
        for sigil_item in sigils:
            if 'sigil' in sigil_item and isinstance(sigil_item['sigil'], str):
                sigil_id = sigil_item['sigil']
                if sigil_id in sigil_index:
                    logger.warning(f"Duplicate sigil ID '{sigil_id}' found. Overwriting with content from '{sigil_item.get('_source_file')}'. Original was '{sigil_index[sigil_id].get('_source_file')}'.")
                sigil_index[sigil_id] = sigil_item
        
        logger.info(f"Created sigil index with {len(sigil_index)} unique entries.")
        return sigil_index    

    def _get_sigil_embedding(self, sigil: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get or generate embedding for a sigil, using a cache key sensitive to content changes and model.
        """
        if not HAVE_SENTENCE_TRANSFORMERS or not self.embedding_model:
            return None
            
        # FEATURE 4: Robust cache key for embeddings
        sigil_id_part = sigil.get('sigil')
        # Use _last_modified (timestamp) added during sigil loading for content versioning
        last_modified_time = sigil.get('_last_modified', 0) 

        if sigil_id_part:
            # Using sigil ID is preferred if available and unique
            cache_key_base = f"id:{sigil_id_part}"
        else:
            # Fallback to hash of content if no ID (less ideal for tracking changes if content hash is not perfect)
            # However, _last_modified should capture most file-based changes.
            source_file_part = sigil.get('_source_file')
            if source_file_part:
                 cache_key_base = f"file:{source_file_part}" # If ID missing but file known
            else: # Absolute fallback
                 cache_key_base = f"hash:{hashlib.md5(str(sigil).encode('utf-8')).hexdigest()}"
        
        # Include model name and modification time in cache key
        cache_key = f"{cache_key_base}::mod:{last_modified_time}::model:{self.embedding_model_name}"

        if self.cache_enabled and cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
            
        text_to_embed = self._get_sigil_text_for_embedding(sigil)
        if not text_to_embed.strip():
            logger.debug(f"No text content to embed for sigil: {sigil_id_part or sigil.get('_source_file', 'N/A')}")
            return None

        try:
            embedding = self.embedding_model.encode(text_to_embed)
            if self.cache_enabled:
                self._embeddings_cache[cache_key] = embedding
                # Conditional save can be added here, e.g., if len(self._embeddings_cache) % N == 0
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for sigil ({sigil_id_part or 'N/A'}): {e}")
            return None
    
    def _get_sigil_text_for_embedding(self, sigil: Dict[str, Any]) -> str:
        """Extracts key textual parts of a sigil for generating its embedding."""
        texts = []
        if sigil.get('sigil'): texts.append(f"Sigil Name: {sigil['sigil']}")
        
        all_tags = []
        if sigil.get('tag'): all_tags.append(str(sigil['tag']))
        if sigil.get('tags'):
            tags_val = sigil['tags']
            if isinstance(tags_val, list): all_tags.extend(str(t) for t in tags_val)
            elif isinstance(tags_val, str): all_tags.append(tags_val)
        if all_tags: texts.append(f"Keywords: {', '.join(list(set(all_tags)))}") # Unique tags

        if sigil.get('principle'): texts.append(f"Core Principle: {sigil['principle']}")
        
        usage_parts = []
        if 'usage' in sigil and isinstance(sigil['usage'], dict):
            if sigil['usage'].get('description'):
                usage_parts.append(f"Description: {sigil['usage']['description']}")
            if sigil['usage'].get('examples'):
                examples = sigil['usage']['examples']
                ex_str = "; ".join(examples[:2]) if isinstance(examples, list) else str(examples) # First 2 examples
                usage_parts.append(f"Examples: {ex_str}")
        if usage_parts:
            texts.append(f"Usage Context: {' '.join(usage_parts)}")
            
        return "\n".join(texts)

    # FEATURE 9: User-Managed Embeddings Cache Persistence (modified original methods)
    def _load_embeddings_cache(self, file_path: Optional[Path] = None) -> None:
        """Load embeddings cache from disk. Uses default path if None."""
        path_to_load = file_path or self._embeddings_cache_path
        if not self.cache_enabled:
            logger.debug("Cache is disabled. Skipping loading embeddings cache.")
            return
        
        if not path_to_load.exists():
            logger.info(f"Embeddings cache not found at {path_to_load}. Starting with an empty cache.")
            self._embeddings_cache = {}
            return
            
        try:
            logger.info(f"Loading embeddings cache from {path_to_load}")
            # Using robust loading for dict saved as 0-d object array
            cache_data = np.load(path_to_load, allow_pickle=True)
            if 'embeddings_dict' in cache_data and cache_data['embeddings_dict'].size > 0 :
                 loaded_cache = cache_data['embeddings_dict'].item()
                 if isinstance(loaded_cache, dict):
                    self._embeddings_cache.update(loaded_cache) # Merge, new values overwrite
                    logger.info(f"Loaded {len(loaded_cache)} embeddings. Total cached: {len(self._embeddings_cache)}")
                 else:
                    logger.warning(f"Embeddings data in {path_to_load} is not a dictionary. Cache not loaded.")
            elif 'embeddings' in cache_data: # Legacy support for old format
                 legacy_embeddings = cache_data.get('embeddings')
                 if hasattr(legacy_embeddings, 'ndim') and legacy_embeddings.ndim == 0 and isinstance(legacy_embeddings.item(), dict):
                     self._embeddings_cache.update(legacy_embeddings.item())
                     logger.info(f"Loaded legacy format embeddings. Total cached: {len(self._embeddings_cache)}")
                 else:
                     logger.warning("Found 'embeddings' key but not in expected 0-d array format. Cache not loaded from this key.")
            else:
                 logger.info(f"No 'embeddings_dict' or compatible 'embeddings' key found in {path_to_load}. Cache remains as is.")

        except Exception as e:
            logger.error(f"Error loading embeddings cache from {path_to_load}: {e}")
            # Decide if to reset: self._embeddings_cache = {} # Or keep existing on error
    
    def _save_embeddings_cache(self, file_path: Optional[Path] = None) -> None:
        """Save embeddings cache to disk. Uses default path if None."""
        path_to_save = file_path or self._embeddings_cache_path
        if not self.cache_enabled or not self._embeddings_cache:
            if not self._embeddings_cache: logger.debug("Embeddings cache is empty. Nothing to save.")
            return
            
        try:
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {len(self._embeddings_cache)} embeddings to cache at {path_to_save}")
            # Save the dictionary as a 0-d object array for robust loading
            np.savez_compressed(
                path_to_save, 
                embeddings_dict=np.array(self._embeddings_cache, dtype=object), # Correct way to save dict
                timestamp=datetime.now(timezone.utc).timestamp(),
                model_name=self.embedding_model_name
            )
            logger.info(f"Embeddings cache saved successfully.")
        except Exception as e:
            logger.error(f"Error saving embeddings cache to {path_to_save}: {e}")

    def export_embeddings_cache(self, export_path: Path) -> bool:
        """Exports the current in-memory embeddings cache to a specified file."""
        if not self.cache_enabled:
            logger.warning("Cache is disabled. Cannot export embeddings.")
            return False
        if not self._embeddings_cache:
            logger.info("Embeddings cache is empty. Nothing to export.")
            # Optionally create an empty cache file if that's desired behavior
            # self._save_embeddings_cache(file_path=export_path) 
            return False # Or True if empty file is "successful export"
        try:
            self._save_embeddings_cache(file_path=export_path) # This method now handles logging
            return True
        except Exception: # _save_embeddings_cache already logs details
            return False

    def import_embeddings_cache(self, import_path: Path, merge: bool = True) -> bool:
        """Imports an embeddings cache from a specified file, optionally merging."""
        if not self.cache_enabled:
            logger.warning("Cache is disabled. Cannot import embeddings.")
            return False
        if not import_path.exists() or not import_path.is_file():
            logger.error(f"Embeddings cache file not found for import: {import_path}")
            return False
        
        try:
            logger.info(f"Importing embeddings from {import_path}. Merge: {merge}")
            # Load into a temporary dict first to manage merge/replace logic
            temp_load_cache = {}
            cache_data = np.load(import_path, allow_pickle=True)

            if 'embeddings_dict' in cache_data and cache_data['embeddings_dict'].size > 0:
                 loaded_entries = cache_data['embeddings_dict'].item()
            elif 'embeddings' in cache_data: # Legacy support
                 loaded_entries = cache_data['embeddings'].item() if hasattr(cache_data['embeddings'], 'ndim') and cache_data['embeddings'].ndim == 0 else {}
            else:
                loaded_entries = {}
                logger.warning(f"No compatible embedding data found in {import_path}.")
                return False

            if not isinstance(loaded_entries, dict):
                logger.error(f"Data in {import_path} is not a dictionary. Import failed.")
                return False

            if merge:
                # Current cache takes precedence if keys collide, or other way?
                # Standard update: imported takes precedence.
                # To have existing take precedence: temp_load_cache.update(self._embeddings_cache); self._embeddings_cache = temp_load_cache
                self._embeddings_cache.update(loaded_entries) 
                logger.info(f"Merged {len(loaded_entries)} embeddings. Total now: {len(self._embeddings_cache)}.")
            else:
                self._embeddings_cache = loaded_entries
                logger.info(f"Replaced cache with {len(loaded_entries)} embeddings from {import_path}.")
            return True
        except Exception as e:
            logger.error(f"Error importing embeddings cache from {import_path}: {e}")
            return False

    # FEATURE 8: Batch Embedding Processor
    def precompute_all_embeddings(self, force_recompute: bool = False, batch_size: int = 32) -> int:
        """
        Generates and caches embeddings for all loaded sigils, using batch processing.
        Useful for pre-populating the cache for embedded systems or before LLM training.
        """
        if not HAVE_SENTENCE_TRANSFORMERS or not self.embedding_model:
            logger.warning("Sentence Transformers not available. Cannot precompute embeddings.")
            return 0

        all_sigils_list = self.load_all_sigils(force_reload=force_recompute) # Force reload if recomputing all
        if not all_sigils_list:
            logger.info("No sigils loaded to precompute embeddings for.")
            return 0
        
        texts_to_embed_batch: List[str] = []
        sigil_cache_keys_batch: List[str] = [] # To map embeddings back to their cache keys

        for sigil_data in all_sigils_list:
            sigil_id_part = sigil_data.get('sigil')
            last_modified_time = sigil_data.get('_last_modified', 0)
            
            if sigil_id_part: cache_key_base = f"id:{sigil_id_part}"
            elif sigil_data.get('_source_file'): cache_key_base = f"file:{sigil_data['_source_file']}"
            else: cache_key_base = f"hash:{hashlib.md5(str(sigil_data).encode('utf-8')).hexdigest()}"
            
            current_cache_key = f"{cache_key_base}::mod:{last_modified_time}::model:{self.embedding_model_name}"

            if force_recompute or current_cache_key not in self._embeddings_cache:
                text = self._get_sigil_text_for_embedding(sigil_data)
                if text.strip(): # Only add if there's actual text
                    texts_to_embed_batch.append(text)
                    sigil_cache_keys_batch.append(current_cache_key)
        
        if not texts_to_embed_batch:
            logger.info("All sigil embeddings seem to be up-to-date in cache (or no text to embed). Force recompute if needed.")
            return 0

        logger.info(f"Starting batch embedding computation for {len(texts_to_embed_batch)} sigils.")
        computed_count = 0
        try:
            # Use show_progress_bar if available and desired for long jobs
            generated_embeddings = self.embedding_model.encode(
                texts_to_embed_batch, 
                batch_size=batch_size, 
                show_progress_bar=True if len(texts_to_embed_batch) > batch_size else False
            )
            
            for i, emb_vector in enumerate(generated_embeddings):
                cache_key_for_sigil = sigil_cache_keys_batch[i]
                self._embeddings_cache[cache_key_for_sigil] = emb_vector
                computed_count += 1
            
            logger.info(f"Successfully computed and cached {computed_count} new/updated embeddings.")
            if computed_count > 0 and self.cache_enabled:
                self._save_embeddings_cache() # Save after a significant batch job
        except Exception as e:
            logger.error(f"Error during batch embedding generation: {e}")
        return computed_count

    # FEATURE 2: SigilAnalytics - Method to update stats
    def _update_sigil_analytics(self, sigil_id: str) -> None:
        """Updates retrieval analytics for a given sigil."""
        if sigil_id: # Ensure sigil_id is valid
            self._sigil_retrieval_counts[sigil_id] += 1
            self._sigil_last_retrieved_time[sigil_id] = time.time()

    # FEATURE 2: SigilAnalytics - Method to get stats
    def get_sigil_analytics(self, sigil_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns retrieval analytics for a specific sigil or overall statistics.
        """
        if sigil_id:
            if sigil_id not in self._sigil_retrieval_counts:
                 return {"id": sigil_id, "retrieval_count": 0, "last_retrieved_iso": None, "message": "Sigil not found in analytics."}
            return {
                "id": sigil_id,
                "retrieval_count": self._sigil_retrieval_counts.get(sigil_id, 0),
                "last_retrieved_iso": datetime.fromtimestamp(self._sigil_last_retrieved_time[sigil_id], tz=timezone.utc).isoformat() 
                                      if sigil_id in self._sigil_last_retrieved_time else None
            }
        else: # Overall statistics
            # Sort by count for top N
            top_5_retrieved = sorted(
                self._sigil_retrieval_counts.items(), 
                key=lambda item: item[1], 
                reverse=True
            )[:5]
            
            return {
                "total_retrievals_tracked_for_session": sum(self._sigil_retrieval_counts.values()),
                "unique_sigils_retrieved_in_session": len(self._sigil_retrieval_counts),
                "top_5_retrieved_in_session": [
                    {"id": sid, "count": count} for sid, count in top_5_retrieved
                ]
            }
            
    def analyze_sigil_match(self, sigil: Dict[str, Any], query: str, similarity_score: float) -> str:
        """(Existing method, adapted for consistency) Generate an explanation for why a sigil matched a query."""
        explanation_parts = []
        sigil_id = sigil.get('sigil', 'UnknownSigil')
        
        query_words = set(word.lower() for word in query.split() if len(word) > 2) # Meaningful words
        
        # Tag matches
        sigil_tags_set = set()
        if 'tag' in sigil and sigil['tag']: sigil_tags_set.add(str(sigil['tag']).lower())
        if 'tags' in sigil:
            tags_val = sigil['tags']
            if isinstance(tags_val, list): sigil_tags_set.update(str(t).lower() for t in tags_val)
            elif isinstance(tags_val, str): sigil_tags_set.add(tags_val.lower())
        
        matched_tags = query_words.intersection(sigil_tags_set)
        if matched_tags:
            explanation_parts.append(f"Query matched tags: {', '.join(sorted(list(matched_tags)))}")
        
        # Principle keyword matches
        if 'principle' in sigil and sigil['principle']:
            principle_lower = str(sigil['principle']).lower()
            # Simple keyword check (could be enhanced with NLP)
            principle_matched_keywords = {qw for qw in query_words if qw in principle_lower}
            if principle_matched_keywords:
                explanation_parts.append(f"Query terms in principle: {', '.join(sorted(list(principle_matched_keywords)))}")
        
        explanation_parts.append(f"Semantic similarity: {similarity_score:.3f}")
        
        if not explanation_parts:
            return f"Sigil '{sigil_id}' matched with similarity {similarity_score:.3f} (no specific keyword overlaps found in summary)."
        else:
            return f"Sigil '{sigil_id}' matched: " + "; ".join(explanation_parts)

    def _generate_rag_signature(self, retrieved_sigils: List[Dict[str, Any]]) -> str:
        """(Existing method) Generate a RAG signature for tracing."""
        if not retrieved_sigils:
            return "/* RAG Signature: No VoxSigil items retrieved for this context. */"
            
        sigil_infos = []
        for item_dict in retrieved_sigils: # item_dict is the sigil dictionary
            sigil_name = item_dict.get('sigil', 'unknown_sigil')
            score = item_dict.get('_similarity_score', 0.0) # Using the stored score
            sigil_infos.append(f"{sigil_name} (score: {score:.3f})")
            
        return f"/* RAG Signature: This response may be influenced by VoxSigils: {', '.join(sigil_infos)}. */"

    def auto_fuse_related_sigils(self, base_sigils: List[Dict[str, Any]], 
                                 max_additional: int = 3) -> List[Dict[str, Any]]:
        """
        (Existing method, adapted) Automatically add related sigils to an initial set.
        Tries to find related sigils based on 'relationships' field and shared tags.
        """
        if not base_sigils or max_additional <= 0:
            return base_sigils
            
        all_system_sigils = self.load_all_sigils()
        if not all_system_sigils: return base_sigils

        # Create an index for quick lookups by sigil ID
        sigil_index_by_id = {s['sigil']: s for s in all_system_sigils if 'sigil' in s}

        # Track IDs already in the list (base + newly added) to avoid duplicates
        current_sigil_ids = {s['sigil'] for s in base_sigils if 'sigil' in s}
        
        fused_sigils_list = list(base_sigils) # Start with a copy
        added_count = 0
        
        # Iterate through a copy of base_sigils for stable iteration if modifying fused_sigils_list
        for sigil_item in list(base_sigils): 
            if added_count >= max_additional: break
            
            source_sigil_id = sigil_item.get('sigil')
            if not source_sigil_id: continue

            # 1. Explicit relationships
            if 'relationships' in sigil_item and isinstance(sigil_item['relationships'], dict):
                for rel_type, rel_targets in sigil_item['relationships'].items():
                    targets_as_list = rel_targets if isinstance(rel_targets, list) else [rel_targets]
                    for target_id in targets_as_list:
                        if isinstance(target_id, str) and target_id in sigil_index_by_id and target_id not in current_sigil_ids:
                            related_s_data = sigil_index_by_id[target_id].copy()
                            related_s_data['_fusion_reason'] = f"related_to:{source_sigil_id}(type:{rel_type})"
                            related_s_data.setdefault('_similarity_score', 0.4) # Assign modest score
                            fused_sigils_list.append(related_s_data)
                            current_sigil_ids.add(target_id)
                            added_count += 1
                            if added_count >= max_additional: break
                    if added_count >= max_additional: break
            
            # 2. Shared tags (if still need more and explicit relations didn't fill quota)
            if added_count < max_additional:
                source_tags = set()
                if 'tag' in sigil_item and sigil_item['tag']: source_tags.add(str(sigil_item['tag']).lower())
                if 'tags' in sigil_item:
                    s_tags_val = sigil_item['tags']
                    if isinstance(s_tags_val, list): source_tags.update(str(t).lower() for t in s_tags_val)
                    elif isinstance(s_tags_val, str): source_tags.add(s_tags_val.lower())
                
                if source_tags:
                    for other_s in all_system_sigils:
                        other_id = other_s.get('sigil')
                        if not other_id or other_id in current_sigil_ids: continue

                        other_s_tags = set()
                        if 'tag' in other_s and other_s['tag']: other_s_tags.add(str(other_s['tag']).lower())
                        if 'tags' in other_s:
                            os_tags_val = other_s['tags']
                            if isinstance(os_tags_val, list): other_s_tags.update(str(t).lower() for t in os_tags_val)
                            elif isinstance(os_tags_val, str): other_s_tags.add(os_tags_val.lower())

                        if not source_tags.isdisjoint(other_s_tags): # If any shared tag
                            shared = source_tags.intersection(other_s_tags)
                            related_s_data = other_s.copy()
                            related_s_data['_fusion_reason'] = f"shared_tags_with:{source_sigil_id}(tags:{','.join(list(shared)[:2])})"
                            related_s_data.setdefault('_similarity_score', 0.3) # Lower score for tag match
                            fused_sigils_list.append(related_s_data)
                            current_sigil_ids.add(other_id)
                            added_count += 1
                            if added_count >= max_additional: break
                    if added_count >= max_additional: break
        
        if added_count > 0:
            logger.info(f"Auto-fused {added_count} additional sigils.")
        return fused_sigils_list
        
    def find_related_sigils(self, sigil_id: str, relationship_types: Optional[List[str]] = None,
                            include_semantic_relatives: bool = True, num_semantic: int = 3, 
                            semantic_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        (Existing method, enhanced) Find sigils related by definition or semantics.
        """
        all_sigils_list = self.load_all_sigils()
        sigil_index = {s['sigil']: s for s in all_sigils_list if 'sigil' in s}
        
        source_sigil_data = sigil_index.get(sigil_id)
        if not source_sigil_data:
            logger.warning(f"Source sigil '{sigil_id}' not found in index for finding relations.")
            return []
        
        related_items: List[Dict[str, Any]] = []
        found_related_ids: set[str] = {sigil_id} # Don't relate to self
        
        # 1. Direct relationships from source sigil
        if 'relationships' in source_sigil_data and isinstance(source_sigil_data['relationships'], dict):
            for rel_type, rel_values in source_sigil_data['relationships'].items():
                if relationship_types and rel_type not in relationship_types: continue
                
                targets = rel_values if isinstance(rel_values, list) else [rel_values]
                for target_id_val in targets:
                    if isinstance(target_id_val, str) and target_id_val in sigil_index and target_id_val not in found_related_ids:
                        r_sigil = sigil_index[target_id_val].copy()
                        r_sigil.update({'_relation_type': rel_type, '_relation_source': sigil_id, '_similarity_score': 0.9}) # High score for direct relation
                        related_items.append(r_sigil)
                        found_related_ids.add(target_id_val)
        
        # 2. Semantic relatives (if enabled and possible)
        if include_semantic_relatives and HAVE_SENTENCE_TRANSFORMERS and self.embedding_model:
            source_embedding = self._get_sigil_embedding(source_sigil_data)
            if source_embedding is not None:
                semantic_candidates = []
                for other_id, other_data in sigil_index.items():
                    if other_id in found_related_ids: continue # Skip already found
                    
                    other_embedding = self._get_sigil_embedding(other_data)
                    if other_embedding is not None:
                        sim = np.dot(source_embedding, other_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(other_embedding))
                        if not np.isnan(sim) and sim >= semantic_threshold:
                            s_match = other_data.copy()
                            s_match.update({'_relation_type': 'semantic', '_relation_source': sigil_id, '_similarity_score': float(sim)})
                            semantic_candidates.append(s_match)
                
                semantic_candidates.sort(key=lambda x: x.get('_similarity_score', 0.0), reverse=True)
                for sem_match in semantic_candidates[:num_semantic]:
                    if sem_match['sigil'] not in found_related_ids: # Double check ID before adding
                        related_items.append(sem_match)
                        found_related_ids.add(sem_match['sigil'])
        
        # Sort all found related items by their assigned/calculated similarity score
        related_items.sort(key=lambda x: x.get('_similarity_score', 0.0), reverse=True)
        
        logger.info(f"Found {len(related_items)} sigils related to '{sigil_id}'.")
        return related_items

# Singleton instance for easy import and use
voxsigil_rag = VoxSigilRAG()

# Helper function using the singleton
def inject_voxsigil_context(prompt: str, num_sigils: int = 5, filter_tag: Optional[str] = None,
                   filter_tags: Optional[List[str]] = None, tag_operator: str = "OR",
                   detail_level: str = "standard", query: Optional[str] = None,
                   min_score_threshold: float = 0.0, include_explanations: bool = False,
                   # Propagate new feature params to singleton call
                   exclude_tags: Optional[List[str]] = None,
                   exclude_sigil_ids: Optional[List[str]] = None,
                   augment_query_flag: bool = True,
                   apply_recency_boost_flag: bool = True,
                   enable_context_optimizer: bool = False,
                   max_context_chars_budget: Optional[int] = None
                   ) -> Tuple[str, List[Dict[str, Any]]]:
    """Helper function to inject VoxSigil context into a prompt, using the singleton instance."""
    return voxsigil_rag.inject_voxsigil_context(
        prompt=prompt, num_sigils=num_sigils, filter_tag=filter_tag, filter_tags=filter_tags,
        tag_operator=tag_operator, detail_level=detail_level, query=query,
        min_score_threshold=min_score_threshold, include_explanations=include_explanations,
        exclude_tags=exclude_tags, exclude_sigil_ids=exclude_sigil_ids,
        augment_query_flag=augment_query_flag,
        apply_recency_boost_flag=apply_recency_boost_flag,
        enable_context_optimizer=enable_context_optimizer,
        max_context_chars_budget=max_context_chars_budget
    )

if __name__ == "__main__":
    # Setup a temporary test VoxSigil-Library for demonstration
    test_lib_path = Path("Temp-VoxSigil-Library")
    test_lib_path.mkdir(exist_ok=True)
    sigils_dir = test_lib_path / "core" # Using 'core' as one of the scanned dirs
    sigils_dir.mkdir(exist_ok=True)
    
    # Create dummy sigil files
    sigil_alpha_content = {
        "sigil": "AlphaTest",
        "principle": "This is the Alpha sigil, focusing on initial test concepts and AI foundations.",
        "tags": ["test", "alpha", "ai_basics"],
        "usage": {"description": "Use for testing alpha features related to AI."}
    }
    sigil_alpha_file = sigils_dir / "alpha.voxsigil"
    with open(sigil_alpha_file, 'w', encoding='utf-8') as f:
        yaml.dump(sigil_alpha_content, f)
    
    time.sleep(0.1) # Ensure modification times are distinct

    sigil_beta_content = {
        "sigil": "BetaTest",
        "principle": "The Beta sigil explores advanced AI reasoning and symbolic manipulation.",
        "tags": ["test", "beta", "symbolic_reasoning", "advanced_ai"],
        "usage": {"description": "For beta tests on symbolic AI."},
        "relationships": {"complements": ["AlphaTest"]}
    }
    sigil_beta_file = sigils_dir / "beta.yaml" # Test with .yaml extension
    with open(sigil_beta_file, 'w', encoding='utf-8') as f:
        yaml.dump(sigil_beta_content, f)

    sigil_gamma_content = {
        "sigil": "GammaExcluded",
        "principle": "Gamma sigil is for features that are often excluded in tests.",
        "tags": ["test", "gamma", "exclusion_target"],
    }
    sigil_gamma_file = sigils_dir / "gamma.json" # Test with .json
    with open(sigil_gamma_file, 'w', encoding='utf-8') as f:
        json.dump(sigil_gamma_content, f)

    # Initialize RAG with the test library (using the global singleton for this test)
    voxsigil_rag.voxsigil_library_path = test_lib_path # Point singleton to test lib
    voxsigil_rag.load_all_sigils(force_reload=True) # Load them
    
    # FEATURE 8: Test Batch Embedding Processor
    if HAVE_SENTENCE_TRANSFORMERS:
        logger.info("\n=== Testing Batch Embedding Processor ===")
        count = voxsigil_rag.precompute_all_embeddings(force_recompute=True)
        logger.info(f"Precomputed {count} embeddings.")
    else:
        logger.warning("Skipping batch embedding test as SentenceTransformers not available.")

    logger.info("\n=== Testing RAG System with New Features ===")
    test_prompt_main = "Explain advanced AI symbolic reasoning using relevant sigils."
    
    # Using the global helper function which uses the configured singleton
    enhanced_prompt, retrieved_items = inject_voxsigil_context(
        prompt=test_prompt_main,
        query="AI symbolic reasoning", # Specific query
        num_sigils=2,
        filter_tags=["test"], # Must have 'test' tag
        exclude_tags=["gamma"], # Exclude sigils with 'gamma' tag (should exclude GammaExcluded)
        # exclude_sigil_ids=["AlphaTest"], # Example of excluding by ID
        augment_query_flag=True,      # FEATURE 5
        apply_recency_boost_flag=True, # FEATURE 7
        include_explanations=True,
        enable_context_optimizer=True, # FEATURE 1
        max_context_chars_budget=300  # Small budget to force optimization
    )
    
    print("\n\n=== Original Prompt ===")
    print(test_prompt_main)
    print("\n=== Retrieved Sigil Items (Count: {}) ===".format(len(retrieved_items)))
    if retrieved_items:
        for idx, item in enumerate(retrieved_items):
            print(f"{idx+1}. ID: {item.get('sigil', 'N/A')}, "
                  f"Score: {item.get('_similarity_score', 0.0):.3f}, "
                  f"Source: {Path(item.get('_source_file','N/A')).name}")
            if '_recency_boost_applied' in item:
                print(f"    Recency Boost: +{item['_recency_boost_applied']*100:.1f}%")
            if '_fusion_reason' in item:
                print(f"    Fusion Reason: {item['_fusion_reason']}")
            if item.get('sigil') == "GammaExcluded":
                 logger.error("TEST VALIDATION FAILED: GammaExcluded sigil was retrieved despite exclusion rule.")

    else:
        print("No sigil items retrieved.")
    
    print("\n=== Enhanced Prompt with VoxSigil Context ===")
    print(enhanced_prompt)

    # FEATURE 2: Test Sigil Analytics
    logger.info("\n=== Sigil Analytics (Overall) ===")
    print(json.dumps(voxsigil_rag.get_sigil_analytics(), indent=2))
    if retrieved_items and retrieved_items[0].get('sigil'):
        logger.info(f"\n=== Analytics for '{retrieved_items[0]['sigil']}' ===")
        print(json.dumps(voxsigil_rag.get_sigil_analytics(sigil_id=retrieved_items[0]['sigil']), indent=2))

    # FEATURE 9: Test Export/Import of Embeddings Cache
    if HAVE_SENTENCE_TRANSFORMERS:
        logger.info("\n=== Testing Embeddings Cache Export/Import ===")
        cache_export_file = Path("./exported_voxsigil_embeddings.npz")
        if voxsigil_rag.export_embeddings_cache(cache_export_file):
            logger.info(f"Successfully exported embeddings to {cache_export_file}")
            
            # Simulate new instance loading this cache
            new_rag_instance = VoxSigilRAG(voxsigil_library_path=test_lib_path, cache_enabled=True)
            new_rag_instance._embeddings_cache = {} # Clear initial cache
            logger.info(f"New instance initially has {len(new_rag_instance._embeddings_cache)} embeddings.")
            if new_rag_instance.import_embeddings_cache(cache_export_file):
                logger.info(f"Successfully imported. New instance now has {len(new_rag_instance._embeddings_cache)} embeddings.")
                # Verify some key exists (if AlphaTest was embedded)
                # This check depends on the exact cache key format, which can be complex.
                # A simpler check is just the count.
            if cache_export_file.exists():
                os.remove(cache_export_file) # Clean up
    
    # Clean up dummy files and directory
    logger.info("\nCleaning up test files...")
    try:
        if sigil_alpha_file.exists(): os.remove(sigil_alpha_file)
        if sigil_beta_file.exists(): os.remove(sigil_beta_file)
        if sigil_gamma_file.exists(): os.remove(sigil_gamma_file)
        if sigils_dir.exists(): sigils_dir.rmdir()
        if test_lib_path.exists(): test_lib_path.rmdir()
        logger.info("Test cleanup complete.")
    except OSError as e:
        logger.error(f"Error during cleanup: {e}. Manual cleanup of 'Temp-VoxSigil-Library' might be needed.")

    logger.info("\n=== VoxSigil RAG Test Run Finished ===")