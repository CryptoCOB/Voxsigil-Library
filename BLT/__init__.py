#!/usr/bin/env python
"""
blt_encoder.py - Implementation of BLT (Bidirectional Language Transformer) Encoder

This file implements the BLT encoder interface for VantaCore.
"""

import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
    from jsonschema import SchemaError, ValidationError
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None
    SchemaError = None
    ValidationError = None
    logging.getLogger(__name__).warning(
        "jsonschema library not installed. Schema validation disabled"
    )

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None
    logging.getLogger(__name__).warning(
        "numpy not installed. Some BLTEncoder features will be limited"
    )

# Try to import the interface, but don't fail if it's not available
try:
    from Vanta.interfaces.blt_encoder_interface import BaseBLTEncoder
except ImportError:
    # Create a minimal base class if the interface is not available
    class BaseBLTEncoder:
        pass


logger = logging.getLogger("VoxSigil.BLTEncoder")


class BLTEncoder(BaseBLTEncoder):
    """
    Implementation of the BLT encoder interface using transformer-based embedding models.

    This implementation supports various embedding models and provides caching
    for efficient reuse of embeddings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the BLT encoder with configuration.

        Args:
            config: Dictionary containing configuration parameters:
                - model_name: Name of the embedding model to use
                - embedding_dim: Dimension of the embeddings
                - cache_enabled: Whether to cache embeddings
                - cache_max_size: Maximum cache size
                - use_gpu: Whether to use GPU for encoding
                - min_patch_size: Minimum size for patches in words (used in create_patches)
                - max_patch_size: Maximum size for patches in words (used in create_patches)
                - entropy_threshold: Threshold for determining high vs low entropy
            **kwargs: Additional keyword arguments to be added to the config dictionary
        """
        self.config = config or {}

        # Add any direct keyword arguments to the config
        for key, value in kwargs.items():
            self.config[key] = value
        self.model_name = self.config.get("model_name", "all-MiniLM-L12-v2")
        self.embedding_dim = self.config.get("embedding_dim", 384)
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_max_size = self.config.get("cache_max_size", 5000)
        self.use_gpu = self.config.get("use_gpu", False)
        self.min_patch_size = self.config.get("min_patch_size", 4)
        self.max_patch_size = self.config.get("max_patch_size", 8)
        self.entropy_threshold = self.config.get("entropy_threshold", 0.5)

        # Initialize embedding cache
        self.embedding_cache = {}

        # Initialize the model (placeholder for actual implementation)
        self._initialize_model()

        logger.info(
            f"BLTEncoder initialized with model '{self.model_name}', embedding_dim={self.embedding_dim}"
        )

    def _initialize_model(self):
        """Initialize the embedding model."""
        # This is a placeholder - in a real implementation, this would load
        # the specified embedding model using a library like sentence-transformers
        logger.info(f"Loading embedding model '{self.model_name}'")

        try:
            # Placeholder for actual model initialization
            # from sentence_transformers import SentenceTransformer
            # self.model = SentenceTransformer(self.model_name, device='cuda' if self.use_gpu else 'cpu')
            pass
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fall back to a dummy model that will produce random embeddings for development
            logger.warning("Using fallback random embedding generator")

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by this encoder.        Returns:
            int: The embedding dimension.
        """
        return self.embedding_dim

    def get_encoder_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing details about the encoder.

        Returns:
            Dict[str, Any]: Encoder details (e.g., model name, version).
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.embedding_cache)
            if hasattr(self, "embedding_cache")
            else 0,
            "cache_max_size": self.cache_max_size,
            "use_gpu": self.use_gpu,
        }

    def encode(self, text_content: str, task_type: str = "general") -> List[float]:
        """
        Encode text content into a vector embedding.

        Args:
            text_content: The text to encode
            task_type: The type of task (affects encoding parameters)

        Returns:
            List[float]: Vector embedding of the text
        """
        # Convert data to text if it's not already a string
        if isinstance(text_content, (list, dict)):
            text_content = json.dumps(text_content)
        elif not isinstance(text_content, str):
            text_content = str(text_content)

        # Forward to the existing text encoding method
        return self._encode_text(text_content, task_type)

    def _encode_text(
        self, text_content: str, task_type: str = "general"
    ) -> List[float]:
        """
        Encode the input text into a vector embedding.

        Args:
            text_content: The text to encode
            task_type: The type of task the embedding will be used for

        Returns:
            List[float]: Vector embedding of the text
        """
        # Check cache if enabled
        cache_key = f"{task_type}:{text_content}"
        if self.cache_enabled and cache_key in self.embedding_cache:
            logger.debug(f"Cache hit for text (len={len(text_content)})")
            return self.embedding_cache[cache_key]

        logger.debug(
            f"Encoding text (len={len(text_content)}) for task type '{task_type}'"
        )

        try:
            # Placeholder for actual encoding - in a real implementation, this would use
            # the loaded model to generate an embedding
            # embedding = self.model.encode(text_content).tolist()

            # For development, generate a deterministic "embedding" based on text hash
            import hashlib

            hasher = hashlib.sha256(text_content.encode("utf-8"))
            embedding = [
                int(hasher.hexdigest()[i : i + 2], 16) / 255.0
                for i in range(0, min(self.embedding_dim * 2, 64), 2)
            ]

            # Pad or truncate to the correct dimension
            if len(embedding) < self.embedding_dim:
                embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
            else:
                embedding = embedding[: self.embedding_dim]

            # Cache the result if enabled
            if self.cache_enabled:
                # Manage cache size
                if len(self.embedding_cache) >= self.cache_max_size:
                    # Simple LRU - remove a random entry
                    # A real implementation would use a proper LRU cache
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))
                self.embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            # Return zeros as fallback
            return [0.0] * self.embedding_dim

    def encode_batch(
        self, texts: List[str], task_type: str = "general"
    ) -> List[List[float]]:
        """
        Encode multiple texts into vector embeddings, with potential batch efficiency.

        Args:
            texts: List of texts to encode
            task_type: The type of task the embeddings will be used for

        Returns:
            List[List[float]]: List of vector embeddings corresponding to input texts
        """
        # Check which texts are not in cache
        uncached_indices = []
        uncached_texts = []
        results = []

        # Initialize results list with placeholders
        for _ in range(len(texts)):
            results.append([0.0] * self.embedding_dim)

        if self.cache_enabled:
            for i, text in enumerate(texts):
                cache_key = f"{task_type}:{text}"
                if cache_key in self.embedding_cache:
                    results[i] = self.embedding_cache[cache_key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            logger.debug(
                f"Batch encoding {len(uncached_texts)} texts for task type '{task_type}'"
            )

            try:
                # Placeholder for actual batch encoding
                # batch_embeddings = self.model.encode(uncached_texts).tolist()

                # For development, encode each text individually
                batch_embeddings = []
                for text in uncached_texts:
                    embedding = self._encode_text(text, task_type)
                    batch_embeddings.append(embedding)

                # Put results in the right positions
                for i, embed in zip(uncached_indices, batch_embeddings):
                    results[i] = embed

                    # Cache if enabled
                    if self.cache_enabled:
                        cache_key = f"{task_type}:{texts[i]}"
                        self.embedding_cache[cache_key] = embed

            except Exception as e:
                logger.error(f"Error in batch encoding: {e}")
                # Fall back to individual encoding
                for i, text in zip(uncached_indices, uncached_texts):
                    results[i] = self._encode_text(text, task_type)

        return results

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Similarity score (typically between 0 and 1)
        """
        # Check dimensions match
        if len(embedding1) != len(embedding2):
            logger.warning(
                f"Embedding dimensions don't match: {len(embedding1)} vs {len(embedding2)}"
            )
            # Truncate to the shorter length
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]

        try:
            # Compute cosine similarity
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Avoid division by zero
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm1 * norm2)

            # Ensure result is within [0, 1] range
            return float(max(0.0, min(1.0, (similarity + 1) / 2)))

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: The query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top matches to return

        Returns:
            List[Dict[str, Any]]: Top matches with similarity scores
        """
        if not candidate_embeddings:
            return []

        try:
            # Compute similarities
            similarities = [
                self.compute_similarity(query_embedding, candidate)
                for candidate in candidate_embeddings
            ]

            # Create result objects with index and similarity
            results = [
                {"index": i, "similarity": sim} for i, sim in enumerate(similarities)
            ]

            # Sort by similarity (descending) and take top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []

    def create_patches(self, text: str) -> List[Dict[str, Any]]:
        """
        Create patches from text for BLT processing.

        Args:
            text: Input text to create patches from

        Returns:
            List[Dict[str, Any]]: List of patch dictionaries with entropy and content
        """
        try:
            # Simple patch creation - split text into chunks and compute basic entropy
            import math

            if not text:
                return []

            # Split text into words for simple patching
            words = text.split()
            patches = []

            # Use the configured min_patch_size and max_patch_size
            min_size = self.min_patch_size if hasattr(self, "min_patch_size") else 4
            max_size = self.max_patch_size if hasattr(self, "max_patch_size") else 8
            # Create patches of appropriate size
            patch_size = min(max_size, max(min_size, len(words) // 10 + 1))

            for i in range(0, len(words), patch_size):
                patch_words = words[i : i + patch_size]
                patch_text = " ".join(patch_words)

                # Compute simple entropy
                if patch_text:
                    byte_array = patch_text.encode("utf-8")
                    freq = {}
                    for byte_val in byte_array:
                        freq[byte_val] = freq.get(byte_val, 0) + 1
                    total_bytes = len(byte_array)

                    if total_bytes > 0:
                        entropy = -sum(
                            (count / total_bytes) * math.log2(count / total_bytes)
                            for count in freq.values()
                            if count > 0
                        )
                    else:
                        entropy = 0.0
                else:
                    entropy = 0.0

                patches.append(
                    {
                        "content": patch_text,
                        "entropy": round(entropy, 5),
                        "index": len(patches),
                        "word_count": len(patch_words),
                    }
                )

            return patches

        except Exception as e:
            logger.error(f"Error creating patches: {e}")
            return []

    # ---------------------------------------------------------------------
    #  Approximate decode helpers
    # ---------------------------------------------------------------------
    def decode(
        self,
        embedding: List[float],
        task_type: str = "general",
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Attempt to recover candidate original texts for a given embedding by searching
        the in-memory cache for the closest vectors and returning the best matches.

        Args:
            embedding: Vector representation to be decoded.
            task_type: Task identifier that was supplied during encoding.
            top_k: Number of candidate texts to return.

        Returns:
            A list (possibly empty) of dicts with ``text`` and ``similarity`` keys,
            sorted by descending similarity.
        """
        if not (self.cache_enabled and self.embedding_cache):
            return []

        # Filter cache entries that match the requested task type
        candidate_entries = (
            (cache_key.split(":", 1)[1], cached_embed)
            for cache_key, cached_embed in self.embedding_cache.items()
            if cache_key.startswith(f"{task_type}:")
        )

        scored: List[Dict[str, Any]] = []
        for text, cached_vector in candidate_entries:
            sim = self.compute_similarity(embedding, cached_vector)
            scored.append({"text": text, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def decode_batch(
        self,
        embeddings: List[List[float]],
        task_type: str = "general",
        top_k: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """
        Decode a batch of embeddings by applying ``decode`` to each vector.

        Args:
            embeddings: List of vectors to decode.
            task_type: Task identifier used during encoding.
            top_k: Number of candidates to return per embedding.

        Returns:
            List of lists; each inner list contains up to ``top_k`` candidate
            decodings for the corresponding input vector.
        """
        return [self.decode(vec, task_type, top_k) for vec in embeddings]


# SigilPatchEncoder moved from sigil_patch_encoder.py to avoid circular imports
class SigilPatchEncoder:
    """
    Production-Grade SigilPatchEncoder for VoxSigil System.

    A comprehensive implementation that provides sophisticated entropy analysis,
    adaptive patch generation, and robust encoding capabilities for the VoxSigil
    Byte Latent Transformer (BLT) architecture.

    Features:
    - Multi-scale entropy analysis with adaptive thresholds
    - Content-aware patch generation with semantic boundaries
    - Robust fallback mechanisms for reliability
    - Performance optimization with intelligent caching
    - Support for both BLT and heuristic processing modes
    - Comprehensive error handling and validation
    - Statistical analysis and monitoring capabilities
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        use_blt: bool = True,
        blt_encoder_kwargs: Optional[dict] = None,
        max_patch_size: int = 512,
        min_patch_size: int = 16,
        adaptive_patching: bool = True,
        enable_caching: bool = True,
        performance_tracking: bool = True,
    ):
        """
        Initialize the production-grade sigil patch encoder.

        Args:
            entropy_threshold: Base threshold for entropy classification (0.0-1.0)
            use_blt: Enable BLT encoder integration when available
            blt_encoder_kwargs: Configuration parameters for BLT encoder
            max_patch_size: Maximum characters per patch
            min_patch_size: Minimum characters per patch
            adaptive_patching: Enable intelligent patch boundary detection
            enable_caching: Enable result caching for performance
            performance_tracking: Enable performance metrics collection
        """
        # Core configuration
        self.entropy_threshold = max(0.0, min(1.0, entropy_threshold))
        self.max_patch_size = max(min_patch_size, max_patch_size)
        self.min_patch_size = max(1, min_patch_size)
        self.adaptive_patching = adaptive_patching
        self.enable_caching = enable_caching
        self.performance_tracking = performance_tracking

        # Performance metrics
        self.stats = (
            {
                "total_analyses": 0,
                "cache_hits": 0,
                "total_patches_created": 0,
                "avg_entropy_computed": 0.0,
                "processing_time_ms": 0.0,
            }
            if performance_tracking
            else None
        )  # Caching infrastructure
        self.cache = {} if enable_caching else None
        self.cache_max_size = 1000  # Limit cache size        # BLT encoder setup with enhanced error handling
        self.use_blt = use_blt
        self.blt_encoder = None

        if self.use_blt:
            try:
                # Use the BLTEncoder class from this same module to avoid circular imports
                # BLTEncoder is defined above in this same file
                ByteLatentTransformerEncoder = BLTEncoder

                kwargs = blt_encoder_kwargs or {}
                # Enhanced BLT configuration
                if "entropy_threshold" not in kwargs:
                    kwargs["entropy_threshold"] = self.entropy_threshold
                if "max_patch_size" not in kwargs:
                    kwargs["max_patch_size"] = self.max_patch_size
                if "min_patch_size" not in kwargs:
                    kwargs["min_patch_size"] = self.min_patch_size

                self.blt_encoder = ByteLatentTransformerEncoder(**kwargs)
                logger.info(
                    f"BLT encoder initialized successfully with {len(kwargs)} parameters"
                )
            except ImportError as ie:
                logger.warning(
                    f"BLT encoder class not available, using heuristic methods: {ie}"
                )
                self.use_blt = False
            except Exception as e:
                logger.error(
                    f"Failed to initialize BLT encoder: {e}. Falling back to heuristic methods"
                )
                self.use_blt = False
                self.blt_encoder = None

        # Content type detection patterns
        self.content_patterns = {
            "code": [
                r"def\s+\w+",
                r"class\s+\w+",
                r"function\s+\w+",
                r"=>",
                r"->",
                r"\{.*\}",
                r"import\s+\w+",
            ],
            "json": [r"^\s*[\{\[]", r'"[\w\s]+"\s*:', r'"\w+"\s*:\s*"', r":\s*[\{\[]"],
            "xml": [r"<\w+[^>]*>", r"</\w+>", r"<\w+\s*/>", r"<!DOCTYPE", r"<\?xml"],
            "markdown": [
                r"^#{1,6}\s+",
                r"^\s*-\s+",
                r"^\s*\*\s+",
                r"```\w*",
                r"\[.*\]\(.*\)",
            ],
            "data": [r"\d+\.\d+", r"\d{4}-\d{2}-\d{2}", r"[A-Z]{2,}", r"\w+:\w+"],
        }

        logger.info(
            f"SigilPatchEncoder initialized (BLT: {self.use_blt}, caching: {enable_caching}, adaptive: {adaptive_patching})"
        )

    def analyze_entropy(
        self, text: str, detailed_analysis: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Advanced entropy analysis with adaptive patch generation.

        Performs sophisticated content analysis to generate semantically-aware patches
        with accurate entropy calculations, supporting multiple content types and
        providing intelligent fallback mechanisms.

        Args:
            text: Input text for analysis
            detailed_analysis: Enable detailed entropy breakdown and statistics

        Returns:
            Tuple of (patch_contents, entropy_scores)
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid or empty text provided for entropy analysis")
            return [], []

        # Performance tracking
        start_time = time.time() if self.performance_tracking else None

        # Check cache first
        cache_key = None
        if self.enable_caching and self.cache is not None:
            cache_key = hashlib.md5(f"{text}_{detailed_analysis}".encode()).hexdigest()
            if cache_key in self.cache:
                if self.stats:
                    self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for entropy analysis (key: {cache_key[:8]})")
                return self.cache[cache_key]

        try:
            # Primary BLT-based analysis
            if self.use_blt and self.blt_encoder:
                try:
                    patches = self.blt_encoder.create_patches(text)
                    if patches and isinstance(patches, list):
                        # Extract enhanced patch information
                        patch_contents = []
                        entropy_scores = []

                        for patch in patches:
                            if isinstance(patch, dict):
                                content = patch.get("content", patch.get("text", ""))
                                entropy = patch.get("entropy", 0.5)
                            else:
                                # Handle different patch formats
                                content = str(getattr(patch, "content", patch))
                                entropy = getattr(patch, "entropy", 0.5)

                            if content:  # Only add non-empty patches
                                patch_contents.append(content)
                                entropy_scores.append(float(entropy))

                        if patch_contents and entropy_scores:
                            logger.debug(
                                f"BLT analysis successful: {len(patch_contents)} patches, avg entropy {sum(entropy_scores) / len(entropy_scores):.3f}"
                            )
                            result = (patch_contents, entropy_scores)
                            self._cache_result(cache_key, result)
                            self._update_stats(
                                start_time, len(patch_contents), entropy_scores
                            )
                            return result

                except Exception as e:
                    logger.warning(
                        f"BLT encoder analysis failed: {e}. Using enhanced fallback"
                    )

            # Enhanced heuristic analysis with content-aware processing
            result = self._enhanced_heuristic_analysis(text, detailed_analysis)
            self._cache_result(cache_key, result)
            self._update_stats(start_time, len(result[0]), result[1])
            return result

        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}. Returning minimal fallback")
            fallback_result = ([text], [0.5])
            self._cache_result(cache_key, fallback_result)
            return fallback_result

    def _cache_result(
        self, cache_key: Optional[str], result: Tuple[List[str], List[float]]
    ) -> None:
        """Cache analysis result if caching is enabled."""
        if self.enable_caching and cache_key and self.cache is not None:
            # Manage cache size
            if len(self.cache) >= self.cache_max_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = result

    def _update_stats(
        self, start_time: Optional[float], patch_count: int, entropy_scores: List[float]
    ) -> None:
        """Update performance statistics if tracking is enabled."""
        if self.performance_tracking and self.stats and start_time is not None:
            self.stats["total_analyses"] += 1
            self.stats["total_patches_created"] += patch_count
            self.stats["processing_time_ms"] += (time.time() - start_time) * 1000
            if entropy_scores:
                avg_entropy = sum(entropy_scores) / len(entropy_scores)
                # Running average update
                total_analyses = self.stats["total_analyses"]
                self.stats["avg_entropy_computed"] = (
                    self.stats["avg_entropy_computed"] * (total_analyses - 1)
                    + avg_entropy
                ) / total_analyses

    def _detect_content_type(self, text: str) -> str:
        """Detect the primary content type of the text."""
        text_sample = text[:500]  # Sample for performance

        # Check patterns for different content types
        for content_type, patterns in self.content_patterns.items():
            match_count = sum(
                1
                for pattern in patterns
                if re.search(pattern, text_sample, re.MULTILINE)
            )
            if match_count >= 2:  # Require multiple pattern matches for confidence
                return content_type

        # Fallback heuristics
        if text_sample.strip().startswith(("(", "[", "{")):
            return "data"
        elif (
            len(text.split("\n")) > 10
            and len(text.split()) / len(text.split("\n")) < 15
        ):
            return "structured"
        else:
            return "natural"

    def _calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy for a text string."""
        if not text:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / text_length
                entropy -= probability * math.log2(probability)

        return entropy

    def _enhanced_heuristic_analysis(
        self, text: str, detailed_analysis: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Enhanced heuristic analysis with content-aware processing.

        Args:
            text: Input text to analyze
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Tuple of (patches, entropy_scores)
        """
        # Detect content type for appropriate processing
        content_type = self._detect_content_type(text)
        logger.debug(f"Detected content type: {content_type}")

        # Use adaptive patching if enabled
        if self.adaptive_patching:
            patches, entropy_scores = self._create_adaptive_patches(text, content_type)
        else:
            # Fallback to simple fixed-size patching
            patches = []
            entropy_scores = []

            # Simple chunking
            for i in range(0, len(text), self.max_patch_size):
                patch = text[i : i + self.max_patch_size]
                patches.append(patch)

                # Basic entropy calculation
                entropy = self._calculate_shannon_entropy(patch) / 8  # Normalize
                entropy_scores.append(max(0.1, min(0.9, entropy)))

        # Ensure we have at least one patch
        if not patches:
            patches = [text]
            entropy_scores = [0.5]

        logger.debug(
            f"Heuristic analysis: {len(patches)} patches, avg entropy {sum(entropy_scores) / len(entropy_scores):.3f}"
        )
        return patches, entropy_scores

    def _create_adaptive_patches(
        self, text: str, content_type: str
    ) -> Tuple[List[str], List[float]]:
        """Create content-aware patches with adaptive sizing."""
        patches = []
        entropy_scores = []

        if content_type in ["code", "json", "xml"]:
            # Use structure-aware patching
            patches, entropy_scores = self._create_structured_patches(text)
        elif content_type == "markdown":
            # Use markdown-aware patching
            patches, entropy_scores = self._create_markdown_patches(text)
        else:
            # Use semantic-aware patching for natural language
            patches, entropy_scores = self._create_semantic_patches(text)

        return patches, entropy_scores

    def _create_structured_patches(self, text: str) -> Tuple[List[str], List[float]]:
        """Create patches for structured content (code, JSON, XML)."""
        lines = text.split("\n")
        patches = []
        entropy_scores = []
        current_patch = []
        current_size = 0
        brace_depth = 0

        for line in lines:
            current_patch.append(line)
            current_size += len(line) + 1

            # Track nesting depth
            brace_depth += line.count("{") + line.count("[") + line.count("(")
            brace_depth -= line.count("}") + line.count("]") + line.count(")")

            # Create patch at structural boundaries or size limits
            should_break = (
                current_size >= self.max_patch_size
                or (brace_depth == 0 and current_size >= self.min_patch_size)
                or line.strip() in ["", "}", "]", ")", "</div>", "</section>"]
            )

            if should_break and current_patch:
                patch_text = "\n".join(current_patch)
                patches.append(patch_text)

                # Calculate entropy based on structure density
                structure_chars = sum(patch_text.count(c) for c in "{}[]()<>")
                structure_ratio = structure_chars / len(patch_text) if patch_text else 0
                base_entropy = self._calculate_shannon_entropy(patch_text)

                # Adjust entropy based on structure (structured content = lower entropy)
                adjusted_entropy = base_entropy * (0.3 + 0.7 * (1 - structure_ratio))
                entropy_scores.append(
                    max(0.1, min(0.9, adjusted_entropy / 8))
                )  # Normalize to 0-1

                current_patch = []
                current_size = 0

        # Add remaining content
        if current_patch:
            patch_text = "\n".join(current_patch)
            patches.append(patch_text)
            entropy_scores.append(self._calculate_shannon_entropy(patch_text) / 8)

        return patches, entropy_scores

    def _create_markdown_patches(self, text: str) -> Tuple[List[str], List[float]]:
        """Create patches for markdown content."""
        # Split by markdown headers and sections
        sections = re.split(r"^(#{1,6}\s+.*)$", text, flags=re.MULTILINE)
        patches = []
        entropy_scores = []

        current_section = []
        for section in sections:
            if not section.strip():
                continue

            current_section.append(section)
            section_text = "\n".join(current_section)

            if len(section_text) >= self.min_patch_size:
                if section.startswith("#") or len(section_text) >= self.max_patch_size:
                    patches.append(section_text)

                    # Markdown entropy estimation
                    markup_chars = sum(section_text.count(c) for c in "#*`[]_()")
                    markup_ratio = (
                        markup_chars / len(section_text) if section_text else 0
                    )
                    base_entropy = self._calculate_shannon_entropy(section_text)

                    # Headers and lists have lower entropy
                    adjusted_entropy = base_entropy * (0.5 + 0.5 * (1 - markup_ratio))
                    entropy_scores.append(max(0.2, min(0.8, adjusted_entropy / 8)))

                    current_section = []

        if current_section:
            section_text = "\n".join(current_section)
            patches.append(section_text)
            entropy_scores.append(self._calculate_shannon_entropy(section_text) / 8)

        return patches, entropy_scores

    def _create_semantic_patches(self, text: str) -> Tuple[List[str], List[float]]:
        """Create semantically-aware patches for natural language."""
        # Split by sentences and paragraphs
        sentences = re.split(r"[.!?]+\s+", text)
        patches = []
        entropy_scores = []
        current_patch = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_patch.append(sentence)
            current_size += len(sentence)

            # Break at semantic boundaries or size limits
            if current_size >= self.max_patch_size or (
                current_size >= self.min_patch_size
                and sentence.endswith((".", "!", "?"))
            ):
                patch_text = ". ".join(current_patch) + "."
                patches.append(patch_text)

                # Natural language entropy is typically higher
                base_entropy = self._calculate_shannon_entropy(patch_text)
                # Add complexity bonus for longer, varied sentences
                complexity_bonus = min(
                    0.2, len(set(patch_text.split())) / len(patch_text.split()) * 0.3
                )
                adjusted_entropy = (base_entropy / 8) + complexity_bonus
                entropy_scores.append(max(0.3, min(0.9, adjusted_entropy)))

                current_patch = []
                current_size = 0

        if current_patch:
            patch_text = ". ".join(current_patch) + "."
            patches.append(patch_text)
            entropy_scores.append(
                max(0.3, self._calculate_shannon_entropy(patch_text) / 8)
            )

        return patches, entropy_scores

    def compute_average_entropy(self, text: str) -> float:
        """
        Compute the average entropy for text.

        Args:
            text: The text to analyze

        Returns:
            Average entropy value
        """
        _, entropy_scores = self.analyze_entropy(text)
        if not entropy_scores:
            return 0.5  # Default medium entropy

        return sum(entropy_scores) / len(entropy_scores)

    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        Create a BLT-aware embedding for the text.

        Args:
            text: The text to encode

        Returns:
            Numpy array embedding or None if encoding fails
        """
        if not text:
            return None

        if self.use_blt and self.blt_encoder:
            try:
                # Use BLT encoder for embedding
                return np.array(self.blt_encoder.encode(text), dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error using BLT encoder for embedding: {e}")
                # Fall back to heuristic method

        # Simple fallback embedding using hashing
        return self._heuristic_encode(text)

    def _heuristic_encode(self, text: str) -> np.ndarray:
        """
        Create a simple embedding for text without BLT.

        Args:
            text: The text to encode

        Returns:
            Numpy array embedding
        """
        # Simple hash-based embedding - for fallback only
        # Define embedding dimension
        dim = 384  # Common embedding dimension

        # Convert text to bytes and hash
        text_bytes = text.encode("utf-8")
        hash_obj = hashlib.sha256(text_bytes)
        hash_bytes = hash_obj.digest()

        # Use the hash to seed a random number generator
        seed = int.from_bytes(hash_bytes[:4], byteorder="little")
        rng = np.random.RandomState(seed)

        # Generate a random vector
        embedding = rng.randn(dim).astype(np.float32)

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Normalize if needed
        if abs(norm1 - 1.0) > 1e-6:
            embedding1 = embedding1 / norm1
        if abs(norm2 - 1.0) > 1e-6:
            embedding2 = embedding2 / norm2

        # Calculate cosine similarity
        similarity = np.dot(
            embedding1, embedding2
        )  # Ensure the result is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))

    def get_performance_stats(self) -> Optional[Dict[str, Any]]:
        """Get current performance statistics."""

        if self.performance_tracking and self.stats:
            return self.stats.copy()
        return None

    def reset_cache(self) -> None:
        """Clear the analysis cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Analysis cache cleared")

    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate data against a given JSON schema.

        Args:
            data: The data (as a dictionary) to validate.
            schema: The JSON schema (as a dictionary) to validate against.

        Returns:
            True if the data is valid against the schema, False otherwise.
        """
        if jsonschema is None or ValidationError is None or SchemaError is None:
            logger.warning(
                "jsonschema library is not installed. Skipping schema validation."
            )
            return (
                True  # Or False, depending on desired behavior when library is missing
            )

        try:
            jsonschema.validate(instance=data, schema=schema)
            return True
        except ValidationError as e:
            logger.warning(f"Schema validation failed: {e.message}")
            return False
        except SchemaError as e:
            logger.error(f"Invalid schema provided: {e.message}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during schema validation: {e}")
            return False


class ByteLatentTransformerEncoder:
    """
    Production-grade Byte Latent Transformer Encoder with advanced semantic encoding capabilities.

    This implementation provides:
    - Sophisticated byte-level and semantic analysis
    - Multi-scale patch embedding with attention mechanisms
    - Entropy-driven adaptive encoding strategies
    - Context-aware transformations
    - Support for ARC GridFormer color validation and correction
    - Production-ready error handling and performance optimization
    """

    def __init__(
        self,
        patch_size: int = 64,
        max_patches: int = 16,
        embedding_dim: int = 128,
        arc_mode: bool = False,
        color_correction_threshold: float = 0.8,
        context_window: int = 256,
        attention_heads: int = 4,
        use_positional_encoding: bool = True,
        entropy_threshold: float = 0.25,
        min_patch_size: int = 8,
        max_patch_size: int = 128,
        **kwargs,
    ):
        """
        Initialize the ByteLatentTransformerEncoder.

        Args:
            patch_size: Default size for text patches (in bytes)
            max_patches: Maximum number of patches to process
            embedding_dim: Dimension of output embeddings
            arc_mode: Enable ARC-specific color validation features
            color_correction_threshold: Confidence threshold for ARC color corrections
            context_window: Size of context window for attention mechanisms
            attention_heads: Number of attention heads for multi-head attention
            use_positional_encoding: Whether to use positional encoding
            entropy_threshold: Threshold for entropy-based routing decisions
            min_patch_size: Minimum patch size for adaptive patching
            max_patch_size: Maximum patch size for adaptive patching
        """
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embedding_dim = embedding_dim
        self.arc_mode = arc_mode
        self.color_correction_threshold = color_correction_threshold
        self.context_window = context_window
        self.attention_heads = attention_heads
        self.use_positional_encoding = use_positional_encoding
        self.entropy_threshold = entropy_threshold
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

        # Initialize ARC-specific components if enabled
        self.arc_palette = list(range(10)) if arc_mode else []
        self.palette_embeddings = self._init_palette_embeddings() if arc_mode else None
        self.correction_history: list[tuple[Any, Any, float]] = []
        self.max_history = 100

        # Initialize encoding components
        self._init_encoding_components()

        # Performance tracking
        self._encoding_stats = {
            "total_encodings": 0,
            "avg_encoding_time": 0.0,
            "cache_hits": 0,
            "entropy_routing_decisions": {"high": 0, "low": 0},
        }

        # Simple LRU cache for frequent encodings
        self._encoding_cache: dict[str, np.ndarray] = {}
        self._cache_max_size = 1000

    def _init_encoding_components(self) -> None:
        """Initialize sophisticated encoding components."""
        # Create basis vectors for different semantic dimensions
        np.random.seed(42)  # Deterministic initialization

        # Semantic basis vectors for different content types
        self.semantic_bases = {
            "textual": np.random.randn(
                self.embedding_dim // 4, self.embedding_dim
            ).astype(np.float32),
            "numerical": np.random.randn(
                self.embedding_dim // 4, self.embedding_dim
            ).astype(np.float32),
            "structural": np.random.randn(
                self.embedding_dim // 4, self.embedding_dim
            ).astype(np.float32),
            "contextual": np.random.randn(
                self.embedding_dim // 4, self.embedding_dim
            ).astype(np.float32),
        }

        # Normalize basis vectors
        for key in self.semantic_bases:
            for i in range(self.semantic_bases[key].shape[0]):
                norm = np.linalg.norm(self.semantic_bases[key][i])
                if norm > 0:
                    self.semantic_bases[key][i] /= norm

        # Positional encoding matrix
        if self.use_positional_encoding:
            self.positional_encoding = self._create_positional_encoding()

        # Attention weight matrices (simplified multi-head attention)
        self.attention_weights = {
            "query": np.random.randn(
                self.attention_heads,
                self.embedding_dim,
                self.embedding_dim // self.attention_heads,
            ).astype(np.float32)
            * 0.1,
            "key": np.random.randn(
                self.attention_heads,
                self.embedding_dim,
                self.embedding_dim // self.attention_heads,
            ).astype(np.float32)
            * 0.1,
            "value": np.random.randn(
                self.attention_heads,
                self.embedding_dim,
                self.embedding_dim // self.attention_heads,
            ).astype(np.float32)
            * 0.1,
        }

    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding matrix."""
        pe = np.zeros((self.max_patches, self.embedding_dim), dtype=np.float32)

        for pos in range(self.max_patches):
            for i in range(0, self.embedding_dim, 2):
                div_term = np.exp(i * -(np.log(10000.0) / self.embedding_dim))
                pe[pos, i] = np.sin(pos * div_term)
                if i + 1 < self.embedding_dim:
                    pe[pos, i + 1] = np.cos(pos * div_term)

        return pe

    def _init_palette_embeddings(self) -> dict[int, np.ndarray]:
        """Initialize ARC color palette embeddings with semantic meaning."""
        palette_embeddings = {}
        color_semantics = {
            0: "background",  # Black - background/empty
            1: "primary",  # Blue - primary element
            2: "secondary",  # Red - secondary element
            3: "nature",  # Green - natural elements
            4: "warning",  # Yellow - attention/warning
            5: "neutral",  # Gray - neutral elements
            6: "accent",  # Magenta - accent elements
            7: "highlight",  # Orange - highlights
            8: "water",  # Cyan - water/flow elements
            9: "special",  # Brown - special elements
        }

        for color, semantic in color_semantics.items():
            # Create semantically meaningful embeddings for each color
            np.random.seed(color + 1000)  # Offset to avoid conflict with main seed
            base_embedding = np.random.randn(self.embedding_dim).astype(np.float32)

            # Add semantic bias based on color meaning
            semantic_weight = 0.3
            if semantic == "background":
                base_embedding[: self.embedding_dim // 4] *= 1 - semantic_weight
            elif semantic == "primary":
                base_embedding[: self.embedding_dim // 4] *= 1 + semantic_weight

            # Normalize
            norm = np.linalg.norm(base_embedding)
            palette_embeddings[color] = base_embedding / (norm + 1e-9)

        return palette_embeddings

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text using sophisticated BLT methodology.

        Args:
            text: Input text to encode
            use_cache: Whether to use encoding cache for performance

        Returns:
            np.ndarray: Normalized embedding vector
        """
        import time

        start_time = time.time()

        # Input validation and normalization
        validated_text = self.validate_input(text)
        if not validated_text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Check cache first
        cache_key = f"{hash(validated_text)}_{self.embedding_dim}"
        if use_cache and cache_key in self._encoding_cache:
            self._encoding_stats["cache_hits"] += 1
            return self._encoding_cache[cache_key].copy()

        # Create adaptive patches based on content entropy
        patches = self.create_patches(validated_text)
        if not patches:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Multi-scale embedding generation
        patch_embeddings = []
        for i, patch in enumerate(patches):
            patch_emb = self._advanced_patch_embedding(patch, position=i)
            patch_embeddings.append(patch_emb)

        # Apply attention mechanism for patch aggregation
        if len(patch_embeddings) > 1:
            final_embedding = self._apply_attention(patch_embeddings)
        else:
            final_embedding = patch_embeddings[0]

        # Apply contextual transformations
        final_embedding = self._apply_contextual_transformations(
            final_embedding, validated_text
        )

        # Final normalization
        norm = np.linalg.norm(final_embedding)
        normalized_embedding = (
            final_embedding / (norm + 1e-9) if norm > 0 else final_embedding
        )

        # Cache management
        if use_cache:
            self._manage_cache(cache_key, normalized_embedding.copy())

        # Update performance stats
        encoding_time = time.time() - start_time
        self._update_stats(encoding_time)

        return normalized_embedding.astype(np.float32)

    def _advanced_patch_embedding(
        self, patch: "Patch", position: int = 0
    ) -> np.ndarray:
        """Generate sophisticated embedding for a single patch."""
        # Multi-component embedding generation
        components = {
            "content": self._content_embedding(patch.content),
            "entropy": self._entropy_embedding(patch.entropy),
            "position": self._positional_embedding(position),
            "structure": self._structural_embedding(patch.content),
        }

        # Weighted combination based on patch characteristics
        weights = self._adaptive_weights(patch)

        final_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for component, weight in weights.items():
            if component in components:
                final_embedding += weight * components[component]

        return final_embedding

    def _content_embedding(self, content: str) -> np.ndarray:
        """Generate content-based embedding using multiple strategies."""
        if not content:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Strategy 1: Character frequency analysis
        char_freq = self._character_frequency_embedding(content)

        # Strategy 2: N-gram analysis
        ngram_emb = self._ngram_embedding(content)

        # Strategy 3: Semantic pattern recognition
        pattern_emb = self._pattern_embedding(content)

        # Combine strategies
        combined = 0.4 * char_freq + 0.3 * ngram_emb + 0.3 * pattern_emb

        # Normalize
        norm = np.linalg.norm(combined)
        return combined / (norm + 1e-9) if norm > 0 else combined

    def _character_frequency_embedding(self, content: str) -> np.ndarray:
        """Create embedding based on character frequency distribution."""
        if not content:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Count character categories
        char_counts = {
            "alpha": sum(1 for c in content if c.isalpha()),
            "digit": sum(1 for c in content if c.isdigit()),
            "space": sum(1 for c in content if c.isspace()),
            "punct": sum(1 for c in content if not c.isalnum() and not c.isspace()),
            "upper": sum(1 for c in content if c.isupper()),
            "lower": sum(1 for c in content if c.islower()),
        }

        total_chars = len(content)
        if total_chars == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Create frequency-based embedding
        freq_vector = np.zeros(self.embedding_dim, dtype=np.float32)

        # Map frequencies to embedding dimensions
        for i, (category, count) in enumerate(char_counts.items()):
            if i < self.embedding_dim:
                freq_vector[i] = count / total_chars

        # Add byte-level entropy information
        content_bytes = content.encode("utf-8", errors="replace")
        entropy = self._shannon_entropy(content_bytes)
        if self.embedding_dim > len(char_counts):
            freq_vector[len(char_counts)] = entropy / 8.0  # Normalize entropy

        return freq_vector

    def _ngram_embedding(self, content: str, n: int = 3) -> np.ndarray:
        """Create embedding based on n-gram analysis."""
        if len(content) < n:
            return self._character_frequency_embedding(content)

        # Generate n-grams
        ngrams = [content[i : i + n] for i in range(len(content) - n + 1)]

        if not ngrams:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Create n-gram frequency distribution
        from collections import Counter

        ngram_counts = Counter(ngrams)

        # Map n-grams to embedding space using hash
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        for ngram, count in ngram_counts.items():
            # Hash n-gram to embedding dimension
            ngram_hash = hash(ngram) % self.embedding_dim
            embedding[ngram_hash] += count / len(ngrams)

        return embedding

    def _pattern_embedding(self, content: str) -> np.ndarray:
        """Recognize and embed structural patterns in content."""
        import re

        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "number": r"\d+",
            "code": r"[{}()\[\]<>]",
            "camelCase": r"[a-z][A-Z]",
            "snake_case": r"[a-z_][a-z_]*[a-z]",
            "CAPS": r"\b[A-Z]{2,}\b",
        }

        pattern_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        for i, (pattern_name, pattern) in enumerate(patterns.items()):
            if i >= self.embedding_dim:
                break
            matches = len(re.findall(pattern, content))
            pattern_embedding[i] = min(matches / max(len(content), 1), 1.0)

        return pattern_embedding

    def _entropy_embedding(self, entropy: float) -> np.ndarray:
        """Create embedding that captures entropy information."""
        entropy_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Normalize entropy to [0, 1] range (assuming max entropy ~8 for bytes)
        normalized_entropy = min(entropy / 8.0, 1.0)

        # Distribute entropy information across embedding
        for i in range(min(8, self.embedding_dim)):
            entropy_embedding[i] = normalized_entropy * np.sin(
                i * np.pi * normalized_entropy
            )

        return entropy_embedding

    def _positional_embedding(self, position: int) -> np.ndarray:
        """Generate positional embedding for patch sequence."""
        if not self.use_positional_encoding or position >= self.max_patches:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        return self.positional_encoding[position].copy()

    def _structural_embedding(self, content: str) -> np.ndarray:
        """Analyze and embed structural characteristics of content."""
        if not content:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        structural_features = {
            "length": len(content),
            "unique_chars": len(set(content)),
            "repetition": len(content) - len(set(content)),
            "line_breaks": content.count("\n"),
            "spaces": content.count(" "),
            "complexity": len(set(content)) / max(len(content), 1),
        }

        # Normalize features
        max_length = 1000  # Reasonable maximum for normalization
        structural_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        for i, (feature, value) in enumerate(structural_features.items()):
            if i >= self.embedding_dim:
                break
            if feature == "length":
                structural_embedding[i] = min(value / max_length, 1.0)
            elif feature == "complexity":
                structural_embedding[i] = value
            else:
                structural_embedding[i] = min(
                    value / max(structural_features["length"], 1), 1.0
                )

        return structural_embedding

    def _adaptive_weights(self, patch: "Patch") -> dict[str, float]:
        """Calculate adaptive weights based on patch characteristics."""
        entropy_ratio = patch.entropy / 8.0  # Normalize entropy

        # High entropy content gets more emphasis on content and structure
        # Low entropy content gets more emphasis on position and pattern
        if entropy_ratio > self.entropy_threshold:
            return {"content": 0.4, "entropy": 0.3, "position": 0.1, "structure": 0.2}
        else:
            return {"content": 0.3, "entropy": 0.2, "position": 0.2, "structure": 0.3}

    def _apply_attention(self, patch_embeddings: list[np.ndarray]) -> np.ndarray:
        """Apply simplified multi-head attention to patch embeddings."""
        if len(patch_embeddings) <= 1:
            return (
                patch_embeddings[0]
                if patch_embeddings
                else np.zeros(self.embedding_dim, dtype=np.float32)
            )

        # Stack embeddings
        embeddings_matrix = np.stack(
            patch_embeddings, axis=0
        )  # [num_patches, embedding_dim]

        # Simplified attention computation
        attention_output = np.zeros_like(embeddings_matrix[0])

        for head in range(self.attention_heads):
            # Extract head-specific dimensions
            head_dim = self.embedding_dim // self.attention_heads
            start_dim = head * head_dim
            end_dim = start_dim + head_dim

            if end_dim > self.embedding_dim:
                end_dim = self.embedding_dim

            head_embeddings = embeddings_matrix[:, start_dim:end_dim]

            # Compute attention scores (simplified)
            scores = np.dot(head_embeddings, head_embeddings.T)
            attention_weights = self._softmax(scores)

            # Apply attention
            attended = np.dot(attention_weights.T, head_embeddings)
            weighted_output = np.mean(attended, axis=0)

            # Assign back to full embedding
            attention_output[start_dim:end_dim] = weighted_output

        return attention_output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _apply_contextual_transformations(
        self, embedding: np.ndarray, original_text: str
    ) -> np.ndarray:
        """Apply context-aware transformations to the embedding."""
        # Detect content type and apply appropriate transformations
        content_type = self._detect_content_type(original_text)

        if content_type in self.semantic_bases:
            # Apply semantic transformation
            transformation_matrix = self.semantic_bases[content_type]
            transformed = np.dot(
                transformation_matrix, embedding[: transformation_matrix.shape[1]]
            )

            # Blend with original
            blend_weight = 0.3
            embedding[: len(transformed)] = (1 - blend_weight) * embedding[
                : len(transformed)
            ] + blend_weight * transformed

        return embedding

    def _detect_content_type(self, text: str) -> str:
        """Detect the primary content type of the text."""
        import re

        # Simple heuristics for content type detection
        if (
            re.search(r"\d+", text)
            and len(re.findall(r"\d+", text)) / max(len(text.split()), 1) > 0.3
        ):
            return "numerical"
        elif re.search(r"[{}()\[\]<>]", text):
            return "structural"
        elif len(text.split()) > 5:
            return "textual"
        else:
            return "contextual"

    def _manage_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Manage the encoding cache with LRU eviction."""
        if len(self._encoding_cache) >= self._cache_max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._encoding_cache))
            del self._encoding_cache[oldest_key]

        self._encoding_cache[cache_key] = embedding

    def _update_stats(self, encoding_time: float) -> None:
        """Update performance statistics."""
        self._encoding_stats["total_encodings"] += 1

        # Update average encoding time (exponential moving average)
        alpha = 0.1
        self._encoding_stats["avg_encoding_time"] = (
            alpha * encoding_time
            + (1 - alpha) * self._encoding_stats["avg_encoding_time"]
        )

    def validate_input(self, text: Any) -> str:
        """Validate and normalize input text."""
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
        """Represents a text patch with metadata."""

        def __init__(self, content: str, start_pos: int, end_pos: int, entropy: float):
            self.content = content
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.entropy = entropy

    def create_patches(self, text: str) -> list[Patch]:
        """Create adaptive patches based on content entropy and structure."""
        validated_text = self.validate_input(text)
        if not validated_text:
            return []

        # Adaptive patching based on content characteristics
        patches = []
        text_bytes = validated_text.encode("utf-8", errors="replace")
        text_len = len(text_bytes)

        # Calculate global entropy to inform patching strategy
        global_entropy = self._shannon_entropy(text_bytes)

        # Adaptive patch size based on entropy
        adaptive_patch_size = self._calculate_adaptive_patch_size(global_entropy)

        current_pos = 0
        while current_pos < text_len and len(patches) < self.max_patches:
            # Determine patch end position
            end_pos = min(current_pos + adaptive_patch_size, text_len)

            # Extract patch bytes and convert to text
            chunk_bytes = text_bytes[current_pos:end_pos]
            chunk_entropy = self._shannon_entropy(chunk_bytes)
            chunk_text = chunk_bytes.decode("utf-8", errors="replace")

            # Create patch with metadata
            patch = self.Patch(chunk_text, current_pos, end_pos, chunk_entropy)
            patches.append(patch)

            # Update position with potential overlap for high-entropy regions
            if chunk_entropy > self.entropy_threshold:
                overlap = (
                    adaptive_patch_size // 4
                )  # 25% overlap for high-entropy content
                current_pos = end_pos - overlap
            else:
                current_pos = end_pos

        # Ensure at least one patch for non-empty text
        if not patches and validated_text:
            chunk_text = validated_text[: min(len(validated_text), self.max_patch_size)]
            entropy = self._shannon_entropy(
                chunk_text.encode("utf-8", errors="replace")
            )
            patches.append(self.Patch(chunk_text, 0, len(chunk_text), entropy))

        return patches

    def _calculate_adaptive_patch_size(self, global_entropy: float) -> int:
        """Calculate adaptive patch size based on content entropy."""
        # High entropy content uses smaller patches for better granularity
        # Low entropy content uses larger patches for efficiency
        entropy_ratio = global_entropy / 8.0  # Normalize to [0,1]

        if entropy_ratio > 0.7:  # High entropy
            return max(self.min_patch_size, self.patch_size // 2)
        elif entropy_ratio < 0.3:  # Low entropy
            return min(self.max_patch_size, self.patch_size * 2)
        else:  # Medium entropy
            return self.patch_size

    def _shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte data."""
        if not data:
            return 0.0

        from collections import Counter

        counts = Counter(data)
        probs = [count / len(data) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate semantic similarity between embeddings."""
        if (
            embedding1 is None
            or embedding2 is None
            or embedding1.size == 0
            or embedding2.size == 0
        ):
            return 0.0

        # Ensure compatible dimensions
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        min_dim = min(emb1_flat.size, emb2_flat.size)

        if min_dim == 0:
            return 0.0

        emb1_compat = emb1_flat[:min_dim]
        emb2_compat = emb2_flat[:min_dim]

        # Calculate cosine similarity
        norm1 = np.linalg.norm(emb1_compat)
        norm2 = np.linalg.norm(emb2_compat)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(emb1_compat, emb2_compat) / (norm1 * norm2)

        # Normalize to [0, 1] range
        return float(max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0)))

    def encode_batch(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Encode multiple texts efficiently."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = []
        for text in texts:
            embedding = self.encode(text, use_cache=use_cache)
            embeddings.append(embedding)

        return np.stack(embeddings, axis=0)

    def get_encoding_stats(self) -> dict[str, Any]:
        """Get performance and usage statistics."""
        return self._encoding_stats.copy()

    def clear_cache(self) -> None:
        """Clear the encoding cache."""
        self._encoding_cache.clear()
        self._encoding_stats["cache_hits"] = 0

    def validate_color(self, color_value: Any) -> int:
        """Validate a color value against the ARC palette (0-9) if ARC mode is enabled."""
        if not self.arc_mode:
            return 0

        try:
            color_int = int(color_value)
            return max(0, min(9, color_int))  # Clamp to ARC palette range
        except (ValueError, TypeError):
            return 0  # Default to background color
