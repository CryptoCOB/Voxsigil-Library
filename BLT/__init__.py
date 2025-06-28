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
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device="cuda" if self.use_gpu else "cpu",
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded embedding model '{self.model_name}' with dimension {self.embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            logger.warning(
                "Using fallback random embedding generator; embeddings may be inconsistent"
            )

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
            if getattr(self, "model", None) is not None:
                embedding = self.model.encode(text_content).tolist()
            else:
                # Fallback: deterministic embedding based on text hash
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

    # === Abstract Methods Implementation ===
    # Required by BaseAgentInterface

    async def execute_task(
        self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a specific task using BLT encoder capabilities."""
        task_type = task.get("type", "encode")

        if task_type == "encode":
            text = task.get("text", "")
            embedding = self.encode(text)
            return {
                "status": "success",
                "result": {
                    "embedding": embedding,
                    "dimension": len(embedding),
                    "text_length": len(text),
                },
            }
        elif task_type == "encode_batch":
            texts = task.get("texts", [])
            embeddings = self.encode_batch(texts)
            return {
                "status": "success",
                "result": {
                    "embeddings": embeddings.tolist()
                    if hasattr(embeddings, "tolist")
                    else embeddings,
                    "count": len(texts),
                    "dimension": self.embedding_dim,
                },
            }
        elif task_type == "similarity":
            text1 = task.get("text1", "")
            text2 = task.get("text2", "")
            similarity = self.compute_similarity(text1, text2)
            return {
                "status": "success",
                "result": {
                    "similarity": similarity,
                    "text1_length": len(text1),
                    "text2_length": len(text2),
                },
            }
        else:
            return {
                "status": "error",
                "error": f"Unknown task type: {task_type}",
                "supported_types": ["encode", "encode_batch", "similarity"],
            }

    async def plan_execution(
        self, goal: str, constraints: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Plan execution steps for BLT encoder goals."""
        constraints = constraints or []

        if "encode" in goal.lower():
            return [
                {
                    "step": 1,
                    "action": "prepare_text",
                    "description": "Clean and preprocess input text",
                },
                {
                    "step": 2,
                    "action": "check_cache",
                    "description": "Check if embedding is already cached",
                },
                {
                    "step": 3,
                    "action": "encode_text",
                    "description": "Generate embedding using the model",
                },
                {
                    "step": 4,
                    "action": "cache_result",
                    "description": "Store result in cache if enabled",
                },
            ]
        elif "similarity" in goal.lower():
            return [
                {
                    "step": 1,
                    "action": "encode_texts",
                    "description": "Encode both input texts",
                },
                {
                    "step": 2,
                    "action": "compute_similarity",
                    "description": "Calculate cosine similarity between embeddings",
                },
                {
                    "step": 3,
                    "action": "normalize_result",
                    "description": "Normalize similarity to [0, 1] range",
                },
            ]
        else:
            return [
                {
                    "step": 1,
                    "action": "analyze_goal",
                    "description": f"Analyze goal: {goal}",
                },
                {
                    "step": 2,
                    "action": "determine_approach",
                    "description": "Determine appropriate BLT encoder approach",
                },
            ]

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available BLT encoder tools."""
        return [
            {
                "name": "encode",
                "description": "Encode text into vector embedding",
                "parameters": ["text", "task_type"],
                "returns": "List[float]",
            },
            {
                "name": "encode_batch",
                "description": "Encode multiple texts efficiently",
                "parameters": ["texts", "use_cache"],
                "returns": "np.ndarray",
            },
            {
                "name": "compute_similarity",
                "description": "Compute similarity between two texts",
                "parameters": ["text1", "text2"],
                "returns": "float",
            },
            {
                "name": "create_patches",
                "description": "Create text patches from content",
                "parameters": ["content", "patch_size", "overlap"],
                "returns": "List[str]",
            },
            {
                "name": "get_encoder_details",
                "description": "Get encoder configuration details",
                "parameters": [],
                "returns": "Dict[str, Any]",
            },
            {
                "name": "clear_cache",
                "description": "Clear the encoding cache",
                "parameters": [],
                "returns": "None",
            },
        ]

    async def save_state(self) -> Dict[str, Any]:
        """Save current BLT encoder state."""
        return {
            "config": self.config,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.embedding_cache)
            if hasattr(self, "embedding_cache")
            else 0,
            "encoding_stats": getattr(self, "_encoding_stats", {}),
            "timestamp": time.time(),
        }

    async def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore BLT encoder from saved state."""
        try:
            self.config = state.get("config", {})
            self.model_name = state.get("model_name", "all-MiniLM-L12-v2")
            self.embedding_dim = state.get("embedding_dim", 384)
            self.cache_enabled = state.get("cache_enabled", True)

            # Restore encoding stats if available
            if "encoding_stats" in state:
                self._encoding_stats = state["encoding_stats"]

            # Re-initialize the model with restored config
            self._initialize_model()

            logger.info(
                f"BLT encoder state restored from timestamp {state.get('timestamp', 'unknown')}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to restore BLT encoder state: {e}")
            return False

    # Required by BLTInterface

    async def build_component(
        self,
        specification: Dict[str, Any],
        build_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a component from specification."""
        build_config = build_config or {}

        component_type = specification.get("type", "encoder")

        if component_type == "encoder":
            # Build a new encoder instance with specified config
            new_config = specification.get("config", {})
            new_config.update(build_config)

            try:
                new_encoder = BLTEncoder(new_config)
                component_id = f"blt_encoder_{int(time.time())}"

                return {
                    "status": "success",
                    "component_id": component_id,
                    "component_type": "BLTEncoder",
                    "config": new_config,
                    "capabilities": [
                        "text_encoding",
                        "batch_encoding",
                        "similarity_computation",
                        "patch_creation",
                    ],
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Failed to build encoder: {e}",
                    "specification": specification,
                }
        else:
            return {
                "status": "error",
                "error": f"Unknown component type: {component_type}",
                "supported_types": ["encoder"],
            }

    async def learn_from_feedback(
        self, component_id: str, feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from component performance feedback."""
        feedback_type = feedback.get("type", "performance")

        if feedback_type == "performance":
            # Adjust cache settings based on performance feedback
            hit_rate = feedback.get("cache_hit_rate", 0)
            if hit_rate < 0.3 and self.cache_max_size < 10000:
                # Increase cache size if hit rate is low
                self.cache_max_size = min(10000, int(self.cache_max_size * 1.5))
                logger.info(
                    f"Increased cache size to {self.cache_max_size} based on feedback"
                )

            return {
                "status": "success",
                "adjustments": {"cache_max_size": self.cache_max_size},
                "feedback_processed": feedback_type,
            }

        elif feedback_type == "accuracy":
            # Log accuracy feedback for potential model adjustments
            accuracy = feedback.get("accuracy", 0)
            logger.info(f"Received accuracy feedback: {accuracy}")

            return {
                "status": "success",
                "feedback_processed": feedback_type,
                "accuracy_logged": accuracy,
            }

        else:
            return {
                "status": "warning",
                "message": f"Unknown feedback type: {feedback_type}",
                "supported_types": ["performance", "accuracy"],
            }

    async def test_component(
        self, component_id: str, test_suite: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test component with provided test suite."""
        results = []
        passed = 0
        failed = 0

        for i, test in enumerate(test_suite):
            test_type = test.get("type", "encode")

            try:
                if test_type == "encode":
                    text = test.get("input", "")
                    expected_dim = test.get("expected_dimension", self.embedding_dim)

                    embedding = self.encode(text)
                    actual_dim = len(embedding)

                    test_passed = actual_dim == expected_dim

                    results.append(
                        {
                            "test_id": i,
                            "type": test_type,
                            "passed": test_passed,
                            "expected_dimension": expected_dim,
                            "actual_dimension": actual_dim,
                            "input_length": len(text),
                        }
                    )

                elif test_type == "similarity":
                    text1 = test.get("text1", "")
                    text2 = test.get("text2", "")
                    expected_range = test.get("expected_range", [0.0, 1.0])

                    similarity = self.compute_similarity(text1, text2)

                    test_passed = expected_range[0] <= similarity <= expected_range[1]

                    results.append(
                        {
                            "test_id": i,
                            "type": test_type,
                            "passed": test_passed,
                            "similarity": similarity,
                            "expected_range": expected_range,
                        }
                    )

                else:
                    results.append(
                        {
                            "test_id": i,
                            "type": test_type,
                            "passed": False,
                            "error": f"Unknown test type: {test_type}",
                        }
                    )
                    failed += 1
                    continue

                if results[-1]["passed"]:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                results.append(
                    {"test_id": i, "type": test_type, "passed": False, "error": str(e)}
                )
                failed += 1

        return {
            "status": "success",
            "component_id": component_id,
            "test_summary": {
                "total": len(test_suite),
                "passed": passed,
                "failed": failed,
                "success_rate": passed / len(test_suite) if test_suite else 0,
            },
            "detailed_results": results,
        }

    async def get_build_metrics(self, component_id: str) -> Dict[str, Any]:
        """Get build and performance metrics."""
        return {
            "component_id": component_id,
            "component_type": "BLTEncoder",
            "model_info": {
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dim,
                "model_loaded": hasattr(self, "model") and self.model is not None,
            },
            "performance_metrics": {
                "cache_enabled": self.cache_enabled,
                "cache_size": len(self.embedding_cache)
                if hasattr(self, "embedding_cache")
                else 0,
                "cache_max_size": self.cache_max_size,
                "encoding_stats": getattr(self, "_encoding_stats", {}),
            },
            "configuration": {
                "use_gpu": self.use_gpu,
                "min_patch_size": self.min_patch_size,
                "max_patch_size": self.max_patch_size,
                "entropy_threshold": self.entropy_threshold,
            },
            "timestamp": time.time(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return health or configuration status (sync version for compatibility)."""
        return {
            "status": "healthy" if hasattr(self, "model") else "degraded",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.embedding_cache)
            if hasattr(self, "embedding_cache")
            else 0,
            "model_available": hasattr(self, "model") and self.model is not None,
            "timestamp": time.time(),
        }

    def process_with_context(self, text: str, bound_context: Dict[str, Any]) -> str:
        """
        Process text using BLT encoder with bound sigil context.

        Args:
            text: The input text to process
            bound_context: Bound context from sigil processing

        Returns:
            Enhanced text with BLT context applied
        """
        try:
            # Extract unified sigil and component information
            unified_sigil = bound_context.get("unified_sigil", "")
            components = bound_context.get("components", [])
            component_count = bound_context.get("component_count", 0)

            # Create enhanced prompt with BLT context
            if unified_sigil and components:
                # Add BLT sigil markers to the text
                sigil_prefix = f"[BLT:{unified_sigil}]"
                component_tags = "[" + "|".join(components) + "]"
                enhanced_text = f"{sigil_prefix} {component_tags} {text}"

                logger.debug(
                    f"BLT enhanced text with {component_count} components: {unified_sigil}"
                )
                return enhanced_text
            else:
                # Return original text if no valid context
                logger.debug("No valid BLT context found, returning original text")
                return text

        except Exception as e:
            logger.warning(f"BLT context processing failed: {e}")
            # Return original text on error
            return text


class ByteLatentTransformerEncoder:
    """
    Advanced BLT encoder that provides byte-level latent transformations.
    This is a specialized encoder for handling low-level data representations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the ByteLatentTransformerEncoder."""
        self.config = config or {}
        self.config.update(kwargs)

        # Default configuration for byte-level operations
        self.byte_embedding_dim = self.config.get("byte_embedding_dim", 256)
        self.latent_compression_ratio = self.config.get(
            "latent_compression_ratio", 0.25
        )
        self.transformer_layers = self.config.get("transformer_layers", 4)
        self.attention_heads = self.config.get("attention_heads", 8)

        self._initialized = True
        logger.info(
            f"ByteLatentTransformerEncoder initialized with {self.byte_embedding_dim}D embeddings"
        )

    def encode_bytes(self, data: bytes) -> List[float]:
        """Encode raw bytes into latent representations."""
        # Convert bytes to embedding representation
        byte_values = list(data[: min(len(data), self.byte_embedding_dim)])

        # Pad or truncate to expected dimension
        if len(byte_values) < self.byte_embedding_dim:
            byte_values.extend([0] * (self.byte_embedding_dim - len(byte_values)))
        else:
            byte_values = byte_values[: self.byte_embedding_dim]

        # Normalize to [-1, 1] range
        normalized = [(b / 127.5) - 1.0 for b in byte_values]
        return normalized

    def encode_text_to_bytes(self, text: str) -> List[float]:
        """Encode text by first converting to bytes, then to latent space."""
        text_bytes = text.encode("utf-8")
        return self.encode_bytes(text_bytes)

    def get_latent_dimension(self) -> int:
        """Get the dimension of latent encodings."""
        return int(self.byte_embedding_dim * self.latent_compression_ratio)

    def compress_to_latent(self, encoding: List[float]) -> List[float]:
        """Compress full encoding to latent space."""
        latent_dim = self.get_latent_dimension()
        # Simple compression by taking every nth element
        stride = len(encoding) // latent_dim
        if stride <= 1:
            return encoding[:latent_dim]
        return [encoding[i * stride] for i in range(latent_dim)]


class SigilPatchEncoder:
    """
    Specialized encoder for creating and managing sigil patches from text/data.
    Sigils are compact symbolic representations of larger content blocks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the SigilPatchEncoder."""
        self.config = config or {}
        self.config.update(kwargs)

        # Configuration for sigil operations
        self.patch_size = self.config.get("patch_size", 64)
        self.sigil_length = self.config.get("sigil_length", 16)
        self.overlap_ratio = self.config.get("overlap_ratio", 0.1)
        self.hash_algorithm = self.config.get("hash_algorithm", "sha256")

        self._patch_cache = {}
        self._sigil_cache = {}

        logger.info(
            f"SigilPatchEncoder initialized with {self.patch_size}-byte patches"
        )

    def create_patches(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping patches from text with sigil markers."""
        text_bytes = text.encode("utf-8")
        patches = []

        overlap_size = int(self.patch_size * self.overlap_ratio)
        step_size = self.patch_size - overlap_size

        for i in range(0, len(text_bytes), step_size):
            patch_bytes = text_bytes[i : i + self.patch_size]
            if len(patch_bytes) == 0:
                break

            # Create sigil for this patch
            sigil = self._create_sigil(patch_bytes)

            patch_info = {
                "index": len(patches),
                "start_byte": i,
                "end_byte": i + len(patch_bytes),
                "size": len(patch_bytes),
                "sigil": sigil,
                "content": patch_bytes.decode("utf-8", errors="replace"),
                "hash": hashlib.sha256(patch_bytes).hexdigest()[:16],
            }
            patches.append(patch_info)

        return patches

    def _create_sigil(self, data: bytes) -> str:
        """Create a compact sigil representation of data."""
        # Use hash to create deterministic sigil
        hasher = getattr(hashlib, self.hash_algorithm)()
        hasher.update(data)
        hash_hex = hasher.hexdigest()

        # Convert to compact sigil format
        sigil_chars = []
        for i in range(0, min(len(hash_hex), self.sigil_length * 2), 2):
            byte_val = int(hash_hex[i : i + 2], 16)
            # Map to printable ASCII range
            char_code = 33 + (byte_val % 94)  # Printable ASCII from ! to ~
            sigil_chars.append(chr(char_code))

        return "".join(sigil_chars[: self.sigil_length])

    def encode_with_sigils(self, text: str) -> Dict[str, Any]:
        """Encode text with sigil patch information."""
        patches = self.create_patches(text)

        # Create overall document sigil
        doc_sigil = self._create_sigil(text.encode("utf-8"))

        return {
            "document_sigil": doc_sigil,
            "patch_count": len(patches),
            "patches": patches,
            "total_size": len(text.encode("utf-8")),
            "encoding_metadata": {
                "patch_size": self.patch_size,
                "sigil_length": self.sigil_length,
                "overlap_ratio": self.overlap_ratio,
                "algorithm": self.hash_algorithm,
            },
        }

    def bind_sigils(self, sigils: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bind multiple sigils into a unified context for BLT processing.

        Args:
            sigils: Dictionary of sigils from different components

        Returns:
            Bound context dictionary ready for BLT processing
        """
        try:
            bound_context = {
                "unified_sigil": "",
                "component_count": len(sigils),
                "components": list(sigils.keys()),
                "binding_metadata": {
                    "binding_algorithm": "sha256_composite",
                    "temporal_decay": getattr(self, "temporal_decay", 0.95),
                    "binding_strength": getattr(self, "binding_strength", 0.8),
                },
            }

            # Create a unified sigil from all component sigils
            if sigils:
                # Combine all sigil information into a single string
                combined_sigil_data = ""
                for component, sigil_data in sigils.items():
                    if isinstance(sigil_data, dict):
                        # Extract pattern and context from sigil data
                        pattern = sigil_data.get("pattern", component)
                        context = sigil_data.get("context", "unknown")
                        combined_sigil_data += f"{component}:{pattern}:{context};"
                    else:
                        combined_sigil_data += f"{component}:{sigil_data};"

                # Create unified sigil
                bound_context["unified_sigil"] = self._create_sigil(
                    combined_sigil_data.encode("utf-8")
                )
                bound_context["raw_data"] = combined_sigil_data

                # Add individual component sigils
                bound_context["individual_sigils"] = {}
                for component, sigil_data in sigils.items():
                    if isinstance(sigil_data, dict):
                        component_text = f"{sigil_data.get('pattern', '')}_{sigil_data.get('context', '')}"
                    else:
                        component_text = str(sigil_data)

                    bound_context["individual_sigils"][component] = self._create_sigil(
                        component_text.encode("utf-8")
                    )

            return bound_context

        except Exception as e:
            logger.warning(f"Failed to bind sigils: {e}")
            # Return a minimal bound context on failure
            return {
                "unified_sigil": "ERR_BIND",
                "component_count": 0,
                "components": [],
                "error": str(e),
            }

    def get_sigil_for_text(self, text: str) -> str:
        """Get just the sigil for a text without full patch processing."""
        return self._create_sigil(text.encode("utf-8"))


# Export all classes
__all__ = ["BLTEncoder", "ByteLatentTransformerEncoder", "SigilPatchEncoder"]
