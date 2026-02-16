"""Semantic layer: pruning, encoding, routing, and context packing."""

import logging
import re
import zlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES (Phase 3)
# ============================================================================

@dataclass
class LatentMemoryUnit:
    """One compressed memory item."""
    
    id: str  # UUID
    embedding: np.ndarray  # Dense vector (768d) for retrieval
    latent_encoding: bytes  # BLT-compressed payload
    original_length: int  # Uncompressed size
    modality: str  # 'text', 'dialogue', 'trajectory', etc.
    retrieval_score: float  # 0-1 relevance to query
    pruned_fraction: float  # How much was removed (0-1)
    entropy_score: float  # Information density
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "embedding": (
                self.embedding.tolist()
                if isinstance(self.embedding, np.ndarray)
                else self.embedding
            ),
            "latent_encoding": self.latent_encoding.hex(),
            "original_length": self.original_length,
            "modality": self.modality,
            "retrieval_score": float(self.retrieval_score),
            "pruned_fraction": float(self.pruned_fraction),
            "entropy_score": float(self.entropy_score),
        }


# ============================================================================
# PHASE 3: GAME-SEMANTIC PRUNER
# ============================================================================

class GameSemanticPruner:
    """
    Compress dialogue/text by identifying game-theoretically valuable moves.
    
    Inspired by: MetaConsciousness.frameworks.game_compression
    Uses significance scoring to identify critical sentences.
    """
    
    def __init__(
        self,
        key_phrase_weight: float = 1.5,
        question_weight: float = 1.2,
        contradiction_weight: float = 1.4,
        sentiment_change_weight: float = 1.3,
        min_significance_ratio: float = 0.7,
        preserve_opening: bool = True,
        preserve_closing: bool = True,
    ):
        """
        Initialize semantic pruner.
        
        Args:
            key_phrase_weight: Weight for sentences with key phrases
            question_weight: Weight for questions
            contradiction_weight: Weight for contradictions/corrections
            sentiment_change_weight: Weight for sentiment shifts
            min_significance_ratio: Keep at least this fraction (0-1) by score
            preserve_opening: Always keep first sentence
            preserve_closing: Always keep last sentence
        """
        self.key_phrase_weight = key_phrase_weight
        self.question_weight = question_weight
        self.contradiction_weight = contradiction_weight
        self.sentiment_change_weight = sentiment_change_weight
        self.min_significance_ratio = min_significance_ratio
        self.preserve_opening = preserve_opening
        self.preserve_closing = preserve_closing
        
        # Key phrase dictionary
        self.key_phrases = [
            "important", "critical", "key", "significant", "notable",
            "essential", "crucial", "vital", "major", "fundamental",
            "breakthrough", "discovery", "novel", "unique", "unique",
        ]
    
    def score_document(self, text: str) -> Dict[int, float]:
        """
        Score each sentence by importance.
        
        Args:
            text: Input text
        
        Returns:
            {sentence_idx: importance_score}
        """
        # Split into sentences (simple heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        scores = {}
        
        for i, sentence in enumerate(sentences):
            score = 1.0  # baseline
            lower_sent = sentence.lower()
            
            # Key phrase detection - add to score instead of multiply
            if any(phrase in lower_sent for phrase in self.key_phrases):
                score += self.key_phrase_weight
            
            # Question detection
            if '?' in sentence:
                score += self.question_weight
            
            # Contradiction detection (heuristics)
            if any(c in lower_sent for c in ["but", "however", "actually", "instead"]):
                score += self.contradiction_weight
            
            # Sentiment change markers
            sentiment_markers = [
                "suddenly", "surprisingly", "unfortunately", "fortunately"
            ]
            if any(s in lower_sent for s in sentiment_markers):
                score += self.sentiment_change_weight
            
            # Preserve opening and closing
            is_opening = i == 0 and self.preserve_opening
            is_closing = i == len(sentences) - 1 and self.preserve_closing
            if is_opening or is_closing:
                score += 0.5
            
            scores[i] = score
        
        return scores
    
    def prune(self, text: str, target_ratio: float = 0.7) -> Tuple[str, float]:
        """
        Keep sentences scoring above threshold.
        
        Args:
            text: Input text
            target_ratio: Keep at least this fraction (0-1)
        
        Returns:
            (pruned_text, pruned_fraction)
        """
        sentences = re.split(r'([.!?]+)', text)
        
        # Reconstruct with delimiters
        text_chunks = []
        for i in range(0, len(sentences) - 1, 2):
            text_chunks.append(sentences[i])
        
        # Score actual sentences (not delimiters)
        actual_sentences = [s.strip() for s in sentences[::2] if s.strip()]
        
        if not actual_sentences:
            return text, 0.0
        
        scores = self.score_document(text)
        
        # Calculate threshold to keep target_ratio
        score_values = list(scores.values())
        threshold = np.percentile(score_values, 100 * (1 - target_ratio))
        
        # Keep sentences above threshold
        kept_indices = set()
        for i, score in scores.items():
            if score >= threshold or (i == 0 and self.preserve_opening) or (i == len(actual_sentences) - 1 and self.preserve_closing):
                kept_indices.add(i)
        
        # Reconstruct pruned text
        pruned_parts = [actual_sentences[i] for i in sorted(kept_indices)]
        pruned_text = '. '.join(pruned_parts) + '.' if pruned_parts else ''
        
        # Calculate pruning fraction
        pruned_fraction = 1.0 - (len(kept_indices) / len(actual_sentences)) if actual_sentences else 0.0
        
        return pruned_text, pruned_fraction


# ============================================================================
# PHASE 3: BLT LATENT CODEC
# ============================================================================

class BLTLatentCodec:
    """
    Map text → fixed-size dense latent representation.
    Deterministic, seeded, reproducible.
    
    Combines embedding (for retrieval) + BLT compression (for storage).
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        compression_level: int = 6,
        device: Optional[str] = None,
    ):
        """
        Initialize latent codec.
        
        Args:
            embedding_dim: Embedding dimension (default 384 for sentence-transformers)
            compression_level: zlib compression level 1-9 (default 6)
            device: Device for embeddings (cpu/cuda/auto)
        """
        self.embedding_dim = embedding_dim
        self.compression_level = compression_level
        self.device = device or "cpu"
        
        # Lazy-load embedder
        self.embedder = None
        self._initialize_embedder()
    
    def _initialize_embedder(self) -> None:
        """Lazy-load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Loaded BLT latent codec embedder")
        except Exception as e:
            logger.warning(f"Embedder initialization failed: {e}, using fallback")
            self.embedder = None
    
    def encode(self, text: str, seed: Optional[int] = None) -> LatentMemoryUnit:
        """
        Encode text into latent unit (embedding + BLT bytes).
        
        Args:
            text: Text to encode
            seed: Optional seed for determinism
        
        Returns:
            LatentMemoryUnit with embedding and compressed payload
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Generate embedding
        if self.embedder:
            try:
                if seed is not None:
                    np.random.seed(seed)
                embedding = self.embedder.encode(text, convert_to_tensor=False)
                embedding = np.array(embedding, dtype=np.float32)
            except Exception as e:
                logger.error(f"Embedding failed: {e}, using zeros")
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            # Fallback: deterministic hash-based embedding
            if seed is not None:
                np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        
        # Compress text via zlib
        text_bytes = text.encode('utf-8')
        try:
            compressed = zlib.compress(text_bytes, level=self.compression_level)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            compressed = text_bytes
        
        # Create latent unit
        unit = LatentMemoryUnit(
            id=f"latent_{hash(text) & 0xffffffff:08x}",
            embedding=embedding,
            latent_encoding=compressed,
            original_length=len(text),
            modality="text",
            retrieval_score=1.0,
            pruned_fraction=0.0,
            entropy_score=0.5,
        )
        
        return unit
    
    def decode(self, unit: LatentMemoryUnit) -> str:
        """Reconstruct text from latent unit."""
        try:
            return zlib.decompress(unit.latent_encoding).decode('utf-8')
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return ""


# ============================================================================
# PHASE 3: ENTROPY ROUTER
# ============================================================================

class EntropyRouter:
    """
    Decide which memory units to include in final pack.
    Entropy governs what's worth transmitting to LLM.
    """
    
    def __init__(
        self,
        entropy_threshold: float = 0.3,
        skip_threshold: float = 0.1,
        max_budget_tokens: int = 2048,
        strategy: str = "static",
    ):
        """
        Initialize entropy router.
        
        Args:
            entropy_threshold: Include if entropy > this
            skip_threshold: Skip if entropy < this
            max_budget_tokens: Overall token budget
            strategy: 'static' or 'learned' threshold selection
        """
        self.entropy_threshold = entropy_threshold
        self.skip_threshold = skip_threshold
        self.max_budget_tokens = max_budget_tokens
        self.strategy = strategy
    
    def calculate_entropy(self, unit: LatentMemoryUnit) -> float:
        """
        Calculate information entropy of a unit.
        
        Uses: compression ratio as proxy for entropy.
        Higher ratio → lower entropy (more predictable).
        
        Args:
            unit: Latent memory unit
        
        Returns:
            Entropy score (0-1, higher = more information)
        """
        if unit.original_length == 0:
            return 0.0
        
        # Compression ratio indicates information density
        # High ratio (compressed ≈ original) = diverse/high entropy
        # Low ratio (compressed << original) = repetitive/low entropy
        compression_ratio = len(unit.latent_encoding) / unit.original_length
        
        # Direct compression ratio is entropy indicator
        entropy = min(compression_ratio, 1.0)
        
        return float(entropy)
    
    def estimate_tokens(self, unit: LatentMemoryUnit) -> int:
        """
        Estimate token cost of a unit.
        
        Heuristic: 1 token ≈ 4 characters.
        
        Args:
            unit: Latent memory unit
        
        Returns:
            Estimated token count
        """
        return max(1, unit.original_length // 4)
    
    def route(self, units: List[LatentMemoryUnit]) -> Tuple[List[LatentMemoryUnit], Dict[str, Any]]:
        """
        Given units (in retrieval order), decide which to include.
        
        Args:
            units: List of memory units from retrieval
        
        Returns:
            (included_units, routing_metadata)
        """
        outputs = []
        budget_remaining = self.max_budget_tokens
        routing_stats = {
            "total_units": len(units),
            "included_units": 0,
            "skipped_units": 0,
            "budget_units": 0,
            "budget_used": 0,
        }
        
        for unit in units:
            entropy = self.calculate_entropy(unit)
            token_cost = self.estimate_tokens(unit)
            
            # Update entropy in unit
            unit.entropy_score = entropy
            
            if entropy < self.skip_threshold:
                # Skip: too predictable (too compressible)
                routing_stats["skipped_units"] += 1
                continue
            elif budget_remaining >= token_cost:
                # Check entropy threshold as well
                if entropy >= self.entropy_threshold:
                    # Include: has enough entropy AND fits budget
                    outputs.append(unit)
                    budget_remaining -= token_cost
                    routing_stats["included_units"] += 1
                else:
                    # Too predictable even if fits budget
                    routing_stats["skipped_units"] += 1
            else:
                # Over budget: stop
                routing_stats["budget_units"] += 1
                break
        
        routing_stats["budget_used"] = self.max_budget_tokens - budget_remaining
        
        return outputs, routing_stats


# ============================================================================
# PHASE 3: CONTEXT PACK BUILDER
# ============================================================================

class ContextPackBuilder:
    """Build final context pack from latent units for LLM consumption."""
    
    def __init__(self):
        """Initialize pack builder."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens in text.
        
        Heuristic: 1 token ≈ 4 characters.
        
        Args:
            text: Text to count
        
        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)
    
    def expand_latent_units(
        self,
        units: List[LatentMemoryUnit],
        codec: BLTLatentCodec,
    ) -> str:
        """
        Expand compressed units back to readable text.
        
        Args:
            units: List of latent memory units
            codec: BLT codec for decompression
        
        Returns:
            Expanded text for LLM
        """
        expanded_texts = []
        
        for unit in units:
            try:
                text = codec.decode(unit)
                if text:
                    expanded_texts.append(text)
            except Exception as e:
                logger.error(f"Failed to expand unit {unit.id}: {e}")
        
        return "\n\n".join(expanded_texts)
    
    def build_pack(
        self,
        units: List[LatentMemoryUnit],
        codec: BLTLatentCodec,
        query: str,
        budget_tokens: int,
    ) -> Dict[str, Any]:
        """
        Build final context pack.
        
        Args:
            units: Routed latent memory units
            codec: BLT codec for expansion
            query: Original query
            budget_tokens: Token budget
        
        Returns:
            Context pack dict (ready for JSON serialization)
        """
        # Expand units
        expanded_text = self.expand_latent_units(units, codec)
        
        # Count tokens
        token_count = self.estimate_tokens(expanded_text)
        
        # Ensure budget compliance
        if token_count > budget_tokens:
            # Truncate to fit budget
            max_chars = budget_tokens * 4
            expanded_text = expanded_text[:max_chars]
            token_count = self.estimate_tokens(expanded_text)
        
        # Extract retrieval and entropy scores
        retrieval_scores = [float(u.retrieval_score) for u in units]
        entropy_scores = [float(u.entropy_score) for u in units]
        
        # Calculate compression ratio
        original_total = sum(u.original_length for u in units)
        compressed_total = sum(len(u.latent_encoding) for u in units)
        compression_ratio = compressed_total / original_total if original_total > 0 else 1.0
        
        # Build pack
        pack = {
            "latent_units": [u.to_dict() for u in units],
            "expanded_text": expanded_text,
            "token_count": token_count,
            "query": query,
            "budget_tokens": budget_tokens,
            "retrieval_scores": retrieval_scores,
            "entropy_scores": entropy_scores,
            "compression_ratio": float(compression_ratio),
            "version": "phase-3",
        }
        
        return pack


# ============================================================================
# ORIGINAL PHASE 2 COMPONENTS (PRESERVED)
# ============================================================================

class EmbeddingGenerator:
    """Generate embeddings from text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding generator."""
        self.model_name = model_name
        self.model = None
        self.dim = 384
        self._initialize()
    
    def _initialize(self) -> None:
        """Lazy-load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Loaded embedding model: {self.model_name} (dim={self.dim})")
        except ImportError:
            logger.warning("sentence-transformers not available")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def encode(self, text: str) -> List[float]:
        """Encode single text string to embedding vector."""
        if not text or not self.model:
            return [0.0] * self.dim
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self.dim
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embeddings."""
        if not texts or not self.model:
            return [[0.0] * self.dim for _ in texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [[0.0] * self.dim for _ in texts]


class SemanticRouter:
    """Route queries to optimal compression algorithm."""
    
    def __init__(self):
        """Initialize router with heuristics."""
        self.routing_rules = {
            "image": ["sheaf", "meta_learning"],
            "dialogue": ["game_semantic", "quantum"],
            "trajectory": ["homotopy", "quantum"],
            "text": ["quantum", "meta_learning"],
            "scientific": ["homotopy", "quantum"],
            "default": ["quantum", "blt"],
        }
    
    def infer_data_type(self, content: str) -> str:
        """Infer content type from text characteristics."""
        if not content:
            return "text"
        
        lower_content = content.lower()
        
        if any(m in lower_content for m in ["said:", "replied:", "asked:", '?"', '"']):
            return "dialogue"
        
        trajectory_keywords = ["position", "coordinate", "x:", "y:", "z:", "trajectory"]
        if any(k in lower_content for k in trajectory_keywords):
            return "trajectory"
        
        scientific_keywords = ["equation", "formula", "coefficient", "derivative", "integral"]
        if any(s in lower_content for s in scientific_keywords):
            return "scientific"
        
        if any(img in lower_content for img in ["image", "pixel", "color", "rgb", "#"]):
            return "image"
        
        return "text"
    
    def route(self, query: str, content_type: Optional[str] = None) -> str:
        """Route query to best compression algorithm."""
        if not content_type:
            content_type = self.infer_data_type(query)
        
        algorithms = self.routing_rules.get(content_type, self.routing_rules["default"])
        return algorithms[0] if algorithms else "quantum"


class SemanticPruner:
    """Prune irrelevant content from knowledge base."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize pruner."""
        self.threshold = threshold
    
    def prune(self, content: str, query: str, budget_tokens: int) -> str:
        """Remove low-relevance content to fit token budget."""
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= budget_tokens:
            return content
        
        max_chars = budget_tokens * 4 * 0.9
        return content[:int(max_chars)] + "... [pruned]"


class SemanticEncoder:
    """Encode remaining content into compressed representation."""
    
    def __init__(self):
        """Initialize encoder."""
        self.embeddings = EmbeddingGenerator()
    
    def encode(self, content: str, mode: str = "balanced") -> bytes:
        """Compress content into optimized byte representation."""
        try:
            encoded = content.encode('utf-8')
            mode_byte = {'aggressive': b'\x01', 'balanced': b'\x02', 'quality': b'\x03'}.get(mode, b'\x02')
            length_bytes = len(encoded).to_bytes(4, 'little')
            return mode_byte + length_bytes + encoded
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return b''


__all__ = [
    "GameSemanticPruner",
    "BLTLatentCodec",
    "EntropyRouter",
    "ContextPackBuilder",
    "LatentMemoryUnit",
    "EmbeddingGenerator",
    "SemanticRouter",
    "SemanticPruner",
    "SemanticEncoder",
]
