import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    from VoxSigilRag.voxsigil_blt import ByteLatentTransformerEncoder
except ImportError:
    ByteLatentTransformerEncoder = None

try:
    import logging
    logger = logging.getLogger("VoxSigilHybridMiddleware")
except ImportError:
    logger = None

class SigilPatchEncoder:
    """
    Combined and enhanced SigilPatchEncoder for VoxSigil.
    Provides both mock (heuristic) and BLT-based patching and entropy analysis.
    Responsible for text analysis, patch generation, and entropy calculation.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        use_blt: bool = True,
        blt_encoder_kwargs: Optional[dict] = None
    ):
        """
        Initialize the sigil patch encoder.

        Args:
            entropy_threshold: Threshold for determining high vs low entropy.
            use_blt: If True, use BLT encoder for patching/entropy; else use mock/heuristic.
            blt_encoder_kwargs: Optional dict for BLT encoder initialization.
        """
        self.entropy_threshold = entropy_threshold
        self.use_blt = use_blt and ByteLatentTransformerEncoder is not None
        self.blt_encoder = None
        if self.use_blt:
            kwargs = blt_encoder_kwargs or {}
            # Pass entropy_threshold if supported
            if "entropy_threshold" not in kwargs:
                kwargs["entropy_threshold"] = entropy_threshold
            self.blt_encoder = ByteLatentTransformerEncoder(**kwargs)
        if logger:
            logger.info(f"SigilPatchEncoder initialized (use_blt={self.use_blt}).")
        else:
            print(f"SigilPatchEncoder initialized (use_blt={self.use_blt}).")

    def analyze_entropy(
        self, text: str
    ) -> Tuple[Optional[List[Union[str, bytes]]], List[float]]:
        """
        Analyze text to produce patches and their entropy scores.

        Returns:
            patches: List of patch strings or bytes.
            entropy_scores: List of entropy scores (floats).
        """
        if not text:
            return None, []

        if self.use_blt and self.blt_encoder is not None:
            # Use BLT encoder's patching
            patches = self.blt_encoder.create_patches(text)
            # Each patch: (bytes, entropy)
            patch_bytes = [p[0] for p in patches]
            entropy_scores = [p[1] for p in patches]
            return patch_bytes, entropy_scores

        # --- Mock/heuristic fallback ---
        patches = []
        entropy_scores = []
        # Heuristic: check for common structured data markers
        if any(c in text for c in ['<', '>', '{', '}', '[', ']']) and len(text) < 500:
            avg_entropy = np.random.uniform(0.1, 0.3)
            num_segments = max(1, len(text) // 50)
            for i in range(num_segments):
                patches.append(text[i*50:(i+1)*50])
                entropy_scores.append(np.random.uniform(avg_entropy - 0.05, avg_entropy + 0.05))
        elif "explain" in text.lower() or "what are" in text.lower() or len(text) > 300:
            avg_entropy = np.random.uniform(0.6, 0.8)
            num_segments = max(1, len(text) // 100)
            for i in range(num_segments):
                patches.append(text[i*100:(i+1)*100])
                entropy_scores.append(np.random.uniform(avg_entropy - 0.1, avg_entropy + 0.1))
        else:
            avg_entropy = np.random.uniform(0.35, 0.55)
            entropy_scores = [avg_entropy]
            patches = [text]

        entropy_scores = [max(0.0, min(1.0, s)) for s in entropy_scores if s is not None]
        if not entropy_scores:
            entropy_scores = [0.5]
        if logger:
            logger.debug(f"Analyzed entropy for text (first 50 chars): '{text[:50]}...'. Avg score: {sum(entropy_scores)/len(entropy_scores):.4f}")
        return patches, entropy_scores

    def compute_average_entropy(self, text: str) -> float:
        """
        Computes the average entropy score for the given text.

        Returns:
            Average entropy score between 0 and 1
        """
        _, entropy_scores = self.analyze_entropy(text)
        if not entropy_scores:
            if logger:
                logger.warning(f"No entropy scores generated for text: '{text[:50]}...'")
            return 0.5
        avg_entropy = sum(entropy_scores) / len(entropy_scores)
        if logger:
            logger.debug(f"Average entropy for text: {avg_entropy:.4f}")
        return avg_entropy

    def get_patch_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about patches in the text.

        Returns:
            Dictionary of patch statistics
        """
        patches, entropy_scores = self.analyze_entropy(text)
        patch_count = len(patches) if patches else 0
        avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0
        max_entropy = max(entropy_scores) if entropy_scores else 0
        min_entropy = min(entropy_scores) if entropy_scores else 0
        high_entropy_patches = sum(1 for e in entropy_scores if e >= self.entropy_threshold)
        low_entropy_patches = sum(1 for e in entropy_scores if e < self.entropy_threshold)
        return {
            "patch_count": patch_count,
            "avg_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "high_entropy_patches": high_entropy_patches,
            "low_entropy_patches": low_entropy_patches,
            "high_entropy_ratio": high_entropy_patches / patch_count if patch_count else 0
        }

    def encode_with_blt(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text using the BLT encoder.

        Returns:
            BLT encoding as a numpy array, or None if BLT unavailable.
        """
        if self.use_blt and self.blt_encoder is not None:
            return self.blt_encoder.encode(text)
        if logger:
            logger.warning("BLT encoder not available; encode_with_blt returns None.")
        return None
