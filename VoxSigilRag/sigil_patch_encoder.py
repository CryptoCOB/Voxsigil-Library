import numpy as np
from typing import List, Tuple, Dict, Any

# Import the ByteLatentTransformerEncoder from the same package
from VoxSigilRag.voxsigil_blt import ByteLatentTransformerEncoder

class SigilPatchEncoder:
    """
    An encoder for creating and analyzing patches in VoxSigil sigils based on entropy.
    This wrapper class simplifies working with the ByteLatentTransformerEncoder.
    """
    
    def __init__(self, entropy_threshold: float = 0.5):
        """
        Initialize the sigil patch encoder.
        
        Args:
            entropy_threshold: Threshold for determining high vs low entropy
        """
        self.entropy_threshold = entropy_threshold
        self.blt_encoder = ByteLatentTransformerEncoder(
            entropy_threshold=entropy_threshold
        )
    
    def analyze_entropy(self, text: str) -> Tuple[List[bytes], List[float]]:
        """
        Analyze the entropy of a text input and split it into patches.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (patches, entropy_scores)
        """
        # Create patches using the BLT encoder
        patches = self.blt_encoder.create_patches(text)
        
        # Extract patches and entropy scores
        patch_bytes = [p[0] for p in patches]
        entropy_scores = [p[1] for p in patches]
        
        return patch_bytes, entropy_scores
    
    def get_patch_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about patches in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of patch statistics
        """
        patches, entropy_scores = self.analyze_entropy(text)
        
        # Calculate statistics
        avg_entropy = sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0
        max_entropy = max(entropy_scores) if entropy_scores else 0
        min_entropy = min(entropy_scores) if entropy_scores else 0
        
        # Count high and low entropy patches
        high_entropy_patches = sum(1 for e in entropy_scores if e >= self.entropy_threshold)
        low_entropy_patches = sum(1 for e in entropy_scores if e < self.entropy_threshold)
        
        return {
            "patch_count": len(patches),
            "avg_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "min_entropy": min_entropy,
            "high_entropy_patches": high_entropy_patches,
            "low_entropy_patches": low_entropy_patches,
            "high_entropy_ratio": high_entropy_patches / len(patches) if patches else 0
        }
    
    def encode_with_blt(self, text: str) -> np.ndarray:
        """
        Encode text using the BLT encoder.
        
        Args:
            text: Text to encode
            
        Returns:
            BLT encoding as a numpy array
        """
        return self.blt_encoder.encode(text)
