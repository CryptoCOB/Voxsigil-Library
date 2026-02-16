
"""
Test module for quantum_compression.py

This module tests the functionality of the quantum compression utilities.
"""

import os
import sys
import unittest
import tempfile # Kept import, though unused
import json
import numpy as np
from pathlib import Path # Use pathlib

# Add project root to sys.path
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except IndexError: # Handle case where script is run from unexpected location
    PROJECT_ROOT = Path(".").resolve() # Fallback
    if str(PROJECT_ROOT) not in sys.path:
         sys.path.insert(0, str(PROJECT_ROOT))


# Local import of the module to test
# Now import the class and the refactored wrappers
from MetaConsciousness.utils.compression.quantum_compression import (
    QuantumCompressor,
    compress, decompress, # Legacy wrappers
    compress_text_blocks, decompress_text_blocks, # Legacy block wrappers
    quantum_compress, quantum_decompress, # Legacy combined wrappers
    quantum_svd, compress_quantum, decompress_quantum, # Misplaced SVD funcs
    estimate_compression_ratio # Misplaced SVD func
)

# Ensure MetaConsciousness path is available for log_event import inside the tested module
# (This might be implicitly handled by the project root addition above)

class TestQuantumCompressor(unittest.TestCase):
    """Test cases for QuantumCompressor class and wrappers."""

    def setUp(self) -> None:
    """Set up test environment."""
        # Test config: low thresholds to encourage compression, sim off for predictability
        self.test_config = {
            "min_compression_length": 10,
            "compression_entropy_threshold": 0.05,
            "max_compression_level": 9,
            "use_quantum_simulation": False,
            "log_level": "ERROR" # Minimize logging during tests
        }
        # Instance for direct testing
        self.compressor = QuantumCompressor(config=self.test_config)
        # Instance configured to always force compression for block tests
        self.force_compressor = QuantumCompressor(config={**self.test_config, "min_compression_length": 1, "compression_entropy_threshold": 0.01}) # noqa

        # Sample texts
        self.sample_text = "This is compressible text with repetition. " * 10
        self.non_compressible_short = "Short"
        self.non_compressible_low_entropy = "abcdefghijklmnop" # High entropy but maybe short
        self.long_low_entropy = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" # noqa
        self.text_with_blocks_template = """
Regular text.
{marker}python
# Code Block 1
def func1():
    # Repetitive content * 15
    pass
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
    # Repetitive content * 15
{marker}
More regular text.
{marker}
# Block 2 - Also repetitive enough
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
var x = 1;
{marker}
Final text."""
        self.text_with_blocks = self.text_with_blocks_template.format(marker="```")

    def test_initialization(self) -> None:
    """Test compressor initialization and config loading."""
        self.assertIsInstance(self.compressor, QuantumCompressor)
        self.assertEqual(self.compressor.config["min_compression_length"], 10)
        self.assertEqual(self.compressor.config["max_compression_level"], 9)
        self.assertFalse(self.compressor.config["use_quantum_simulation"])

    def test_compression_decompression(self) -> None:
    """Test basic compress/decompress cycle."""
        compressed, metadata = self.compressor.compress(self.sample_text)
        self.assertTrue(metadata["compressed"])
        self.assertIsInstance(compressed, bytes)
        # Metadata should indicate non-simulation
        self.assertFalse(metadata.get("simulated", True)) # False expected

        decompressed = self.compressor.decompress(compressed, metadata)
        self.assertEqual(decompressed, self.sample_text)

    def test_compression_skipping(self) -> None:
    """Test skipping compression based on thresholds."""
        # Test too short
        compressed_short, metadata_short = self.compressor.compress(self.non_compressible_short)
        self.assertFalse(metadata_short["compressed"])
        self.assertEqual(metadata_short.get("reason"), "too_short")
        # Check data is original bytes
        self.assertEqual(compressed_short, self.non_compressible_short.encode('utf-8'))

        # Test low entropy (using a dedicated low entropy string)
        compressed_entropy, metadata_entropy = self.compressor.compress(self.long_low_entropy)
        self.assertFalse(metadata_entropy["compressed"])
        self.assertEqual(metadata_entropy.get("reason"), "low_entropy")
        self.assertEqual(compressed_entropy, self.long_low_entropy.encode('utf-8'))


    def test_force_compression(self) -> None:
    """Test forcing compression."""
        # Force compress short text
        compressed, metadata = self.compressor.compress(self.non_compressible_short, force_compression=True)
        self.assertTrue(metadata["compressed"])
        decompressed = self.compressor.decompress(compressed, metadata)
        self.assertEqual(decompressed, self.non_compressible_short)

        # Force compress low entropy text
        compressed_ent, metadata_ent = self.compressor.compress(self.long_low_entropy, force_compression=True)
        self.assertTrue(metadata_ent["compressed"])
        decompressed_ent = self.compressor.decompress(compressed_ent, metadata_ent)
        self.assertEqual(decompressed_ent, self.long_low_entropy)

    def test_compression_levels(self) -> None:
    """Test different compression levels."""
        compressed_0, meta_0 = self.compressor.compress(self.sample_text, level=0, force_compression=True)
        compressed_9, meta_9 = self.compressor.compress(self.sample_text, level=9, force_compression=True)

        self.assertTrue(meta_0["compressed"])
        self.assertTrue(meta_9["compressed"])
        # Level 9 should be smaller or equal (zlib doesn't guarantee smaller for all data)
        self.assertLessEqual(len(compressed_9), len(compressed_0))
        self.assertEqual(meta_0["level"], 0)
        self.assertEqual(meta_9["level"], 9)

        decomp_0 = self.compressor.decompress(compressed_0, meta_0)
        decomp_9 = self.compressor.decompress(compressed_9, meta_9)
        self.assertEqual(decomp_0, self.sample_text)
        self.assertEqual(decomp_9, self.sample_text)

    def test_text_blocks(self) -> None:
    """Test compression and decompression of text blocks using instance methods."""
        # Use the force_compressor instance for this test
        compressed = self.force_compressor.compress_text_blocks(self.text_with_blocks)
        self.assertIn("COMPRESSED:META=", compressed, "Compression marker not found in text blocks result")
        self.assertNotIn("Repetitive content * 15", compressed) # Original content should be gone

        decompressed = self.force_compressor.decompress_text_blocks(compressed)
        self.assertNotIn("COMPRESSED:META=", decompressed)
        self.assertIn("Repetitive content * 15", decompressed)
        self.assertIn("var x = 1;", decompressed)
        # Basic check for overall structure preservation
        self.assertTrue(decompressed.startswith("\nRegular text.\n"))
        self.assertTrue(decompressed.endswith("\nFinal text."))

    def test_quantum_compression_wrappers(self) -> None:
    """Test the quantum_compress and quantum_decompress top-level wrappers."""
        # The wrappers now instantiate a *default* compressor.
        # To test compression *happening*, the default config needs low thresholds
        # OR we need to provide text guaranteed to compress under defaults.
        # Let's use a very long repetitive text that *should* compress by default.
        long_repetitive_text = "Repeat this line for default compression test. " * 50
        block_text = f"Some text\n```\n{long_repetitive_text}\n```\nMore text"

        compressed = quantum_compress(block_text) # Uses default compressor instance implicitly

        # Check if the block inside was compressed
        # This assertion might still fail if default thresholds are too high
        self.assertIn("COMPRESSED:META=", compressed, "Top-level wrapper failed to compress block")

        # Decompress and verify
        decompressed = quantum_decompress(compressed)
        self.assertNotIn("COMPRESSED:META=", decompressed)
        self.assertIn("Repeat this line for default compression test.", decompressed)

    def test_svd_compression_misplaced(self) -> None:
    """Test SVD-based functions (acknowledging they are misplaced)."""
        # Create test data
        data = np.random.rand(20, 15)
        data[5:10, 5:10] += 2 # Add structure

        # Compress
        # Ensure rank is set low enough to actually compress for testing ratio
        compressed_info = compress_quantum(data, rank=5)
        self.assertTrue(compressed_info["compressed"])
        self.assertEqual(compressed_info["method"], "svd")
        self.assertIn("U", compressed_info)
        self.assertIn("S", compressed_info)
        self.assertIn("Vt", compressed_info)
        self.assertLessEqual(compressed_info["rank"], 5)
        self.assertGreater(compressed_info["compression_ratio"], 1.0) # Original/Compressed

        # Decompress
        decompressed_data = decompress_quantum(compressed_info)
        self.assertIsNotNone(decompressed_data)
        self.assertEqual(data.shape, decompressed_data.shape)
        # Check if lossy reconstruction is 'close enough' (MSE check)
        mse = np.mean((data - decompressed_data) ** 2)
        self.assertLess(mse, 0.1, "SVD reconstruction error too high") # Expect low error for rank 5 on 20x15

    def test_stats_tracking(self) -> None:
    """Test statistics tracking."""
        self.compressor.reset_stats() # Start fresh

        # Operation 1: Compressible text
        _, meta1 = self.compressor.compress(self.sample_text)
        # Operation 2: Non-compressible text (skipped by threshold)
        _, meta2 = self.compressor.compress(self.non_compressible_short)
        # Operation 3: Decompress first result
        self.compressor.decompress(meta1.get("_compressed_bytes", b""), meta1) # Pass dummy bytes if needed
        # Operation 4: Decompress non-compressed data
        self.compressor.decompress(self.non_compressible_short.encode('utf-8'), meta2)

        stats = self.compressor.get_stats()

        self.assertEqual(stats["compress_calls"], 2, "Compress calls mismatch") # FIXED: Both calls count
        self.assertEqual(stats["compress_success"], 1, "Compress success mismatch")
        self.assertEqual(stats["compress_skipped_length"], 1, "Compress skip length mismatch")
        self.assertEqual(stats["decompress_calls"], 2, "Decompress calls mismatch")
        # Decompress should succeed even if input wasn't compressed by *this* call,
        # as long as metadata flag is correct or absent/defaults to True.
        # Let's refine: The second decompress call will have metadata['compressed']=False
        self.assertEqual(stats["decompress_success"], 1, "Decompress success mismatch") # Only first one truly decompressed
        self.assertGreater(stats["total_compress_time_ms"], 0)
        self.assertGreater(stats["total_decompress_time_ms"], 0)
        self.assertGreater(stats["total_original_bytes"], 0)
        self.assertGreater(stats["total_compressed_bytes_final"], 0)
        self.assertLess(stats["average_compression_ratio"], 1.0) # Overall ratio should be < 1

    def test_config_loading(self) -> None:
    """Test loading configuration from file."""
         with tempfile.TemporaryDirectory() as tmpdir:
              config_path = os.path.join(tmpdir, "test_comp_config.json")
              test_config_data = {
                   "min_compression_length": 55,
                   "compression_entropy_threshold": 0.99,
                   "use_quantum_simulation": True, # Override default
                   "quantum_circuit_depth": 3
              }
              with open(config_path, "w") as f:
                   json.dump(test_config_data, f)

              # Test init with config file
              compressor = QuantumCompressor(config_path=config_path)
              self.assertEqual(compressor.config["min_compression_length"], 55)
              self.assertEqual(compressor.config["compression_entropy_threshold"], 0.99)
              self.assertTrue(compressor.config["use_quantum_simulation"])
              self.assertEqual(compressor.config["quantum_circuit_depth"], 3)
              # Check default value is still there if not overridden
              self.assertEqual(compressor.config["max_compression_level"], 9)

              # Test init with dict overriding file
              compressor_override = QuantumCompressor(config={"min_compression_length": 66}, config_path=config_path)
              self.assertEqual(compressor_override.config["min_compression_length"], 66) # Dict overrides file
              self.assertEqual(compressor_override.config["compression_entropy_threshold"], 0.99) # Value from file retained


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Run tests directly if needed

# --- END OF FILE test_quantum_compression.py ---