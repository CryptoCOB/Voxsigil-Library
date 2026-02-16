import unittest
import sys
import os
import json
import base64
import re
import numpy as np
import zlib
from typing import Union, Dict, Any

# Ensure the parent directory is in the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MetaConsciousness.utils.compression.quantum_compression import (
    compress, decompress, 
    compress_text_blocks, decompress_text_blocks,
    default_compressor, log_event, CONFIG
)

class TestSpecificCompression(unittest.TestCase):
    def setUp(self):
        # Test data
        self.code_block = """```
def test_function():
    print('Hello world')
    return True
```"""
        
        self.medium_text = "This is a medium length text that should be long enough to be compressed by our algorithm. " * 5
        
        # Enable debug logging
        self.setup_logging()
        
        # Save original config
        self.original_config = default_compressor.config.copy()
        
        # Configure compressor for testing - critical to ensure consistent test results
        default_compressor.config.update({
            "min_compression_length": 10,  # Shorter threshold for tests
            "compression_entropy_threshold": 1.0,  # Allow all text to compress
            "use_quantum_simulation": False,  # IMPORTANT: Disable quantum sim for reliable tests
            "debug_mode": True,
        })
    
    def tearDown(self):
        # Restore original config
        default_compressor.config = self.original_config
    
    def setup_logging(self):
        """Setup detailed logging for tests"""
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('compression_debug')
        self.logger.setLevel(logging.DEBUG)
    
    def log_debug(self, message, data=None):
        """Helper to log debug info"""
        self.logger.debug(message)
        if data:
            if isinstance(data, bytes):
                preview = data[:50].decode('utf-8', errors='replace') + '...' if len(data) > 50 else data.decode('utf-8', errors='replace')
                self.logger.debug(f"Data (bytes): {preview}")
            elif isinstance(data, str):
                preview = data[:50] + '...' if len(data) > 50 else data
                self.logger.debug(f"Data (str): {preview}")
            else:
                self.logger.debug(f"Data: {data}")
    
    def test_direct_compression_decompression(self):
        """Test the most basic direct compression and decompression"""
        sample_text = "Test sample text" * 20  # Make sure it's long enough
        
        # Raw compression with zlib
        raw_data = sample_text.encode('utf-8')
        self.log_debug(f"Raw data length: {len(raw_data)}")
        
        # Try direct zlib compression/decompression
        zlib_compressed = zlib.compress(raw_data)
        self.log_debug(f"Zlib compressed length: {len(zlib_compressed)}")
        
        # Base85 encode
        encoded = base64.b85encode(zlib_compressed)
        self.log_debug(f"Base85 encoded: {encoded[:50]}...")
        
        # And back
        decoded = base64.b85decode(encoded)
        self.assertEqual(decoded, zlib_compressed, "Base85 roundtrip failed")
        
        decompressed = zlib.decompress(decoded)
        self.assertEqual(decompressed, raw_data, "Zlib roundtrip failed")
        
        # Now let's test the actual compressor
        self.log_debug("Testing QuantumCompressor directly...")
        
        # Force config to ensure compression
        temp_config = {
            "min_compression_length": 10,
            "compression_entropy_threshold": 1.0,  # Always compress
            "debug_mode": True,
            "use_quantum_simulation": False  # Disable quantum sim for basic test
        }
        test_compressor = default_compressor.__class__(config=temp_config)
        
        # Direct compression with force_compression=True
        compressed, metadata = test_compressor.compress(sample_text, force_compression=True)
        self.log_debug(f"Compression metadata: {metadata}")
        self.assertTrue(metadata["compressed"], "Compression didn't happen despite forced settings")
        
        # Direct decompression
        decompressed = test_compressor.decompress(compressed, metadata)
        self.assertEqual(sample_text, decompressed, "Direct compression/decompression failed")
    
    def test_detailed_code_block_compression(self):
        """Detailed test of code block compression with diagnostics"""
        # Log original data for clarity
        self.log_debug("Code block length:", len(self.code_block))
        self.log_debug("Code block entropy:", default_compressor.calculate_entropy(self.code_block))
    
        # Test full compression/decompression pipeline
        self.log_debug("Now testing the full pipeline functions")
        compressed_text = compress_text_blocks(self.code_block)
        self.log_debug("Result of compress_text_blocks:", compressed_text)
        decompressed_text = decompress_text_blocks(compressed_text)
        self.log_debug("Result of decompress_text_blocks:", decompressed_text)
        
        # Verify
        self.assertEqual(self.code_block, decompressed_text, 
                         "Decompressed text should match original code block")
    
    def test_detailed_legacy_functions(self):
        """Detailed test of legacy compression functions with diagnostics"""
        # Step 1: Compress with legacy function and force compression
        compressed_data, metadata = compress(self.medium_text, force_compression=True)
        self.log_debug("Original text:", self.medium_text)
        self.log_debug(f"Original text length: {len(self.medium_text)}")
        self.log_debug(f"Compressed data type: {type(compressed_data)}")
        if isinstance(compressed_data, bytes):
            preview = compressed_data[:50].decode('utf-8', errors='replace') + '...' if len(compressed_data) > 50 else compressed_data.decode('utf-8', errors='replace')
            self.log_debug(f"Compressed data preview: {preview}")
        self.log_debug("Compression metadata:", metadata)
        
        # Try direct decompression of individual steps to diagnose
        try:
            # Step 1: Base85 decode
            decoded = base64.b85decode(compressed_data)
            self.log_debug(f"Base85 decoded length: {len(decoded)}")
            
            # Step 2: zlib decompress
            decompressed_bytes = zlib.decompress(decoded)
            self.log_debug(f"Zlib decompressed length: {len(decompressed_bytes)}")
            
            # Step 3: tobytes + decode
            decompressed_text = decompressed_bytes.decode('utf-8', errors='replace')
            self.log_debug(f"Decoded text preview: {decompressed_text[:50]}...")
        except Exception as e:
            self.log_debug(f"Error in manual decompression steps: {e}")
        
        # Step 2: Decompress with legacy function
        # Pass both data and metadata to ensure proper decompression
        self.log_debug("Now attempting legacy decompression...")
        decompressed = decompress((compressed_data, metadata))
        self.log_debug(f"Decompressed text type: {type(decompressed)}")
        self.log_debug(f"Decompressed text length: {len(decompressed)}")
        
        if len(decompressed) > 50:
            preview = decompressed[:50] + '...'
        else:
            preview = decompressed
        self.log_debug(f"Decompressed text preview: {preview}")
        
        # Step 3: Check if null bytes are present
        null_byte_count = decompressed.count('\x00') if isinstance(decompressed, str) else 0
        self.log_debug(f"Null byte count in decompressed text: {null_byte_count}")
        
        # Verification
        self.assertEqual(self.medium_text, decompressed,
                         "Decompressed text should match original")

if __name__ == '__main__':
    unittest.main()
