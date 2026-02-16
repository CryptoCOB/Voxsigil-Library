"""
Isolated tests for debugging failing quantum compression features.
"""
import unittest
import os
import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if (parent_dir not in sys.path):
    sys.path.insert(0, parent_dir)

from MetaConsciousness.utils.compression.quantum_compression import (
    QuantumCompressor, compress, decompress, 
    quantum_compress, quantum_decompress,
    compress_text_blocks, decompress_text_blocks
)
from MetaConsciousness.memory.compression.rag_compression import CompressedRAGStore

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compression_debug")

class TestFailingCompression(unittest.TestCase):
    """Isolated tests for failing compression features."""
    
    def setUp(self):
        """Set up for tests."""
        # Show full diffs for better debugging
        self.maxDiff = None
        
        # Use default config but force quantum simulation off for baseline testing
        self.compressor = QuantumCompressor(config={
            "min_compression_length": 50,
            "use_quantum_simulation": False,
            "debug_mode": True
        })
        
        # Test data
        self.medium_text = "This is a medium length text that should be long enough to be compressed by our algorithm. " * 5
        self.code_block = "```\ndef test_function():\n    print('Hello world')\n    return True\n```"
        
        # RAG test data
        self.doc_large = {
            "content": "This is a much larger document with enough content to trigger compression. " * 10,
            "metadata": {"id": "2", "type": "large"}
        }
        self.docs = [
            {
                "content": "Small document",
                "metadata": {"id": "1", "type": "small"}
            },
            self.doc_large
        ]
        
        self.rag_store = CompressedRAGStore(compressor=self.compressor)
    
    def test_basic_compression_roundtrip(self):
        """Test very basic compression/decompression without any complexity."""
        original = "Simple test string that is long enough to trigger compression but not complex."
        
        # Trace each step
        logger.debug(f"Original text length: {len(original)}")
        
        # Direct byte encoding/decoding test
        bytes_data = original.encode('utf-8')
        decoded = bytes_data.decode('utf-8')
        self.assertEqual(original, decoded)
        logger.debug(f"Basic UTF-8 roundtrip successful")
        
        # Simplest compression test
        compressed = zlib.compress(bytes_data)
        decompressed = zlib.decompress(compressed)
        decoded_again = decompressed.decode('utf-8')
        self.assertEqual(original, decoded_again)
        logger.debug(f"Basic zlib roundtrip successful")
        
        # Now test our library's basic function
        compressed_data, metadata = self.compressor.compress(original)
        logger.debug(f"Compression metadata: {metadata}")
        
        decompressed_text = self.compressor.decompress(compressed_data, metadata)
        self.assertEqual(original, decompressed_text)
        logger.debug(f"QuantumCompressor roundtrip successful")
    
    def test_failing_code_block_compression(self):
        """Test compression of code blocks (failing test)."""
        # Debug original code block
        logger.debug(f"Original code block: {repr(self.code_block)}")
        
        # Compress code blocks
        text_with_compressed_blocks = self.compressor.compress_text_blocks(self.code_block)
        logger.debug(f"Compressed blocks result: {repr(text_with_compressed_blocks)}")
        
        # Should contain the COMPRESSED marker
        self.assertIn("COMPRESSED", text_with_compressed_blocks)
        
        # Decompress code blocks
        decompressed_text = self.compressor.decompress_text_blocks(text_with_compressed_blocks)
        logger.debug(f"Decompressed result: {repr(decompressed_text)}")
        
        # Compare character by character if they differ
        if self.code_block != decompressed_text:
            logger.debug("Character-by-character comparison:")
            min_len = min(len(self.code_block), len(decompressed_text))
            for i in range(min_len):
                if self.code_block[i] != decompressed_text[i]:
                    logger.debug(f"Position {i}: '{self.code_block[i]}' vs '{decompressed_text[i]}'")
                    break
        
        # Should get original back after decompression
        self.assertEqual(self.code_block, decompressed_text)
    
    def test_failing_compression_decompression(self):
        """Test basic compression and decompression (failing test)."""
        # Medium text should be compressed
        compressed_medium, metadata_medium = self.compressor.compress(self.medium_text)
        logger.debug(f"Medium text compression metadata: {metadata_medium}")
        
        # Check the compression bytes
        logger.debug(f"Compressed data type: {type(compressed_medium)}")
        logger.debug(f"Compressed data: {compressed_medium[:50]}...")
        
        # Decompress should restore the original text
        decompressed = self.compressor.decompress(compressed_medium, metadata_medium)
        logger.debug(f"Decompressed result type: {type(decompressed)}")
        logger.debug(f"Decompressed result: {decompressed[:50]}...")
        
        # Compare initial part of texts
        logger.debug(f"Original text starts with: {self.medium_text[:50]}...")
        
        # Should get original back after decompression
        self.assertEqual(self.medium_text, decompressed)
    
    def test_failing_legacy_functions(self):
        """Test legacy functions for backward compatibility (failing test)."""
        # Use legacy compress function
        compressed = compress(self.medium_text)
        logger.debug(f"Legacy compressed type: {type(compressed)}")
        
        # Use legacy decompress function
        decompressed = decompress(compressed)
        logger.debug(f"Legacy decompressed type: {type(decompressed)}")
        
        # Should get original back after decompression
        self.assertEqual(self.medium_text, decompressed)
        
        # Quantum compress/decompress should work too
        quantum_compressed = quantum_compress(self.medium_text)
        quantum_decompressed = quantum_decompress(quantum_compressed)
        
        logger.debug(f"Quantum compressed: {quantum_compressed[:50]}...")
        logger.debug(f"Quantum decompressed: {quantum_decompressed[:50]}...")
        
        # But decompressed should match original
        self.assertEqual(self.medium_text, quantum_decompressed)
    
    def test_failing_rag_document_decompression(self):
        """Test RAG document decompression (failing test)."""
        # Compress large document
        compressed = self.rag_store.compress_document(self.doc_large["content"], self.doc_large["metadata"])
        logger.debug(f"RAG compression info: {compressed['compression']}")
        
        # Decompress the document
        decompressed = self.rag_store.decompress_document(compressed)
        logger.debug(f"Decompressed content starts with: {decompressed['content'][:50]}...")
        
        # Content should be restored after decompression
        self.assertEqual(self.doc_large["content"], decompressed["content"])
    
    def test_failing_rag_batch_processing(self):
        """Test RAG batch document processing (failing test)."""
        # Process multiple documents for storage
        compressed_docs = self.rag_store.process_documents_for_storage(self.docs)
        
        # First doc shouldn't be compressed, second should be
        self.assertFalse(compressed_docs[0]["compression"]["compressed"])
        self.assertTrue(compressed_docs[1]["compression"]["compressed"])
        
        # Log compression details
        logger.debug(f"Doc 1 compression: {compressed_docs[0]['compression']}")
        logger.debug(f"Doc 2 compression: {compressed_docs[1]['compression']}")
        
        # Process for retrieval - this decompresses the documents
        decompressed_docs = self.rag_store.process_documents_for_retrieval(compressed_docs)
        
        # Log first part of decompressed content
        logger.debug(f"Decompressed doc 1: {decompressed_docs[0]['content']}")
        logger.debug(f"Decompressed doc 2 starts with: {decompressed_docs[1]['content'][:50]}...")
        
        # Should restore original content after decompression
        self.assertEqual(self.docs[0]["content"], decompressed_docs[0]["content"])
        self.assertEqual(self.docs[1]["content"], decompressed_docs[1]["content"])

# Required for direct execution of this file
if __name__ == "__main__":
    # Essential imports needed when running isolated tests
    import zlib
    
    # Run just these tests
    unittest.main(verbosity=2)
