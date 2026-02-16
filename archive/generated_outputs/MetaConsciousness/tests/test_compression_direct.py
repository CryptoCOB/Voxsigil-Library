import sys
import os
import json
import base64
import re
import zlib

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MetaConsciousness.utils.compression.quantum_compression import (
    compress, decompress, default_compressor, log_event
)

def main():
    # Enable debug mode
    default_compressor.config["debug_mode"] = True
    default_compressor.config["use_quantum_simulation"] = False
    
    print("=== Testing Code Block Compression/Decompression ===")
    
    # Original code block
    code_block = """```
def test_function():
    print('Hello world')
    return True
```"""
    
    print("Original code block:", code_block)
    
    # Step 1: Manually compress just the contents (without the markers)
    content = code_block.strip("```").strip()
    print("Content to compress:", content)
    
    # Force compression
    compressed_data, metadata = default_compressor.compress(content, force_compression=True)
    print("Compressed data:", compressed_data.decode('ascii'))
    print("Metadata:", metadata)
    
    # Step 2: Manually reconstruct the compressed block with metadata
    metadata_b64 = base64.b64encode(json.dumps(metadata).encode()).decode()
    compressed_block = f"```COMPRESSED:{metadata_b64}\n{compressed_data.decode('ascii')}\n```"
    print("Reconstructed compressed block:", compressed_block)
    
    # Step 3: Manually decompress
    # First extract the parts
    pattern = re.compile(r"```COMPRESSED:([A-Za-z0-9+/=]+)\n(.*?)\n```", re.DOTALL)
    match = pattern.search(compressed_block)
    
    if match:
        metadata_b64 = match.group(1)
        content = match.group(2)
        print("Extracted metadata_b64:", metadata_b64[:30] + "...")
        print("Extracted content:", content[:30] + "...")
        
        # Decode metadata
        metadata_json = base64.b64decode(metadata_b64).decode()
        metadata = json.loads(metadata_json)
        print("Decoded metadata:", metadata)
        
        # Decompress content
        decompressed = default_compressor.decompress(content, metadata)
        print("Decompressed content:", decompressed)
        
        # Reconstruct original code block
        reconstructed = f"```{decompressed}```"
        print("Reconstructed original:", reconstructed)
        
        # Check if they match
        print("Match:", reconstructed == code_block)
    else:
        print("Failed to match the compressed block pattern")

if __name__ == "__main__":
    main()
