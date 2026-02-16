"""
Compression Utilities

This package provides utilities for various compression techniques
used throughout the MetaConsciousness framework.
"""

import logging
from typing import Dict, Any, List, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Version
__version__ = '0.2.0'

def get_available_compression_methods() -> List[str]:
    """
    Get a list of available compression methods.
    
    Returns:
        List of available compression method names
    """
    methods = []
    
    # Check for other compression methods
    try:
        from .lz_compression import is_available as lz_available
        if lz_available():
            methods.append("lz")
    except ImportError:
        pass
    
    try:
        from .huffman import is_available as huffman_available
        if huffman_available():
            methods.append("huffman")
    except ImportError:
        pass
    
    return methods

def compress(data: Any, method: str = "auto", **kwargs) -> Dict[str, Any]:
    """
    Compress data using the specified method.
    
    Args:
        data: Data to compress
        method: Compression method to use or "auto" to select best method
        **kwargs: Additional arguments for the compression method
        
    Returns:
        Dictionary with compressed data and metadata
    """
    available_methods = get_available_compression_methods()
    
    if not available_methods:
        logger.error("No compression methods available")
        return {"error": "No compression methods available", "compressed": False}
    
    if method == "auto":
        method = available_methods[0]  # Use first available method
    
    if method not in available_methods:
        logger.error(f"Compression method '{method}' not available")
        return {"error": f"Compression method '{method}' not available", "compressed": False}
    
    if method == "lz":
        try:
            from .lz_compression import compress as lz_compress
            return lz_compress(data, **kwargs)
        except ImportError:
            logger.error("LZ compression not available")
            return {"error": "LZ compression not available", "compressed": False}
    elif method == "huffman":
        try:
            from .huffman import compress as huffman_compress
            return huffman_compress(data, **kwargs)
        except ImportError:
            logger.error("Huffman compression not available")
            return {"error": "Huffman compression not available", "compressed": False}
    else:
        logger.error(f"Unsupported compression method: {method}")
        return {"error": f"Unsupported compression method: {method}", "compressed": False}

def decompress(compressed_data: Dict[str, Any]) -> Any:
    """
    Decompress data that was compressed using compress().
    
    Args:
        compressed_data: Compressed data dictionary from compress()
        
    Returns:
        Decompressed data
    """
    method = compressed_data.get("method")
    
    if method == "lz":
        try:
            from .lz_compression import decompress as lz_decompress
            return lz_decompress(compressed_data)
        except ImportError:
            logger.error("LZ decompression not available")
            return None
    elif method == "huffman":
        try:
            from .huffman import decompress as huffman_decompress
            return huffman_decompress(compressed_data)
        except ImportError:
            logger.error("Huffman decompression not available")
            return None
    else:
        logger.error(f"Unsupported decompression method: {method}")
        return None
