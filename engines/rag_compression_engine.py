#!/usr/bin/env python
"""
RAG Compression Engine for VoxSigil BLT module.

This module provides compression functionality for the BLT-enhanced RAG system.
"""

import base64
import binascii
import bz2
import hashlib
import json
import logging
import lzma
import math
import zlib
from typing import Any, Dict, Optional, Tuple, Union

# HOLO-1.5 Mesh Infrastructure
from .base import BaseEngine, vanta_engine, CognitiveMeshRole

# Constants for compression modes
MODE_ZLIB = "zlib"
MODE_LZMA = "lzma"
MODE_BZ2 = "bz2"
MODE_SYMBOLIC_DIGEST = "symbolic_digest"
MODE_PASSTHROUGH = "passthrough"  # No compression

SYMBOLIC_STRATEGY_FIRST_LINES = "first_lines"
SYMBOLIC_STRATEGY_FIRST_CHARS = "first_chars"
DEFAULT_ENCODING = "utf-8"


class RAGCompressionError(Exception):
    """Exception raised for RAG compression errors."""
    pass


@vanta_engine(
    name="rag_compression_engine",
    subsystem="rag_optimization_subsystem",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="RAG compression and optimization engine for data synthesis and compression",
    capabilities=["data_compression", "rag_optimization", "synthesis", "integration", "fusion"]
)
class RAGCompressionEngine(BaseEngine):
    """Compression engine for RAG content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the RAG compression engine.

        Args:
            config: Optional configuration dictionary
        """
        # Initialize BaseEngine with HOLO-1.5 mesh capabilities
        super().__init__(None, config)  # Will set vanta_core later if needed
        
        # Initialize logger first to ensure it's always available
        self.logger = logging.getLogger("VoxSigilSystem.RAGCompressionEngine")

        self.config = self._get_default_config()
        if config:
            self.update_config(config)
        self._init_metrics()
        self.logger.info(f"RAGCompressionEngine initialized with config: {self.config}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration for the compression engine."""
        return {
            "default_mode": MODE_ZLIB,
            "zlib_level": 9,
            "lzma_preset": None,
            "bz2_compresslevel": 9,
            "symbolic_digest_strategy": SYMBOLIC_STRATEGY_FIRST_LINES,
            "symbolic_preserve_ratio": 0.3,
            "symbolic_preserve_chars": 512,
            "min_entropy_for_compression": 1.5,
            "min_length_for_compression": 64,
            "encoding": DEFAULT_ENCODING,
            "store_original_length": True,
            "error_on_decompression_failure": False,
        }

    def _init_metrics(self) -> None:
        """Initialize metrics for tracking compression performance."""
        self.metrics = {
            "compress_requests": 0,
            "decompress_requests": 0,
            "successful_compressions": 0,
            "successful_decompressions": 0,
            "bytes_original_total": 0,
            "bytes_compressed_total": 0,
            "avg_compression_ratio": 0.0,
            "last_op_details": None,
            "errors": {"compression": 0, "decompression": 0},
            "skipped_compressions_entropy": 0,
            "skipped_compressions_length": 0,
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the compression engine configuration.

        Args:
            new_config: Dictionary of configuration values to update
        """
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
                self.logger.info(f"Config '{key}' updated to: {value}")
            else:
                self.logger.warning(f"Attempted to update unknown config key: {key}")

    def _estimate_entropy(self, text: str) -> float:
        """
        Estimate the entropy of a text string.

        Args:
            text: The text to analyze

        Returns:
            float: Calculated entropy value
        """
        if not text:
            return 0.0
        byte_array = text.encode(self.config["encoding"])
        freq = {}
        for byte_val in byte_array:
            freq[byte_val] = freq.get(byte_val, 0) + 1
        total_bytes = len(byte_array)
        if total_bytes == 0:
            return 0.0
        entropy = -sum(
            (count / total_bytes) * math.log2(count / total_bytes)
            for count in freq.values()
            if count > 0
        )
        return round(entropy, 5)

    def _update_avg_ratio(self, new_ratio: float) -> None:
        """
        Update the average compression ratio metric.

        Args:
            new_ratio: The new ratio to incorporate into the average
        """
        current_avg = self.metrics["avg_compression_ratio"]
        count = self.metrics["successful_compressions"]
        if count <= 1:  # First compression
            self.metrics["avg_compression_ratio"] = new_ratio
        else:  # Running average
            self.metrics["avg_compression_ratio"] = current_avg + (
                (new_ratio - current_avg) / count
            )

    def _wrap_with_metadata(
        self, compressed_data: Union[str, bytes], mode: str, original_length: int
    ) -> str:
        """
        Wrap compressed data with metadata for storage and retrieval.

        Args:
            compressed_data: The compressed data (bytes or string)
            mode: The compression mode used
            original_length: The original length of the uncompressed data

        Returns:
            str: JSON string containing wrapped data with metadata
        """
        if isinstance(compressed_data, bytes):
            compressed_data_str = base64.b64encode(compressed_data).decode(
                DEFAULT_ENCODING
            )
        else:
            compressed_data_str = (
                compressed_data  # Assumed to be string for symbolic digest
            )
        metadata = {"mode_used": mode, "compressed_payload": compressed_data_str}
        if self.config["store_original_length"]:
            metadata["original_length"] = original_length
        return json.dumps(metadata)

    def _unwrap_metadata(
        self, wrapped_data_str: str
    ) -> Optional[Tuple[str, bytes, Optional[int]]]:
        """
        Extract data and metadata from a wrapped compressed payload.

        Args:
            wrapped_data_str: The wrapped data string (JSON)

        Returns:
            Optional[Tuple[str, bytes, Optional[int]]]: Tuple of (mode, payload_bytes, original_length)
            or None if unwrapping fails
        """
        try:
            metadata = json.loads(wrapped_data_str)
            mode = metadata.get("mode_used")
            payload_str = metadata.get("compressed_payload")
            original_length = metadata.get("original_length")

            if not mode or payload_str is None:
                self.logger.error(
                    "Metadata unwrapping failed: 'mode_used' or 'compressed_payload' missing."
                )
                return None

            # For byte-based compression modes, decode from base64. For others, encode to bytes.
            if mode in [MODE_ZLIB, MODE_LZMA, MODE_BZ2]:
                # Ensure payload_str is a string before encoding
                if isinstance(payload_str, str):
                    payload_str_safe = payload_str
                elif isinstance(payload_str, (bytes, bytearray, memoryview)):
                    # Convert bytes-like objects to string first
                    if isinstance(payload_str, memoryview):
                        # Convert memoryview to bytes first
                        payload_str_safe = bytes(payload_str).decode(DEFAULT_ENCODING)
                    elif hasattr(payload_str, "decode"):
                        payload_str_safe = payload_str.decode(DEFAULT_ENCODING)
                    else:
                        payload_str_safe = str(payload_str)
                else:
                    payload_str_safe = str(payload_str)
                payload_bytes = base64.b64decode(
                    payload_str_safe.encode(DEFAULT_ENCODING)
                )
            elif mode == MODE_SYMBOLIC_DIGEST or mode == MODE_PASSTHROUGH:
                # Ensure payload_str is a string before encoding
                if isinstance(payload_str, str):
                    payload_str_safe = payload_str
                elif isinstance(payload_str, (bytes, bytearray, memoryview)):
                    # Convert bytes-like objects to string first
                    if isinstance(payload_str, memoryview):
                        # Convert memoryview to bytes first
                        payload_str_safe = bytes(payload_str).decode(DEFAULT_ENCODING)
                    elif hasattr(payload_str, "decode"):
                        payload_str_safe = payload_str.decode(DEFAULT_ENCODING)
                    else:
                        payload_str_safe = str(payload_str)
                else:
                    payload_str_safe = str(payload_str)
                payload_bytes = payload_str_safe.encode(
                    self.config["encoding"]
                )  # payload_str is already the content
            else:
                self.logger.error(
                    f"Unrecognized mode '{mode}' during metadata unwrapping."
                )
                return None
            return mode, payload_bytes, original_length
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError during metadata unwrapping: {e}")
        except binascii.Error as e:  # Catches base64 decoding errors
            self.logger.error(f"Base64 decoding error during metadata unwrapping: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during metadata unwrapping: {e}")
        return None

    def _compress_symbolic_digest(self, text: str) -> str:
        """
        Create a symbolic digest of text (partial representation).

        Args:
            text: Text to create a digest for

        Returns:
            str: Symbolic digest representation
        """
        if not text:
            return ""

        strategy = self.config["symbolic_digest_strategy"]
        preserve_ratio = self.config["symbolic_preserve_ratio"]
        preserve_chars = self.config["symbolic_preserve_chars"]

        # Calculate a SHA-256 hash for the text (used as an integrity check)
        text_hash = hashlib.sha256(text.encode(self.config["encoding"])).hexdigest()[
            :16
        ]

        if strategy == SYMBOLIC_STRATEGY_FIRST_LINES:
            lines = text.split("\n")
            total_lines = len(lines)
            keep_lines = min(
                int(total_lines * preserve_ratio),
                int(preserve_chars / 40),  # Approx 40 chars per line as heuristic
                max(
                    5, total_lines // 2
                ),  # Keep at least 5 lines or half, whichever is greater
            )

            kept_text = "\n".join(lines[:keep_lines])
            if keep_lines < total_lines:
                kept_text += f"\n...{total_lines - keep_lines} more lines..."

        elif strategy == SYMBOLIC_STRATEGY_FIRST_CHARS:
            if len(text) <= preserve_chars:
                kept_text = text
            else:
                kept_text = (
                    text[:preserve_chars]
                    + f"...{len(text) - preserve_chars} more chars..."
                )
        else:
            # Fallback strategy - use first_chars
            kept_text = text[: min(len(text), preserve_chars)]
            if len(text) > preserve_chars:
                kept_text += f"...{len(text) - preserve_chars} more chars..."

        # Add the hash for integrity
        result = f"{kept_text}\n[SHA256:{text_hash}]"
        return result

    def compress(self, text: str, mode: Optional[str] = None) -> Optional[str]:
        """
        Compress text using the specified or default compression mode.

        Args:
            text: Text to compress
            mode: Optional compression mode override

        Returns:
            Optional[str]: Wrapped compressed data with metadata, or None on failure
        """
        self.metrics["compress_requests"] += 1
        if not text:
            self.logger.warning("Compression attempt on empty text.")
            return self._wrap_with_metadata("", MODE_PASSTHROUGH, 0)

        original_length_bytes = len(text.encode(self.config["encoding"]))
        current_mode = mode or self.config["default_mode"]

        # Check if text is too short for compression
        if original_length_bytes < self.config[
            "min_length_for_compression"
        ] and current_mode not in [MODE_SYMBOLIC_DIGEST, MODE_PASSTHROUGH]:
            self.logger.info(
                f"Text length ({original_length_bytes} bytes) < min_length ({self.config['min_length_for_compression']}). Using passthrough."
            )
            self.metrics["skipped_compressions_length"] += 1
            current_mode = MODE_PASSTHROUGH

        # Calculate entropy and check if it's worth compressing
        entropy = self._estimate_entropy(text)
        if (
            current_mode not in [MODE_SYMBOLIC_DIGEST, MODE_PASSTHROUGH]
            and entropy < self.config["min_entropy_for_compression"]
        ):
            self.logger.info(
                f"Text entropy ({entropy:.3f}) < min_entropy ({self.config['min_entropy_for_compression']}). Using passthrough."
            )
            self.metrics["skipped_compressions_entropy"] += 1
            current_mode = MODE_PASSTHROUGH

        compressed_payload: Optional[Union[bytes, str]] = None
        try:
            text_bytes = text.encode(self.config["encoding"])

            if current_mode == MODE_ZLIB:
                compressed_payload = zlib.compress(
                    text_bytes, level=self.config["zlib_level"]
                )
            elif current_mode == MODE_LZMA:
                compressed_payload = lzma.compress(
                    text_bytes, preset=self.config["lzma_preset"]
                )
            elif current_mode == MODE_BZ2:
                compressed_payload = bz2.compress(
                    text_bytes, compresslevel=self.config["bz2_compresslevel"]
                )
            elif current_mode == MODE_SYMBOLIC_DIGEST:
                compressed_payload = self._compress_symbolic_digest(text)  # returns str
            elif current_mode == MODE_PASSTHROUGH:
                compressed_payload = text  # str
            else:
                self.logger.error(f"Unsupported compression mode: {current_mode}")
                self.metrics["errors"]["compression"] += 1
                return None

            if compressed_payload is None:  # Should not happen if logic is correct
                self.logger.error(
                    f"Compression payload is None for mode {current_mode}"
                )
                return None

            compressed_length_bytes = (
                len(compressed_payload)
                if isinstance(compressed_payload, (bytes, bytearray, memoryview))
                else len(compressed_payload.encode(self.config["encoding"]))
            )
            ratio = (
                compressed_length_bytes / original_length_bytes
                if original_length_bytes
                else 1.0
            )

            # If compressed is larger than original (for byte modes), use passthrough
            if (
                isinstance(compressed_payload, bytes)
                and compressed_length_bytes >= original_length_bytes
            ):
                self.logger.info(
                    f"Mode '{current_mode}' resulted in larger/equal size ({compressed_length_bytes} vs {original_length_bytes}). Using passthrough."
                )
                current_mode = MODE_PASSTHROUGH
                compressed_payload = text
                compressed_length_bytes = original_length_bytes
                ratio = 1.0

            wrapped_output = self._wrap_with_metadata(
                compressed_payload, current_mode, original_length_bytes
            )

            self.metrics["successful_compressions"] += 1
            self.metrics["bytes_original_total"] += original_length_bytes
            self.metrics["bytes_compressed_total"] += compressed_length_bytes
            self._update_avg_ratio(ratio)
            self.metrics["last_op_details"] = {
                "ratio": round(ratio, 5),
                "entropy": entropy,
                "mode": current_mode,
                "original_length_bytes": original_length_bytes,
                "compressed_length_bytes": compressed_length_bytes,
            }
            self.logger.debug(
                f"Mode '{current_mode}': Compressed {original_length_bytes} to {compressed_length_bytes} bytes. Ratio: {ratio:.3f}"
            )
            return wrapped_output

        except Exception as e:
            self.logger.error(f"Error during compression: {e}")
            self.metrics["errors"]["compression"] += 1
            return None

    def decompress(self, compressed_data: str) -> Optional[str]:
        """
        Decompress data that was previously compressed with the compress method.

        Args:
            compressed_data: The wrapped compressed data string

        Returns:
            Optional[str]: The decompressed text, or None on failure
        """
        self.metrics["decompress_requests"] += 1

        if not compressed_data:
            self.logger.warning("Decompression attempt on empty data.")
            return ""

        try:
            # Unwrap the metadata to get the mode and payload
            unwrap_result = self._unwrap_metadata(compressed_data)
            if not unwrap_result:
                self.logger.error("Failed to unwrap metadata for decompression.")
                self.metrics["errors"]["decompression"] += 1
                if self.config["error_on_decompression_failure"]:
                    raise RAGCompressionError("Failed to unwrap metadata")
                return None

            mode, payload_bytes, original_length = unwrap_result

            # Decompress based on the mode
            decompressed_text = None
            if mode == MODE_ZLIB:
                try:
                    decompressed_bytes = zlib.decompress(payload_bytes)
                    decompressed_text = decompressed_bytes.decode(
                        self.config["encoding"]
                    )
                except zlib.error as e:
                    self.logger.error(f"Zlib decompression error: {e}")
                    self.metrics["errors"]["decompression"] += 1
                    if self.config["error_on_decompression_failure"]:
                        raise RAGCompressionError(f"Zlib decompression error: {e}")
                    return None
            elif mode == MODE_LZMA:
                try:
                    decompressed_bytes = lzma.decompress(payload_bytes)
                    decompressed_text = decompressed_bytes.decode(
                        self.config["encoding"]
                    )
                except lzma.LZMAError as e:
                    self.logger.error(f"LZMA decompression error: {e}")
                    self.metrics["errors"]["decompression"] += 1
                    if self.config["error_on_decompression_failure"]:
                        raise RAGCompressionError(f"LZMA decompression error: {e}")
                    return None
            elif mode == MODE_BZ2:
                try:
                    decompressed_bytes = bz2.decompress(payload_bytes)
                    decompressed_text = decompressed_bytes.decode(
                        self.config["encoding"]
                    )
                except Exception as e:
                    self.logger.error(f"BZ2 decompression error: {e}")
                    self.metrics["errors"]["decompression"] += 1
                    if self.config["error_on_decompression_failure"]:
                        raise RAGCompressionError(f"BZ2 decompression error: {e}")
                    return None
            elif mode == MODE_PASSTHROUGH:
                # No compression was applied, just decode the bytes
                try:
                    decompressed_text = payload_bytes.decode(self.config["encoding"])
                except UnicodeDecodeError as e:
                    self.logger.error(f"Unicode decode error in passthrough mode: {e}")
                    # Try with errors='replace' as fallback
                    decompressed_text = payload_bytes.decode(
                        self.config["encoding"], errors="replace"
                    )
            elif mode == MODE_SYMBOLIC_DIGEST:
                # Symbolic digest can't be fully decompressed - it's lossy
                self.logger.warning(
                    "Attempted to decompress a symbolic digest, which is lossy. Returning partial content."
                )
                try:
                    # Extract content part (excluding the hash)
                    digest_text = payload_bytes.decode(self.config["encoding"])
                    if "[SHA256:" in digest_text:
                        decompressed_text = digest_text.split("[SHA256:")[0].strip()
                    else:
                        decompressed_text = digest_text
                except Exception as e:
                    self.logger.error(f"Error handling symbolic digest: {e}")
                    decompressed_text = payload_bytes.decode(
                        self.config["encoding"], errors="replace"
                    )
            else:
                self.logger.error(
                    f"Unknown compression mode '{mode}' during decompression"
                )
                self.metrics["errors"]["decompression"] += 1
                if self.config["error_on_decompression_failure"]:
                    raise RAGCompressionError(f"Unknown compression mode: {mode}")
                return None

            if decompressed_text is not None:
                self.metrics["successful_decompressions"] += 1
                # Validate original length if available
                if original_length is not None:
                    actual_length = len(
                        decompressed_text.encode(self.config["encoding"])
                    )
                    if (
                        abs(actual_length - original_length) > 0.1 * original_length
                    ):  # >10% diff
                        self.logger.warning(
                            f"Decompressed length ({actual_length}) differs significantly from original ({original_length})"
                        )
                return decompressed_text

            self.logger.error("Decompression produced None result")
            self.metrics["errors"]["decompression"] += 1
            if self.config["error_on_decompression_failure"]:
                raise RAGCompressionError("Decompression produced None result")
            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error during decompression: {e}")
            self.metrics["errors"]["decompression"] += 1
            if self.config["error_on_decompression_failure"]:
                raise RAGCompressionError(f"JSON decode error: {e}")
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error during decompression: {e}")
            self.metrics["errors"]["decompression"] += 1
            if self.config["error_on_decompression_failure"]:
                raise RAGCompressionError(f"Decompression error: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current compression engine metrics.

        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        return self.metrics.copy()  # Return a copy to prevent external modification


# For direct testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create compression engine
    engine = RAGCompressionEngine()

    # Test different compression modes
    test_text = "This is a test of the RAG compression engine. " * 10

    for mode in [
        MODE_ZLIB,
        MODE_LZMA,
        MODE_BZ2,
        MODE_SYMBOLIC_DIGEST,
        MODE_PASSTHROUGH,
    ]:
        print(f"\nTesting {mode} compression:")
        compressed = engine.compress(test_text, mode=mode)
        if compressed:
            print(f"  Compressed size: {len(compressed)} bytes")
            decompressed = engine.decompress(compressed)
            if decompressed:
                print(f"  Decompression successful: {decompressed[:50]}...")
                if mode != MODE_SYMBOLIC_DIGEST:  # Symbolic digest is lossy
                    print(f"  Original/decompressed match: {test_text == decompressed}")
            else:
                print("  Decompression failed!")
        else:
            print("  Compression failed!")

    # Print metrics
    print("\nCompression Metrics:")
    metrics = engine.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
