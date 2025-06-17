import binascii
import zlib
import lzma
import bz2
import json
import logging
import math
import base64
from typing import Dict, Any, Tuple, Optional, Union

# --- Constants ---
MODE_ZLIB = "zlib"
MODE_LZMA = "lzma"
MODE_BZ2 = "bz2"
MODE_SYMBOLIC_DIGEST = "symbolic_digest"
MODE_PASSTHROUGH = "passthrough" # No compression

SYMBOLIC_STRATEGY_FIRST_LINES = "first_lines"
SYMBOLIC_STRATEGY_FIRST_CHARS = "first_chars"
# Placeholder for more advanced strategies
# SYMBOLIC_STRATEGY_KEYWORD_SENTENCES = "keyword_sentences"
# SYMBOLIC_STRATEGY_SEMANTIC_CHUNKS = "semantic_chunks"

DEFAULT_ENCODING = 'utf-8'

# --- Logger Setup ---
logger = logging.getLogger("VoxSigilRAG.ProductionCompression")
# Example basic logging configuration, adjust as needed for your application
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RAGCompressionEngine:
    """
    Production-Grade VoxSigil-Aware RAG Compression Engine.

    Features:
    - Multiple compression algorithms (zlib, lzma, bz2, passthrough).
    - Configurable symbolic digest strategies.
    - Metadata-wrapped compressed output for reliable decompression.
    - Adjustable compression levels.
    - Comprehensive error handling and logging.
    - Extensible design for new algorithms and features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = self._get_default_config()
        if config:
            self.update_config(config) # Apply user config over defaults
        self._init_metrics()
        logger.info(f"RAGCompressionEngine initialized with config: {self.config}")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "default_mode": MODE_ZLIB,
            "zlib_level": 9,  # 0-9 (9 is max compression, slowest)
            "lzma_preset": None, # 0-9, or None for default. Higher is more compression.
                                # LZMA can also take `format` and `filters` if needed.
            "bz2_compresslevel": 9, # 1-9 (9 is max compression)
            "symbolic_digest_strategy": SYMBOLIC_STRATEGY_FIRST_LINES,
            "symbolic_preserve_ratio": 0.3,  # For line-based strategy
            "symbolic_preserve_chars": 512,  # For char-based strategy
            "min_entropy_for_compression": 1.5, # Skip compression for low entropy data
            "min_length_for_compression": 64, # Skip compression for very short strings
            "encoding": DEFAULT_ENCODING,
            "store_original_length": True, # Store original length in metadata
            "error_on_decompression_failure": False # If True, raises exception, else returns placeholder
        }

    def _init_metrics(self) -> None:
        self.metrics = {
            "compress_requests": 0,
            "decompress_requests": 0,
            "successful_compressions": 0,
            "successful_decompressions": 0,
            "bytes_original_total": 0,
            "bytes_compressed_total": 0,
            "avg_compression_ratio": 0.0,
            "last_op_details": None, # To store ratio, entropy, mode of the last operation
            "errors": {
                "compression": 0,
                "decompression": 0
            },
            "skipped_compressions_entropy": 0,
            "skipped_compressions_length": 0,
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Updates the engine's configuration."""
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Config '{key}' updated to: {value}")
            else:
                logger.warning(f"Attempted to update unknown config key: {key}")

    def _estimate_entropy(self, text: str) -> float:
        """Estimates Shannon entropy of the text."""
        if not text:
            return 0.0
        
        byte_array = text.encode(self.config["encoding"])
        freq = {}
        for byte_val in byte_array:
            freq[byte_val] = freq.get(byte_val, 0) + 1
        
        total_bytes = len(byte_array)
        entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) 
                       for count in freq.values() if count > 0)
        return round(entropy, 5)

    def _wrap_with_metadata(self, compressed_data: Union[str, bytes], mode: str, original_length: int) -> str:
        """Wraps compressed data with metadata for reliable decompression."""
        if isinstance(compressed_data, bytes):
            # Encode binary data to string (e.g., base64) to ensure JSON compatibility
            compressed_data_str = base64.b64encode(compressed_data).decode(DEFAULT_ENCODING)
        else:
            compressed_data_str = compressed_data

        metadata = {
            "mode_used": mode,
            "compressed_payload": compressed_data_str
        }
        if self.config["store_original_length"]:
            metadata["original_length"] = original_length
        
        return json.dumps(metadata)

    def _unwrap_metadata(self, wrapped_data_str: str) -> Optional[Tuple[str, bytes, Optional[int]]]:
        """Extracts mode, compressed payload, and original length from metadata."""
        try:
            metadata = json.loads(wrapped_data_str)
            mode = metadata.get("mode_used")
            payload_str = metadata.get("compressed_payload")
            original_length = metadata.get("original_length")

            if not mode or payload_str is None: # payload can be empty string for symbolic digest
                logger.error("Metadata unwrapping failed: 'mode_used' or 'compressed_payload' missing.")
                return None

            # Decode payload if it was base64 encoded (for binary compression outputs)
            if mode in [MODE_ZLIB, MODE_LZMA, MODE_BZ2]:
                 payload_bytes = base64.b64decode(payload_str.encode(DEFAULT_ENCODING))
            else: # For symbolic digest or passthrough, payload might be string directly
                 payload_bytes = payload_str.encode(self.config["encoding"]) # Assuming it was stored as string

            return mode, payload_bytes, original_length
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError during metadata unwrapping: {e}")
            return None
        except binascii.Error as e: # For base64 decoding errors
            logger.error(f"Base64 decoding error during metadata unwrapping: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during metadata unwrapping: {e}")
            return None


    def compress(self, text: str, mode: Optional[str] = None) -> Optional[str]:
        """
        Compresses the input text using the specified or default mode.
        Returns a JSON string containing metadata and the compressed payload, or None on error.
        """
        self.metrics["compress_requests"] += 1
        if not text:
            logger.warning("Compression attempt on empty text. Returning empty.")
            # Or handle as an error / specific passthrough if needed
            return self._wrap_with_metadata("", MODE_PASSTHROUGH, 0)


        original_length = len(text.encode(self.config["encoding"]))
        
        # Check min length for compression
        if original_length < self.config["min_length_for_compression"]:
            logger.info(f"Text length ({original_length} bytes) is less than min_length_for_compression "
                        f"({self.config['min_length_for_compression']}). Skipping compression (passthrough).")
            self.metrics["skipped_compressions_length"] += 1
            self.metrics["successful_compressions"] += 1 # Still counts as a successful operation
            self.metrics["bytes_original_total"] += original_length
            self.metrics["bytes_compressed_total"] += original_length # No change
            self._update_avg_ratio(1.0)
            op_details = {"ratio": 1.0, "entropy": self._estimate_entropy(text), "mode": MODE_PASSTHROUGH, "original_length": original_length, "compressed_length": original_length}
            self.metrics["last_op_details"] = op_details
            return self._wrap_with_metadata(text, MODE_PASSTHROUGH, original_length)

        entropy = self._estimate_entropy(text)

        # Check min entropy for compression (except for symbolic digest or passthrough)
        current_mode = mode or self.config["default_mode"]
        if current_mode not in [MODE_SYMBOLIC_DIGEST, MODE_PASSTHROUGH] and \
           entropy < self.config["min_entropy_for_compression"]:
            logger.info(f"Text entropy ({entropy:.3f}) is less than min_entropy_for_compression "
                        f"({self.config['min_entropy_for_compression']}). Switching to passthrough.")
            self.metrics["skipped_compressions_entropy"] += 1
            current_mode = MODE_PASSTHROUGH # Override mode to passthrough

        compressed_payload: Optional[Union[bytes, str]] = None
        try:
            if current_mode == MODE_ZLIB:
                compressed_payload = zlib.compress(text.encode(self.config["encoding"]), 
                                                   level=self.config["zlib_level"])
            elif current_mode == MODE_LZMA:
                compressed_payload = lzma.compress(text.encode(self.config["encoding"]),
                                                   preset=self.config["lzma_preset"])
            elif current_mode == MODE_BZ2:
                compressed_payload = bz2.compress(text.encode(self.config["encoding"]),
                                                  compresslevel=self.config["bz2_compresslevel"])
            elif current_mode == MODE_SYMBOLIC_DIGEST:
                compressed_payload = self._compress_symbolic_digest(text) # Returns str
            elif current_mode == MODE_PASSTHROUGH:
                compressed_payload = text # Store as str
            else:
                logger.error(f"Unsupported compression mode: {current_mode}")
                self.metrics["errors"]["compression"] += 1
                return None
            
            if compressed_payload is None: # Should not happen if logic is correct
                logger.error(f"Compression payload is None for mode {current_mode}")
                self.metrics["errors"]["compression"] += 1
                return None

            # For binary outputs, use their length. For string outputs (symbolic, passthrough), encode to measure.
            compressed_length = len(compressed_payload) if isinstance(compressed_payload, bytes) else len(compressed_payload.encode(self.config["encoding"]))
            ratio = compressed_length / original_length if original_length else 1.0

            wrapped_output = self._wrap_with_metadata(compressed_payload, current_mode, original_length)
            
            self.metrics["successful_compressions"] += 1
            self.metrics["bytes_original_total"] += original_length
            self.metrics["bytes_compressed_total"] += compressed_length
            self._update_avg_ratio(ratio)
            
            op_details = {"ratio": round(ratio, 5), "entropy": entropy, "mode": current_mode, "original_length": original_length, "compressed_length": compressed_length}
            self.metrics["last_op_details"] = op_details
            logger.debug(f"Mode '{current_mode}': Compressed from {original_length} to {compressed_length} bytes. Ratio: {ratio:.3f}, Entropy: {entropy:.3f}")
            return wrapped_output

        except Exception as e:
            logger.error(f"Error during compression with mode '{current_mode}': {e}", exc_info=True)
            self.metrics["errors"]["compression"] += 1
            return None

    def _compress_symbolic_digest(self, text: str) -> str:
        """Implements symbolic digest strategies."""
        strategy = self.config["symbolic_digest_strategy"]
        
        if strategy == SYMBOLIC_STRATEGY_FIRST_LINES:
            lines = text.splitlines()
            preserve_ratio = self.config["symbolic_preserve_ratio"]
            num_lines_to_keep = max(1, int(len(lines) * preserve_ratio)) if lines else 0
            digest_lines = lines[:num_lines_to_keep]
            # Store as a JSON array of lines to distinguish from simple string
            return json.dumps(digest_lines, ensure_ascii=False) 
        
        elif strategy == SYMBOLIC_STRATEGY_FIRST_CHARS:
            preserve_chars = self.config["symbolic_preserve_chars"]
            # Simply return the substring, will be wrapped by _wrap_with_metadata
            return text[:preserve_chars] 
            
        # Add other strategies here:
        # elif strategy == SYMBOLIC_STRATEGY_KEYWORD_SENTENCES:
        #     # Requires NLP - placeholder for future extension
        #     logger.warning("SYMBOLIC_STRATEGY_KEYWORD_SENTENCES not yet implemented, using first_chars.")
        #     return text[:self.config["symbolic_preserve_chars"]]
            
        else:
            logger.warning(f"Unknown symbolic digest strategy: {strategy}. Defaulting to first chars.")
            return text[:self.config["symbolic_preserve_chars"]]


    def decompress(self, wrapped_data_str: str) -> Optional[str]:
        """
        Decompresses data that was compressed and wrapped by this engine.
        Returns the original text, or a placeholder/None on error.
        """
        self.metrics["decompress_requests"] += 1
        if not wrapped_data_str:
            logger.warning("Decompression attempt on empty wrapped data. Returning empty string.")
            return "" # Or None, depending on desired behavior for empty input

        unwrapped = self._unwrap_metadata(wrapped_data_str)
        if unwrapped is None:
            self.metrics["errors"]["decompression"] += 1
            return self._handle_decompression_error("Metadata unwrapping failed.")

        mode, compressed_payload_bytes, original_length = unwrapped
        decompressed_text: Optional[str] = None

        try:
            if mode == MODE_ZLIB:
                decompressed_text = zlib.decompress(compressed_payload_bytes).decode(self.config["encoding"])
            elif mode == MODE_LZMA:
                decompressed_text = lzma.decompress(compressed_payload_bytes).decode(self.config["encoding"])
            elif mode == MODE_BZ2:
                decompressed_text = bz2.decompress(compressed_payload_bytes).decode(self.config["encoding"])
            elif mode == MODE_SYMBOLIC_DIGEST:
                # Payload for symbolic digest might be JSON array of lines or just a string (for first_chars)
                try:
                    # Attempt to parse as JSON list of lines first
                    digest_lines_or_str = json.loads(compressed_payload_bytes.decode(self.config["encoding"]))
                    if isinstance(digest_lines_or_str, list):
                        decompressed_text = "\n".join(digest_lines_or_str)
                    else: # Assumed it was a string (e.g. first_chars)
                        decompressed_text = digest_lines_or_str
                except json.JSONDecodeError: 
                    # If not valid JSON, assume it was a simple string payload (e.g., first_chars)
                    decompressed_text = compressed_payload_bytes.decode(self.config["encoding"])
            elif mode == MODE_PASSTHROUGH:
                decompressed_text = compressed_payload_bytes.decode(self.config["encoding"])
            else:
                self.metrics["errors"]["decompression"] += 1
                return self._handle_decompression_error(f"Unsupported decompression mode found in metadata: {mode}")

            self.metrics["successful_decompressions"] += 1
            logger.debug(f"Successfully decompressed data using mode '{mode}'.")
            return decompressed_text

        except Exception as e:
            self.metrics["errors"]["decompression"] += 1
            return self._handle_decompression_error(f"Error during decompression with mode '{mode}': {e}", exc_info=True)

    def _handle_decompression_error(self, message: str, exc_info=False) -> Optional[str]:
        logger.error(message, exc_info=exc_info)
        if self.config["error_on_decompression_failure"]:
            raise RAGCompressionError(message) # Consider defining a custom exception
        return "<decompression_failed: see logs>"


    def _update_avg_ratio(self, current_ratio: float) -> None:
        """Helper to update the average compression ratio."""
        if self.metrics["successful_compressions"] > 0:
            # More numerically stable way to calculate running average
            self.metrics["avg_compression_ratio"] = (
                self.metrics["avg_compression_ratio"] * (self.metrics["successful_compressions"] -1) + current_ratio
            ) / self.metrics["successful_compressions"]
            self.metrics["avg_compression_ratio"] = round(self.metrics["avg_compression_ratio"], 5)
        else: # Should not happen if called after successful_compressions increment
            self.metrics["avg_compression_ratio"] = current_ratio


    def get_stats(self) -> Dict[str, Any]:
        """Returns a copy of the current compression statistics."""
        stats = self.metrics.copy()
        # Calculate overall ratio if totals are available
        if stats["bytes_original_total"] > 0:
            stats["overall_effective_ratio"] = round(
                stats["bytes_compressed_total"] / stats["bytes_original_total"], 5
            )
        else:
            stats["overall_effective_ratio"] = 0.0
        return stats

    def reset_metrics(self) -> None:
        """Resets all collected metrics to their initial state."""
        self._init_metrics()
        logger.info("Compression metrics have been reset.")

# Custom Exception
class RAGCompressionError(Exception):
    pass

# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for example
    logger.setLevel(logging.DEBUG)

    engine = RAGCompressionEngine(config={
        "min_entropy_for_compression": 0.5, # Lower for testing small strings
        "min_length_for_compression": 10
    })

    test_texts = [
        "This is a test string. It has some repetition, repetition, repetition.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", # Low entropy
        "Short", # Too short
        "This is another slightly longer test string with reasonable entropy for its size.",
        "A symbolic digest\nwill take the first few lines\nbased on the ratio.\nThis line might be cut.\nThis one too.",
        json.dumps({"complex_object": True, "data": [1,2,3], "text": "nested field"}), # JSON string
        "" # Empty string
    ]

    for i, text in enumerate(test_texts):
        print(f"\n--- Original Text {i+1} ({len(text)} chars) ---")
        print(text)
        
        print("\n--- Compressing with ZLIB (default) ---")
        wrapped_zlib = engine.compress(text)
        if wrapped_zlib:
            print(f"Wrapped ZLIB: {wrapped_zlib[:100]}... (len: {len(wrapped_zlib)})")
            decompressed_zlib = engine.decompress(wrapped_zlib)
            print(f"Decompressed ZLIB: {decompressed_zlib}")
            assert decompressed_zlib == text, f"ZLIB Decompression mismatch for text {i+1}!"

        engine.update_config({"default_mode": MODE_LZMA})
        print("\n--- Compressing with LZMA ---")
        wrapped_lzma = engine.compress(text)
        if wrapped_lzma:
            print(f"Wrapped LZMA: {wrapped_lzma[:100]}... (len: {len(wrapped_lzma)})")
            decompressed_lzma = engine.decompress(wrapped_lzma)
            print(f"Decompressed LZMA: {decompressed_lzma}")
            assert decompressed_lzma == text, f"LZMA Decompression mismatch for text {i+1}!"

        engine.update_config({"default_mode": MODE_SYMBOLIC_DIGEST, "symbolic_preserve_ratio": 0.6})
        print("\n--- Compressing with Symbolic Digest (first_lines, ratio 0.6) ---")
        wrapped_symb_lines = engine.compress(text, mode=MODE_SYMBOLIC_DIGEST) # Explicit mode override
        if wrapped_symb_lines:
            print(f"Wrapped Symbolic (lines): {wrapped_symb_lines}")
            decompressed_symb_lines = engine.decompress(wrapped_symb_lines)
            print(f"Decompressed Symbolic (lines): {decompressed_symb_lines}")
            # Note: Symbolic digest is lossy, so direct assert text == decompressed_text is not appropriate
            # Instead, check if the decompressed version is what you'd expect from the digest

        engine.update_config({"symbolic_digest_strategy": SYMBOLIC_STRATEGY_FIRST_CHARS, "symbolic_preserve_chars": 20})
        print("\n--- Compressing with Symbolic Digest (first_chars, 20) ---")
        wrapped_symb_chars = engine.compress(text, mode=MODE_SYMBOLIC_DIGEST)
        if wrapped_symb_chars:
            print(f"Wrapped Symbolic (chars): {wrapped_symb_chars}")
            decompressed_symb_chars = engine.decompress(wrapped_symb_chars)
            print(f"Decompressed Symbolic (chars): {decompressed_symb_chars}")

        engine.update_config({"default_mode": MODE_ZLIB}) # Reset for next iteration

    print("\n\n--- Final Engine Stats ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    engine.reset_metrics()
    print("\n--- Metrics after reset ---")
    print(engine.get_stats())

    # Test decompression failure handling
    print("\n--- Testing Decompression Failure ---")
    engine.update_config({"error_on_decompression_failure": False})
    bad_data_1 = "this is not json"
    print(f"Decompressing bad_data_1: {engine.decompress(bad_data_1)}")

    bad_data_2 = json.dumps({"mode_used": "unknown_mode", "compressed_payload": "data"})
    print(f"Decompressing bad_data_2: {engine.decompress(bad_data_2)}")
    
    bad_data_3 = json.dumps({"mode_used": MODE_ZLIB, "compressed_payload": "not_base64_or_hex"})
    print(f"Decompressing bad_data_3: {engine.decompress(bad_data_3)}")

    try:
        engine.update_config({"error_on_decompression_failure": True})
        engine.decompress(bad_data_1)
    except Exception as e: # RAGCompressionError once defined properly
        print(f"Caught expected exception: {e}")