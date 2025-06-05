#!/usr/bin/env python
"""
ARC Utilities Module (arc_utils.py)

Provides utility functions and shared resources for ARC LLM components,
including caching, common data structures, and helper functions.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Logger Setup ---
logger = logging.getLogger("ARC.Utils")

# --- Global LLM Response Cache ---
# Thread-safe dictionary for caching LLM responses
# Key: cache key (hash), Value: (timestamp, response_data)
LLM_RESPONSE_CACHE: Dict[str, Tuple[float, Any]] = {}

# Cache configuration
CACHE_TTL_SECONDS = int(os.getenv("ARC_LLM_CACHE_TTL", "3600"))  # 1 hour default
MAX_CACHE_SIZE = int(os.getenv("ARC_LLM_CACHE_MAX_SIZE", "1000"))


def get_cache_key(content: str, model_id: str, temperature: float, service: str) -> str:
    """
    Generate a cache key for LLM responses.

    Args:
        content: The prompt/content being cached
        model_id: Model identifier
        temperature: Temperature parameter
        service: Service name (ollama, lmstudio, etc.)

    Returns:
        Hexadecimal cache key
    """
    key_string = (
        f"model:{model_id}|svc:{service}|temp:{temperature:.2f}|content:{content}"
    )
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


def cache_response(cache_key: str, response_data: Any) -> None:
    """
    Cache an LLM response with timestamp.

    Args:
        cache_key: Unique cache key
        response_data: Response data to cache
    """
    global LLM_RESPONSE_CACHE

    # Clean up expired entries if cache is getting large
    if len(LLM_RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        cleanup_expired_cache_entries()

    # Remove oldest entries if still at max size
    if len(LLM_RESPONSE_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        oldest_key = min(
            LLM_RESPONSE_CACHE.keys(), key=lambda k: LLM_RESPONSE_CACHE[k][0]
        )
        del LLM_RESPONSE_CACHE[oldest_key]
        logger.debug(f"Removed oldest cache entry: {oldest_key[:8]}...")

    LLM_RESPONSE_CACHE[cache_key] = (time.time(), response_data)
    logger.debug(f"Cached response with key: {cache_key[:8]}...")


def get_cached_response(cache_key: str) -> Optional[Any]:
    """
    Retrieve a cached response if it exists and is not expired.

    Args:
        cache_key: Cache key to lookup

    Returns:
        Cached response data or None if not found/expired
    """
    global LLM_RESPONSE_CACHE

    if cache_key not in LLM_RESPONSE_CACHE:
        return None

    timestamp, response_data = LLM_RESPONSE_CACHE[cache_key]

    # Check if expired
    if time.time() - timestamp > CACHE_TTL_SECONDS:
        del LLM_RESPONSE_CACHE[cache_key]
        logger.debug(f"Removed expired cache entry: {cache_key[:8]}...")
        return None

    logger.debug(f"Cache hit for key: {cache_key[:8]}...")
    return response_data


def cleanup_expired_cache_entries() -> int:
    """
    Remove expired entries from the cache.

    Returns:
        Number of entries removed
    """
    global LLM_RESPONSE_CACHE

    current_time = time.time()
    expired_keys = [
        key
        for key, (timestamp, _) in LLM_RESPONSE_CACHE.items()
        if current_time - timestamp > CACHE_TTL_SECONDS
    ]

    for key in expired_keys:
        del LLM_RESPONSE_CACHE[key]

    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    return len(expired_keys)


def clear_cache() -> None:
    """Clear all cached responses."""
    global LLM_RESPONSE_CACHE
    LLM_RESPONSE_CACHE.clear()
    logger.info("LLM response cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    global LLM_RESPONSE_CACHE

    current_time = time.time()
    active_entries = 0
    expired_entries = 0

    for timestamp, _ in LLM_RESPONSE_CACHE.values():
        if current_time - timestamp > CACHE_TTL_SECONDS:
            expired_entries += 1
        else:
            active_entries += 1

    return {
        "total_entries": len(LLM_RESPONSE_CACHE),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "max_cache_size": MAX_CACHE_SIZE,
    }


# --- ARC Grid Utilities ---
def validate_arc_grid(grid: Any) -> bool:
    """
    Validate that a grid conforms to ARC format.

    Args:
        grid: Grid data to validate

    Returns:
        True if valid ARC grid, False otherwise
    """
    if not isinstance(grid, list):
        return False

    if not grid:  # Empty grid
        return False

    # Check that all rows are lists
    if not all(isinstance(row, list) for row in grid):
        return False

    # Check that all rows have same length
    if len(set(len(row) for row in grid)) != 1:
        return False

    # Check that all values are integers in valid range (0-9 for ARC)
    for row in grid:
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False

    return True


def grid_to_string(grid: list) -> str:
    """
    Convert an ARC grid to a string representation.

    Args:
        grid: ARC grid as list of lists

    Returns:
        String representation of the grid
    """
    if not validate_arc_grid(grid):
        return "Invalid grid"

    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def string_to_grid(grid_str: str) -> Optional[list]:
    """
    Parse a string representation back to an ARC grid.

    Args:
        grid_str: String representation of grid

    Returns:
        ARC grid as list of lists, or None if invalid
    """
    try:
        lines = grid_str.strip().split("\n")
        grid = []

        for line in lines:
            if not line.strip():
                continue
            row = [int(char) for char in line.strip()]
            grid.append(row)

        if validate_arc_grid(grid):
            return grid
        else:
            return None
    except (ValueError, TypeError):
        return None


# --- File and Path Utilities ---
def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename by removing/replacing invalid characters.

    Args:
        name: Original name

    Returns:
        Safe filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    safe_name = name
    for char in invalid_chars:
        safe_name = safe_name.replace(char, "_")

    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(" .")

    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


# --- Token Counting Utilities ---
def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 characters per token for English text
    # This is a rough approximation and should be replaced with proper tokenization
    return max(1, len(text) // 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum token count

    Returns:
        Truncated text
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    # Rough truncation based on character estimate
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    # Truncate and add ellipsis
    return text[: max_chars - 3] + "..."


# --- Error Handling Utilities ---
class ARCError(Exception):
    """Base exception for ARC-related errors."""

    pass


class ARCValidationError(ARCError):
    """Exception for validation errors."""

    pass


class ARCCacheError(ARCError):
    """Exception for cache-related errors."""

    pass


# --- Initialization ---
logger.info("ARC utilities module initialized")
