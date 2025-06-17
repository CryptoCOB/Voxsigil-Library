"""
Sleep Time Computation for Vanta Core
====================================

Provides utilities for computing sleep and backoff times for various operations.
"""

import logging
import random
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SleepTimeComputer:
    """Computes appropriate sleep times for operations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sleep time computer

        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.base_delay = self.config.get("base_delay", 1.0)
        self.max_delay = self.config.get("max_delay", 60.0)
        self.jitter = self.config.get("jitter", True)
        self.backoff_factor = self.config.get("backoff_factor", 2.0)

    def compute_backoff(self, attempt: int) -> float:
        """Compute backoff time for retry attempts

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Sleep time in seconds
        """
        delay = min(self.base_delay * (self.backoff_factor**attempt), self.max_delay)

        if self.jitter:
            # Add jitter of ±15%
            jitter_factor = 1.0 + random.uniform(-0.15, 0.15)
            delay *= jitter_factor

        return delay

    def compute_adaptive_sleep(self, load_factor: float) -> float:
        """Compute adaptive sleep time based on system load

        Args:
            load_factor: System load factor (0.0 to 1.0)

        Returns:
            Sleep time in seconds
        """
        # Higher load = longer sleep
        base_sleep = self.config.get("adaptive_base_sleep", 0.1)
        max_sleep = self.config.get("adaptive_max_sleep", 5.0)

        # Ensure load factor is between 0 and 1
        load_factor = max(0.0, min(1.0, load_factor))

        # Compute sleep time
        sleep_time = base_sleep + (max_sleep - base_sleep) * load_factor

        if self.jitter:
            # Add smaller jitter of ±5%
            jitter_factor = 1.0 + random.uniform(-0.05, 0.05)
            sleep_time *= jitter_factor

        return sleep_time

    def get_throttled_sleep(self, rate_limit: float) -> float:
        """Get sleep time for rate limiting

        Args:
            rate_limit: Target rate limit in operations per second

        Returns:
            Sleep time in seconds
        """
        if rate_limit <= 0:
            return 0.0

        base_sleep = 1.0 / rate_limit

        if self.jitter:
            # Add small jitter of ±5%
            jitter_factor = 1.0 + random.uniform(-0.05, 0.05)
            base_sleep *= jitter_factor

        return base_sleep


# Default instance
default_sleep_computer = SleepTimeComputer()


def get_sleep_computer(config: Optional[Dict[str, Any]] = None) -> SleepTimeComputer:
    """Get a sleep time computer instance

    Args:
        config: Configuration options

    Returns:
        SleepTimeComputer instance
    """
    return SleepTimeComputer(config=config)


def sleep_with_backoff(attempt: int) -> None:
    """Sleep with exponential backoff

    Args:
        attempt: Current attempt number (0-based)
    """
    sleep_time = default_sleep_computer.compute_backoff(attempt)
    logger.debug(f"Sleeping for {sleep_time:.2f}s (attempt {attempt + 1})")
    time.sleep(sleep_time)


class CognitiveState:
    """Represents the cognitive state for sleep time computation"""

    def __init__(self, load=0.0, complexity=0.0, fatigue=0.0):
        self.cognitive_load = load
        self.task_complexity = complexity
        self.fatigue_level = fatigue

    def update(self, load=None, complexity=None, fatigue=None):
        if load is not None:
            self.cognitive_load = load
        if complexity is not None:
            self.task_complexity = complexity
        if fatigue is not None:
            self.fatigue_level = fatigue

    def get_sleep_recommendation(self):
        """Calculate recommended sleep time based on cognitive state"""
        base_sleep = 0.1  # Base sleep time
        load_factor = self.cognitive_load * 0.05
        complexity_factor = self.task_complexity * 0.03
        fatigue_factor = self.fatigue_level * 0.02
        return base_sleep + load_factor + complexity_factor + fatigue_factor


# Alias for backward compatibility - some code expects SleepTimeCompute
SleepTimeCompute = SleepTimeComputer


# Alternative factory function for compatibility
def create_sleep_time_compute(config=None):
    """Create SleepTimeCompute instance for backward compatibility."""
    return SleepTimeComputer(config=config)
