"""
Retry Policy for VoxSigil Supervisor
===================================

Provides retry policies for handling failures in task execution.
"""

import logging
import random
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RetryPolicy:
    """Base class for retry policies"""

    def __init__(self, max_retries: int = 3):
        """Initialize a retry policy

        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries

    def should_retry(self, attempt: int, error: Optional[Exception] = None) -> bool:
        """Determine if a retry should be attempted

        Args:
            attempt: Current attempt number (0-based)
            error: The error that occurred, if any

        Returns:
            True if retry should be attempted, False otherwise
        """
        _ = error  # mark as used to satisfy linters
        return attempt < self.max_retries

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        _ = attempt  # mark as used to satisfy linters
        return 0.0


class ExponentialBackoffPolicy(RetryPolicy):
    """Retry policy with exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        """Initialize an exponential backoff policy

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add jitter to delay
        """
        super().__init__(max_retries)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Get delay with exponential backoff

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (2**attempt), self.max_delay)

        if self.jitter:
            # Add jitter of 15%
            jitter_factor = 1.0 + random.uniform(-0.15, 0.15)
            delay *= jitter_factor

        return delay


# Default retry policy
default_retry_policy = ExponentialBackoffPolicy()


def retry_with_policy(
    func: Callable[..., Any],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with retry policy

    Args:
        func: Function to execute
        *args: Positional arguments for func
        policy: Retry policy to use
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        Exception: The last exception if all retries fail
    """
    if policy is None:
        policy = default_retry_policy

    last_error = None
    attempt = 0

    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not policy.should_retry(attempt, error=e):
                break

            delay = policy.get_delay(attempt)
            logger.warning(f"Retry attempt {attempt + 1} after error: {e}. Waiting {delay:.2f}s")
            time.sleep(delay)
            attempt += 1

    if last_error:
        raise last_error
    return None
