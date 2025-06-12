from __future__ import annotations

"""Interface for hybrid middleware implementations used in Vanta."""

from abc import ABC, abstractmethod
from .specialized_interfaces import MiddlewareInterface


class BaseHybridMiddleware(MiddlewareInterface, ABC):
    """Base hybrid middleware contract."""

    @abstractmethod
    def get_middleware_capabilities(self) -> list[str]:
        """Return a list of supported capability identifiers."""

    @abstractmethod
    def configure_middleware(self, config: dict) -> bool:
        """Configure middleware with the provided settings."""
