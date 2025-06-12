from __future__ import annotations

"""Base interface for BLT encoder implementations."""

from abc import ABC, abstractmethod
from .specialized_interfaces import BLTInterface


class BaseBLTEncoder(BLTInterface, ABC):
    """Interface defining core methods for BLT encoder components."""

    @abstractmethod
    async def encode(self, text: str) -> list[float]:
        """Encode text into an embedding vector."""

    @abstractmethod
    def get_status(self) -> dict:
        """Return health or configuration status."""
