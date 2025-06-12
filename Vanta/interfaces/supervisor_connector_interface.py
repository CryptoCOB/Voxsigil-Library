from __future__ import annotations

"""Supervisor connector interface for Vanta components."""

from abc import ABC, abstractmethod


class BaseSupervisorConnector(ABC):
    """Abstract base class for supervisor connectors."""

    @abstractmethod
    def get_sigil_content_as_dict(self, sigil_ref: str) -> dict | None:
        """Retrieve sigil content as a dictionary."""

    @abstractmethod
    def create_sigil(self, sigil_ref: str, content: dict, sigil_type: str) -> None:
        """Create or update a sigil entry."""
