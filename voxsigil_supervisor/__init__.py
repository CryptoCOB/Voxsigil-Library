# voxsigil_supervisor/__init__.py
"""
VoxSigil Supervisor Package.

This package contains the core logic for the VoxSigilSupervisor agent,
including its engine, interfaces to external systems (RAG, LLM, Memory),
and strategic components for reasoning and evaluation.
"""

__version__ = "0.1.0"

# Setup package-level logger
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())