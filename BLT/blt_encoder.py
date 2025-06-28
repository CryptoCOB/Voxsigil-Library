"""Compatibility shim for older imports expecting ``blt_encoder`` module."""

# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
from BLT import BLTEncoder, ByteLatentTransformerEncoder, SigilPatchEncoder

__all__ = ["BLTEncoder", "ByteLatentTransformerEncoder", "SigilPatchEncoder"]
