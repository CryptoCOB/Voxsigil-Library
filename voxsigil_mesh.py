"""Lightweight symbolic cognition mesh for VoxSigil."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple

from VoxSigilRag.voxsigil_rag_compression import RAGCompressionEngine


class VoxSigilMesh:
    """Minimal message mesh that compresses all transmissions."""

    def __init__(self, gui_hook: Callable[[str], None] | None = None) -> None:
        self.nodes: Dict[str, Any] = {}
        self.history: List[Tuple[str, str | None]] = []
        self.gui_hook = gui_hook
        self.blt = RAGCompressionEngine()

    def register(self, name: str, node: Any) -> None:
        self.nodes[name] = node

    def transmit(self, sender: str, message: str) -> None:
        compressed = self.blt.compress(message)
        self.history.append((sender, compressed))
        if self.gui_hook:
            try:
                self.gui_hook(compressed)
            except Exception:
                pass
        for name, node in self.nodes.items():
            if name == sender:
                continue
            if hasattr(node, "receive"):
                try:
                    node.receive(compressed)
                except Exception:
                    pass

    # --- Convenience methods -------------------------------------------------

    def broadcast(self, sender: str, message: str) -> None:
        """Alias for transmit for backwards compatibility."""
        self.transmit(sender, message)

    def test_broadcast(self, msg: str = "⚡ TEST_SIGNAL") -> None:
        """Emit a test signal to all registered nodes."""
        self.broadcast("Tester", msg)
