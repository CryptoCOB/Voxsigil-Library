from functools import wraps
from typing import Any, Callable

try:
    from VoxSigilRag.voxsigil_rag_compression import RAGCompressionEngine
    _compressor = RAGCompressionEngine()
except Exception:  # pragma: no cover - optional dependency
    _compressor = None


def compress_outbound(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to compress outbound text messages using BLT."""

    @wraps(func)
    def wrapper(self, message: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(message, str) and _compressor is not None:
            try:
                message = _compressor.compress(message) or message
            except Exception:
                pass
        return func(self, message, *args, **kwargs)

    return wrapper
