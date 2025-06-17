"""Core modules for Vanta."""


# Use lazy imports to avoid circular dependencies
def _lazy_import_core():
    """Lazy import for core modules to avoid circular dependencies."""
    globals_ = {}

    try:
        from .UnifiedAgentRegistry import UnifiedAgentRegistry

        globals_["UnifiedAgentRegistry"] = UnifiedAgentRegistry
    except ImportError:
        pass

    try:
        from .UnifiedAsyncBus import UnifiedAsyncBus

        globals_["UnifiedAsyncBus"] = UnifiedAsyncBus
    except ImportError:
        pass

    try:
        from .UnifiedMemoryInterface import UnifiedMemoryInterface

        globals_["UnifiedMemoryInterface"] = UnifiedMemoryInterface
    except ImportError:
        pass

    try:
        from .UnifiedVantaCore import UnifiedVantaCore, get_vanta_core

        globals_["UnifiedVantaCore"] = UnifiedVantaCore
        globals_["get_vanta_core"] = get_vanta_core
    except ImportError:
        pass

    try:
        from .VantaBLTMiddleware import VantaMiddlewareSuite

        globals_["VantaMiddlewareSuite"] = VantaMiddlewareSuite
    except ImportError:
        pass

    try:
        from .VantaCognitiveEngine import VantaCognitiveEngine

        globals_["VantaCognitiveEngine"] = VantaCognitiveEngine
    except ImportError:
        pass

    try:
        from .VantaOrchestrationEngine import VantaOrchestrationEngine

        globals_["VantaOrchestrationEngine"] = VantaOrchestrationEngine
    except ImportError:
        pass

    return globals_


# Module-level getattr to handle lazy imports
def __getattr__(name):
    """Lazy import handler for core modules."""
    globals_dict = _lazy_import_core()
    if name in globals_dict:
        return globals_dict[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "UnifiedVantaCore",
    "get_vanta_core",
    "UnifiedAgentRegistry",
    "UnifiedAsyncBus",
    "UnifiedMemoryInterface",
    "VantaBLTMiddleware",
    "VantaCognitiveEngine",
    "VantaOrchestrationEngine",
]
