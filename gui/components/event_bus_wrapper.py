"""
Event bus wrapper for GUI components
This module provides wrappers to ensure all GUI components have proper event bus support
"""

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def with_event_bus(tab_class: Callable, event_bus: Any) -> Any:
    """
    Create a wrapper to inject event bus into a tab that doesn't accept it natively
    """
    try:
        # Try direct instantiation with event_bus
        return tab_class(event_bus=event_bus)
    except TypeError:
        # If it doesn't accept event_bus, create an instance without it
        instance = tab_class()

        # Try to add the event_bus to the instance
        if hasattr(instance, "set_event_bus"):
            instance.set_event_bus(event_bus)
        else:
            # Add event_bus directly to the instance
            instance.event_bus = event_bus
            logger.info(f"Injected event_bus into {tab_class.__name__}")

            # Add a simple set_event_bus method to the instance
            def set_event_bus(new_event_bus):
                instance.event_bus = new_event_bus

            instance.set_event_bus = set_event_bus.__get__(instance, type(instance))

            # Patch instance to expose subscribe method
            if not hasattr(instance, "subscribe_to_event"):

                def subscribe_to_event(event_type, callback):
                    if instance.event_bus and hasattr(instance.event_bus, "subscribe"):
                        instance.event_bus.subscribe(event_type, callback)
                        return True
                    return False

                instance.subscribe_to_event = subscribe_to_event.__get__(
                    instance, type(instance)
                )

        return instance


# Apply wrappers to all tab classes that need event bus support
def apply_event_bus_wrapper(module_path: str, class_name: str, event_bus: Any) -> Any:
    """Load a module and class, then apply event_bus wrapper"""
    try:
        module = importlib.import_module(module_path)
        tab_class = getattr(module, class_name)
        return with_event_bus(tab_class, event_bus)
    except Exception as e:
        logger.error(
            f"Failed to create wrapped instance of {module_path}.{class_name}: {e}"
        )
        return None


# Specific wrappers for the problematic tabs
def create_enhanced_training_tab(event_bus=None):
    """Create an EnhancedTrainingTab with event bus support"""
    return apply_event_bus_wrapper(
        "gui.components.enhanced_training_tab", "EnhancedTrainingTab", event_bus
    )


def create_enhanced_visualization_tab(event_bus=None):
    """Create an EnhancedVisualizationTab with event bus support"""
    return apply_event_bus_wrapper(
        "gui.components.enhanced_visualization_tab",
        "EnhancedVisualizationTab",
        event_bus,
    )


def create_enhanced_novel_reasoning_tab(event_bus=None):
    """Create an EnhancedNovelReasoningTab with event bus support"""
    return apply_event_bus_wrapper(
        "gui.components.enhanced_novel_reasoning_tab",
        "EnhancedNovelReasoningTab",
        event_bus,
    )


def create_enhanced_neural_tts_tab(event_bus=None):
    """Create an EnhancedNeuralTTSTab with event bus support"""
    return apply_event_bus_wrapper(
        "gui.components.enhanced_neural_tts_tab", "EnhancedNeuralTTSTab", event_bus
    )
