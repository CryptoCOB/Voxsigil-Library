#!/usr/bin/env python3
"""
Active Flag Registry - Runtime flag management
===============================================

Manages runtime configuration flags and publishes changes to the event system.
Provides centralized flag storage and change notification.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Set

logger = logging.getLogger(__name__)


class ActiveFlagRegistry:
    """
    Centralized registry for runtime configuration flags.

    Manages flag values, change notifications, and persistence.
    """

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.flags = {}
        self.flag_metadata = {}
        self.change_listeners = {}
        self.lock = threading.RLock()

        # Initialize with default flags
        self.initialize_default_flags()

    def initialize_default_flags(self):
        """Initialize with default system flags"""
        default_flags = {
            # Core system flags
            "VANTA_DEBUG": {
                "value": False,
                "description": "Enable debug mode with verbose logging",
                "type": bool,
                "category": "core",
            },
            "VANTA_MUSIC": {
                "value": False,
                "description": "Enable music generation capabilities",
                "type": bool,
                "category": "features",
            },
            "VANTA_TRAINING": {
                "value": False,
                "description": "Enable training mode",
                "type": bool,
                "category": "training",
            },
            "VANTA_STREAMING": {
                "value": True,
                "description": "Enable live data streaming to GUI",
                "type": bool,
                "category": "gui",
            },
            "VANTA_COMPRESSION": {
                "value": True,
                "description": "Enable data compression",
                "type": bool,
                "category": "performance",
            },
            # GUI flags
            "GUI_VERBOSE": {
                "value": False,
                "description": "Verbose GUI logging",
                "type": bool,
                "category": "gui",
            },
            "GUI_AUTO_REFRESH": {
                "value": True,
                "description": "Auto-refresh GUI components",
                "type": bool,
                "category": "gui",
            },
            "GUI_DARK_THEME": {
                "value": True,
                "description": "Use dark theme",
                "type": bool,
                "category": "gui",
            },
            # Component flags
            "ARC_SOLVER_ACTIVE": {
                "value": True,
                "description": "ARC solver enabled",
                "type": bool,
                "category": "components",
            },
            "BLT_RAG_ACTIVE": {
                "value": True,
                "description": "BLT RAG system active",
                "type": bool,
                "category": "components",
            },
            "GRIDFORMER_ACTIVE": {
                "value": True,
                "description": "GridFormer system active",
                "type": bool,
                "category": "components",
            },
            # Performance flags
            "PERFORMANCE_MONITORING": {
                "value": True,
                "description": "Enable performance monitoring",
                "type": bool,
                "category": "performance",
            },
            "MEMORY_OPTIMIZATION": {
                "value": True,
                "description": "Enable memory optimization",
                "type": bool,
                "category": "performance",
            },
            # Development flags
            "DEV_MODE": {
                "value": False,
                "description": "Development mode with extra debugging",
                "type": bool,
                "category": "development",
            },
            "EXPERIMENTAL_FEATURES": {
                "value": False,
                "description": "Enable experimental features",
                "type": bool,
                "category": "development",
            },
        }

        with self.lock:
            for flag_name, metadata in default_flags.items():
                self.flags[flag_name] = metadata["value"]
                self.flag_metadata[flag_name] = {
                    "description": metadata["description"],
                    "type": metadata["type"],
                    "category": metadata["category"],
                    "created": time.time(),
                    "last_modified": time.time(),
                }

        logger.info(f"Initialized {len(default_flags)} default flags")

    def set_flag(self, name: str, value: Any, notify: bool = True) -> bool:
        """
        Set a flag value.

        Args:
            name: Flag name
            value: Flag value
            notify: Whether to notify listeners

        Returns:
            True if flag was set successfully
        """
        try:
            with self.lock:
                old_value = self.flags.get(name)

            # Determine expected type if metadata exists, otherwise assume current value type
            expected_type = (
                self.flag_metadata[name]["type"] if name in self.flag_metadata else type(value)
            )

            # Type validation / conversion
            if not isinstance(value, expected_type):
                # Attempt to convert the provided value to the expected type
                try:
                    if expected_type is bool:
                        value = (
                            bool(value)
                            if not isinstance(value, str)
                            else value.lower() in ["true", "1", "on", "yes"]
                        )
                    elif expected_type is int:
                        value = int(value)
                    elif expected_type is float:
                        value = float(value)
                    elif expected_type is str:
                        value = str(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Cannot convert flag {name} value {value} to type {expected_type}"
                    )
                    return False

                # Set the flag
                self.flags[name] = value

                # Update metadata
                if name not in self.flag_metadata:
                    self.flag_metadata[name] = {
                        "description": f"Runtime flag: {name}",
                        "type": type(value),
                        "category": "runtime",
                        "created": time.time(),
                        "last_modified": time.time(),
                    }
                else:
                    self.flag_metadata[name]["last_modified"] = time.time()

                # Notify if value changed
                if notify and old_value != value:
                    self._notify_change(name, value, old_value)

                logger.debug(f"Flag {name} set to {value}")
                return True

        except Exception as e:
            logger.error(f"Error setting flag {name}: {e}")
            return False

    def get_flag(self, name: str, default=None):
        """Get a flag value"""
        with self.lock:
            return self.flags.get(name, default)

    def has_flag(self, name: str) -> bool:
        """Check if a flag exists"""
        with self.lock:
            return name in self.flags

    def get_all_flags(self) -> Dict[str, Any]:
        """Get all flags as a dictionary"""
        with self.lock:
            return self.flags.copy()

    def get_flags_by_category(self, category: str) -> Dict[str, Any]:
        """Get all flags in a specific category"""
        with self.lock:
            result = {}
            for name, value in self.flags.items():
                if name in self.flag_metadata:
                    if self.flag_metadata[name].get("category") == category:
                        result[name] = value
            return result

    def get_flag_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a flag"""
        with self.lock:
            return self.flag_metadata.get(name, {}).copy()

    def remove_flag(self, name: str, notify: bool = True) -> bool:
        """Remove a flag"""
        try:
            with self.lock:
                if name in self.flags:
                    old_value = self.flags.pop(name)
                    self.flag_metadata.pop(name, None)

                    if notify:
                        self._notify_change(name, None, old_value)

                    logger.info(f"Removed flag {name}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error removing flag {name}: {e}")
            return False

    def add_change_listener(self, flag_name: str, callback: Callable[[str, Any, Any], None]):
        """Add a change listener for a specific flag"""
        with self.lock:
            if flag_name not in self.change_listeners:
                self.change_listeners[flag_name] = []
            self.change_listeners[flag_name].append(callback)

    def remove_change_listener(self, flag_name: str, callback: Callable):
        """Remove a change listener"""
        with self.lock:
            if flag_name in self.change_listeners:
                try:
                    self.change_listeners[flag_name].remove(callback)
                    if not self.change_listeners[flag_name]:
                        del self.change_listeners[flag_name]
                except ValueError:
                    pass

    def _notify_change(self, name: str, new_value: Any, old_value: Any):
        """Notify listeners of flag changes"""
        try:
            # Notify specific listeners
            if name in self.change_listeners:
                for callback in self.change_listeners[name]:
                    try:
                        callback(name, new_value, old_value)
                    except Exception as e:
                        logger.error(f"Error in flag change listener: {e}")

            # Publish to event bus
            if self.event_bus:
                self.event_bus.publish(
                    "flag.changed",
                    {
                        "flag": name,
                        "value": new_value,
                        "old_value": old_value,
                        "timestamp": time.time(),
                    },
                )

        except Exception as e:
            logger.error(f"Error notifying flag change: {e}")

    def export_flags(self) -> Dict[str, Any]:
        """Export all flags and metadata"""
        with self.lock:
            return {
                "flags": self.flags.copy(),
                "metadata": self.flag_metadata.copy(),
                "exported_at": time.time(),
            }

    def import_flags(self, data: Dict[str, Any], overwrite: bool = False):
        """Import flags from exported data"""
        try:
            flags_data = data.get("flags", {})
            metadata_data = data.get("metadata", {})

            with self.lock:
                for name, value in flags_data.items():
                    if name not in self.flags or overwrite:
                        self.flags[name] = value

                for name, metadata in metadata_data.items():
                    if name not in self.flag_metadata or overwrite:
                        self.flag_metadata[name] = metadata.copy()

            logger.info(f"Imported {len(flags_data)} flags")

        except Exception as e:
            logger.error(f"Error importing flags: {e}")

    def get_categories(self) -> Set[str]:
        """Get all flag categories"""
        with self.lock:
            categories = set()
            for metadata in self.flag_metadata.values():
                category = metadata.get("category")
                if category:
                    categories.add(category)
            return categories


# Global instance
_flag_registry = None


def get_flag_registry(event_bus=None) -> ActiveFlagRegistry:
    """Get the global flag registry instance"""
    global _flag_registry
    if _flag_registry is None:
        _flag_registry = ActiveFlagRegistry(event_bus)
    return _flag_registry


def set_flag(name: str, value: Any) -> bool:
    """Convenience function to set a flag"""
    return get_flag_registry().set_flag(name, value)


def get_flag(name: str, default=None):
    """Convenience function to get a flag"""
    return get_flag_registry().get_flag(name, default)


# For testing
def test_flag_registry():
    """Test the flag registry"""
    registry = ActiveFlagRegistry()
    # Test setting flags
    assert registry.set_flag("TEST_FLAG", True)
    assert registry.get_flag("TEST_FLAG") is True

    # Test type conversion
    assert registry.set_flag("TEST_FLAG", "false")
    assert registry.get_flag("TEST_FLAG") is False

    # Test categories
    categories = registry.get_categories()
    assert "core" in categories

    print("Flag registry tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_flag_registry()
