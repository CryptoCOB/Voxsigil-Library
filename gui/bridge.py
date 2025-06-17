#!/usr/bin/env python3
"""
GUI Bridge - Automatic GUI component registration and streaming
===============================================================

Enhanced bridge that automatically registers components to the GUI based on
ui_spec configurations and sets up streaming connections.
"""

import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class GUIBridge:
    """
    Bridge for automatically adding components to the GUI with streaming support.

    Handles registration of components based on ui_spec and automatic
    setup of event streaming connections.
    """

    def __init__(self, main_gui=None, event_bus=None):
        self.main_gui = main_gui
        self.event_bus = event_bus
        self.registered_components = {}
        self.widget_resolvers = {}

        # Register default widget resolvers
        self.register_default_resolvers()

    def register_default_resolvers(self):
        """Register default widget type resolvers"""

        def control_center_resolver(ui_spec: Dict, **kwargs):
            from .components.control_center_tab import ControlCenterTab

            return ControlCenterTab(
                event_bus=kwargs.get("event_bus"), training_engine=kwargs.get("training_engine")
            )

        def processing_engines_resolver(ui_spec: Dict, **kwargs):
            from .components.processing_engines_tab import ProcessingEnginesTab

            return ProcessingEnginesTab(event_bus=kwargs.get("event_bus"))

        def system_health_resolver(ui_spec: Dict, **kwargs):
            from .components.system_health_dashboard import SystemHealthDashboard

            return SystemHealthDashboard(event_bus=kwargs.get("event_bus"))

        # Register resolvers
        self.widget_resolvers["control_center"] = control_center_resolver
        self.widget_resolvers["processing_engines"] = processing_engines_resolver
        self.widget_resolvers["system_health"] = system_health_resolver

    def register_widget_resolver(self, widget_type: str, resolver: Callable):
        """Register a widget resolver for a specific type"""
        self.widget_resolvers[widget_type] = resolver
        logger.info(f"Registered widget resolver for type: {widget_type}")

    def resolve_widget(self, ui_spec: Dict, **kwargs):
        """Resolve a widget instance from ui_spec"""
        widget_type = ui_spec.get("type", "unknown")

        if widget_type in self.widget_resolvers:
            try:
                return self.widget_resolvers[widget_type](ui_spec, **kwargs)
            except Exception as e:
                logger.error(f"Error resolving widget type {widget_type}: {e}")
                return None
        else:
            logger.warning(f"No resolver found for widget type: {widget_type}")
            return None

    def add_to_gui(self, mod_id: str, ui_spec: Dict, **kwargs) -> bool:
        """
        Add a component to the GUI based on ui_spec.

        Args:
            mod_id: Module identifier
            ui_spec: UI specification dictionary
            **kwargs: Additional arguments for widget creation

        Returns:
            True if successfully added
        """
        try:
            if not self.main_gui:
                logger.warning("No main GUI instance available")
                return False

            tab_name = ui_spec.get("tab", mod_id)
            priority = ui_spec.get("priority", 999)

            # Resolve widget
            widget = self.resolve_widget(ui_spec, event_bus=self.event_bus, **kwargs)

            if not widget:
                logger.error(f"Could not resolve widget for {mod_id}")
                return False

            # Add to GUI
            if hasattr(self.main_gui, "add_tab"):
                self.main_gui.add_tab(tab_name, widget, priority)
            elif hasattr(self.main_gui, "addTab"):
                self.main_gui.addTab(widget, tab_name)
            else:
                logger.error("Main GUI does not support tab addition")
                return False

            # Setup streaming if specified
            if ui_spec.get("stream") and self.event_bus:
                self.setup_streaming(widget, ui_spec)

            # Register component
            self.registered_components[mod_id] = {
                "widget": widget,
                "ui_spec": ui_spec,
                "tab_name": tab_name,
            }

            logger.info(f"Added component {mod_id} as tab '{tab_name}'")
            return True

        except Exception as e:
            logger.error(f"Error adding component {mod_id} to GUI: {e}")
            return False

    def setup_streaming(self, widget, ui_spec: Dict):
        """Setup streaming connections for a widget"""
        try:
            stream_topics = ui_spec.get("stream_topics", [])
            if isinstance(stream_topics, str):
                stream_topics = [stream_topics]

            # Default topic if not specified
            if not stream_topics and ui_spec.get("stream"):
                widget_type = ui_spec.get("type", "unknown")
                stream_topics = [f"{widget_type}.update"]

            # Subscribe to topics
            for topic in stream_topics:
                if hasattr(widget, "update_stream"):
                    self.event_bus.subscribe(topic, widget.update_stream)
                elif hasattr(widget, "on_stream_update"):
                    self.event_bus.subscribe(topic, widget.on_stream_update)
                else:
                    logger.warning(f"Widget does not have streaming methods for topic {topic}")

            logger.info(f"Setup streaming for {len(stream_topics)} topics")

        except Exception as e:
            logger.error(f"Error setting up streaming: {e}")

    def remove_from_gui(self, mod_id: str) -> bool:
        """Remove a component from the GUI"""
        try:
            if mod_id not in self.registered_components:
                return False

            component_info = self.registered_components[mod_id]
            widget = component_info["widget"]
            tab_name = component_info["tab_name"]

            # Remove from GUI
            if hasattr(self.main_gui, "remove_tab"):
                self.main_gui.remove_tab(tab_name)
            elif hasattr(self.main_gui, "removeTab"):
                # Find tab index and remove
                for i in range(self.main_gui.count()):
                    if self.main_gui.tabText(i) == tab_name:
                        self.main_gui.removeTab(i)
                        break

            # Cleanup widget
            if hasattr(widget, "cleanup"):
                widget.cleanup()

            # Remove from registry
            del self.registered_components[mod_id]

            logger.info(f"Removed component {mod_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing component {mod_id}: {e}")
            return False

    def get_registered_components(self) -> Dict[str, Dict]:
        """Get all registered components"""
        return self.registered_components.copy()

    def get_component_widget(self, mod_id: str):
        """Get widget for a specific component"""
        component_info = self.registered_components.get(mod_id)
        return component_info["widget"] if component_info else None

    def update_component_streaming(self, mod_id: str, new_topics: list):
        """Update streaming topics for a component"""
        try:
            if mod_id not in self.registered_components:
                return False

            component_info = self.registered_components[mod_id]
            widget = component_info["widget"]
            ui_spec = component_info["ui_spec"]

            # Update ui_spec
            ui_spec["stream_topics"] = new_topics

            # Re-setup streaming
            self.setup_streaming(widget, ui_spec)

            return True

        except Exception as e:
            logger.error(f"Error updating streaming for {mod_id}: {e}")
            return False


# Global bridge instance
_gui_bridge = None


def get_gui_bridge(main_gui=None, event_bus=None) -> GUIBridge:
    """Get the global GUI bridge instance"""
    global _gui_bridge
    if _gui_bridge is None:
        _gui_bridge = GUIBridge(main_gui, event_bus)
    elif main_gui and not _gui_bridge.main_gui:
        _gui_bridge.main_gui = main_gui
    elif event_bus and not _gui_bridge.event_bus:
        _gui_bridge.event_bus = event_bus
    return _gui_bridge


def auto_register_component(mod_id: str, ui_spec: Dict, **kwargs) -> bool:
    """Convenience function to auto-register a component"""
    bridge = get_gui_bridge()
    return bridge.add_to_gui(mod_id, ui_spec, **kwargs)


# For testing
def test_gui_bridge():
    """Test the GUI bridge"""

    class MockGUI:
        def __init__(self):
            self.tabs = []

        def add_tab(self, name, widget, priority=999):
            self.tabs.append((name, widget, priority))

    class MockWidget:
        def __init__(self):
            self.stream_calls = []

        def update_stream(self, event):
            self.stream_calls.append(event)

    class MockEventBus:
        def __init__(self):
            self.subscriptions = {}

        def subscribe(self, topic, handler):
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
            self.subscriptions[topic].append(handler)

    # Create test instances
    gui = MockGUI()
    bus = MockEventBus()
    bridge = GUIBridge(gui, bus)

    # Register a test resolver
    def test_resolver(ui_spec, **kwargs):
        return MockWidget()

    bridge.register_widget_resolver("test_type", test_resolver)

    # Test adding component
    ui_spec = {
        "tab": "Test Tab",
        "type": "test_type",
        "stream": True,
        "stream_topics": ["test.update"],
    }

    success = bridge.add_to_gui("test_mod", ui_spec)
    assert success
    assert len(gui.tabs) == 1
    assert gui.tabs[0][0] == "Test Tab"

    print("GUI bridge tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gui_bridge()
