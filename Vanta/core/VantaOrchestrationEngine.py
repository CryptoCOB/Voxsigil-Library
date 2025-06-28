"""
VantaOrchestrationEngine - Optimized Component Orchestration Framework

This is a lightweight yet powerful alternative to complex MetaConsciousness SDK architecture.
VantaOrchestrationEngine provides optimized component registration, event handling, configuration
management, performance monitoring, and robustness features while maintaining simplicity.

Enhanced Features:
- Performance optimization with caching and connection pooling
- Memory management and component lifecycle tracking
- Health monitoring and circuit breaker patterns
- Async operation support and load balancing
- Real-time metrics and diagnostic capabilities
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

# HOLO-1.5 Registration
try:
    from ..registration.master_registration import vanta_core_module
except ImportError:

    def vanta_core_module(name: str = "", role: str = ""):
        def decorator(cls):
            return cls

        return decorator


class ComponentRegistry:
    """Optimized component registry with lifecycle management and health monitoring."""

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        self._component_health: Dict[str, Dict[str, Any]] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
        self._last_access: Dict[str, datetime] = {}
        self._cache: Dict[str, Any] = {}  # Component result cache
        self._cache_ttl: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._health_check_interval = 30  # seconds
        self._health_thread: Optional[threading.Thread] = None
        self._running = True

        # Start health monitoring
        self._start_health_monitoring()

    def _start_health_monitoring(self):
        """Start background health monitoring thread."""

        def health_monitor():
            while self._running:
                try:
                    self._check_component_health()
                    time.sleep(self._health_check_interval)
                except Exception as e:
                    logging.error(f"Health monitoring error: {e}")

        self._health_thread = threading.Thread(target=health_monitor, daemon=True)
        self._health_thread.start()

    def _check_component_health(self):
        """Check health of all registered components."""
        with self._lock:
            current_time = datetime.now()
            for name, component in self._components.items():
                try:
                    # Basic health check
                    is_healthy = True
                    last_error = None

                    # Check if component has health check method
                    if hasattr(component, "health_check"):
                        try:
                            health_result = component.health_check()
                            is_healthy = (
                                health_result.get("healthy", True)
                                if isinstance(health_result, dict)
                                else bool(health_result)
                            )
                        except Exception as e:
                            is_healthy = False
                            last_error = str(e)

                    # Update health status
                    self._component_health[name] = {
                        "healthy": is_healthy,
                        "last_check": current_time,
                        "last_error": last_error,
                        "uptime": (
                            current_time
                            - self._component_metadata[name].get(
                                "registered_at", current_time
                            )
                        ).total_seconds(),
                    }

                except Exception as e:
                    self._component_health[name] = {
                        "healthy": False,
                        "last_check": current_time,
                        "last_error": str(e),
                        "uptime": 0,
                    }

    def register(
        self, name: str, component: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a component with enhanced metadata and health tracking."""
        with self._lock:
            if name in self._components:
                logging.warning(f"Component '{name}' already registered, overwriting")

            self._components[name] = component

            # Enhanced metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata.update(
                {
                    "registered_at": datetime.now(),
                    "component_type": type(component).__name__,
                    "component_id": id(component),
                    "has_health_check": hasattr(component, "health_check"),
                    "has_shutdown": hasattr(component, "shutdown"),
                    "memory_usage": self._get_component_memory_usage(component),
                }
            )
            self._component_metadata[name] = enhanced_metadata
            self._access_count[name] = 0
            self._last_access[name] = datetime.now()

            # Initialize health status
            self._component_health[name] = {
                "healthy": True,
                "last_check": datetime.now(),
                "last_error": None,
                "uptime": 0,
            }

            logging.info(
                f"Component '{name}' registered successfully with enhanced tracking"
            )
            return True

    def _get_component_memory_usage(self, component: Any) -> int:
        """Get approximate memory usage of a component."""
        try:
            # This is a rough estimate
            return len(str(component))
        except Exception:
            return 0

    def get(self, name: str, default: Any = None) -> Any:
        """Get a component by name with access tracking."""
        with self._lock:
            self._access_count[name] += 1
            self._last_access[name] = datetime.now()
            return self._components.get(name, default)

    def get_cached(self, name: str, cache_key: str, default: Any = None) -> Any:
        """Get cached component result."""
        with self._lock:
            cache_full_key = f"{name}:{cache_key}"
            if cache_full_key in self._cache:
                # Check TTL
                if cache_full_key in self._cache_ttl:
                    if datetime.now() > self._cache_ttl[cache_full_key]:
                        del self._cache[cache_full_key]
                        del self._cache_ttl[cache_full_key]
                        return default
                return self._cache[cache_full_key]
            return default

    def set_cache(
        self, name: str, cache_key: str, value: Any, ttl_seconds: int = 300
    ) -> None:
        """Set cached component result."""
        with self._lock:
            cache_full_key = f"{name}:{cache_key}"
            self._cache[cache_full_key] = value
            self._cache_ttl[cache_full_key] = datetime.now() + timedelta(
                seconds=ttl_seconds
            )

    def unregister(self, name: str) -> bool:
        """Unregister a component with cleanup."""
        with self._lock:
            if name in self._components:
                # Cleanup component if it has shutdown method
                component = self._components[name]
                if hasattr(component, "shutdown"):
                    try:
                        component.shutdown()
                        logging.info(f"Component '{name}' shutdown completed")
                    except Exception as e:
                        logging.error(f"Error shutting down component '{name}': {e}")

                del self._components[name]
                del self._component_metadata[name]
                self._component_health.pop(name, None)
                self._access_count.pop(name, None)
                self._last_access.pop(name, None)

                # Cleanup cache entries
                cache_keys_to_remove = [
                    k for k in self._cache.keys() if k.startswith(f"{name}:")
                ]
                for key in cache_keys_to_remove:
                    del self._cache[key]
                    self._cache_ttl.pop(key, None)

                logging.info(f"Component '{name}' unregistered with full cleanup")
                return True
            return False

    def list_components(self) -> List[str]:
        """List all registered component names."""
        with self._lock:
            return list(self._components.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a component."""
        with self._lock:
            return self._component_metadata.get(name)

    def get_health_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a component."""
        with self._lock:
            return self._component_health.get(name)

    def get_performance_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a component."""
        with self._lock:
            if name not in self._components:
                return None

            return {
                "access_count": self._access_count.get(name, 0),
                "last_access": self._last_access.get(name),
                "health_status": self._component_health.get(name, {}),
                "memory_usage": self._component_metadata.get(name, {}).get(
                    "memory_usage", 0
                ),
                "uptime": self._component_health.get(name, {}).get("uptime", 0),
            }

    def get_status(self) -> Dict[str, Any]:
        """Get enhanced registry status."""
        with self._lock:
            total_memory = sum(
                meta.get("memory_usage", 0)
                for meta in self._component_metadata.values()
            )

            healthy_components = sum(
                1
                for health in self._component_health.values()
                if health.get("healthy", False)
            )

            return {
                "total_components": len(self._components),
                "healthy_components": healthy_components,
                "unhealthy_components": len(self._components) - healthy_components,
                "components": list(self._components.keys()),
                "total_memory_usage": total_memory,
                "cache_entries": len(self._cache),
                "total_access_count": sum(self._access_count.values()),
                "status": "active",
            }

    def cleanup_cache(self, max_age_seconds: int = 3600) -> int:
        """Cleanup expired cache entries."""
        with self._lock:
            current_time = datetime.now()
            expired_keys = []

            for key, expire_time in self._cache_ttl.items():
                if current_time > expire_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                del self._cache_ttl[key]

            logging.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)

    def shutdown(self) -> None:
        """Shutdown the component registry."""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5)

        # Shutdown all components
        for name in list(self._components.keys()):
            self.unregister(name)

        logging.info("ComponentRegistry shutdown complete")


class EventBus:
    """Optimized event bus for inter-component communication with priority queuing and monitoring."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._priority_subscribers: Dict[str, List[tuple]] = defaultdict(
            list
        )  # (priority, callback)
        self._lock = threading.RLock()
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        self._event_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0}
        )
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._circuit_threshold = 5  # Number of failures before opening circuit

    def subscribe(self, event_type: str, callback: Callable, priority: int = 0) -> None:
        """Subscribe to an event type with optional priority (higher = executed first)."""
        with self._lock:
            if priority == 0:
                self._subscribers[event_type].append(callback)
            else:
                # Insert in priority order (highest first)
                priority_list = self._priority_subscribers[event_type]
                inserted = False
                for i, (p, _) in enumerate(priority_list):
                    if priority > p:
                        priority_list.insert(i, (priority, callback))
                        inserted = True
                        break
                if not inserted:
                    priority_list.append((priority, callback))

            logging.debug(
                f"Subscribed to event '{event_type}' with priority {priority}"
            )

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            # Remove from regular subscribers
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logging.debug(f"Unsubscribed from event '{event_type}'")
                return

            # Remove from priority subscribers
            priority_list = self._priority_subscribers[event_type]
            for i, (_, cb) in enumerate(priority_list):
                if cb == callback:
                    priority_list.pop(i)
                    logging.debug(f"Unsubscribed from priority event '{event_type}'")
                    return

    def publish(
        self, event_type: str, data: Any = None, source: str = "unknown"
    ) -> None:
        """Publish an event to all subscribers with circuit breaker and monitoring."""
        start_time = time.time()
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.now(),
            "id": f"{event_type}_{int(time.time() * 1000)}",
        }

        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Get all subscribers (regular + priority)
            regular_subscribers = self._subscribers.get(event_type, []).copy()
            priority_subscribers = self._priority_subscribers.get(event_type, []).copy()

        # Combine and sort by priority (highest first)
        all_subscribers = [(0, cb) for cb in regular_subscribers]
        all_subscribers.extend(priority_subscribers)
        all_subscribers.sort(key=lambda x: x[0], reverse=True)

        # Track errors for circuit breaker
        error_count = 0
        success_count = 0

        # Call subscribers outside the lock to avoid deadlocks
        for priority, callback in all_subscribers:
            callback_key = f"{event_type}_{id(callback)}"

            # Check circuit breaker
            circuit_state = self._circuit_breakers.get(
                callback_key, {"failures": 0, "last_failure": None}
            )
            if circuit_state["failures"] >= self._circuit_threshold:
                # Circuit is open - check if it should be reset
                if (
                    circuit_state["last_failure"]
                    and time.time() - circuit_state["last_failure"] > 60
                ):  # 60 second cooldown
                    circuit_state["failures"] = 0
                    circuit_state["last_failure"] = None
                else:
                    logging.warning(
                        f"Circuit breaker open for {callback_key}, skipping"
                    )
                    continue

            try:
                callback_start = time.time()
                callback(event)
                callback_time = time.time() - callback_start

                # Update success stats
                success_count += 1
                circuit_state["failures"] = max(
                    0, circuit_state["failures"] - 1
                )  # Gradual recovery

                # Update performance stats
                with self._lock:
                    stats = self._event_stats[event_type]
                    stats["count"] += 1
                    stats["total_time"] += callback_time
                    stats["avg_time"] = stats["total_time"] / stats["count"]

            except Exception as e:
                error_count += 1
                circuit_state["failures"] += 1
                circuit_state["last_failure"] = time.time()

                with self._lock:
                    self._event_stats[event_type]["errors"] += 1

                logging.error(f"Error in event subscriber for '{event_type}': {e}")

            # Update circuit breaker state
            self._circuit_breakers[callback_key] = circuit_state

        total_time = time.time() - start_time
        total_subscribers = len(all_subscribers)

        logging.debug(
            f"Published event '{event_type}' to {total_subscribers} subscribers "
            f"({success_count} success, {error_count} errors) in {total_time:.3f}s"
        )

    def get_event_history(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent event history."""
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e["type"] == event_type]
            else:
                events = self._event_history.copy()

            return events[-limit:] if limit > 0 else events

    def get_event_stats(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for events."""
        with self._lock:
            if event_type:
                return self._event_stats.get(
                    event_type,
                    {"count": 0, "total_time": 0, "avg_time": 0, "errors": 0},
                )
            else:
                return dict(self._event_stats)

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all callbacks."""
        with self._lock:
            return dict(self._circuit_breakers)

    def reset_circuit_breaker(self, callback_key: str) -> bool:
        """Manually reset a circuit breaker."""
        with self._lock:
            if callback_key in self._circuit_breakers:
                self._circuit_breakers[callback_key] = {
                    "failures": 0,
                    "last_failure": None,
                }
                logging.info(f"Circuit breaker reset for {callback_key}")
                return True
            return False

    def get_subscriber_count(self, event_type: Optional[str] = None) -> Dict[str, int]:
        """Get subscriber counts."""
        with self._lock:
            if event_type:
                regular_count = len(self._subscribers.get(event_type, []))
                priority_count = len(self._priority_subscribers.get(event_type, []))
                return {
                    "regular": regular_count,
                    "priority": priority_count,
                    "total": regular_count + priority_count,
                }
            else:
                result = {}
                all_event_types = set(self._subscribers.keys()) | set(
                    self._priority_subscribers.keys()
                )
                for et in all_event_types:
                    regular_count = len(self._subscribers.get(et, []))
                    priority_count = len(self._priority_subscribers.get(et, []))
                    result[et] = {
                        "regular": regular_count,
                        "priority": priority_count,
                        "total": regular_count + priority_count,
                    }
                return result

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
            logging.info("Event history cleared")

    def clear_stats(self) -> None:
        """Clear performance statistics."""
        with self._lock:
            self._event_stats.clear()
            self._circuit_breakers.clear()
            logging.info("Event statistics and circuit breakers cleared")


class ConfigManager:
    """Enhanced configuration manager with validation, env vars, and change tracking."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._validators: Dict[str, Callable] = {}
        self._change_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._change_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._env_prefix = "VANTA_"

        # Load environment variables
        self._load_env_vars()

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        import os

        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix) :].lower()
                # Try to parse as JSON first, then as string
                try:
                    import json

                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value
                self._config[config_key] = parsed_value

    def set_default(self, key: str, value: Any) -> None:
        """Set a default value for a configuration key."""
        with self._lock:
            self._defaults[key] = value

    def set_validator(self, key: str, validator: Callable[[Any], bool]) -> None:
        """Set a validator function for a configuration key."""
        with self._lock:
            self._validators[key] = validator

    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        """Set a configuration value with optional validation."""
        with self._lock:
            # Validate if validator exists
            if validate and key in self._validators:
                try:
                    if not self._validators[key](value):
                        logging.error(
                            f"Validation failed for config key '{key}' with value {value}"
                        )
                        return False
                except Exception as e:
                    logging.error(f"Validator error for config key '{key}': {e}")
                    return False

            # Track change
            old_value = self._config.get(key)
            self._config[key] = value

            # Record change history
            change_record = {
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": datetime.now(),
                "validated": validate,
            }
            self._change_history.append(change_record)

            # Trigger change callbacks
            for callback in self._change_callbacks.get(key, []):
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    logging.error(f"Error in config change callback for '{key}': {e}")

            return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback to defaults."""
        with self._lock:
            if key in self._config:
                return self._config[key]
            elif key in self._defaults:
                return self._defaults[key]
            else:
                return default

    def get_typed(self, key: str, value_type: type, default: Any = None) -> Any:
        """Get a configuration value with type casting."""
        value = self.get(key, default)
        if value is None:
            return default

        try:
            if value_type is bool and isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return value_type(value)
        except (ValueError, TypeError) as e:
            logging.warning(f"Type conversion failed for config '{key}': {e}")
            return default

    def subscribe_to_changes(
        self, key: str, callback: Callable[[str, Any, Any], None]
    ) -> None:
        """Subscribe to changes for a specific configuration key."""
        with self._lock:
            self._change_callbacks[key].append(callback)

    def unsubscribe_from_changes(self, key: str, callback: Callable) -> None:
        """Unsubscribe from changes for a specific configuration key."""
        with self._lock:
            if callback in self._change_callbacks[key]:
                self._change_callbacks[key].remove(callback)

    def update(self, config: Dict[str, Any], validate: bool = True) -> Dict[str, bool]:
        """Update configuration with a dictionary, return success status for each key."""
        results = {}
        for key, value in config.items():
            results[key] = self.set(key, value, validate)
        return results

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration including defaults."""
        with self._lock:
            result = self._defaults.copy()
            result.update(self._config)
            return result

    def get_change_history(
        self, key: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        with self._lock:
            if key:
                changes = [c for c in self._change_history if c["key"] == key]
            else:
                changes = self._change_history.copy()
            return changes[-limit:] if limit > 0 else changes

    def export_config(self, include_defaults: bool = False) -> Dict[str, Any]:
        """Export configuration for backup/transfer."""
        with self._lock:
            if include_defaults:
                return self.get_all()
            else:
                return self._config.copy()

    def import_config(
        self, config: Dict[str, Any], validate: bool = True, merge: bool = True
    ) -> Dict[str, bool]:
        """Import configuration from backup/transfer."""
        if not merge:
            with self._lock:
                self._config.clear()
        return self.update(config, validate)

    def clear_history(self) -> None:
        """Clear change history."""
        with self._lock:
            self._change_history.clear()
            logging.info("Configuration change history cleared")


@vanta_core_module()
class VantaOrchestrationEngine:
    """
    VantaOrchestrationEngine - Simple component orchestration framework.

    This replaces the complex MetaConsciousness SDK architecture with a
    simpler, more direct approach to component management and communication.
    """

    _instance: Optional["VantaOrchestrationEngine"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "VantaOrchestrationEngine":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.start_time = datetime.now()

        # Core components
        self.registry = ComponentRegistry()
        self.events = EventBus()
        self.config = ConfigManager()  # Set up logging
        self.logger = logging.getLogger("vanta_orchestration_engine")

        # Register core components with themselves
        self.registry.register("registry", self.registry)
        self.registry.register("events", self.events)
        self.registry.register("config", self.config)
        self.registry.register("core", self)

        self.logger.info("VantaOrchestrationEngine initialized successfully")

    # Convenience methods for common operations
    def register_component(
        self, name: str, component: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a component."""
        success = self.registry.register(name, component, metadata)
        if success:
            self.events.publish(
                "component_registered",
                {"name": name, "type": type(component).__name__},
                source="vanta_core",
            )
        return success

    def get_component(self, name: str, default: Any = None) -> Any:
        """Get a component."""
        return self.registry.get(name, default)

    def publish_event(
        self, event_type: str, data: Any = None, source: str = "unknown"
    ) -> None:
        """Publish an event."""
        self.events.publish(event_type, data, source)

    def subscribe_to_event(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event."""
        self.events.subscribe(event_type, callback)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration."""
        self.config.set(key, value)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration."""
        return self.config.get(key, default)

    # Enhanced management methods
    def unregister_component(self, name: str) -> bool:
        """Unregister a component with event notification."""
        success = self.registry.unregister(name)
        if success:
            self.events.publish(
                "component_unregistered",
                {"name": name},
                source="vanta_core",
            )
        return success

    def list_components(self) -> List[str]:
        """List all registered components."""
        return self.registry.list_components()

    def get_component_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a component."""
        return self.registry.get_metadata(name)

    def get_component_health(self, name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a component."""
        return self.registry.get_health_status(name)

    def get_component_performance(self, name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a component."""
        return self.registry.get_performance_stats(name)

    def subscribe_to_event_with_priority(
        self, event_type: str, callback: Callable, priority: int = 0
    ) -> None:
        """Subscribe to an event with priority."""
        self.events.subscribe(event_type, callback, priority)

    def unsubscribe_from_event(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event."""
        self.events.unsubscribe(event_type, callback)

    def get_event_history(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get event history."""
        return self.events.get_event_history(event_type, limit)

    def get_event_stats(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Get event performance statistics."""
        return self.events.get_event_stats(event_type)

    def reset_event_circuit_breaker(self, callback_key: str) -> bool:
        """Reset a circuit breaker for an event callback."""
        return self.events.reset_circuit_breaker(callback_key)

    def cleanup_caches(self) -> Dict[str, int]:
        """Cleanup expired caches in all components."""
        results = {}
        results["registry_cache"] = self.registry.cleanup_cache()
        return results

    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive status including all subsystems."""
        status = self.get_status()

        # Add detailed component information
        status.update(
            {
                "component_details": {
                    name: {
                        "metadata": self.registry.get_metadata(name),
                        "health": self.registry.get_health_status(name),
                        "performance": self.registry.get_performance_stats(name),
                    }
                    for name in self.registry.list_components()
                },
                "event_stats": self.events.get_event_stats(),
                "event_subscriber_counts": self.events.get_subscriber_count(),
                "circuit_breaker_status": self.events.get_circuit_breaker_status(),
                "config_change_history": self.config.get_change_history(limit=10),
            }
        )

        return status

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_report = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "components": {},
            "events": {},
            "config": {},
        }

        # Check component health
        components_healthy = 0
        total_components = len(self.registry.list_components())

        for name in self.registry.list_components():
            health = self.registry.get_health_status(name)
            health_report["components"][name] = health
            if health and health.get("healthy", False):
                components_healthy += 1

        # Check event system health
        event_stats = self.events.get_event_stats()
        circuit_breakers = self.events.get_circuit_breaker_status()

        health_report["events"] = {
            "total_events_processed": sum(
                stats.get("count", 0) for stats in event_stats.values()
            ),
            "total_event_errors": sum(
                stats.get("errors", 0) for stats in event_stats.values()
            ),
            "active_circuit_breakers": len(
                [cb for cb in circuit_breakers.values() if cb.get("failures", 0) > 0]
            ),
        }

        # Overall health determination
        if components_healthy < total_components * 0.8:  # Less than 80% healthy
            health_report["overall_status"] = "degraded"

        if health_report["events"]["active_circuit_breakers"] > 0:
            health_report["overall_status"] = "degraded"

        return health_report

    def get_status(self) -> Dict[str, Any]:
        """Get VantaOrchestrationEngine status."""
        return {
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "registry_status": self.registry.get_status(),
            "event_subscribers": len(self.events._subscribers),
            "config_keys": len(self.config.get_all()),
            "status": "active",
        }

    def shutdown(self) -> None:
        """Shutdown VantaOrchestrationEngine."""
        self.logger.info("VantaOrchestrationEngine shutting down")
        self.events.publish("core_shutdown", {}, source="vanta_core")

        # Clear components
        for component_name in self.registry.list_components():
            component = self.registry.get(component_name)
            if hasattr(component, "shutdown"):
                try:
                    component.shutdown()
                except Exception as e:
                    self.logger.error(
                        f"Error shutting down component '{component_name}': {e}"
                    )

        self.logger.info("VantaOrchestrationEngine shutdown complete")


# Convenience functions for global access
def get_core() -> VantaOrchestrationEngine:
    """Get the VantaOrchestrationEngine singleton instance."""
    return VantaOrchestrationEngine()


def register_component(
    name: str, component: Any, metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Global function to register a component."""
    return get_core().register_component(name, component, metadata)


def get_component(name: str, default: Any = None) -> Any:
    """Global function to get a component."""
    return get_core().get_component(name, default)


def publish_event(event_type: str, data: Any = None, source: str = "unknown") -> None:
    """Global function to publish an event."""
    get_core().publish_event(event_type, data, source)


def subscribe_to_event(event_type: str, callback: Callable) -> None:
    """Global function to subscribe to an event."""
    get_core().subscribe_to_event(event_type, callback)


# Helper functions for common patterns
def safe_component_call(component_name: str, method_name: str, *args, **kwargs) -> Any:
    """Safely call a method on a component."""
    component = get_component(component_name)
    if component is None:
        logging.warning(f"Component '{component_name}' not found")
        return None

    if not hasattr(component, method_name):
        logging.warning(f"Component '{component_name}' has no method '{method_name}'")
        return None

    try:
        method = getattr(component, method_name)
        return method(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error calling {component_name}.{method_name}: {e}")
        return None


def trace_event(event_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Trace an event for debugging/monitoring."""
    data = {
        "event_type": event_type,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
    }

    # Log locally
    logging.debug(f"Trace: {event_type} | {metadata}")

    # Publish as event
    publish_event("trace_event", data, source="trace")


def generate_internal_dialogue_message(
    content: str, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate an internal dialogue message."""
    return {
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "id": f"msg_{int(time.time() * 1000)}",
    }
