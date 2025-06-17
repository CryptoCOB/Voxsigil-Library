#!/usr/bin/env python3
"""
Trace Store - Event trace storage and retrieval
================================================

Manages storage and retrieval of event traces for debugging and monitoring.
Provides efficient trace lookup and cleanup capabilities.
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceStore:
    """
    Storage and management for event traces.

    Provides efficient storage, retrieval, and cleanup of event traces
    with support for filtering and querying.
    """

    def __init__(self, event_bus=None, max_traces: int = 10000, retention_hours: int = 24):
        self.event_bus = event_bus
        self.max_traces = max_traces
        self.retention_seconds = retention_hours * 3600

        # Storage
        self.traces = {}  # event_id -> trace_data
        self.trace_index = defaultdict(deque)  # indexed by type, status, etc.
        self.trace_timeline = deque()  # (timestamp, event_id) for cleanup

        # Thread safety
        self.lock = threading.RLock()

        # Stats
        self.stats = {"total_traces": 0, "active_traces": 0, "traces_cleaned": 0}

        # Setup event subscriptions
        if self.event_bus:
            self.event_bus.subscribe("trace.get", self.handle_trace_request)
            self.event_bus.subscribe("trace.store", self.handle_trace_store)
            self.event_bus.subscribe("*", self.auto_trace_event)  # Trace all events

        # Start cleanup timer
        self.cleanup_timer = threading.Timer(300, self.cleanup_old_traces)  # 5 minutes
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()

    def store_trace(
        self,
        event_id: str = None,
        event_type: str = "unknown",
        status: str = "active",
        data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Store a trace entry.

        Args:
            event_id: Unique event ID (generated if not provided)
            event_type: Type of event
            status: Event status
            data: Event data
            metadata: Additional metadata

        Returns:
            Event ID
        """
        if event_id is None:
            event_id = str(uuid.uuid4())

        if data is None:
            data = {}

        if metadata is None:
            metadata = {}

        timestamp = time.time()

        trace_entry = {
            "event_id": event_id,
            "event_type": event_type,
            "status": status,
            "data": data,
            "metadata": metadata,
            "timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        }

        with self.lock:
            # Store trace
            self.traces[event_id] = trace_entry

            # Update indexes
            self.trace_index["type:" + event_type].append(event_id)
            self.trace_index["status:" + status].append(event_id)

            # Add to timeline
            self.trace_timeline.append((timestamp, event_id))

            # Update stats
            self.stats["total_traces"] += 1
            self.stats["active_traces"] = len(self.traces)

            # Cleanup if needed
            if len(self.traces) > self.max_traces:
                self._cleanup_oldest()

        logger.debug(f"Stored trace {event_id} of type {event_type}")

        # Publish trace event
        if self.event_bus:
            self.event_bus.publish(
                "trace.event",
                {
                    "event_id": event_id,
                    "type": event_type,
                    "status": status,
                    "timestamp": timestamp,
                },
            )

        return event_id

    def get_trace(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by event ID"""
        with self.lock:
            return self.traces.get(event_id, {}).copy() if event_id in self.traces else None

    def update_trace(
        self,
        event_id: str,
        status: str = None,
        data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Update an existing trace"""
        with self.lock:
            if event_id not in self.traces:
                return False

            trace = self.traces[event_id]

            if status is not None:
                # Update index if status changed
                old_status = trace["status"]
                if old_status != status:
                    self.trace_index["status:" + old_status].remove(event_id)
                    self.trace_index["status:" + status].append(event_id)

                trace["status"] = status

            if data is not None:
                trace["data"].update(data)

            if metadata is not None:
                trace["metadata"].update(metadata)

            trace["last_updated"] = time.time()

            return True

    def get_traces_by_type(self, event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get traces by event type"""
        with self.lock:
            event_ids = list(self.trace_index["type:" + event_type])[-limit:]
            return [self.traces[eid].copy() for eid in event_ids if eid in self.traces]

    def get_traces_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get traces by status"""
        with self.lock:
            event_ids = list(self.trace_index["status:" + status])[-limit:]
            return [self.traces[eid].copy() for eid in event_ids if eid in self.traces]

    def get_recent_traces(self, limit: int = 100, since: float = None) -> List[Dict[str, Any]]:
        """Get recent traces"""
        with self.lock:
            recent_ids = []

            # Get from timeline
            for timestamp, event_id in reversed(self.trace_timeline):
                if since and timestamp < since:
                    break
                if event_id in self.traces:
                    recent_ids.append(event_id)
                if len(recent_ids) >= limit:
                    break

            return [self.traces[eid].copy() for eid in recent_ids]

    def search_traces(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search traces by text query"""
        query_lower = query.lower()
        results = []

        with self.lock:
            for trace in self.traces.values():
                # Search in event type, status, and data
                searchable_text = (
                    f"{trace['event_type']} {trace['status']} {str(trace['data'])}".lower()
                )
                if query_lower in searchable_text:
                    results.append(trace.copy())
                    if len(results) >= limit:
                        break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get trace store statistics"""
        with self.lock:
            return self.stats.copy()

    def cleanup_old_traces(self):
        """Clean up old traces based on retention policy"""
        current_time = time.time()
        cutoff_time = current_time - self.retention_seconds

        with self.lock:
            cleaned_count = 0

            # Clean from timeline and traces
            while self.trace_timeline and self.trace_timeline[0][0] < cutoff_time:
                _, event_id = self.trace_timeline.popleft()
                if event_id in self.traces:
                    trace = self.traces.pop(event_id)

                    # Remove from indexes
                    event_type = trace["event_type"]
                    status = trace["status"]

                    if event_id in self.trace_index["type:" + event_type]:
                        self.trace_index["type:" + event_type].remove(event_id)
                    if event_id in self.trace_index["status:" + status]:
                        self.trace_index["status:" + status].remove(event_id)

                    cleaned_count += 1

            self.stats["traces_cleaned"] += cleaned_count
            self.stats["active_traces"] = len(self.traces)

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old traces")

        # Schedule next cleanup
        self.cleanup_timer = threading.Timer(300, self.cleanup_old_traces)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()

    def _cleanup_oldest(self):
        """Clean up oldest traces when limit exceeded"""
        cleanup_count = len(self.traces) - self.max_traces + 100  # Clean extra

        for _ in range(cleanup_count):
            if not self.trace_timeline:
                break

            _, event_id = self.trace_timeline.popleft()
            if event_id in self.traces:
                trace = self.traces.pop(event_id)

                # Remove from indexes
                event_type = trace["event_type"]
                status = trace["status"]

                try:
                    self.trace_index["type:" + event_type].remove(event_id)
                except ValueError:
                    pass
                try:
                    self.trace_index["status:" + status].remove(event_id)
                except ValueError:
                    pass

        self.stats["traces_cleaned"] += cleanup_count

    def handle_trace_request(self, event):
        """Handle trace get requests from event bus"""
        try:
            data = event.get("data", {})
            event_id = data.get("event_id")

            if event_id:
                trace = self.get_trace(event_id)
                if self.event_bus:
                    self.event_bus.publish(
                        "trace.reply",
                        {"event_id": event_id, "trace": trace, "found": trace is not None},
                    )
        except Exception as e:
            logger.error(f"Error handling trace request: {e}")

    def handle_trace_store(self, event):
        """Handle trace store requests from event bus"""
        try:
            data = event.get("data", {})
            event_id = self.store_trace(**data)

            if self.event_bus:
                self.event_bus.publish("trace.stored", {"event_id": event_id, "success": True})
        except Exception as e:
            logger.error(f"Error handling trace store: {e}")

    def auto_trace_event(self, event):
        """Automatically trace events (if enabled)"""
        try:
            # Only trace certain event types to avoid overwhelming storage
            event_type = event.get("type", "unknown")

            # Skip trace events to avoid recursion
            if event_type.startswith("trace."):
                return

            # Only trace important events
            important_types = [
                "user.command",
                "command.reply",
                "flag.changed",
                "system.status",
                "training.status",
                "agent.detail",
                "engine.stats",
                "error",
            ]

            if event_type in important_types:
                self.store_trace(
                    event_type=event_type,
                    status="auto_traced",
                    data=event.get("data", {}),
                    metadata={"auto_traced": True, "source": "event_bus"},
                )

        except Exception as e:
            logger.warning(f"Error auto-tracing event: {e}")

    def clear_all_traces(self):
        """Clear all traces (for testing/cleanup)"""
        with self.lock:
            self.traces.clear()
            self.trace_index.clear()
            self.trace_timeline.clear()
            self.stats = {"total_traces": 0, "active_traces": 0, "traces_cleaned": 0}
        logger.info("Cleared all traces")


# Global instance
_trace_store = None


def get_trace_store(event_bus=None) -> TraceStore:
    """Get the global trace store instance"""
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore(event_bus)
    return _trace_store


# For testing
def test_trace_store():
    """Test the trace store"""
    store = TraceStore()

    # Test storing traces
    event_id1 = store.store_trace(event_type="test", status="active", data={"key": "value1"})
    event_id2 = store.store_trace(event_type="test", status="completed", data={"key": "value2"})

    # Test retrieval
    trace1 = store.get_trace(event_id1)
    assert trace1 is not None
    assert trace1["event_type"] == "test"

    # Test by type
    test_traces = store.get_traces_by_type("test")
    assert len(test_traces) == 2

    # Test by status
    active_traces = store.get_traces_by_status("active")
    assert len(active_traces) == 1

    # Test update
    assert store.update_trace(event_id1, status="completed")
    updated_trace = store.get_trace(event_id1)
    assert updated_trace["status"] == "completed"

    print("Trace store tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_trace_store()
