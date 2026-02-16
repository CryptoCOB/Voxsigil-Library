import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, Optional, List

logger = logging.getLogger(__name__)

class DependencyMonitor(FileSystemEventHandler):
    def __init__(self, callback: Callable, refresh_interval: int = 5, 
                file_patterns: List[str] = None):
        self.callback = callback
        self.last_update = 0
        self.refresh_interval = refresh_interval
        self.file_patterns = file_patterns or ['.py']
        self.observer = None

    def on_modified(self, event):
        if not event.is_directory and any(event.src_path.endswith(pattern) for pattern in self.file_patterns):
            current_time = time.time()
            if current_time - self.last_update > self.refresh_interval:
                logger.info(f"Change detected in {event.src_path}, updating dependencies...")
                self.callback()
                self.last_update = current_time

    def start_monitoring(self, path: str) -> bool:
        """Start monitoring the specified path for changes."""
        try:
            self.observer = Observer()
            self.observer.schedule(self, path, recursive=True)
            self.observer.start()
            logger.info(f"Started monitoring {path} for changes")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self) -> None:
        """Stop monitoring for changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped monitoring for changes")
