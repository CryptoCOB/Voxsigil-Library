"""Reconstructed task_fabric module.
No .pyc present; interface inferred.
Provides TaskFabric – a lightweight orchestrator for distributed tasks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import time
import threading

@dataclass
class FabricTask:
    task_id: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: str = "pending"  # pending|running|completed|failed
    result: Any = None
    error: Optional[Exception] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def run(self):
        self.status = "running"
        self.started_at = time.time()
        try:
            self.result = self.fn(*self.args, **self.kwargs)
            self.status = "completed"
        except Exception as e:  # pragma: no cover - placeholder
            self.error = e
            self.status = "failed"
        finally:
            self.finished_at = time.time()

class TaskFabric:
    """Simple in‑memory task orchestration fabric."""
    def __init__(self):
        self._tasks: Dict[str, FabricTask] = {}
        self._lock = threading.Lock()

    def submit(self, task_id: str, fn: Callable[..., Any], *args, **kwargs) -> FabricTask:
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task id already exists: {task_id}")
            task = FabricTask(task_id, fn, args, kwargs)
            self._tasks[task_id] = task
        threading.Thread(target=task.run, daemon=True).start()
        return task

    def get(self, task_id: str) -> Optional[FabricTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[FabricTask]:
        return list(self._tasks.values())

    def stats(self) -> Dict[str, Any]:
        total = len(self._tasks)
        running = sum(1 for t in self._tasks.values() if t.status == "running")
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        failed = sum(1 for t in self._tasks.values() if t.status == "failed")
        return {
            "total": total,
            "running": running,
            "completed": completed,
            "failed": failed,
        }

__all__ = ["TaskFabric", "FabricTask"]
