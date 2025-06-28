"""
Core GridFormer Module
======================

Provides grid formation, transformation and pattern-detection
capabilities for the VoxSigil system, with live hooks into the
HOLO-1.5 cognitive mesh.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# HOLO-1.5 base class & decorator
from .base import BaseCore, CognitiveMeshRole, vanta_core_module

if TYPE_CHECKING:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

logger = logging.getLogger(__name__)


@vanta_core_module(
    name="grid_former",
    subsystem="grid_processing",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="Core grid formation and transformation engine for spatial reasoning",
    capabilities=[
        "grid_transformation",
        "pattern_detection",
        "spatial_reasoning",
        "grid_analysis",
    ],
    cognitive_load=3.0,
    symbolic_depth=3,
    collaboration_patterns=[
        "spatial_reasoning",
        "pattern_analysis",
        "grid_processing",
    ],
)
class GridFormer(BaseCore):
    """Grid formation / transformation engine (HOLO-1.5)."""

    # ───────────────────────────────────────────────
    # Construction
    # ───────────────────────────────────────────────
    def __init__(
        self, vanta_core: "UnifiedVantaCore", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(vanta_core, config or {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("GridFormer initialised and registered with HOLO-1.5 mesh")

    # ───────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────
    def transform_grid(
        self, input_grid: List[List[int]], transformation_type: str
    ) -> List[List[int]]:
        """Apply a simple grid-level transformation."""

        if not input_grid or not input_grid[0]:
            self.logger.warning("Empty grid supplied to transform_grid")
            return input_grid

        rows, cols = len(input_grid), len(input_grid[0])

        match transformation_type.lower():
            case "rotate":
                # 90-degree clockwise rotation
                output = [
                    [input_grid[rows - 1 - j][i] for j in range(rows)]
                    for i in range(cols)
                ]
            case "mirror":
                # Horizontal mirror
                output = [row[::-1] for row in input_grid]
            case "invert":
                # Binary invert (0 ↔ 1)
                output = [[1 - val for val in row] for row in input_grid]
            case _:
                self.logger.error("Unknown transformation type %s", transformation_type)
                return input_grid

        self.logger.debug("Applied %s transformation", transformation_type)
        self._publish_mesh_event("grid_transformed", {"type": transformation_type})
        return output

    def detect_patterns(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Return a list of recognised patterns in *grid*."""

        patterns: List[Dict[str, Any]] = []
        if not grid or not grid[0]:
            self.logger.warning("Empty grid supplied to detect_patterns")
            return patterns

        rows, cols = len(grid), len(grid[0])

        # Symmetry checks
        horizontal_sym = all(grid[i] == grid[rows - 1 - i] for i in range(rows // 2))
        vertical_sym = all(
            all(grid[r][c] == grid[r][cols - 1 - c] for r in range(rows))
            for c in range(cols // 2)
        )

        if horizontal_sym:
            patterns.append(
                {"type": "symmetry", "axis": "horizontal", "confidence": 1.0}
            )
        if vertical_sym:
            patterns.append({"type": "symmetry", "axis": "vertical", "confidence": 1.0})

        # Density
        density = sum(sum(row) for row in grid) / (rows * cols)
        patterns.append(
            {
                "type": "density",
                "value": density,
                "distribution": (
                    "uniform"
                    if 0.2 < density < 0.8
                    else "sparse"
                    if density <= 0.2
                    else "dense"
                ),
            }
        )

        self.logger.debug("detect_patterns found %d pattern(s)", len(patterns))
        self._publish_mesh_event("grid_patterns_detected", {"count": len(patterns)})
        return patterns

    def generate_grid(
        self, pattern_spec: Dict[str, Any], size: Tuple[int, int]
    ) -> List[List[int]]:
        """Generate a new grid given a *pattern_spec* and *size* (rows, cols)."""

        rows, cols = size
        density: float = pattern_spec.get("density", 0.5)
        symmetry: str = pattern_spec.get("symmetry", "none")

        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # Determine the editable quadrant depending on symmetry type
        edit_rows = rows // 2 if symmetry in {"horizontal", "both"} else rows
        edit_cols = cols // 2 if symmetry in {"vertical", "both"} else cols

        for r in range(edit_rows):
            for c in range(edit_cols):
                if random.random() < density:
                    grid[r][c] = 1

        # Apply symmetry reflections
        if symmetry in {"horizontal", "both"}:
            for r in range(edit_rows):
                grid[rows - 1 - r][:edit_cols] = grid[r][:edit_cols]

        if symmetry in {"vertical", "both"}:
            for r in range(rows):
                for c in range(edit_cols):
                    grid[r][cols - 1 - c] = grid[r][c]

        self.logger.debug("Generated %dx%d grid with %s", rows, cols, pattern_spec)
        self._publish_mesh_event("grid_generated", {"rows": rows, "cols": cols})
        return grid

    # ───────────────────────────────────────────────
    # ⚙️  Mesh-Ready Helper Features
    # ───────────────────────────────────────────────
    def _publish_mesh_event(self, topic: str, payload: Dict[str, Any]) -> None:  # ⚙️ 1
        """Emit an event onto VantaCore’s event bus (if available)."""
        try:
            if hasattr(self.vanta_core, "emit_event"):
                self.vanta_core.emit_event(topic, payload | {"source": "grid_former"})
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to publish mesh event %s: %s", topic, exc)

    def on_mesh_command(self, command: Dict[str, Any]) -> Any:  # ⚙️ 2
        """Generic handler so the mesh can remote-control this module."""
        ctype = command.get("type")
        if ctype == "transform":
            return self.transform_grid(command["grid"], command["mode"])
        if ctype == "detect":
            return self.detect_patterns(command["grid"])
        if ctype == "generate":
            return self.generate_grid(command["spec"], tuple(command["size"]))
        self.logger.warning("Unknown mesh command %s", ctype)

    def heartbeat(self) -> Dict[str, Any]:  # ⚙️ 3
        """Cheap status ping consumed by HoloMesh health monitors."""
        return {
            "component": "grid_former",
            "timestamp": time.time(),
            "status": "ok",
            "load_estimate": self._synthetic_load_estimate(),
        }

    def to_sigil_repr(self, grid: List[List[int]]) -> str:  # ⚙️ 4
        """Return a *very* compact sigil-string encoding of the grid (for logs/UI)."""
        return "".join(
            "".join("█" if cell else "·" for cell in row) + "\n" for row in grid
        )

    async def async_process_request(self, request: Dict[str, Any]) -> Any:  # ⚙️ 5
        """Coroutine wrapper so async agents can await heavy grid work."""
        # (Could off-load to a thread pool if CPU-intensive in real-life.)
        return self.on_mesh_command(request)

    # ───────────────────────────────────────────────
    # Internals
    # ───────────────────────────────────────────────
    def _synthetic_load_estimate(self) -> float:
        """Placeholder load metric until real telemetry wired."""
        return random.uniform(0.05, 0.25)
