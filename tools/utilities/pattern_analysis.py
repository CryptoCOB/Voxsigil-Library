#!/usr/bin/env python3
"""
Pattern Analysis Utilities for VoxSigil System
Provides pattern recognition, analysis, and matching functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class PatternAnalysis:
    """
    Advanced pattern analysis system for VoxSigil.
    Handles pattern recognition, matching, and analysis across various data types.
    """

    def __init__(self):
        self.pattern_cache = {}
        self.similarity_threshold = 0.8
        self.pattern_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, Any]:
        """Initialize common pattern templates."""
        return {
            "spatial": {
                "symmetry": ["horizontal", "vertical", "diagonal", "rotational"],
                "shapes": ["rectangle", "square", "line", "cross", "L-shape"],
                "arrangements": ["grid", "cluster", "linear", "scattered"],
            },
            "color": {
                "gradients": ["increasing", "decreasing", "alternating"],
                "distributions": ["uniform", "clustered", "random"],
                "relationships": ["complementary", "analogous", "triadic"],
            },
            "transformation": {
                "geometric": ["rotation", "reflection", "translation", "scaling"],
                "logical": ["inversion", "completion", "extension", "filtering"],
                "compositional": ["addition", "subtraction", "intersection", "union"],
            },
        }

    def analyze_grid_pattern(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        Analyze patterns in a grid structure.

        Args:
            grid: 2D grid to analyze

        Returns:
            Pattern analysis results
        """
        if not grid or not grid[0]:
            return {"error": "Empty grid"}

        analysis = {
            "dimensions": (len(grid), len(grid[0])),
            "symmetries": self._detect_symmetries(grid),
            "shapes": self._detect_shapes(grid),
            "color_patterns": self._analyze_color_patterns(grid),
            "spatial_relationships": self._analyze_spatial_relationships(grid),
            "complexity_score": self._calculate_complexity_score(grid),
        }

        return analysis

    def _detect_symmetries(self, grid: List[List[int]]) -> List[str]:
        """Detect symmetries in the grid."""
        symmetries = []

        # Horizontal symmetry
        if self._is_horizontally_symmetric(grid):
            symmetries.append("horizontal")

        # Vertical symmetry
        if self._is_vertically_symmetric(grid):
            symmetries.append("vertical")

        # Rotational symmetry
        if self._is_rotationally_symmetric(grid):
            symmetries.append("rotational")

        # Diagonal symmetries
        if self._is_diagonally_symmetric(grid, "main"):
            symmetries.append("main_diagonal")

        if self._is_diagonally_symmetric(grid, "anti"):
            symmetries.append("anti_diagonal")

        return symmetries

    def _is_horizontally_symmetric(self, grid: List[List[int]]) -> bool:
        """Check if grid is horizontally symmetric."""
        height = len(grid)
        for i in range(height // 2):
            if grid[i] != grid[height - 1 - i]:
                return False
        return True

    def _is_vertically_symmetric(self, grid: List[List[int]]) -> bool:
        """Check if grid is vertically symmetric."""
        for row in grid:
            if row != row[::-1]:
                return False
        return True

    def _is_rotationally_symmetric(self, grid: List[List[int]]) -> bool:
        """Check if grid has 180-degree rotational symmetry."""
        height, width = len(grid), len(grid[0])

        for i in range(height):
            for j in range(width):
                if grid[i][j] != grid[height - 1 - i][width - 1 - j]:
                    return False
        return True

    def _is_diagonally_symmetric(self, grid: List[List[int]], diagonal_type: str) -> bool:
        """Check if grid is symmetric along a diagonal."""
        height, width = len(grid), len(grid[0])

        if height != width:  # Can only check diagonal symmetry for square grids
            return False

        if diagonal_type == "main":
            # Main diagonal (top-left to bottom-right)
            for i in range(height):
                for j in range(width):
                    if grid[i][j] != grid[j][i]:
                        return False
        elif diagonal_type == "anti":
            # Anti-diagonal (top-right to bottom-left)
            for i in range(height):
                for j in range(width):
                    if grid[i][j] != grid[width - 1 - j][height - 1 - i]:
                        return False

        return True

    def _detect_shapes(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect shapes and objects in the grid."""
        shapes = []
        height, width = len(grid), len(grid[0])
        visited = [[False] * width for _ in range(height)]

        for i in range(height):
            for j in range(width):
                if not visited[i][j] and grid[i][j] != 0:  # Assuming 0 is background
                    shape = self._extract_connected_component(grid, i, j, visited)
                    if shape:
                        shape_info = self._analyze_shape(shape)
                        shapes.append(shape_info)

        return shapes

    def _extract_connected_component(
        self, grid: List[List[int]], start_i: int, start_j: int, visited: List[List[bool]]
    ) -> List[Tuple[int, int]]:
        """Extract connected component using flood fill."""
        height, width = len(grid), len(grid[0])
        color = grid[start_i][start_j]
        component = []
        stack = [(start_i, start_j)]

        while stack:
            i, j = stack.pop()
            if i < 0 or i >= height or j < 0 or j >= width or visited[i][j] or grid[i][j] != color:
                continue

            visited[i][j] = True
            component.append((i, j))

            # Add neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((i + di, j + dj))

        return component

    def _analyze_shape(self, component: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze a connected component to determine shape properties."""
        if not component:
            return {}

        # Calculate bounding box
        min_i = min(pos[0] for pos in component)
        max_i = max(pos[0] for pos in component)
        min_j = min(pos[1] for pos in component)
        max_j = max(pos[1] for pos in component)

        width = max_j - min_j + 1
        height = max_i - min_i + 1
        area = len(component)
        bounding_area = width * height

        shape_info = {
            "area": area,
            "bounding_box": (min_i, min_j, max_i, max_j),
            "width": width,
            "height": height,
            "density": area / bounding_area if bounding_area > 0 else 0,
            "type": self._classify_shape(component, width, height, area),
        }

        return shape_info

    def _classify_shape(
        self, component: List[Tuple[int, int]], width: int, height: int, area: int
    ) -> str:
        """Classify the shape based on its properties."""
        bounding_area = width * height
        density = area / bounding_area if bounding_area > 0 else 0

        # Rectangle/square detection
        if density > 0.9:
            if abs(width - height) <= 1:
                return "square"
            else:
                return "rectangle"

        # Line detection
        if width == 1 or height == 1:
            return "line"

        # L-shape detection (simplified)
        if density < 0.7 and (width > 2 and height > 2):
            return "L-shape"

        # Cross detection (simplified)
        if density < 0.6 and width > 2 and height > 2:
            return "cross"

        return "irregular"

    def _analyze_color_patterns(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze color usage patterns in the grid."""
        height, width = len(grid), len(grid[0])
        color_counts = {}
        color_positions = {}

        # Count colors and track positions
        for i in range(height):
            for j in range(width):
                color = grid[i][j]
                color_counts[color] = color_counts.get(color, 0) + 1
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((i, j))

        # Analyze patterns
        total_cells = height * width
        analysis = {
            "unique_colors": len(color_counts),
            "color_distribution": {k: v / total_cells for k, v in color_counts.items()},
            "dominant_color": max(color_counts, key=color_counts.get),
            "color_entropy": self._calculate_color_entropy(color_counts, total_cells),
            "spatial_clustering": self._analyze_color_clustering(color_positions),
        }

        return analysis

    def _calculate_color_entropy(self, color_counts: Dict[int, int], total_cells: int) -> float:
        """Calculate color entropy (diversity measure)."""
        if total_cells == 0:
            return 0.0

        entropy = 0.0
        for count in color_counts.values():
            if count > 0:
                p = count / total_cells
                entropy -= p * np.log2(p) if p > 0 else 0

        return entropy

    def _analyze_color_clustering(
        self, color_positions: Dict[int, List[Tuple[int, int]]]
    ) -> Dict[str, float]:
        """Analyze spatial clustering of colors."""
        clustering = {}

        for color, positions in color_positions.items():
            if len(positions) < 2:
                clustering[color] = 1.0  # Single cell is perfectly clustered
                continue

            # Calculate average distance between positions
            total_distance = 0
            count = 0

            for i, pos1 in enumerate(positions):
                for pos2 in positions[i + 1 :]:
                    distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                    total_distance += distance
                    count += 1

            avg_distance = total_distance / count if count > 0 else 0
            # Normalize clustering score (inverse of average distance)
            clustering[color] = 1.0 / (1.0 + avg_distance)

        return clustering

    def _analyze_spatial_relationships(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between objects."""
        shapes = self._detect_shapes(grid)
        relationships = []

        for i, shape1 in enumerate(shapes):
            for j, shape2 in enumerate(shapes[i + 1 :], i + 1):
                relationship = self._calculate_relationship(shape1, shape2)
                relationships.append({"shape1_id": i, "shape2_id": j, "relationship": relationship})

        return relationships

    def _calculate_relationship(
        self, shape1: Dict[str, Any], shape2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate spatial relationship between two shapes."""
        bbox1 = shape1.get("bounding_box", (0, 0, 0, 0))
        bbox2 = shape2.get("bounding_box", (0, 0, 0, 0))

        # Calculate centers
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        # Calculate distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

        # Determine relative position
        dx = center2[1] - center1[1]  # x-axis difference
        dy = center2[0] - center1[0]  # y-axis difference

        if abs(dx) > abs(dy):
            position = "right" if dx > 0 else "left"
        else:
            position = "below" if dy > 0 else "above"

        return {
            "distance": distance,
            "relative_position": position,
            "overlap": self._check_overlap(bbox1, bbox2),
        }

    def _check_overlap(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two bounding boxes overlap."""
        return not (
            bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]
        )

    def _calculate_complexity_score(self, grid: List[List[int]]) -> float:
        """Calculate overall complexity score for the grid."""
        height, width = len(grid), len(grid[0])

        # Size complexity
        size_complexity = min((height * width) / 900, 1.0)  # Normalize to max 30x30

        # Color complexity
        unique_colors = len(set(sum(grid, [])))
        color_complexity = min(unique_colors / 10, 1.0)  # Normalize to max 10 colors

        # Shape complexity
        shapes = self._detect_shapes(grid)
        shape_complexity = min(len(shapes) / 10, 1.0)  # Normalize to max 10 shapes

        # Symmetry bonus (lower complexity for symmetric patterns)
        symmetries = self._detect_symmetries(grid)
        symmetry_bonus = len(symmetries) * 0.1

        # Overall complexity
        complexity = (size_complexity + color_complexity + shape_complexity) / 3
        complexity = max(0, complexity - symmetry_bonus)  # Apply symmetry bonus

        return complexity

    def compare_patterns(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """
        Compare two patterns and return similarity score.

        Args:
            pattern1: First pattern analysis
            pattern2: Second pattern analysis

        Returns:
            Similarity score between 0 and 1
        """
        if not pattern1 or not pattern2:
            return 0.0

        similarity_scores = []

        # Compare dimensions
        dim1 = pattern1.get("dimensions", (0, 0))
        dim2 = pattern2.get("dimensions", (0, 0))

        if dim1[0] > 0 and dim1[1] > 0 and dim2[0] > 0 and dim2[1] > 0:
            dim_similarity = 1.0 - abs(dim1[0] - dim2[0]) / max(dim1[0], dim2[0])
            dim_similarity *= 1.0 - abs(dim1[1] - dim2[1]) / max(dim1[1], dim2[1])
            similarity_scores.append(dim_similarity)

        # Compare symmetries
        sym1 = set(pattern1.get("symmetries", []))
        sym2 = set(pattern2.get("symmetries", []))

        if sym1 or sym2:
            sym_similarity = len(sym1.intersection(sym2)) / len(sym1.union(sym2))
            similarity_scores.append(sym_similarity)

        # Compare color patterns
        color1 = pattern1.get("color_patterns", {})
        color2 = pattern2.get("color_patterns", {})

        if color1 and color2:
            color_similarity = self._compare_color_patterns(color1, color2)
            similarity_scores.append(color_similarity)

        # Compare complexity
        comp1 = pattern1.get("complexity_score", 0)
        comp2 = pattern2.get("complexity_score", 0)

        if comp1 > 0 or comp2 > 0:
            comp_similarity = 1.0 - abs(comp1 - comp2)
            similarity_scores.append(comp_similarity)

        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    def _compare_color_patterns(self, color1: Dict[str, Any], color2: Dict[str, Any]) -> float:
        """Compare color patterns between two analyses."""
        # Compare number of unique colors
        colors1 = color1.get("unique_colors", 0)
        colors2 = color2.get("unique_colors", 0)

        if colors1 == 0 and colors2 == 0:
            return 1.0

        color_sim = (
            1.0 - abs(colors1 - colors2) / max(colors1, colors2) if max(colors1, colors2) > 0 else 0
        )

        # Compare entropy
        entropy1 = color1.get("color_entropy", 0)
        entropy2 = color2.get("color_entropy", 0)

        if entropy1 > 0 or entropy2 > 0:
            entropy_sim = 1.0 - abs(entropy1 - entropy2) / max(entropy1, entropy2, 1)
        else:
            entropy_sim = 1.0

        return (color_sim + entropy_sim) / 2

    def find_pattern_matches(
        self,
        target_pattern: Dict[str, Any],
        pattern_database: List[Dict[str, Any]],
        threshold: float = None,
    ) -> List[Tuple[int, float]]:
        """
        Find matching patterns in a database.

        Args:
            target_pattern: Pattern to match
            pattern_database: Database of patterns to search
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples
        """
        if threshold is None:
            threshold = self.similarity_threshold

        matches = []

        for i, pattern in enumerate(pattern_database):
            similarity = self.compare_patterns(target_pattern, pattern)
            if similarity >= threshold:
                matches.append((i, similarity))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def extract_transformation_pattern(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Extract transformation pattern between input and output grids.

        Args:
            input_grid: Input grid
            output_grid: Output grid

        Returns:
            Transformation pattern analysis
        """
        input_analysis = self.analyze_grid_pattern(input_grid)
        output_analysis = self.analyze_grid_pattern(output_grid)

        transformation = {
            "input_pattern": input_analysis,
            "output_pattern": output_analysis,
            "dimension_change": self._analyze_dimension_change(input_grid, output_grid),
            "color_mapping": self._analyze_color_mapping(input_grid, output_grid),
            "spatial_transformation": self._analyze_spatial_transformation(input_grid, output_grid),
            "complexity_change": output_analysis.get("complexity_score", 0)
            - input_analysis.get("complexity_score", 0),
        }

        return transformation

    def _analyze_dimension_change(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Dict[str, Any]:
        """Analyze changes in grid dimensions."""
        input_dim = (len(input_grid), len(input_grid[0]) if input_grid else 0)
        output_dim = (len(output_grid), len(output_grid[0]) if output_grid else 0)

        return {
            "input_dimensions": input_dim,
            "output_dimensions": output_dim,
            "size_change": output_dim[0] * output_dim[1] - input_dim[0] * input_dim[1],
            "aspect_ratio_change": (output_dim[1] / output_dim[0] if output_dim[0] > 0 else 0)
            - (input_dim[1] / input_dim[0] if input_dim[0] > 0 else 0),
        }

    def _analyze_color_mapping(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Dict[str, Any]:
        """Analyze color transformations between grids."""
        input_colors = set(sum(input_grid, []))
        output_colors = set(sum(output_grid, []))

        return {
            "input_colors": list(input_colors),
            "output_colors": list(output_colors),
            "new_colors": list(output_colors - input_colors),
            "removed_colors": list(input_colors - output_colors),
            "preserved_colors": list(input_colors.intersection(output_colors)),
        }

    def _analyze_spatial_transformation(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Dict[str, Any]:
        """Analyze spatial transformations between grids."""
        # Simplified spatial transformation analysis
        if len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
            # Same size - check for geometric transformations
            if self._is_rotation(input_grid, output_grid):
                return {
                    "type": "rotation",
                    "angle": self._detect_rotation_angle(input_grid, output_grid),
                }
            elif self._is_reflection(input_grid, output_grid):
                return {
                    "type": "reflection",
                    "axis": self._detect_reflection_axis(input_grid, output_grid),
                }
            else:
                return {
                    "type": "modification",
                    "changes": self._count_cell_changes(input_grid, output_grid),
                }
        else:
            return {
                "type": "resize",
                "scale_factor": self._calculate_scale_factor(input_grid, output_grid),
            }

    def _is_rotation(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a rotation of grid1."""
        # Simplified rotation detection
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False

        # Check 90-degree rotations
        for angle in [90, 180, 270]:
            rotated = self._rotate_grid(grid1, angle)
            if rotated == grid2:
                return True

        return False

    def _is_reflection(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if grid2 is a reflection of grid1."""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False

        # Check horizontal reflection
        if grid1[::-1] == grid2:
            return True

        # Check vertical reflection
        if [row[::-1] for row in grid1] == grid2:
            return True

        return False

    def _rotate_grid(self, grid: List[List[int]], angle: int) -> List[List[int]]:
        """Rotate grid by specified angle (90, 180, 270 degrees)."""
        if angle == 90:
            return [
                [grid[len(grid) - 1 - j][i] for j in range(len(grid))] for i in range(len(grid[0]))
            ]
        elif angle == 180:
            return [row[::-1] for row in grid[::-1]]
        elif angle == 270:
            return [
                [grid[j][len(grid[0]) - 1 - i] for j in range(len(grid))]
                for i in range(len(grid[0]))
            ]
        else:
            return grid

    def _detect_rotation_angle(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """Detect rotation angle between grids."""
        for angle in [90, 180, 270]:
            if self._rotate_grid(grid1, angle) == grid2:
                return angle
        return 0

    def _detect_reflection_axis(self, grid1: List[List[int]], grid2: List[List[int]]) -> str:
        """Detect reflection axis between grids."""
        if grid1[::-1] == grid2:
            return "horizontal"
        elif [row[::-1] for row in grid1] == grid2:
            return "vertical"
        return "unknown"

    def _count_cell_changes(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """Count number of cell changes between grids."""
        changes = 0
        for i in range(min(len(grid1), len(grid2))):
            for j in range(min(len(grid1[0]), len(grid2[0]))):
                if grid1[i][j] != grid2[i][j]:
                    changes += 1
        return changes

    def _calculate_scale_factor(
        self, grid1: List[List[int]], grid2: List[List[int]]
    ) -> Tuple[float, float]:
        """Calculate scale factor between grids."""
        h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
        h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0

        scale_h = h2 / h1 if h1 > 0 else 0
        scale_w = w2 / w1 if w1 > 0 else 0

        return (scale_h, scale_w)
